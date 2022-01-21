import torch
from torch import nn
from dataset import GenerateIterator, GenerateIterator_eval
from myargs import args
import numpy as np
from tqdm import tqdm
from models import Classifier, Discriminator, EMA
from vat import VAT, ConditionalEntropyLoss
import torch.nn.functional as F
import random
import check_img
seed = args.seed
print("seed: ", seed, " , orth: ", args.orth, " , source: ", args.source, ", dann:", args.dann, ", batch size:", args.batch_size)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# discriminator network
feature_discriminator = Discriminator(large=args.large).cuda()
# classifier network.
classifier = Classifier(large=args.large).cuda()
# state_dict = torch.load(f'./checkpoint/epoch_14_source_False_orth_True_r_{args.r}')['classifier_dict']
# classifier.load_state_dict(state_dict)
# set the midpoint
midpoint = 21 if args.src == 'signs' or args.tgt == 'signs' else 5
if args.src == 'cifar' or args.src == 'stl':
    midpoint = 4
# loss functions
cent = ConditionalEntropyLoss().cuda()
r_ = torch.zeros(midpoint * 2)
if args.src == 'signs' or args.tgt == 'signs':
    r_ = torch.zeros(43)
if args.src == 'cifar' or args.src == 'stl':
    r_ = torch.zeros(9)

r_[:midpoint] = args.r[::-1][0]
r_[midpoint:] = args.r[::-1][1]
xent = nn.CrossEntropyLoss(weight=r_, reduction='mean').cuda()
xent_disc = nn.CrossEntropyLoss(reduction='mean').cuda()
sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()
vat_loss = VAT(classifier).cuda()
print("mid points:", midpoint)
# optimizer.
optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# datasets.
iterator_train = GenerateIterator(args)
iterator_val = GenerateIterator_eval(args)

# loss params.
dw = 1e-2
cw = 1
sw = 1
tw = 1e-2
bw = 1e-2
pre_epoch = 15
if args.src == 'cifar' or args.tgt == 'cifar':
    dw = args.dw
    tw = 1e-1
    sw = 0 if args.src == 'stl' else 1
    pre_epoch = 15
if args.src == 'signs' or args.tgt == 'signs':
    dw = 1e-2
    sw = 1
    tw = 1e-2
    bw = 1e-2
    pre_epoch = 15
if args.src == 'digits' or args.tgt == 'digits':
    dw = 1e-2
    sw = 1
    tw = 1e-1
    bw = 1e-2
    pre_epoch = 1
    args.num_epoch = 10
if args.src == 'mnistm' or args.tgt == 'mnistm':
    dw = 1e-1
    sw = 0
    state_dict = torch.load(f'./checkpoint/epoch_14_source_False_orth_True_r_{args.r}')['classifier_dict']
    classifier.load_state_dict(state_dict)
    pre_epoch = 1
if args.src == 'svhn' or args.tgt == 'mnist':
    sw = 0
if args.dann:
    sw = 0
    tw = 0
    bw = 0
print(f"dw:{dw}, cw:{cw}, sw:{sw}, tw:{tw}, bw:{bw}, pepoch:{pre_epoch}")

''' Exponential moving average (simulating teacher model) '''
ema = EMA(0.998)
ema.register(classifier)

# training..
for epoch in range(1, args.num_epoch+1):
    iterator_train.dataset.shuffledata()
    pbar = tqdm(iterator_train, disable=False,
                bar_format="{percentage:.0f}%,{elapsed},{remaining},{desc}")

    loss_main_sum, n_total = 0, 0
    loss_domain_sum, loss_src_class_sum, \
    loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
    loss_disc_sum = 0

    for images_source, labels_source, images_target, labels_target in pbar:

        images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()
        if args.balance:
            img_s_0 = images_source[labels_source < midpoint]
            img_s_1 = images_source[labels_source >= midpoint]
            img_t_0 = images_target[labels_target < midpoint]
            img_t_1 = images_target[labels_target >= midpoint]
            label_s_0 = labels_source[labels_source < midpoint]
            label_s_1 = labels_source[labels_source >= midpoint]
            label_t_0 = labels_target[labels_target < midpoint]
            label_t_1 = labels_target[labels_target >= midpoint]
            length = min(len(img_s_0), len(img_t_0), len(img_t_1), len(img_s_1))
            images_source = torch.cat((img_s_0[:length], img_s_1[:length]), dim=0)
            images_target = torch.cat((img_t_0[:length], img_t_1[:length]), dim=0)
            labels_source = torch.cat((label_s_0[:length], label_s_1[:length]), dim=0)
            labels_target = torch.cat((label_t_0[:length], label_t_1[:length]), dim=0)

        if args.source:
            ' Classifier losses setup. '
            # supervised/source classification.
            feats_source, pred_source = classifier(images_source)
            feats_target, pred_target = classifier(images_target, track_bn=True)
            loss_src_class = xent(pred_source, labels_source)

            # combined loss.
            loss_main = (
                    cw * loss_src_class
            )
            # Update classifier.
            optimizer_cls.zero_grad()
            loss_main.backward()
            optimizer_cls.step()
            # # Polyak averaging.
            # ema(classifier)  # TODO: move ema into the optimizer step fn.

            loss_main_sum += loss_main.item()
            n_total += 1

            pbar.set_description('loss {:.3f},'.format(
                loss_main_sum / n_total,
            ))
        else:
            # pass images through the classifier network.
            feats_source, pred_source = classifier(images_source)
            feats_target, pred_target = classifier(images_target, track_bn=True)
            ' Discriminator losses setup. '
            # discriminator loss.
            real_logit_disc = feature_discriminator(feats_source.detach())
            fake_logit_disc = feature_discriminator(feats_target.detach())

            loss_disc = 0.5 * (
                    xent_disc(real_logit_disc, torch.ones(real_logit_disc.size(0), device='cuda').long()) +
                    xent_disc(fake_logit_disc, torch.zeros(fake_logit_disc.size(0), device='cuda').long())
            )

            ' Update network(s) '
            # Update discriminator.
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            ' Classifier losses setup. '
            # supervised/source classification.
            loss_src_class = xent(pred_source, labels_source)
            # conditional entropy loss.
            loss_trg_cent = cent(pred_target)

            # virtual adversarial loss.
            loss_src_vat = vat_loss(images_source, pred_source)
            loss_trg_vat = vat_loss(images_target, pred_target)

            # domain loss.
            real_logit = feature_discriminator(feats_source)
            fake_logit = feature_discriminator(feats_target)

            # 1: src; 0: tgt
            pred_src = F.softmax(pred_source, dim=1).detach()
            pred_tgt = F.softmax(pred_target, dim=1).detach()

            # if args.r[0] > args.r[1]:
            #     pred_src = torch.cat([pred_src[:, midpoint:].sum(1, True), pred_src[:, :midpoint].sum(1, True)], dim=1)
            #     pred_tgt = torch.cat([pred_tgt[:, midpoint:].sum(1, True), pred_tgt[:, :midpoint].sum(1, True)], dim=1)
            # else:
            #     pred_src = torch.cat([pred_src[:, :midpoint].sum(1, True), pred_src[:, midpoint:].sum(1, True)], dim=1)
            #     pred_tgt = torch.cat([pred_tgt[:, :midpoint].sum(1, True), pred_tgt[:, midpoint:].sum(1, True)], dim=1)

            p_c_tgt = pred_tgt[:, :midpoint].sum() / len(pred_tgt)
            p_not_c_tgt = pred_tgt[:, midpoint:].sum() / len(pred_tgt)
            p_c_tgt_x = pred_tgt[:, :midpoint].sum(1)
            p_not_c_tgt_x = pred_tgt[:, midpoint:].sum(1)

            mask = (labels_source < midpoint).float()
            pred_src = torch.ones((len(pred_src))).cuda() * (args.r[0]/(args.r[0] + p_c_tgt)) * mask \
                       + torch.ones((len(pred_src))).cuda() * (args.r[1]/(args.r[1] + p_not_c_tgt)) * (1-mask)

            pred_src = pred_src.unsqueeze(1)
            pred_src = torch.cat([1-pred_src, pred_src], dim=1)

            pred_tgt = torch.ones((len(pred_src))).cuda() * (args.r[0]/(args.r[0] + p_c_tgt)) * p_c_tgt_x \
                       + torch.ones((len(pred_src))).cuda() * (args.r[1]/(args.r[1] + p_not_c_tgt)) * p_not_c_tgt_x
            pred_tgt = pred_tgt.unsqueeze(1)
            pred_tgt = torch.cat([1-pred_tgt, pred_tgt], dim=1)

            correct_src = torch.ones(real_logit.size(0)).cuda().eq(pred_src.max(1)[1]).sum() / len(real_logit)
            correct_tgt = torch.zeros(fake_logit.size(0)).cuda().eq(pred_tgt.max(1)[1]).sum() / len(fake_logit)

            if args.orth and epoch >= pre_epoch:
                # # 1: src; 0: tgt
                # domain_src = torch.sigmoid(real_logit)
                # #print("before:", domain_src.size())
                # domain_src = torch.cat([1-domain_src, domain_src], dim=1)
                # domain_tgt = torch.sigmoid(fake_logit)
                # domain_tgt = torch.cat([1-domain_tgt, domain_tgt], dim=1)
                domain_src = torch.softmax(real_logit, dim=1)
                #print("before:", domain_src)
                domain_src = domain_src / (pred_src + 1e-3)
                domain_src = domain_src / domain_src.sum(1, True)
                #print("after:", domain_src)
                #real_logit = domain_src[:, 1].unsqueeze(1)
                real_logit = torch.log(domain_src+1e-3)

                domain_tgt = torch.softmax(fake_logit, dim=1)
                domain_tgt = domain_tgt / (pred_tgt + 1e-3)
                domain_tgt = domain_tgt / domain_tgt.sum(1, True)
                #fake_logit = domain_tgt[:, 1].unsqueeze(1)
                fake_logit = torch.log(domain_tgt + 1e-3)


            loss_domain = 0.5 * (
                    xent_disc(real_logit, torch.zeros(real_logit.size(0), device='cuda').long()) +
                    xent_disc(fake_logit, torch.ones(fake_logit.size(0), device='cuda').long())
            )

            # combined loss.
            loss_main = (
                    dw * loss_domain +
                    cw * loss_src_class +
                    sw * loss_src_vat +
                    tw * loss_trg_cent +
                    tw * loss_trg_vat
            )

            # Update classifier.
            optimizer_cls.zero_grad()
            loss_main.backward()
            optimizer_cls.step()

            # # Polyak averaging.
            # ema(classifier)  # TODO: move ema into the optimizer step fn.

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_src_class.item()
            loss_src_vat_sum += loss_src_vat.item()
            loss_trg_cent_sum += loss_trg_cent.item()
            loss_trg_vat_sum += loss_trg_vat.item()
            loss_main_sum += loss_main.item()
            loss_disc_sum += loss_disc.item()
            n_total += 1

            pbar.set_description('loss {:.3f},'
                                 # ' domain {:.3f},'
                                 # ' s cls {:.3f},'
                                 # ' s vat {:.3f},'
                                 # ' t c-ent {:.3f},'
                                 # ' t vat {:.3f},'
                                 # ' disc {:.3f}'
                                 ' src w1 {:.3f},'
                                 ' tgt w1 {:.3f}'.format(
                loss_main_sum / n_total,
                # loss_domain_sum / n_total,
                # loss_src_class_sum / n_total,
                # loss_src_vat_sum / n_total,
                # loss_trg_cent_sum / n_total,
                # loss_trg_vat_sum / n_total,
                # loss_disc_sum / n_total,
                correct_src.item(),
                correct_tgt.item()
            ))

    # validate.
    if epoch % 1 == 0:
        classifier.eval()
        feature_discriminator.eval()

        with torch.no_grad():
            preds_val, gts_val = [], []
            val_loss = 0
            for images_target, labels_target in iterator_val:
                images_target, labels_target = images_target.cuda(), labels_target.cuda()

                # cross entropy based classification
                _, pred_val = classifier(images_target)
                pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)

                preds_val.extend(pred_val)
                gts_val.extend(labels_target)

            preds_val = np.asarray(preds_val)
            gts_val = np.asarray(gts_val)

            score_cls_val = (np.mean(preds_val == gts_val)).astype(np.float)
            print('\n({}) acc. v {:.3f}\n'.format(epoch, score_cls_val))

        feature_discriminator.train()
        classifier.train()

    if epoch % 14 == 0 and args.save:
        save_dict = {
            "args": vars(args),
            "classifier_dict": classifier.state_dict(),
            "disc_dict": feature_discriminator.state_dict()
        }
        path = f'./checkpoint/epoch_{epoch}_source_{args.source}_orth_{args.orth}_r_{args.r}'
        print('Save to ...', path)
        torch.save(save_dict, path)
