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
from utils import im_weights_update
import copy

seed = args.seed
print("seed: ", seed, " , orth: ", args.orth, " , source: ", args.source, ", dann:", args.dann, ", iw:", args.iw, ", adda:", args.adda,  ", batch size:", args.batch_size)
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
sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()
vat_loss = VAT(classifier).cuda()
print("mid points:", midpoint)
# optimizer.
optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# datasets.
# iterator_train, r_s = GenerateIterator(args, r_tgt=[1,1])
# iterator_val = GenerateIterator_eval(args, r_tgt=[1,1])

iterator_train, r_s = GenerateIterator(args)
iterator_val = GenerateIterator_eval(args)
if args.iw:
    class_num = len(r_s)
    im_weights = torch.ones((class_num, 1), requires_grad=False).cuda()
    cov_mat = torch.tensor(np.zeros((class_num, class_num), dtype=np.float32),
                       requires_grad=False).cuda()
    pseudo_target_label = torch.tensor(np.zeros((class_num, 1), dtype=np.float32),
                                   requires_grad=False).cuda()

# loss params.
dw = 1e-2
cw = 1
sw = 1
tw = 1e-2
bw = 1e-2
pre_epoch = 1
start_epoch = 1
if args.src == 'cifar' or args.tgt == 'cifar':
    dw = args.dw
    tw = 1e-1
    sw = 0 if args.src == 'stl' else 1
    pre_epoch = 5
if args.src == 'signs' or args.tgt == 'signs':
    dw = 1e-2
    sw = 1
    tw = 1e-2
    bw = 1e-2
    pre_epoch = 1
if args.src == 'digits' or args.tgt == 'digits':
    dw = 1e-2
    sw = 1
    tw = 1e-2
    bw = 1e-2
    pre_epoch = 1
if args.src == 'mnistm' or args.tgt == 'mnistm':
    dw = 1e-1
    sw = 1
    # state_dict = torch.load(f'./checkpoint/epoch_14_source_False_orth_True_r_{args.r}')['classifier_dict']
    # classifier.load_state_dict(state_dict)
    pre_epoch = 0
if args.src == 'svhn' and args.tgt == 'mnist':
    sw = 0
if args.src == 'svhn2mnist' and args.tgt == 'mnist':
    pre_epoch = 10
if args.src == 'mnist' and args.tgt == 'svhn':
    dw = args.dw
if args.dann:
    sw = 0
    tw = 0
    bw = 0
print(f"dw:{dw}, cw:{cw}, sw:{sw}, tw:{tw}, bw:{bw}, pre epoch:{pre_epoch}, start epoch:{start_epoch}")

''' Exponential moving average (simulating teacher model) '''
ema = EMA(0.998)
ema.register(classifier)
best_val_acc = -1
p_c_tgt = 0.5
p_not_c_tgt = 0.5
# training..
p_t_old = torch.tensor(np.zeros((len(r_s)), dtype=np.float32),
                       requires_grad=False).cuda()
for epoch in range(args.num_epoch):
    iterator_train.dataset.shuffledata()
    cnt = 1.
    if args.iw:
        cov_mat[:] = 0.0
        pseudo_target_label[:] = 0.0
    pbar = tqdm(iterator_train, disable=False,
                bar_format="{percentage:.0f}%,{elapsed},{remaining},{desc}")

    loss_main_sum, n_total = 0, 0
    loss_domain_sum, loss_src_class_sum, \
    loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
    loss_disc_sum = 0

    p_t_now = copy.deepcopy(p_t_old)

    print("estimate pt:", p_t_now)
    p_t_old[:] = 0.0

    # setup target encoder for adda
    if epoch == args.pretrain_epoch and args.adda:
        print("copy network weights")
        classifier_tgt = Classifier(large=args.large).cuda()
        if args.resume:
            path = f'./checkpoint/epoch_{args.resume_epoch}_source_{args.source}_adda_True_r_{args.r}'
            print("load from ", path)
            state_dict = torch.load(path)['classifier_dict']
            classifier_tgt.load_state_dict(state_dict)
            classifier.load_state_dict(state_dict)
        else:
            classifier_tgt.load_state_dict(copy.deepcopy(classifier.state_dict()))
        # only update the feature extractor. The cycada repo uses a learning rate = 0.1 * pretrain learning rate
        optimizer_tgt = torch.optim.Adam(classifier_tgt.feature_extractor.parameters(), lr=args.lr,
                                             betas=(args.beta1, args.beta2))

    for images_source, labels_source, images_target, labels_target in pbar:

        images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()
        if args.balance:
            # make the dataset balance
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

            loss_main_sum += loss_main.item()
            n_total += 1

            pbar.set_description('loss {:.3f},'.format(
                loss_main_sum / n_total,
            ))
        elif args.iw:
            # pass images through the classifier network.
            feats_source, pred_source = classifier(images_source)
            feats_target, pred_target = classifier(images_target, track_bn=True)

            ys_onehot = torch.zeros(len(labels_source), class_num).cuda()
            ys_onehot.scatter_(1, labels_source.view(-1, 1), 1)
            # Compute weights on source data.
            weights = torch.mm(ys_onehot, im_weights)

            # Update weights ---
            # Compute the aggregated distribution of pseudo-label on the target domain.
            pseudo_target_label += torch.sum(
                F.softmax(pred_target, dim=1), dim=0).view(-1, 1).detach()
            # Update the covariance matrix on the source domain as well.
            cov_mat += torch.mm(F.softmax(pred_source,
                                            dim=1).transpose(1, 0), ys_onehot).detach()
            cnt += len(pred_source)

            # set up the loss function for source
            sigmoid_xent_src = nn.BCEWithLogitsLoss(weight=weights, reduction='mean').cuda()

            ' Discriminator losses setup. '
            # discriminator loss.
            real_logit_disc = feature_discriminator(feats_source.detach())
            fake_logit_disc = feature_discriminator(feats_target.detach())


            loss_disc = 0.5 * (
                    sigmoid_xent_src(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
                    sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
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

            loss_domain = 0.5 * (
                    sigmoid_xent_src(real_logit, torch.zeros_like(real_logit, device='cuda')) +
                    sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
            )
            if epoch >= start_epoch:
                # combined loss.
                loss_main = (
                        dw * loss_domain +
                        cw * loss_src_class +
                        sw * loss_src_vat +
                        tw * loss_trg_cent +
                        tw * loss_trg_vat
                )
            else:
                loss_main = cw * loss_src_class + sw * loss_src_vat + tw * loss_trg_cent + tw * loss_trg_vat


            # Update classifier.
            optimizer_cls.zero_grad()
            loss_main.backward()
            optimizer_cls.step()

            loss_main_sum += loss_main.item()
            n_total += 1

            pbar.set_description('loss {:.3f},'.format(
                loss_main_sum / n_total,
            ))
        elif args.adda:
            # pretraining stage
            if epoch < args.pretrain_epoch:
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

                loss_main_sum += loss_main.item()
                n_total += 1

                pbar.set_description('loss {:.3f},'.format(
                    loss_main_sum / n_total,
                ))
            # alignment stage
            else:
                feats_source, pred_source = classifier(images_source)
                feats_source = feats_source.detach()
                pred_source = pred_source.detach()
                feats_target, pred_target = classifier_tgt(images_target, track_bn=True)

                # discriminator loss.
                for i in range(1):
                    real_logit_disc = feature_discriminator(feats_source.detach())
                    fake_logit_disc = feature_discriminator(feats_target.detach())

                    loss_disc = 0.5 * (
                            sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
                            sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
                    )

                    # Update discriminator.
                    optimizer_disc.zero_grad()
                    loss_disc.backward()
                    optimizer_disc.step()

                # domain loss.
                real_logit = feature_discriminator(feats_source)
                fake_logit = feature_discriminator(feats_target)

                # 1: src; 0: tgt
                pred_src = F.softmax(pred_source, dim=1).detach()
                pred_tgt = F.softmax(pred_target, dim=1).detach()

                p_t_old += pred_tgt.sum(0)
                cnt += len(pred_tgt)
                pred = r_s / (r_s + p_t_now)
                pred_src = pred[labels_source.long()]

                pred_tgt = pred_tgt @ pred

                # label_t = torch.argmax(pred_tgt, dim=1)
                # pred_tgt = pred[label_t]

                pred_src = pred_src.unsqueeze(1)
                pred_src = torch.cat([1 - pred_src, pred_src], dim=1)

                pred_tgt = pred_tgt.unsqueeze(1)
                pred_tgt = torch.cat([1 - pred_tgt, pred_tgt], dim=1)
                correct_src = torch.ones_like(real_logit.squeeze()).eq(pred_src.max(1)[1]).sum() / len(real_logit)
                correct_tgt = torch.zeros_like(fake_logit.squeeze()).eq(pred_tgt.max(1)[1]).sum() / len(fake_logit)

                if args.orth and epoch >= max(pre_epoch, 1):
                    #  1: src; 0: tgt
                    domain_src = torch.sigmoid(real_logit)
                    domain_src = torch.cat([1 - domain_src, domain_src], dim=1)
                    domain_tgt = torch.sigmoid(fake_logit)
                    domain_tgt = torch.cat([1 - domain_tgt, domain_tgt], dim=1)

                    domain_src = domain_src / (pred_src + 1e-7)
                    domain_src = domain_src / domain_src.sum(1, True)
                    real_logit = domain_src[:, 1].unsqueeze(1)
                    real_logit = torch.log(real_logit + 1e-7) - torch.log(1 - real_logit + 1e-7)

                    domain_tgt = domain_tgt / (pred_tgt + 1e-7)
                    domain_tgt = domain_tgt / domain_tgt.sum(1, True)
                    fake_logit = domain_tgt[:, 1].unsqueeze(1)
                    fake_logit = torch.log(fake_logit + 1e-7) - torch.log(1 - fake_logit + 1e-7)

                loss_domain = 0.5 * (
                        sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +
                        sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
                )

                # combined loss.
                loss_main = (
                        loss_domain
                )

                # Update classifier.
                optimizer_tgt.zero_grad()
                loss_main.backward()
                optimizer_tgt.step()

                loss_domain_sum += loss_domain.item()
                loss_main_sum += loss_main.item()
                loss_disc_sum += loss_disc.item()
                n_total += 1

                pbar.set_description('loss {:.3f},'
                                     ' src w1 {:.3f},'
                                     ' tgt w1 {:.3f}'.format(
                    loss_main_sum / n_total,
                    correct_src.item(),
                    correct_tgt.item()
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
                    sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
                    sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
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

            p_t_old += pred_tgt.sum(0)
            cnt += len(pred_tgt)
            pred = r_s / (r_s + p_t_now)
            pred_src = pred[labels_source.long()]
            pred_tgt = pred_tgt @ pred

            # p_c_tgt = pred_tgt[:, :midpoint].sum() / len(pred_tgt)
            # p_not_c_tgt = pred_tgt[:, midpoint:].sum() / len(pred_tgt)
            # p_c_tgt_x = pred_tgt[:, :midpoint].sum(1)
            # p_not_c_tgt_x = pred_tgt[:, midpoint:].sum(1)
            # mask = (labels_source < midpoint).float()

            # pred_src = torch.ones((len(pred_src))).cuda() * (args.r[0]/(args.r[0] + p_c_tgt)) * mask \
            #            + torch.ones((len(pred_src))).cuda() * (args.r[1]/(args.r[1] + p_not_c_tgt)) * (1-mask)

            pred_src = pred_src.unsqueeze(1)
            pred_src = torch.cat([1-pred_src, pred_src], dim=1)

            # pred_tgt = torch.ones((len(pred_src))).cuda() * (args.r[0]/(args.r[0] + p_c_tgt)) * p_c_tgt_x \
            #            + torch.ones((len(pred_src))).cuda() * (args.r[1]/(args.r[1] + p_not_c_tgt)) * p_not_c_tgt_x
            pred_tgt = pred_tgt.unsqueeze(1)
            pred_tgt = torch.cat([1-pred_tgt, pred_tgt], dim=1)
            correct_src = torch.ones_like(real_logit.squeeze()).eq(pred_src.max(1)[1]).sum() / len(real_logit)
            correct_tgt = torch.zeros_like(fake_logit.squeeze()).eq(pred_tgt.max(1)[1]).sum() / len(fake_logit)

            if args.orth and epoch >= max(pre_epoch,1):
                #  1: src; 0: tgt
                domain_src = torch.sigmoid(real_logit)
                domain_src = torch.cat([1-domain_src, domain_src], dim=1)
                domain_tgt = torch.sigmoid(fake_logit)
                domain_tgt = torch.cat([1-domain_tgt, domain_tgt], dim=1)

                domain_src = domain_src / (pred_src + 1e-7)
                domain_src = domain_src / domain_src.sum(1, True)
                real_logit = domain_src[:, 1].unsqueeze(1)
                real_logit = torch.log(real_logit+1e-7) - torch.log(1-real_logit + 1e-7)

                domain_tgt = domain_tgt / (pred_tgt + 1e-7)
                domain_tgt = domain_tgt / domain_tgt.sum(1, True)
                fake_logit = domain_tgt[:, 1].unsqueeze(1)
                fake_logit = torch.log(fake_logit + 1e-7) - torch.log(1-fake_logit + 1e-7)


            loss_domain = 0.5 * (
                    sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +
                    sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
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
                correct_src.item(),
                correct_tgt.item()
            ))


    p_t_old /= cnt

    if args.iw:
        pseudo_target_label /= cnt
        cov_mat /= cnt
        # Recompute the importance weight by solving a QP.
        im_weights = im_weights_update(r_s.cpu().detach().numpy(), pseudo_target_label.cpu().detach().numpy(), cov_mat.cpu().detach().numpy(),
                                        im_weights, ma=0.5)
    # validate.
    if epoch % 1 == 0:

        def eval(cls, feature):

            global best_val_acc
            cls.eval()
            feature.eval()
            with torch.no_grad():
                preds_val, gts_val = [], []
                val_loss = 0
                for images_target, labels_target in iterator_val:
                    images_target, labels_target = images_target.cuda(), labels_target.cuda()

                    # cross entropy based classification
                    _, pred_val = cls(images_target)
                    pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)

                    preds_val.extend(pred_val)
                    gts_val.extend(labels_target)

                preds_val = np.asarray(preds_val)
                gts_val = np.asarray(gts_val)

                score_cls_val = (np.mean(preds_val == gts_val)).astype(np.float)
                best_val_acc = max(score_cls_val, best_val_acc)
                print('\n({}) acc. v {:.3f}, best: {:.3f}\n'.format(epoch, score_cls_val, best_val_acc))

            feature.train()
            cls.train()

        if args.adda and epoch >= args.pretrain_epoch:
            eval(classifier_tgt, feature_discriminator)
        else:
            eval(classifier, feature_discriminator)

    if epoch == args.pretrain_epoch - 1 and args.save:
        save_dict = {
            "args": vars(args),
            "classifier_dict": classifier.state_dict(),
            "disc_dict": feature_discriminator.state_dict()
        }
        #path = f'./checkpoint/epoch_{epoch}_source_{args.source}_orth_{args.orth}_r_{args.r}'
        path = f'./checkpoint/epoch_{epoch}_source_{args.source}_adda_{args.adda}_r_{args.r}'
        print('Save to ...', path)
        torch.save(save_dict, path)
