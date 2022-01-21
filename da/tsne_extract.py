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

seed = args.seed
print("seed: ", seed, " , orth: ", args.orth, " , source: ", args.source)
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
state_dict = torch.load(f'./checkpoint/epoch_50_source_False_orth_{args.orth}_r_{args.r}')['classifier_dict']
classifier.load_state_dict(state_dict)
# set the midpoint
midpoint = 21 if args.src == 'signs' or args.tgt == 'signs' else 5
# loss functions
cent = ConditionalEntropyLoss().cuda()
r_ = torch.zeros(midpoint * 2)
if args.src == 'signs' or args.tgt == 'signs':
    r_ = torch.zeros(43)
r_[:midpoint] = args.r[::-1][0]
r_[midpoint:] = args.r[::-1][1]
xent = nn.CrossEntropyLoss(weight=r_, reduction='mean').cuda()
sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()
vat_loss = VAT(classifier).cuda()

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
if args.src == 'signs' or args.tgt == 'signs':
    dw = 1e-2
    sw = 1
    tw = 1e-2
    bw = 1e-2
    pre_epoch = 15
if args.src == 'digits' or args.tgt == 'digits':
    dw = 1e-2
    sw = 1
    tw = 1e-2
    bw = 1e-2
    pre_epoch = 1
if args.src == 'mnistm' or args.tgt == 'mnistm':
    dw = 1e-1
    sw = 0
if args.src == 'svhn' or args.tgt == 'mnist':
    sw = 0

print(f"dw:{dw}, cw:{cw}, sw:{sw}, tw:{tw}, bw:{bw}, pepoch:{pre_epoch}")

''' Exponential moving average (simulating teacher model) '''
ema = EMA(0.998)
ema.register(classifier)

# training..
src_mat = np.empty((8192, 4096))
tgt_mat = np.empty((8192, 4096))
scr_label = np.empty((8192))
tgt_label = np.empty((8192))
cnt = 0
for epoch in range(1):
    iterator_train.dataset.shuffledata()
    pbar = tqdm(iterator_train, disable=False,
                bar_format="{percentage:.0f}%,{elapsed},{remaining},{desc}")
    classifier.eval()
    for images_source, labels_source, images_target, labels_target in pbar:
        images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()

        feats_source, pred_source = classifier(images_source)
        feats_target, pred_target = classifier(images_target)
        length = len(images_source)
        feats_source = feats_source.view(feats_source.size(0),-1).detach().cpu().numpy()
        feats_target = feats_target.view(feats_target.size(0),-1).detach().cpu().numpy()
        labels_source = labels_source.cpu().numpy()
        labels_target = labels_target.cpu().numpy()

        src_mat[cnt: cnt+length] = feats_source
        tgt_mat[cnt: cnt+length] = feats_target
        scr_label[cnt: cnt+length] = labels_source
        tgt_label[cnt: cnt+length] = labels_target

        cnt += length
        if cnt >= 8192:
            break
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
    np.save('src_mat_' + str(args.orth), src_mat)
    np.save('tgt_mat_' + str(args.orth), tgt_mat)
    np.save('src_label_' + str(args.orth), scr_label)
    np.save('tgt_label_' + str(args.orth), tgt_label)



