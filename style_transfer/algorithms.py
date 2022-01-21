# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import copy
import numpy as np
from inv_hvp import neum, sto
import networks
from misc import random_pairs_of_minibatches, ParamDict
import torch.nn.utils.prune as prune

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'Fish',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'EC',
    'Swap',
    'TRM',
    'TRM2',
    'TRM_DRO'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(4)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(4)]
        if self.hparams['opt'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    @staticmethod
    def weighted_cross_entropy_loss(x, y, weights=None):
        if weights is None:
            weights = torch.ones(y.size()).cuda()
        x = F.softmax(x, dim=1)
        loss = - weights * ((1-y) * torch.log(x[:, 0]) + y * torch.log(x[:, 1]))

        return torch.sum(loss) / len(x)

    def update(self, minibatches, **kwargs):
        self.network.train()
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        feature = self.featurizer(all_x)
        if 'weights' in kwargs.keys():
            loss = self.weighted_cross_entropy_loss(self.classifier(feature), all_y, weights=kwargs['weights'])
        else:
            loss = F.cross_entropy(self.classifier(feature), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)

    def predict_feature(self, x):
        return self.featurizer(x)

    def predict_classifier(self, feature):
        return self.classifier(feature)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


class EC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(EC, self).__init__(input_shape, num_classes, num_domains,
                                 hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.latent_dim = 64
        self.input_dim = self.featurizer.n_outputs
        self.num_domains = num_domains
        del self.featurizer

        self.classifiers = [torch.nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), nn.ReLU(True),
                                                nn.Linear(self.latent_dim, num_classes)).cuda() for i in
                            range(num_domains)]
        self.optimizer = [torch.optim.Adam(
            self.classifiers[i].parameters(),
            lr=1e-2,
            weight_decay=self.hparams['weight_decay']
        ) for i in range(num_domains)]

    def update_ec(self, minibatches, feature):
        features = feature.detach()
        start = 0
        for i in range(self.num_domains):
            loss = F.cross_entropy(self.classifiers[i](features[start: start + minibatches[i][1].size(0)]),
                                   minibatches[i][1])
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            start += minibatches[i][1].size(0)

    def predict_envs(self, env, x):
        return self.classifiers[env](x)

    def predict(self, x):
        pass


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        if self.hparams["opt"] == 'Adam':
            self.disc_opt = torch.optim.Adam(
                (list(self.discriminator.parameters()) +
                 list(self.class_embeddings.parameters())),
                lr=self.hparams["lr_d"],
                weight_decay=self.hparams['weight_decay_d'],
                betas=(self.hparams['beta1'], 0.9))

            self.gen_opt = torch.optim.Adam(
                (list(self.featurizer.parameters()) +
                 list(self.classifier.parameters())),
                lr=self.hparams["lr_g"],
                weight_decay=self.hparams['weight_decay_g'],
                betas=(self.hparams['beta1'], 0.9))
        else:
            self.disc_opt = torch.optim.SGD(
                (list(self.discriminator.parameters()) +
                 list(self.class_embeddings.parameters())),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.gen_opt = torch.optim.SGD(
                (list(self.featurizer.parameters()) +
                 list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )

    def update(self, minibatches):
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device='cuda')
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
                                    hparams, conditional=True, class_balance=True)


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                          hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        # domain number
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()
        self.scheduler.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                  num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                    num_domains, hparams, gaussian=False)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches):

        penalty_weight = (
            self.hparams['irm_lambda'] if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 0)

        nll = 0.
        penalty = 0.
        grad = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # # TRM -----
        # loss_swap = 0.0
        # # updating featurizer
        # self.featurizer.eval()
        # all_feature = self.featurizer(all_x).detach()
        #
        # for i in range(30):
        #     all_logits_idx = 0
        #     loss_erm = 0.
        #     for j, (x, y) in enumerate(minibatches):
        #         # j-th domain
        #         feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
        #         all_logits_idx += x.shape[0]
        #         loss_erm += F.cross_entropy(self.clist[j](feature), y)
        #     for opt in self.olist:
        #         opt.zero_grad()
        #     loss_erm.backward()
        #     for opt in self.olist:
        #         opt.step()
        #
        # self.featurizer.train()
        # all_feature = self.featurizer(all_x)
        # feature_split = list()
        # y_split = list()
        # all_logits_idx = 0
        # for i, (x, y) in enumerate(minibatches):
        #     feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
        #     all_logits_idx += x.shape[0]
        #     feature_split.append(feature)
        #     y_split.append(y)
        #
        # for Q, (x, y) in enumerate(minibatches):
        #     sample_list = list(range(len(minibatches)))
        #     sample_list.remove(Q)
        #     # calculate the swapping loss and product of gradient
        #     loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])
        #               for i in range(len(minibatches)) if i in sample_list]
        #     loss_P = torch.max(torch.tensor(loss_P))
        #     loss_swap += loss_P.item()
        #
        # loss_swap /= len(minibatches)
        #
        # # # TRM -----
        self.network.train()
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            loss_ = F.cross_entropy(logits, y)
            nll += loss_
            # grad_P = autograd.grad(loss_, self.classifier.weight, create_graph=True)
            # vec_grad_P = nn.utils.parameters_to_vector(grad_P)
            # grad += vec_grad_P @ vec_grad_P
            # penalty += vec_grad_P @ vec_grad_P
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        grad /= len(minibatches)
        # loss = nll + (penalty_weight * penalty)
        loss = nll + (penalty_weight * grad)
        if self.update_count == self.hparams['irm_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam' and \
                self.hparams['irm_lambda'] != 1:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.update_count += 1
        # return {'loss': loss.item(), 'nll': nll.item(),
        #         'penalty': penalty.item(), 'irm_w_grad': grad.item(), 'trm': loss_swap}
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    def update(self, minibatches):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 0.

        nll = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = (mean + penalty_weight * penalty)

        if self.update_count == self.hparams['vrex_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam':
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                       hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        # print(losses)
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.update_count = 0

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)
            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)
            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)
        That is, when calling .step(), we want grads to be Gi + beta * Gj
        For computational efficiency, we do not compute second derivatives.
        """
        if self.update_count < self.hparams['iters']:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            # updating original network
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.update_count += 1
            return {'loss': loss.item()}

        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"

            inner_net = copy.deepcopy(self.network)
            if self.hparams["opt"] == 'Adam':
                inner_opt = torch.optim.Adam(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            else:
                inner_opt = torch.optim.SGD(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )

            # updating original network
            inner_obj = F.cross_entropy(inner_net(xi), yi)
            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # update bn
            F.cross_entropy(self.network(xi), yi)
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()
            # self.optimizer.load_state_dict(inner_opt.state_dict())
            # # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()
        self.scheduler.step()
        self.update_count += 1
        return {'loss': objective}


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = networks.Featurizer(input_shape, hparams)
        classifier = nn.Linear(featurizer.n_outputs, num_classes)
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = WholeFish(input_shape, num_classes, hparams)
        self.optimizer_inner_state = None
        self.step = 0
        if self.hparams['opt'] == 'SGD':
            self.optimizer_inner = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_inner = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

    def create_clone(self, device):
        self.network_inner = WholeFish(self.input_shape, self.num_classes, self.hparams,
                                       weights=self.network.state_dict()).to(device)

        if self.step > 0 and self.step % self.hparams['sch_size'] == 0:
            self.hparams["lr"] *= 0.1

        if self.hparams['opt'] == 'SGD':
            self.optimizer_inner = torch.optim.SGD(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_inner = torch.optim.Adam(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

        for param_group in self.optimizer_inner.param_groups:
            param_group['lr'] = self.hparams["lr"]

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):

        self.create_clone(minibatches[0][0].device)
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)
        self.step += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class TRM(Algorithm):
    """
    TRM
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer_new = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda() for i in range(4)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(4)]

        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer_new.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_f = torch.optim.Adam(
                self.featurizer_new.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler_f = torch.optim.lr_scheduler.StepLR(self.optimizer_f, step_size=self.hparams['sch_size'],
                                                           gamma=0.1)
        self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=self.hparams['sch_size'],
                                                           gamma=0.1)
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    def update(self, minibatches):
        if self.hparams['class_balanced']:
            minibatches_trm = minibatches[len(minibatches) // 2:]
            minibatches = minibatches[:len(minibatches) // 2]
        else:
            minibatches_trm = minibatches

        loss_swap = 0.0
        loss_Q_sum = 0.0
        loss_P_sum_collect = 0.0
        loss_cos_sum = 0.0
        trm = 0.0
        # updating featurizer
        if self.update_count >= self.hparams['iters']:
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)
            self.featurizer_new.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer_new(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)
            all_feature = all_feature.detach()

            for i in range(self.hparams['n']):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches_trm):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            self.featurizer_new.train()
            all_feature = self.featurizer_new(all_x)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches_trm):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            for Q, (x, y) in enumerate(minibatches_trm):
                sample_list = list(range(len(minibatches_trm)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                loss_Q_sum += loss_Q.item()
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)
                loss_P = [
                    F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i]) * (self.alpha[Q, i].data.detach())
                    if i in sample_list else 0. for i in range(len(minibatches_trm))]
                loss_P_sum = sum(loss_P)
                loss_P_sum_collect += loss_P_sum.item()
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches_trm)
            trm /= len(minibatches_trm)
            loss_Q_sum /= len(minibatches_trm)
            loss_P_sum_collect /= len(minibatches_trm)
            loss_cos_sum = loss_P_sum_collect - loss_swap.item()
        else:
            self.featurizer_new.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer_new(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # updating scheduler
        self.scheduler_f.step()
        self.scheduler_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap, 'Q_loss': loss_Q_sum, 'P_loss': loss_P_sum_collect,
                'cos_avg': loss_cos_sum}

    def predict(self, x):
        return self.classifier(self.featurizer_new(x))

    def train(self):
        self.featurizer_new.train()

    def eval(self):
        self.featurizer_new.eval()

