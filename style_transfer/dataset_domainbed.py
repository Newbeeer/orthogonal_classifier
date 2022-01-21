# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict

import PIL
import torch
from PIL import Image, ImageFile
from utils import TensorDataset, save_image, TensorDataset_ori
from torchvision import transforms
import torchvision.datasets.folder
from torchvision.datasets import CIFAR100, MNIST, ImageFolder, SVHN
from torchvision.transforms.functional import rotate
import tqdm
import io
import functools



ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "RotatedMNIST",
    "ColoredMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "ColoredCOCO",
    "PlacesCOCO",
    "Celeba"
]

NUM_ENVIRONMENTS = {
    # Debug
    "Debug28": 3,
    "Debug224": 3,
    # Small images
    "RotatedMNIST": 6,
    "ColoredMNIST": 3,
    # Big images
    "VLCS": 4,
    "PACS": 4,
    "OfficeHome": 4,
    "TerraIncognita": 4,
    "DomainNet": 6,
}


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8


class Debug(MultipleDomainDataset):
    DATASET_SIZE = 16
    INPUT_SHAPE = None  # Subclasses should override

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.environments = [0, 1, 2]
        self.datasets = []
        for _ in range(len(self.environments)):
            self.datasets.append(
                TensorDataset(
                    torch.randn(self.DATASET_SIZE, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (self.DATASET_SIZE,))
                )
            )

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class MNIST_SVHN(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        pre_process_mnist = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)
             )])
        original_dataset_tr = MNIST(root, train=True, transform=pre_process_mnist, download=True)
        original_dataset_te = MNIST(root, train=False, transform=pre_process_mnist, download=True)
        images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_images_mnist = torch.stack([images, images, images], dim=1)
        print("MNIST:", original_images_mnist.size())
        original_labels_mnist = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images_mnist))
        original_images_mnist = original_images_mnist[shuffle]
        original_labels_mnist = original_labels_mnist[shuffle]

        # SVHN
        pre_process_svhn = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(28),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0.5, 0.5, 0.5),
                                              std=(0.5, 0.5, 0.5)
                                          )])
        original_dataset_tr = SVHN(root, split='train', transform=pre_process_svhn, download=True)
        original_dataset_te = SVHN(root, split='test', transform=pre_process_svhn, download=True)
        print("SVHN train:", original_dataset_tr.data.shape)
        original_images_svhn = torch.cat((torch.from_numpy(original_dataset_tr.data),
                                     torch.from_numpy(original_dataset_te.data)))
        original_labels_svhn = torch.cat((torch.from_numpy(original_dataset_tr.labels),
                                     torch.from_numpy(original_dataset_te.labels)))

        shuffle = torch.randperm(len(original_images_svhn))
        original_images_svhn = original_images_svhn[shuffle]
        original_labels_svhn = original_labels_svhn[shuffle]
        self.datasets = []
        self.datasets.append(TensorDataset_ori(original_images_mnist, original_labels_mnist))
        self.datasets.append(TensorDataset_ori(original_images_svhn, original_labels_svhn))
        self.input_shape = (3, 28, 28)
        self.num_classes = 10

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [180, 180, 180]])
        self.random_colors = torch.randint(255, (10, 3)).float()
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ColoredMNIST(MultipleEnvironmentMNIST):
    # random seed
    ENVIRONMENT_NAMES = [10, 1]

    def __init__(self, root, bias, test_envs, hparams, eval=False):
        # MY_COMBINE setting:
        # 0: random seed for environmental color;
        # 1: use default colors (True) or random colors;
        # 2: Bernoulli parameters for environmental color;
        # 3: designated environmental color number;
        # 4: random seed for bkgd colors
        # 5: Color digit?
        # 6: Color bkgd?
        # 7: Bernoulli parameters for bkgd colors
        if hparams['alg'] == 'ERM':
            MY_COMBINE = [[2, True, bias[0], None, 13,  True, True, bias[1], bias[2]]]
        else:
            if eval:
                MY_COMBINE = [[2, True, bias[0], None, 13, True, True, bias[1], bias[2]]]
            else:
                MY_COMBINE = [[2, True, bias[0], None, 13, True, True, 0, 0],
                              [2, True, bias[0], None, 13, True, True, bias[1], bias[2]],
                              [2, True, bias[0], None, 13, True, True, bias[3], bias[4]]]

        print("MY COMBINE:", MY_COMBINE)
        self.vis = False
        self.input_shape = (3, 28, 28,)

        # Binary classification
        self.num_classes = 2
        super(ColoredMNIST, self).__init__(root, MY_COMBINE, self.color_dataset, (3, 28, 28,), self.num_classes)

        # TODO: set up verbose mode

    def color_dataset(self, images, labels, environment):
        # set the seed

        original_seed = torch.cuda.initial_seed()
        torch.manual_seed(environment[0])
        shuffle = torch.randperm(len(self.colors))
        self.colors_ = self.colors[shuffle] if environment[1] else torch.randint(255, (10, 3)).float()
        torch.manual_seed(environment[0])
        ber_digit = self.torch_bernoulli_(environment[2], len(labels))

        torch.manual_seed(environment[4])
        shuffle = torch.randperm(len(self.colors)-1)
        bkgd_colors = self.colors[shuffle] * 0.75
        torch.manual_seed(environment[4])
        ber_bkgd = {0: self.torch_bernoulli_(environment[7], len(labels)),
                    1: self.torch_bernoulli_(environment[8], len(labels))}
        images = torch.stack([images, images, images], dim=1)
        torch.manual_seed(original_seed)
        # binarize the images
        images = (images > 0).float()
        masks = (1 - images)
        total_len = len(images)
        # Converting to binary classification
        # Random labels
        labels = self.torch_bernoulli_(0.5, total_len)
        color_label = torch.zeros(ber_digit.size())
        bkgd_label = torch.zeros(ber_digit.size())
        if self.vis:
            image_collect = torch.empty(20, *self.input_shape)
            current_label = 0
            current_cnt = 0
            total_cnt = 0
        # Apply the color to the image
        for img_idx in range(total_len):
            # change digit colors
            if ber_digit[img_idx] > 0:
                if environment[5]:
                    color_label[img_idx] = labels[img_idx].long()
                    images[img_idx] = images[img_idx] * self.colors_[labels[img_idx].long()].view(-1, 1, 1)
            else:
                # unbiased uniform sampling
                color = torch.randint(2, [1])[0] if environment[3] is None else environment[3]
                color_label[img_idx] = color
                if environment[5]:
                    images[img_idx] = images[img_idx] * self.colors_[color].view(-1, 1, 1)
            # change bkpg colors
            if ber_bkgd[int(labels[img_idx])][img_idx] > 0:
                if environment[6]:
                    bkgd_label[img_idx] = labels[img_idx].long()
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + masks[img_idx] * bkgd_colors[
                        labels[img_idx].long()].view(-1, 1, 1)
            else:
                color = torch.randint(2, [1])[0]
                bkgd_label[img_idx] = color
                if environment[6]:
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + masks[img_idx] * bkgd_colors[color].view(
                        -1, 1, 1)
            if self.vis:
                # visualize 20 images for sanity check
                import matplotlib.pyplot as plt
                image_collect[total_cnt] = images[img_idx]
                current_cnt += 1
                total_cnt += 1
                if total_cnt == 20:
                    break
        if self.vis:
            save_image(image_collect, 'mnist.png', nrow=10)
            print(f"Visualization for {environment} Done")
            exit(0)
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        color_label = color_label.view(-1).long()
        bkgd_label = bkgd_label.view(-1).long()

        return TensorDataset(True, x, y, color_label, bkgd_label)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=PIL.Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        self.environments = [f.name for f in os.scandir(root) if f.is_dir()]
        self.environments = sorted(self.environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        print("enviroments:", self.environments)
        for i, environment in enumerate(self.environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)



class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENT_NAMES = ["A", "C", "P", "S"]

    def __init__(self, root, bias, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENT_NAMES = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


import h5py
import numpy as np


class MultipleEnvironmentCOCO(MultipleDomainDataset):
    def __init__(self, environments, dataset_transform, input_shape,
                 num_classes, places=False):
        super().__init__()
        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [180, 180, 180]])
        self.random_colors = torch.randint(255, (10, 3)).float()
        h5pyfname = '../data/coco'
        train_file = os.path.join(h5pyfname, 'train.h5py')
        val_file = os.path.join(h5pyfname, 'validtest.h5py')
        test_file = os.path.join(h5pyfname, 'idtest.h5py')
        train_data = h5py.File(train_file, 'r')
        val_data = h5py.File(val_file, 'r')
        test_data = h5py.File(test_file, 'r')
        original_images = np.concatenate(
            (train_data['resized_images'].value, test_data['resized_images'].value, val_data['resized_images'].value),
            axis=0)
        original_labels = np.concatenate((train_data['y'].value, test_data['y'].value, val_data['y'].value), axis=0)

        original_masks = np.concatenate(
            (train_data['resized_mask'].value, test_data['resized_mask'].value, val_data['resized_mask'].value), axis=0)

        print('image size:{}, label size:{}, mask:{}'.format(original_images.shape, original_labels.shape,
                                                             original_masks.shape))

        if places:
            places_file = os.path.join('../data/places/cocoplaces', 'places.h5py')
            places_data = h5py.File(places_file, 'r')
            self.places = places_data['resized_place'].value
            print('place size:{}'.format(self.places.shape))
            self.places = torch.from_numpy(self.places)

        original_images = torch.from_numpy(original_images)
        original_labels = torch.from_numpy(original_labels)
        original_masks = torch.from_numpy(original_masks)
        shuffle = torch.randperm(len(original_images))

        total_len = len(original_images)

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        original_masks = original_masks[shuffle]
        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            masks = original_masks[i::len(self.environments)]
            #             images = original_images
            #             labels = original_labels
            #   masks = original_masks
            self.datasets.append(dataset_transform(images, labels, masks, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ColoredCOCO(MultipleEnvironmentCOCO):
    # random seed
    ENVIRONMENT_NAMES = [1, 2, 3]

    def __init__(self, root, test_envs, hparams):
        # MY_COMBINE setting:
        # 0: random seed for shuffle color;
        # 1: use default colors or random colors;
        # 2: Bernoulli parameters for digit colors;
        # 3: designated color number,
        # 4: random seed for bkgd colors
        # 5: Color digit? 6: Color Bkgd?
        # 7: Bernoulli parameters for bkgd colors
        # 8: removing data-augmentation or not
        MY_COMBINE = [[2, True, 0, None, 11, True, True, 0, True], [2, True, 0.9, None, 12, True, True, 0, False],
                      [2, True, 0.7, None, 13, True, True, 0, False]]
        self.vis = False
        print("MY COMBINE:", MY_COMBINE)
        super(ColoredCOCO, self).__init__(MY_COMBINE, self.color_dataset, (3, 64, 64,), 10, False)

        self.input_shape = (3, 64, 64,)
        self.num_classes = 10

    def color_dataset(self, images, labels, masks, environment):

        # shuffle the colors
        torch.manual_seed(environment[0])
        shuffle = torch.randperm(len(self.colors))
        self.colors_ = self.colors[shuffle] if environment[1] else torch.randint(255, (10, 3)).float()
        # set the bernoulli r.v.
        torch.manual_seed(environment[0])
        ber = self.torch_bernoulli_(environment[2], len(labels))
        print("bernoulli:", len(ber), sum(ber))

        torch.manual_seed(environment[4])
        shuffle = torch.randperm(len(self.colors))
        bkgd_colors = torch.randint(255, (10, 3)).float()
        torch.manual_seed(environment[4])
        ber_obj = self.torch_bernoulli_(environment[7], len(labels))

        total_len = 16 if self.vis else len(images)
        # Apply the color to the image
        for img_idx in range(total_len):
            if ber[img_idx] > 0:
                if environment[5]:
                    place_img = 0.75 * np.multiply(np.ones((3, 64, 64), dtype='float32'),
                                                   self.colors_[labels[img_idx]][:, None, None]) / 255.0
                    images[img_idx] = place_img * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx]
            else:
                if environment[5]:
                    color = torch.randint(10, [1])[0] if environment[3] is None else environment[3]
                    place_img = 0.75 * np.multiply(np.ones((3, 64, 64), dtype='float32'),
                                                   self.colors_[color][:, None, None]) / 255.0
                    images[img_idx] = place_img * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx]

            if ber_obj[img_idx] > 0:
                if environment[6]:
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx] * \
                                      bkgd_colors[labels[img_idx].long()].view(-1, 1, 1) / 255.0
            else:
                if environment[6]:
                    color = torch.randint(5, [1])[0]
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx] * \
                                      bkgd_colors[color].view(-1, 1, 1) / 255.0
            if self.vis:
                # visualize 10 images for sanity check
                import matplotlib.pyplot as plt
                plt.imsave('test_{}.png'.format(img_idx), images[img_idx].numpy().transpose(1, 2, 0))
        if self.vis:
            print("Visualization Done")
            exit(0)
        x = images.float()
        y = labels.view(-1).long()

        return TensorDataset(environment[8], x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


import sys
sys.path.append('../')
from datasets.celeba_dataset import CelebA_group


class Celeba(MultipleDomainDataset):

    def __init__(self, gnames:list, image_size=64, dg=False, oracle=False, young=False):
        super().__init__()
        self.gnames = gnames
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.CenterCrop(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        self.dg = dg

        if self.dg:
            assert len(gnames) == 4, "only four groups are allowable for dg"

            dataset_list = []
            len_minor = 100000
            for i in range(len(self.gnames)):
                if i < len(self.gnames)//2:
                    dataset_list.append(
                        CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[i], transform=transform, label=0))
                else:
                    dataset_list.append(
                        CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[i], transform=transform, label=1))
                print(f"dataset {i} len: {len(list(dataset_list[i].filename))}")
                len_minor = min(len_minor,len(list(dataset_list[i].filename)))

            len_major = len_minor

            # env 0: non-even
            self.env0 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[0], transform=transform, label=0)
            self.env0.filename = list(dataset_list[0].filename)[:len_major*2] + list(dataset_list[2].filename)[:len_major*2]
            self.env0.label = torch.cat([dataset_list[0].label[:len_major*2], dataset_list[2].label[:len_major*2]], dim=0)

            # env 1: evenly sample from each group
            self.env1 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[0], transform=transform, label=0)
            self.env1.filename = list(dataset_list[0].filename)[2*len_major: 3*len_major] + list(dataset_list[1].filename)[:len_minor] \
                   + list(dataset_list[2].filename)[2*len_major: 3*len_major] + list(dataset_list[3].filename)[:len_minor]
            self.env1.label = torch.cat([dataset_list[0].label[2*len_major: 3*len_major], dataset_list[1].label[:len_minor],
                                    dataset_list[2].label[2*len_major: 3*len_major], dataset_list[3].label[:len_minor]], dim=0)

            self.datasets = [self.env0, self.env1]
            assert len(self.env1.filename) == len(self.env1.label)
            print(f" env 0 len:{len(self.env0.label)}, env 1 len:{len(self.env1.label)}")
            if oracle:
                # M_B , M_NB / F_B, F_NB
                self.datasets = [self.env1]
        elif young:
            dataset_list = []
            self.gnames_d = ['male_nonblond_refine', 'male_blond_refine', 'female_blond_refine', 'female_nonblond_refine']
            for i in range(len(self.gnames_d)):
                if i < len(self.gnames)//2:
                    dataset_list.append(
                        CelebA_group('/data/scratch/ylxu/domainbed', self.gnames_d[i], transform=transform, label=0))
                else:
                    dataset_list.append(
                        CelebA_group('/data/scratch/ylxu/domainbed', self.gnames_d[i], transform=transform, label=1))
                print(f"dataset {i} len: {len(list(dataset_list[i].filename))}")
            len_minor = len(list(dataset_list[1].filename))
            # null env
            self.env_test = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[0], transform=transform, label=0)
            self.env_test.filename = list(dataset_list[1].filename)[:len_minor] + list(dataset_list[3].filename)[:len_minor]
            self.env_test.label = torch.cat([dataset_list[1].label[:len_minor], dataset_list[3].label[:len_minor]], dim=0)

            assert len(gnames) == 4, "only four groups are allowable"
            dataset_1 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[0], transform=transform, label=0)
            dataset_2 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[1], transform=transform, label=1)
            dataset_3 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[2], transform=transform, label=0)
            dataset_4 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[3], transform=transform, label=1)
            len_ = min(len(list(dataset_1.filename)), len(list(dataset_2.filename)))
            # len_ = max(len(list(dataset_1.filename)), len(list(dataset_2.filename)))
            dataset_1.filename = list(dataset_1.filename)[:len_] + list(dataset_2.filename)[:len_]
            dataset_1.label = torch.cat([dataset_1.label[:len_], dataset_2.label[:len_]])
            dataset_1.attr = torch.cat([dataset_1.attr, dataset_2.attr], dim=0)
            assert len(dataset_1.filename) == len(dataset_1.label)

            len_ = min(len(list(dataset_3.filename)), len(list(dataset_4.filename)))
            # len_ = max(len(list(dataset_1.filename)), len(list(dataset_2.filename)))
            dataset_3.filename = list(dataset_3.filename)[:len_] + list(dataset_4.filename)[:len_]
            dataset_3.label = torch.cat([dataset_3.label[:len_], dataset_4.label[:len_]])
            dataset_3.attr = torch.cat([dataset_3.attr, dataset_4.attr], dim=0)
            assert len(dataset_3.filename) == len(dataset_3.label)

            self.datasets = [self.env_test, dataset_1, dataset_3]
            print("datasets len:", len(dataset_1.filename)+len(dataset_3.filename))
        else:
            assert len(gnames) == 2, "only two groups are allowable"
            dataset_1 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[0], transform=transform, label=0)
            dataset_2 = CelebA_group('/data/scratch/ylxu/domainbed', self.gnames[1], transform=transform, label=1)
            print(f"dataset 1 len: {len(list(dataset_1.filename))}, dataset 2 len: {len(list(dataset_2.filename))}")
            # len_ = min(len(list(dataset_1.filename)), len(list(dataset_2.filename)))
            dataset_1.filename = list(dataset_1.filename) + list(dataset_2.filename)
            dataset_1.label = torch.cat([dataset_1.label, dataset_2.label])
            dataset_1.attr = torch.cat([dataset_1.attr, dataset_2.attr], dim=0)
            assert len(dataset_1.filename) == len(dataset_1.label)
            self.datasets = [dataset_1]
            print("datasets len:", len(dataset_1.label))

        self.input_shape = (3, image_size, image_size)
        # male & female
        self.num_classes = 2


    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
