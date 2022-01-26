# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import PIL
import torch
from PIL import Image, ImageFile
from utils import TensorDataset, save_image
from torchvision import transforms
from torchvision.datasets import MNIST
import sys
sys.path.append('../')
from datasets.celeba_dataset import CelebA_group



ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "CMNIST",
    "Celeba"
]

NUM_ENVIRONMENTS = {
    "CMNIST": 3,
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


class CMNIST(MultipleEnvironmentMNIST):
    # random seed
    ENVIRONMENT_NAMES = [10, 1]
    def __init__(self, root, bias, dg=False):
        # config setting:
        # 0: random seed for environmental color;
        # 1: use default colors (True) or random colors;
        # 2: Bernoulli parameters for environmental color;
        # 3: designated environmental color number;
        # 4: random seed for bkgd colors
        # 5: Color digit?
        # 6: Color bkgd?
        # 7: Bernoulli parameters for bkgd colors
        if not dg:
            config = [[2, True, bias[0], None, 13,  True, True, bias[1]]]
        else:
            config = [[2, True, bias[0], None, 13, True, True, 0, 0],
                       [2, True, bias[0], None, 13, True, True, bias[1]],
                       [2, True, bias[0], None, 13, True, True, bias[2]]]

        print("config:", config)
        self.vis = False
        self.input_shape = (3, 28, 28,)

        # Binary classification
        self.num_classes = 2
        super(CMNIST, self).__init__(root, config, self.color_dataset, (3, 28, 28,), self.num_classes)

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
        ber_bkgd = self.torch_bernoulli_(environment[7], len(labels))

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
            if ber_bkgd[img_idx] > 0:
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

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        color_label = color_label.view(-1).long()
        bkgd_label = bkgd_label.view(-1).long()

        return TensorDataset(True, x, y, color_label, bkgd_label)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class Celeba(MultipleDomainDataset):

    def __init__(self, root: str, group_names: list, stage: int, image_size=128, dg=False):
        super().__init__()
        self.group_names = group_names
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.CenterCrop(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        if stage in [1,2]:
            assert len(self.group_names) == 4, "only four groups are allowable"
            dataset_list = []
            align_len = 100000
            for i in range(len(self.group_names)):
                label = 0 if i < len(self.group_names) // 2 else 1
                dataset_list.append(CelebA_group(root, self.group_names[i], transform=transform, label=label))
                print(f"dataset {self.group_names[i]} length: {len(list(dataset_list[i].filename))}")
                align_len = min(align_len, len(list(dataset_list[i].filename)))

            # env 0: non-even
            self.env0 = CelebA_group(root, self.group_names[0], transform=transform, label=None)
            self.env0.filename = list(dataset_list[0].filename)[:align_len*2] + list(dataset_list[2].filename)[:align_len*2]
            self.env0.label = torch.cat([dataset_list[0].label[:align_len*2], dataset_list[2].label[:align_len*2]], dim=0)

            # env 1: evenly sample from each group
            self.env1 = CelebA_group(root, self.group_names[0], transform=transform, label=None)
            self.env1.filename = list(dataset_list[0].filename)[2*align_len: 3*align_len] + list(dataset_list[1].filename)[:align_len] \
                   + list(dataset_list[2].filename)[2*align_len: 3*align_len] + list(dataset_list[3].filename)[:align_len]
            self.env1.label = torch.cat([dataset_list[0].label[2*align_len: 3*align_len], dataset_list[1].label[:align_len],
                                    dataset_list[2].label[2*align_len: 3*align_len], dataset_list[3].label[:align_len]], dim=0)

            self.datasets = [self.env0, self.env1]
            print(f" env 0 len:{len(self.env0.label)}, env 1 len:{len(self.env1.label)}")
            if not dg:
                # only use the env1 if not using domain generalization algorithms
                self.datasets = [self.env1]
        elif stage == 3:
            assert len(group_names) == 2, "only two groups are allowable for w_x"
            dataset_1 = CelebA_group(root, self.group_names[0], transform=transform, label=0)
            dataset_2 = CelebA_group(root, self.group_names[1], transform=transform, label=1)
            min_length = min(len(dataset_1.filename), len(dataset_2.filename))

            print(f"{self.group_names[0]} length: {len(list(dataset_1.filename))}, {self.group_names[1]} length: {len(list(dataset_2.filename))}")
            dataset_1.filename = list(dataset_1.filename[:min_length]) + list(dataset_2.filename[:min_length])
            dataset_1.label = torch.cat([dataset_1.label[:min_length], dataset_2.label[:min_length]])
            dataset_1.attr = torch.cat([dataset_1.attr[:min_length], dataset_2.attr[:min_length]], dim=0)
            self.datasets = [dataset_1]
        else:
            raise NotImplementedError

        self.input_shape = (3, image_size, image_size)
        # male vs. female
        self.num_classes = 2


    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
