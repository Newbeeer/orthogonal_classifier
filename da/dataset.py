import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.io import loadmat


class Dataset(data.Dataset):
    def __init__(self, src, tgt, r_src, iseval, dataratio=1.0, r_tgt=None):
        if r_src[0] == 1 or r_src[0] == 0:
            r_src[0] = int(r_src[0])
        if r_src[1] == 1 or r_src[1] == 0:
            r_src[1] = int(r_src[1])
        self.eval = iseval
        if r_tgt is None:
            r_tgt = r_src

        r = r_src
        if src == 'mnist':
            print(f"dataset : mnist, file path : {f'da/data/mnist/mnist32_train_{r}.mat'}")
            data = loadmat(f'da/data/mnist/mnist32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'svhn':
            print(f"dataset : svhn, file path : {f'da/data/svhn/svhn32_train_{r}.mat'}")
            data = loadmat(f'da/data/svhn/svhn32_train_{r}.mat')
            self.datalist_svhn = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_svhn
        elif src == 'mnistm':
            print(f"dataset : mnistm, file path : {f'da/data/mnistm/mnistm32_train_{r}.mat'}")
            data = loadmat(f'da/data/mnistm32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'digits':
            print(f"dataset : digits, file path : {f'da/data/digits32_train_{r}.mat'}")
            data = loadmat(f'da/data/digits32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'signs':
            print(f"dataset : signs, file path : {f'da/data/signs32_train_{r}.mat'}")
            data = loadmat(f'da/data/signs32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'da/data/GTSRB/gtsrb32_train_{r}.mat'}")
            data = loadmat(f'da/data/GTSRB/gtsrb32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'cifar':
            print(f"dataset : cifar, file path : {f'da/data/cifar/cifar32_train_{r}.mat'}")
            data = loadmat(f'da/data/cifar/cifar32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'stl':
            print(f"dataset : stl, file path : {f'da/data/stl/stl32_train_{r}.mat'}")
            data = loadmat(f'da/data/stl/stl32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist

        r = r_tgt
        if tgt == 'mnist':
            print(f"dataset : mnist, file path : {f'da/data/mnist/mnist32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/mnist/mnist32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'svhn':
            print(f"dataset : svhn, file path : {f'da/data/svhn/svhn32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/svhn/svhn32_train_{r[::-1]}.mat')
            self.datalist_svhn = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_svhn
        elif tgt == 'mnistm':
            print(f"dataset : mnistm, file path : {f'da/data/mnistm/mnistm32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/mnistm/mnistm32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'digits':
            print(f"dataset : digits, file path : {f'da/data/digits32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/digits32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'signs':
            print(f"dataset : signs, file path : {f'da/data/signs32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/signs32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'da/data/GTSRB/gtsrb32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/GTSRB/gtsrb32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'cifar':
            print(f"dataset : cifar, file path : {f'da/data/cifar/cifar32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/cifar/cifar32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'stl':
            print(f"dataset : stl, file path : {f'da/data/stl/stl32_train_{r[::-1]}.mat'}")
            data = loadmat(f'da/data/stl/stl32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize_32 = transforms.Resize(32)
        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)

    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):

        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source = self.datalist_src[index_src]['image']
        #print("raw source:", image_source)
        image_source = self.totensor(image_source)
        image_source = self.normalize(image_source).float()
        image_target = self.datalist_target[index_target]['image']
        #print("raw tgt:", image_target)
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target).float()

        return image_source, self.datalist_src[index_src]['label'], image_target, self.datalist_target[index_target]['label']


class Dataset_eval(data.Dataset):
    def __init__(self, tgt, r):

        if tgt == 'mnist':
            print(f"dataset : mnist, file path : {f'da/data/mnist/mnist32_test_{r}.mat'}")
            data = loadmat(f'da/data/mnist/mnist32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'svhn':
            print(f"dataset : svhn, file path : {f'da/data/svhn/svhn32_test_{r}.mat'}")
            data = loadmat(f'da/data/svhn/svhn32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0])]
        elif tgt == 'mnistm':
            print(f"dataset : mnistm, file path : {f'da/data/mnistm/mnistm32_test_{r}.mat'}")
            data = loadmat(f'da/data/mnistm/mnistm32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'digits':
            print(f"dataset : digits, file path : {f'da/data/digits32_test_{r}.mat'}")
            data = loadmat(f'da/data/digits32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'signs':
            print(f"dataset : signs, file path : {f'da/data/signs32_test_{r}.mat'}")
            data = loadmat(f'da/data/signs32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'da/data/GTSRB/gtsrb32_test_{r}.mat'}")
            data = loadmat(f'da/data/GTSRB/gtsrb32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'cifar':
            print(f"dataset : cifar, file path : {f'da/data/cifar/cifar32_test_{r}.mat'}")
            data = loadmat(f'da/data/cifar/cifar32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'stl':
            print(f"dataset : stl, file path : {f'da/data/stl/stl32_test_{r}.mat'}")
            data = loadmat(f'da/data/stl/stl32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.datalist_target)

    def __getitem__(self, index):

        image_target = self.datalist_target[index]['image']
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target).float()

        return image_target, self.datalist_target[index]['label']


def GenerateIterator(args, iseval=False, r_tgt=None):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size if not iseval else args.batch_size_eval,
        'shuffle': True,
        'num_workers': args.workers,
        'drop_last': True,
    }
    dataset = Dataset(args.src, args.tgt, args.r, iseval, r_tgt=r_tgt)
    y_scr = torch.tensor([k['label'] for k in dataset.datalist_src]).unsqueeze(1)
    y_tgt = torch.tensor([k['label'] for k in dataset.datalist_target]).unsqueeze(1)

    # calculate label prior
    y_onehot_s = torch.FloatTensor(len(y_scr), max(y_scr)+1)
    y_onehot_s.zero_()
    y_onehot_s.scatter_(1, y_scr, 1)

    y_onehot_t = torch.FloatTensor(len(y_tgt), max(y_tgt)+1)
    y_onehot_t.zero_()
    y_onehot_t.scatter_(1, y_tgt, 1)
    r_s = y_onehot_s.sum(0) / len(y_onehot_s)
    r_t = y_onehot_t.sum(0) / len(y_onehot_t)
    print("(Class Priors) src :{}, tgt:{}".format(r_s, r_t))
    return data.DataLoader(dataset, **params), r_s.cuda()


def GenerateIterator_eval(args, r_tgt=None):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size_eval,
        'num_workers': args.workers,
    }
    r = args.r[::-1] if r_tgt is None else r_tgt
    return data.DataLoader(Dataset_eval(args.tgt, r), **params)
