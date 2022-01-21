import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np
import gzip
import pickle

import cv2
import check_img
import torch
from skimage.transform import downscale_local_mean, resize
r = [0.3, 0.7]

def make_imbalance(label, img, r=[1., 1.]):
    label = np.array(label)
    img = np.array(img)
    mask = (label >= 4)

    label_0 = label[mask == 0]
    label_1 = label[mask == 1]
    img_0 = img[mask == 0]
    img_1 = img[mask == 1]

    min_len = min(len(label_0), len(label_1))

    label_0 = label_0[: min_len]
    label_1 = label_1[: min_len]
    img_0 = img_0[:min_len]
    img_1 = img_1[:min_len]
    if r[0] > r[1]:
        min_len = int(min_len * r[1] / r[0])
        label_1 = label_1[:min_len]
        img_1 = img_1[:min_len]
    else:
        min_len = int(min_len * r[0] / r[1])
        label_0 = label_0[:min_len]
        img_0 = img_0[:min_len]


    return np.concatenate((label_0, label_1), axis=0), np.concatenate((img_0, img_1), axis=0)

def stl_resize(x):
    H, W, C = 32, 32, 3
    x = x.reshape(-1, 96, 96, 3)
    resized_x = np.empty((len(x), H, W, 3), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        #resized_x[i] = resize(img, (H, W))
        resized_x[i] = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    # Retile to make RGB
    resized_x = resized_x.transpose(0, 3, 1, 2)
    return resized_x

def main():

    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
    root = '/data/scratch/ylxu/data'

    def loadfile(data_file: str, labels_file):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                root, base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(root, base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    trainx, trainy = loadfile(train_list[0][0], train_list[1][0])
    testx, testy = loadfile(test_list[0][0], test_list[1][0])

    cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
    trainy = cls_mapping[trainy]
    testy = cls_mapping[testy]

    # Remove all samples from skipped classes
    train_mask = trainy != -1
    test_mask = testy != -1

    trainx = trainx[train_mask]
    trainy = trainy[train_mask]
    testx = testx[test_mask]
    testy = testy[test_mask]

    trainx = downscale_local_mean(trainx, (1, 1, 3, 3)) / 255.
    testx = downscale_local_mean(testx, (1, 1, 3, 3)) / 255.
    print("Train x y:", trainx.shape, len(trainy))
    print("Test x y:", testx.shape, len(testy))
    trainy, trainx = make_imbalance(trainy, trainx, r)
    # print(trainx.shape)
    # check_img.save_image(torch.from_numpy(trainx[:64]/255.), filename='t.png')
    # exit(0)
    trainx = trainx.transpose(0, 2, 3, 1)
    savemat(f'/data/scratch/ylxu/dirt-t/data/stl32_train_{r}.mat', {'X': trainx, 'y': trainy})


    testy, testx = make_imbalance(testy, testx, r)
    testx = testx.transpose(0, 2, 3, 1)
    savemat(f'/data/scratch/ylxu/dirt-t/data/stl32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading stl32_train_{r}.mat for sanity check")
    data = loadmat(f'/data/scratch/ylxu/dirt-t/data/stl32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 4).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 4).astype(np.float32).sum())

    print(f"Loading stl32_test_{r}.mat for sanity check")
    data = loadmat(f'/data/scratch/ylxu/dirt-t/data/stl32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 4).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 4).astype(np.float32).sum())

if __name__ == '__main__':
    main()
