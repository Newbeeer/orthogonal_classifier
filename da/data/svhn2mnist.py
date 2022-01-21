import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np

r = [1, 1]

def make_imbalance(label, img, r=[1., 1.]):
    label = np.array(label)
    img = np.array(img)
    mask = (label >= 5)[0, :]
    print(mask.shape, label.shape)
    label_0 = label[:, mask == 0]
    label_1 = label[:, mask == 1]

    img_0 = img[mask == 0]
    img_1 = img[mask == 1]

    min_len = min(label_0.shape[1], label_1.shape[1])

    label_0 = label_0[:, : min_len]
    label_1 = label_1[:, : min_len]
    img_0 = img_0[:min_len]
    img_1 = img_1[:min_len]

    label_0 = label_0[:, :int(label_0.shape[1] * r[0])]
    label_1 = label_1[:, :int(label_1.shape[1] * r[1])]

    img_0 = img_0[:int(img_0.shape[0] * r[0])]
    img_1 = img_1[:int(img_1.shape[0] * r[1])]

    return np.concatenate((label_0, label_1), axis=1), np.concatenate((img_0, img_1), axis=0)

def main():


    print("Loading train_32x32.mat for sanity check")
    data = loadmat('../svhn2mnist32.mat')
    trainx = data['X']
    trainy = data['y']

    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'svhn2mnist32_{r}.mat', {'X': trainx, 'y': trainy})
    data = loadmat(f'svhn2mnist32_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

    # print("Loading test_32x32.mat for sanity check")
    # data = loadmat('test_32x32.mat')
    # testx = data['X']
    # testy = data['y']
    # testy[data['y'] == 10] = 0
    # testy, testx = make_imbalance(testy, testx, r)
    # savemat(f'svhn32_test_{r}.mat', {'X': testx, 'y': testy})
    # data = loadmat(f'svhn32_test_{r}.mat')
    # print(data['X'].shape, data['X'].min(), data['X'].max())
    # print(data['y'].shape, data['y'].min(), data['y'].max())
    # print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

if __name__ == '__main__':
    main()
