import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
r = [0.9, 0.1]

def make_imbalance(label, img, r=[1., 1.]):

    label = np.array(label)
    img = np.array(img)
    mask = label >= 5

    label_0 = label[mask == 0]
    label_1 = label[mask == 1]
    img_0 = img[mask == 0]
    img_1 = img[mask == 1]

    min_len = min(len(label_0), len(label_1), 25000)
    label_0 = label_0[: min_len]
    label_1 = label_1[: min_len]
    img_0 = img_0[:min_len]
    img_1 = img_1[:min_len]

    label_0 = label_0[:int(len(label_0) * r[0])]
    label_1 = label_1[:int(len(label_1) * r[1])]
    img_0 = img_0[:int(len(img_0) * r[0])]
    img_1 = img_1[:int(len(img_1) * r[1])]

    return np.concatenate((label_0, label_1), axis=0), np.concatenate((img_0, img_1), axis=0)



def main():


    train = loadmat(os.path.join('./SynthDigits', 'synth_train_32x32.mat'))
    test = loadmat(os.path.join('./SynthDigits', 'synth_test_32x32.mat'))
    trainx = train['X'].transpose((3, 0, 1, 2))
    trainy = train['y'][:, 0]
    print("Train:", trainx.shape, trainy.shape)
    print("Train y:", trainy)
    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'digits32_train_{r}.mat', {'X': trainx, 'y': trainy})

    testx = test['X'].transpose((3, 0, 1, 2))
    testy = test['y'][:, 0]
    testy, testx = make_imbalance(testy, testx, r)
    savemat(f'digits32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading digits32_train_{r}.mat for sanity check")
    data = loadmat(f'digits32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

    print(f"Loading digits32_test_{r}.mat for sanity check")
    data = loadmat(f'digits32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

if __name__ == '__main__':
    main()
