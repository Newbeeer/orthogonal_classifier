import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='dataset setup')
parser.add_argument('--r', type=float, nargs='+', default=[0.7, 0.3])
args = parser.parse_args()
r = args.r

def make_imbalance(label, img, r=[1., 1.]):
    label = np.array(label)
    img = np.array(img)
    mask = (label >= 5)[:, 0]

    label_0 = label[mask == 0]
    label_1 = label[mask == 1]
    img_0 = img[:, :, :, mask == 0]
    img_1 = img[:, :, :, mask == 1]
    
    min_len = min(len(label_0), len(label_1))
    label_0 = label_0[: min_len]
    label_1 = label_1[: min_len]
    img_0 = img_0[:, :, :, :min_len]
    img_1 = img_1[:, :, :, :min_len]

    label_0 = label_0[:int(len(label_0) * r[0])]
    label_1 = label_1[:int(len(label_1) * r[1])]
    img_0 = img_0[:, :, :, :int(img_0.shape[3] * r[0])]
    img_1 = img_1[:, :, :, :int(img_1.shape[3] * r[1])]

    return np.concatenate((label_0, label_1), axis=0), np.concatenate((img_0, img_1), axis=3)

def main():
    if os.path.exists('./svhn/test_32x32.mat') and os.path.exists('./svhn/train_32x32.mat'):
        print("Using existing data")

    else:
        print( "Opening subprocess to download data from URL")
        subprocess.check_output(
            '''
            mkdir svhn
            wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -P svhn
            wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -P svhn
            ''',
            shell=True)

    print("Loading train_32x32.mat for sanity check")
    data = loadmat('./svhn/train_32x32.mat')
    trainx = data['X']
    trainy = data['y']
    trainy[data['y']==10] = 0
    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'./svhn/svhn32_train_{r}.mat', {'X': trainx, 'y': trainy})
    data = loadmat(f'./svhn/svhn32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

    print("Loading test_32x32.mat for sanity check")
    data = loadmat('./svhn/test_32x32.mat')
    testx = data['X']
    testy = data['y']
    testy[data['y'] == 10] = 0
    testy, testx = make_imbalance(testy, testx, r)
    savemat(f'./svhn/svhn32_test_{str(r)}.mat', {'X': testx, 'y': testy})
    data = loadmat(f'./svhn/svhn32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

if __name__ == '__main__':
    main()
