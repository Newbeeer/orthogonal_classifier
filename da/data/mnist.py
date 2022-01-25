import os
import numpy as np
import subprocess
from scipy.io import loadmat, savemat
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser(description='dataset setup')
parser.add_argument('--r', type=float, nargs='+', default=[0.7, 0.3])
args = parser.parse_args()
r = args.r

def make_imbalance(label, img, r=[1., 1.]):
    label = np.array(label)
    img = np.array(img)
    mask = label >= 5
    label_0 = label[mask == 0]
    label_1 = label[mask == 1]
    img_0 = img[mask == 0]
    img_1 = img[mask == 1]

    min_len = min(len(label_0), len(label_1))
    label_0 = label_0[: min_len]
    label_1 = label_1[: min_len]
    img_0 = img_0[:min_len]
    img_1 = img_1[:min_len]

    label_0 = label_0[:int(len(label_0) * r[0])]
    label_1 = label_1[:int(len(label_1) * r[1])]
    img_0 = img_0[:int(len(img_0) * r[0])]
    img_1 = img_1[:int(len(img_1) * r[1])]

    return np.concatenate((label_0, label_1), axis=0), np.concatenate((img_0, img_1), axis=0)

def mnist_resize(x):
    H, W, C = 32, 32, 3
    x = x.reshape(-1, 28, 28)
    resized_x = np.empty((len(x), H, W), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W), mode='reflect')

    # Retile to make RGB
    resized_x = resized_x.reshape(-1, H, W, 1)
    resized_x = np.tile(resized_x, (1, 1, 1, C))
    return resized_x

def main():
    if os.path.exists('mnist/mnist.npz'):
        print("Using existing mnist.npz")

    else:
        print("Opening subprocess to download data from URL")
        subprocess.check_output(
            '''
            mkdir mnist
            wget https://s3.amazonaws.com/img-datasets/mnist.npz -P mnist
            ''',
            shell=True)

    print( "Resizing mnist.npz to (32, 32, 3)")
    data = np.load('mnist/mnist.npz')
    trainx = data['x_train']
    trainy = data['y_train']
    trainy, trainx = make_imbalance(trainy, trainx, r)
    trainx = mnist_resize(trainx)
    savemat(f'mnist/mnist32_train_{r}.mat', {'X': trainx, 'y': trainy})

    testx = data['x_test']
    testy = data['y_test']
    testy, testx = make_imbalance(testy, testx, r)
    testx = mnist_resize(testx)
    savemat(f'mnist/mnist32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading mnist32_train_{r}.mat for sanity check")
    data = loadmat(f'mnist/mnist32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

    print(f"Loading mnist32_test_{r}.mat for sanity check")
    data = loadmat(f'mnist/mnist32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

if __name__ == '__main__':
    main()
