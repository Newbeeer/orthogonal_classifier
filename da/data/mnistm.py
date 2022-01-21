import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np
from scipy.io import loadmat, savemat
from skimage.transform import resize
import matplotlib.pyplot as plt
r = [0.5, 0.5]


def process():
    """Download the MNIST data."""
    # import essential packages
    from six.moves import urllib
    import gzip
    import pickle
    from torchvision import datasets

    # process and save as torch files
    print("Processing...")

    # load MNIST-M images from pkl file
    with gzip.open('keras_mnistm.pkl.gz', "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        mnist_m_data = u.load()
    mnist_m_train_data = np.array(mnist_m_data["train"])
    mnist_m_test_data = np.array(mnist_m_data["test"])

    # get MNIST labels
    mnist_train_labels = datasets.MNIST(root='/home/ylxu/data', train=True, download=True).train_labels.numpy()
    mnist_test_labels = datasets.MNIST(root='/home/ylxu/data', train=False, download=True).test_labels.numpy()

    # save MNIST-M dataset
    training_set = (mnist_m_train_data, mnist_train_labels)
    test_set = (mnist_m_test_data, mnist_test_labels)
    return training_set, test_set

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
    x = x.reshape(-1, 28, 28, 3)
    resized_x = np.empty((len(x), H, W, 3), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W), mode='reflect')
        # if i==1:
        #     plt.imsave("test.jpg", (resized_x[i] * 255).astype(np.uint8))
        #     exit(0)

    return resized_x

def main():

    train, test = process()
    trainx = train[0]
    trainy = train[1]
    print("Train:", trainx.shape, trainy.shape)
    trainy, trainx = make_imbalance(trainy, trainx, r)
    trainx = mnist_resize(trainx)
    savemat(f'mnistm32_train_{r}.mat', {'X': trainx, 'y': trainy})

    testx = test[0]
    testy = test[1]
    testy, testx = make_imbalance(testy, testx, r)
    testx = mnist_resize(testx)
    savemat(f'mnistm32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading mnistm32_train_{r}.mat for sanity check")
    data = loadmat(f'mnistm32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

    print(f"Loading mnistm32_test_{r}.mat for sanity check")
    data = loadmat(f'mnistm32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 5).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 5).astype(np.float32).sum())

if __name__ == '__main__':
    main()
