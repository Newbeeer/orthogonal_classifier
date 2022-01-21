import subprocess
import os
from scipy.io import loadmat, savemat
import numpy as np
import gzip
import pickle
from skimage.transform import resize

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

def main():

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    root = '/home/ylxu/data'
    trainx = []
    trainy = []
    for file_name, checksum in train_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            trainx.append(entry['data'])
            if 'labels' in entry:
                trainy.extend(entry['labels'])
            else:
                trainy.extend(entry['fine_labels'])
    trainx = np.vstack(trainx).reshape(-1, 3, 32, 32)
    trainx = trainx.transpose(0, 2, 3, 1)

    testx = []
    testy = []
    for file_name, checksum in test_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            testx.append(entry['data'])
            if 'labels' in entry:
                testy.extend(entry['labels'])
            else:
                testy.extend(entry['fine_labels'])
    testx = np.vstack(testx).reshape(-1, 3, 32, 32)
    testx = testx.transpose(0, 2, 3, 1)

    cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
    trainy = cls_mapping[trainy]
    testy = cls_mapping[testy]

    # Remove all samples from skipped classes
    train_mask = trainy != -1
    test_mask = testy != -1

    trainx = trainx[train_mask] / 255.
    trainy = trainy[train_mask]
    testx = testx[test_mask] / 255.
    testy = testy[test_mask]

    print("Train x y:", trainx.shape, len(trainy))
    print("Test x y:", testx.shape, len(testy))

    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'/home/ylxu/dirt-t/data/cifar32_train_{r}.mat', {'X': trainx, 'y': trainy})

    testy, testx = make_imbalance(testy, testx, r)
    savemat(f'/home/ylxu/dirt-t/data/cifar32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading cifar32_train_{r}.mat for sanity check")
    data = loadmat(f'/home/ylxu/dirt-t/data/cifar32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 4).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 4).astype(np.float32).sum())

    print(f"Loading cifar32_test_{r}.mat for sanity check")
    data = loadmat(f'/home/ylxu/dirt-t/data/cifar32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 4).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 4).astype(np.float32).sum())

if __name__ == '__main__':
    main()
