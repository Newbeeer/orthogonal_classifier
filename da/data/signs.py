import click

@click.command()
@click.option('--width', type=int, default=32)
@click.option('--height', type=int, default=32)
@click.option('--ignore_roi', is_flag=True, default=False)



def prepare(width, height, ignore_roi):
    import os
    import sys
    import numpy as np
    import tables
    import pandas as pd
    import tqdm
    import cv2
    r = [0.7, 0.3]
    from scipy.io import loadmat, savemat
    import numpy as np
    def make_imbalance(label, img, r=[1., 1.]):
        if r[0] == 0:
            return label, img
        label = np.array(label)
        img = np.array(img)
        mask = label >= 21
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
    synsigns_path = '.'


    data_path = os.path.join(synsigns_path, 'synthetic_data')
    labels_path = os.path.join(data_path, 'train_labelling.txt')

    if not os.path.exists(labels_path):
        print('Labels path {} does not exist'.format(labels_path))
        sys.exit(0)

    # Open the file that lists the image files along with their ground truth class
    lines = [line.strip() for line in open(labels_path, 'r').readlines()]
    lines = [line for line in lines if line != '']


    train_X_arr = []
    y = []
    for line in tqdm.tqdm(lines):
        image_filename, gt, _ = line.split()
        image_path = os.path.join(data_path, image_filename)

        if not os.path.exists(image_path):
            print('Could not find image file {} mentioned in annotations'.format(image_path))
            return
        image_data = cv2.imread(image_path)[:, :, ::-1]
        image_data = cv2.resize(image_data, (32, 32), interpolation=cv2.INTER_AREA)
        train_X_arr.append(image_data)
        y.append(int(gt))

    y = np.array(y, dtype=np.int32)
    x = np.empty((len(train_X_arr), 32, 32, 3))
    for i in range(len(train_X_arr)):
        x[i] = train_X_arr[i]

    # Create train/test split
    np.random.seed(0)
    shuffle = np.random.permutation(len(x))

    x = x[shuffle]
    y = y[shuffle]
    n = 95000
    trainx, trainy = x[:n], y[:n]
    testx, testy = x[n:], y[n:]
    print('train_X.shape={}'.format(trainx.shape))
    print('train_y.shape={}'.format(trainy.shape))

    print('test_X.shape={}'.format(testx.shape))
    print('test_y.shape={}'.format(testy.shape))

    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'signs32_train_{r}.mat', {'X': trainx, 'y': trainy})

    testy, testx = make_imbalance(testy, testx, r)
    savemat(f'signs32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading signs32_train_{r}.mat for sanity check")
    data = loadmat(f'signs32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 21).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 21).astype(np.float32).sum())

    print(f"Loading signs32_test_{r}.mat for sanity check")
    data = loadmat(f'signs32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 21).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 21).astype(np.float32).sum())

if __name__ == '__main__':
    prepare()