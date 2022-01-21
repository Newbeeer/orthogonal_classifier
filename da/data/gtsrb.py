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
    r = [0.3, 0.7]
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
    path = './GTSRB'

    output_path = os.path.join(path, 'gtsrb.h5')
    print('Creating {}...'.format(output_path))

    filters = tables.Filters(complevel=9, complib='blosc')
    train_X_arr = []
    train_y_arr = []


    train_path = os.path.join(path, 'Final_Training', 'Images')
    #test_path = os.path.join(path, 'Final_Test', 'Images')

    if not os.path.exists(train_path):
        print('ERROR!!! Training images path {} does not exist'.format(train_path))
        return



    def load_image_dir(X_arr, y_arr, dir_path, anno_path):
        if not os.path.exists(anno_path):
            print('ERROR!!! Could not find annotations file {}'.format(anno_path))
            return False

        annotations = pd.read_csv(anno_path, sep=';')

        for index, row in tqdm.tqdm(annotations.iterrows(), desc='Images', total=len(annotations.index)):
            image_filename = row['Filename']
            image_path = os.path.join(dir_path, image_filename)
            if not os.path.exists(image_path):
                print('ERROR!!!  Could not find image file {} mentioned in annotations'.format(image_path))
                return False
            image_data = cv2.imread(image_path)[:, :, ::-1]
            # if not ignore_roi:
            #     # Crop out the region of interest
            #     roi_x1 = int(row['Roi.X1'])
            #     roi_x2 = int(row['Roi.X2'])
            #     roi_y1 = int(row['Roi.Y1'])
            #     roi_y2 = int(row['Roi.Y2'])
            #     image_data = image_data[roi_y1:roi_y2, roi_x1:roi_x2, :]

            image_data = cv2.resize(image_data, (width, height), interpolation=cv2.INTER_AREA)
            class_id = int(row['ClassId'])
            X_arr.append(image_data)
            y_arr.append(np.array([class_id], dtype=np.int32))

        return True

    print('Processing training data...')
    for clf_dir_name in tqdm.tqdm(os.listdir(train_path), desc='Class'):
        clf_ndx = int(clf_dir_name)
        clf_path = os.path.join(train_path, clf_dir_name)
        anno_path = os.path.join(clf_path, 'GT-{:05d}.csv'.format(clf_ndx))
        success = load_image_dir(train_X_arr, train_y_arr, clf_path, anno_path)


    x = np.empty((len(train_X_arr), 32, 32, 3))
    y = np.empty((len(train_X_arr)))
    for i in range(len(train_X_arr)):
        x[i] = train_X_arr[i]
        y[i] = int(train_y_arr[i])

    # Create train/test split
    np.random.seed(0)
    # Following Asymmetric Tri-training protocol
    # https://arxiv.org/pdf/1702.08400.pdf
    shuffle = np.random.permutation(len(x))

    x = x[shuffle]
    y = y[shuffle]
    n = 31367
    trainx, trainy = x[:n], y[:n]
    testx, testy = x[n:], y[n:]
    print('train_X.shape={}'.format(trainx.shape))
    print('train_y.shape={}'.format(trainy.shape))

    print('test_X.shape={}'.format(testx.shape))
    print('test_y.shape={}'.format(testy.shape))
    # print('Processing test data...')
    # test_anno_path = os.path.join(path, 'GT-final_test.csv')
    # success = load_image_dir(test_X_arr, test_y_arr, test_path, test_anno_path)
    # if not success:
    #     f_out.close()
    #     os.remove(output_path)
    #     return
    # print('test_X.shape={}'.format(f_out.root.gtsrb.test_X_u8.shape))
    # print('test_y.shape={}'.format(f_out.root.gtsrb.test_y.shape))

    print("Train:", trainx.shape, trainy.shape)
    trainy, trainx = make_imbalance(trainy, trainx, r)
    savemat(f'gtsrb32_train_{r}.mat', {'X': trainx, 'y': trainy})


    testy, testx = make_imbalance(testy, testx, r)
    savemat(f'gtsrb32_test_{r}.mat', {'X': testx, 'y': testy})

    print(f"Loading gtsrb32_train_{r}.mat for sanity check")
    data = loadmat(f'gtsrb32_train_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 21).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 21).astype(np.float32).sum())

    print(f"Loading gtsrb32_test_{r}.mat for sanity check")
    data = loadmat(f'gtsrb32_test_{r}.mat')
    print(data['X'].shape, data['X'].min(), data['X'].max())
    print(data['y'].shape, data['y'].min(), data['y'].max())
    print("Label 0 size:", (data['y'] < 21).astype(np.float32).sum(), "Label 1 size:", (data['y'] >= 21).astype(np.float32).sum())

if __name__ == '__main__':
    prepare()