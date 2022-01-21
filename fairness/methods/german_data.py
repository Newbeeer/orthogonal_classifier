import numpy as np
import pandas as pd
#import tensorflow as tf
import pickle as pkl
import torch
import torch.utils.data
#tfd = tf.contrib.distributions


def create_german_datasets(batch=64):
    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    def transform_to_binary(df, labels):
        d = np.zeros([df.shape[0], 58])
        u = np.zeros([df.shape[0], 1])
        y = np.zeros([df.shape[0], 1])
        idx = 0
        for j in range(len(labels)):
            if type(labels[j]) is list:
                if len(labels[j]) > 2:
                    for i in range(df.shape[0]):
                        d[i, idx + int(labels[j].index(df[i, j]))] = 1
                    idx += len(labels[j])
                else:
                    for i in range(df.shape[0]):
                        d[i, idx] = int(labels[j].index(df[i, j]))
                    idx += 1
            else:
                if j != 12 and j != len(labels) - 1:
                    for i in range(df.shape[0]):
                        d[i, idx] = float(df[i, j] > labels[j])
                    idx += 1
                elif j == len(labels) - 1:
                    for i in range(df.shape[0]):
                        y[i] = float(df[i, j] > labels[j])
                else:
                    for i in range(df.shape[0]):
                        u[i] = float(df[i, j] > labels[j])
        return d.astype(np.bool), u.astype(np.bool), y.astype(np.bool)  # observation, protected, label

    d = pd.read_csv('../data/german.data.txt', header=None, sep=' ').to_numpy()
    labels = gather_labels(d)
    ds = transform_to_binary(d, labels)

    idx = np.arange(d.shape[0])
    np.random.seed(4)
    np.random.shuffle(idx)
    cf = int(d.shape[0] * 0.9)

    german = tuple([a[idx[:cf]].astype(np.float32) for a in ds])

    print(german[0].shape,german[1].shape,german[2].shape)

    return german


def create_torch_dataloader(batch=64):

    data = create_german_datasets(batch=batch)
    class German(torch.utils.data.Dataset):

        def __init__(self,data,train):
            self.train = train

            if self.train:
                self.x = data[0][:800].astype(np.float32)
                self.u = data[1][:800].astype(np.float32)
                self.y = data[2][:800].astype(np.float32)
            else:
                self.x = data[0][800:].astype(np.float32)
                self.u = data[1][800:].astype(np.float32)
                self.y = data[2][800:].astype(np.float32)

        def __getitem__(self, index):

            return self.x[index],self.u[index],self.y[index]

        def __len__(self):
            return len(self.x)

    German_train = German(data,train=True)
    German_test = German(data,train=False)
    train_loader = torch.utils.data.DataLoader(dataset=German_train, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=German_test, batch_size=batch, shuffle=False)
    return train_loader,test_loader


if __name__ == '__main__':
    #save_adult_datasets()
    create_torch_dataloader()