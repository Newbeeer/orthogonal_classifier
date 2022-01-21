import numpy as np
import tensorflow as tf
import pandas as pd
import torch
import torch.utils.data
tfd = tf.contrib.distributions


def create_health_dataset(batch=64):
    d = pd.read_csv('../health.csv')
    d = d[d['YEAR_t'] == 'Y3']
    sex = d['sexMISS'] == 0
    age = d['age_MISS'] == 0
    d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    d = d[sex & age]

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    ages = d[['age_%d5' % (i) for i in range(0, 9)]]
    sexs = d[['sexMALE', 'sexFEMALE']]
    charlson = d['CharlsonIndexI_max']

    x = d.drop(
        ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1).as_matrix()
    labels = gather_labels(x)
    xs = np.zeros_like(x)
    for i in range(len(labels)):
        xs[:, i] = x[:, i] > labels[i]
    x = xs[:, np.nonzero(np.mean(xs, axis=0) > 0.05)[0]].astype(np.float32)

    u = np.expand_dims(sexs.as_matrix()[:, 0], 1)
    v = ages.as_matrix()
    u = np.concatenate([v, u], axis=1).astype(np.float32)
    y = (charlson.as_matrix() > 0).astype(np.float32)

    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    cf = int(0.8 * d.shape[0])
    train_ = (x[idx[:cf]], u[idx[:cf]], y[idx[:cf]])
    test_ = (x[idx[cf:]], u[idx[cf:]], y[idx[cf:]])

    label_train = np.argmax(train_[1][:,:9],axis=1) + 9 * (train_[1][:,9] == 1).astype(np.float32)

    label_test = np.argmax(test_[1][:, :9], axis=1) + 9 * (test_[1][:, 9] == 1).astype(np.float32)
    train_ = (x[idx[:cf]], label_train, y[idx[:cf]])
    test_ = (x[idx[cf:]], label_test, y[idx[cf:]])
    for i in range(3):
        print(train_[i].shape,test_[i].shape)
    return train_,test_

def create_torch_dataloader(batch=64):

    train_,test_ = create_health_dataset(batch=batch)
    class Health(torch.utils.data.Dataset):

        def __init__(self,data,train):
            self.train = train

            if self.train:


                self.x = data[0].astype(np.float32)
                self.u = data[1].astype(np.float32)
                self.y = data[2].astype(np.float32)
            else:

                self.x = data[0].astype(np.float32)
                self.u = data[1].astype(np.float32)
                self.y = data[2].astype(np.float32)

        def __getitem__(self, index):

            return self.x[index],self.u[index],self.y[index]

        def __len__(self):
            return len(self.x)

    Health_train = Health(train_,train=True)
    Health_test = Health(test_,train=False)
    train_loader = torch.utils.data.DataLoader(dataset=Health_train, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=Health_test, batch_size=batch, shuffle=False)
    return train_loader,test_loader


if __name__ == '__main__':
    #save_adult_datasets()
    #create_torch_dataloader()