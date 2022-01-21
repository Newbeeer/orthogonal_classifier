import os
import csv
import imageio
from scipy.io import loadmat, savemat
import numpy as np
directory = './svhn2mnist'
f = open('svhn_mnist_cyc.csv', 'w')
writer = csv.writer(f)

x = []
y = []
for idx, filename in enumerate(os.listdir(directory)):
    writer.writerow([int(filename[0]), filename])
    img = imageio.imread(os.path.join('svhn2mnist', filename))
    x.append(img[None, :, :, :]/255.)
    y.append(int(filename[0]))

    # print(img.shape)
    # print(int(filename[0]), filename)

x = np.concatenate(x, axis=0)
y = np.array(y)
print(x.shape, y.shape)
savemat(f'svhn2mnist32.mat', {'X': x, 'y': y})

print("svhn2mnist32.mat for sanity check")
data = loadmat(f'svhn2mnist32.mat')
print(data['X'].shape, data['X'].min(), data['X'].max())
print(data['y'].shape, data['y'].min(), data['y'].max())
