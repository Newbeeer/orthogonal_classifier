import numpy as np
import torch

def demographic_parity(y_, u):

    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


def equalized_odds(y, y_, u):
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    g = g / uc
    return np.abs(g[0, 1] - g[0, 0]) + np.abs(g[1, 1] - g[1, 0])


def equalizied_opportunity(y, y_logits, u):
    y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if y[i] < 0.999:
            continue
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


def accuracy(y, y_logits):
    y_ = (y_logits > 0.0).astype(np.float32)
    return np.mean((y_ == y).astype(np.float32))

def loss_dp(u_hat, u, y):

    distance = 0
    for i in range(2):
        for j in range(2):

            group = y==i
            distance += torch.abs(u_hat[group] - u[group]).sum() / (y==i).float().sum()
    return distance

def loss_eo(u_hat, u, y):

    distance = 0
    for i in range(2):
        for j in range(2):

            group = torch.logical_and(y==i, u ==j)
            distance += torch.abs(u_hat[group] - u[group]).sum() / (y[u==j]==i).float().sum()

    return distance