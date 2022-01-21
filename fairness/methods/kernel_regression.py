import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import  pdist,squareform,cdist
import torch
class KernelRegression:
    def __init__(self,bandwidth,X,Y):
        self.bandwidth=bandwidth
        self.X = X
        self.Y = Y

    def predict_u(self,data,train=False):
        #size=self.X.shape[0]
        size = self.X.size(0)

        x2 = (data ** 2).sum(1).unsqueeze(1)
        y2 = (self.X ** 2).sum(1).unsqueeze(0)

        distance = x2 + y2 - 2 * (data @ self.X.transpose(0,1))
        if train:
            distance = distance + 1000 * (torch.eye(size)).cuda()
        kernel_dist=self.rbf_kernel(distance)

        sum = torch.sum(kernel_dist, dim=1).unsqueeze(1)
        y_onehot = torch.FloatTensor(self.Y.size(0), 18).zero_().cuda()
        y_onehot.scatter_(1, self.Y.detach().unsqueeze(1).long(), 1)
        #sum=np.sum(kernel_dist,axis=1).reshape((size,1))
        weight = kernel_dist/sum
        #print(weight.size(),y_onehot.size())
        pred=weight @ y_onehot

        #print(pred.size(),pred.sum(1),pred)
        return pred
    def rbf_kernel(self,X):

        return torch.exp(-X/(self.bandwidth**2))


