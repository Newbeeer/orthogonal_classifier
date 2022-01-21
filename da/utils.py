import numpy as np
import torch
import torch.nn as nn
from cvxopt import matrix, solvers

def create_im_weights_update(class_inst, ma, class_num):

    # Label importance weights.
    class_inst.ma = ma
    class_inst.im_weights = nn.Parameter(
        torch.ones(class_num, 1), requires_grad=False)

def im_weights_update(source_y, target_y, cov, im_weights, ma):
    """
    Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
    :param source_y:    The marginal label distribution of the source domain.
    :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
    :param cov:         The covariance matrix of predicted-label and true label of the source domain.
    :param device:      Device of the operation.
    :return:
    """
    # Convert all the vectors to column vectors.
    dim = cov.shape[0]
    source_y = source_y.reshape(-1, 1).astype(np.double)
    target_y = target_y.reshape(-1, 1).astype(np.double)
    cov = cov.astype(np.double)

    P = matrix(np.dot(cov.T, cov), tc="d")
    q = -matrix(np.dot(cov, target_y), tc="d")
    G = matrix(-np.eye(dim), tc="d")
    h = matrix(np.zeros(dim), tc="d")
    A = matrix(source_y.reshape(1, -1), tc="d")
    b = matrix([1.0], tc="d")
    sol = solvers.qp(P, q, G, h, A, b)
    new_im_weights = np.array(sol["x"])

    # EMA for the weights
    im_weights.data = (1 - ma) * torch.tensor(
        new_im_weights, dtype=torch.float32).cuda() + ma * im_weights.data

    return im_weights