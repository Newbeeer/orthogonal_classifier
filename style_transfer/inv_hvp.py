#! /usr/bin/env python3

import torch
from torch.autograd import grad
import torch.nn.functional as F


def sto(v, model, batch, chunk=4):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.
    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = v.detach()
    h_estimate = v
    cnt = 0.
    batch_x = torch.chunk(batch[0], chunk, dim=0)
    batch_y = torch.chunk(batch[1], chunk, dim=0)
    model.eval()
    for i in range(len(batch_x)):
        y = model(batch_x[i])
        loss = F.cross_entropy(y, batch_y[i])
        hv = hvp(loss, model.weight, h_estimate)
        h_estimate = v + h_estimate - hv
        h_estimate = h_estimate.detach()
        cnt += 1

        # not converge
        if torch.max(abs(h_estimate)) > 10:
             break


    model.train()
    return h_estimate.detach()

def neum(v, model, batch):
    v = v.detach()
    h_estimate = v
    cnt = 0.
    model.eval()
    iter = 10
    for i in range(iter):
        model.weight.grad *= 0
        y = model(batch[0].detach())
        loss = F.cross_entropy(y, batch[1].detach())
        hv = hvp(loss, model.weight, v)
        v -= hv
        v = v.detach()
        h_estimate = v + h_estimate
        h_estimate = h_estimate.detach()
        # not converge
        if torch.max(abs(h_estimate)) > 10:
             break
        cnt += 1

    model.train()
    return h_estimate.detach()

def s_test(v_q, model, batch,  damp=0.01, scale=25.0):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.
    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = v_q.detach()
    h_estimate = v
    cnt = 0.
    for x, t in batch:
        t = t.view(1)
        x = x.view(1, -1)
        x, t = x.cuda(), t.cuda()
        y = model(x)
        loss = F.cross_entropy(y, t)
        hv = hvp(loss, model.weight, h_estimate)
        h_estimate = v + h_estimate - hv
        h_estimate = h_estimate.detach()
        cnt += 1
        break
    return h_estimate


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
    first_grads = torch.nn.utils.parameters_to_vector(first_grads)
    # Elementwise products
    elemwise_products = first_grads @ v
    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)
    return_grads = torch.nn.utils.parameters_to_vector(return_grads)
    return return_grads