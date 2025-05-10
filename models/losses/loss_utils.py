# Functions and classes used by different loss functions
import numpy as np
import torch
from torch import Tensor

EPS = 1e-5


def metrics_mean(l):
    # Compute the mean and return as Python number
    metrics = {}
    for e in l:
        for metric_name in e:
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(e[metric_name])

    for metric_name in metrics:
        metrics[metric_name] = np.mean(np.array(metrics[metric_name]))

    return metrics


def squared_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    '''
    Compute squared Euclidean distance
    Input: x is Nxd matrix
           y is Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def sigmoid(tensor: Tensor, temp: float) -> Tensor:
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x: Tensor, similarity: str = 'cosine') -> Tensor:
    """computes the affinity matrix between an input vector and itself"""
    if similarity == 'cosine':
        x = torch.mm(x, x.t())
    elif similarity == 'euclidean':
        x = x.unsqueeze(0)
        x = torch.cdist(x, x, p=2)
        x = x.squeeze(0)
        # The greater the distance the smaller affinity
        x = -x
    else:
        raise NotImplementedError(f"Incorrect similarity measure: {similarity}")
    return x