import numpy as np
import scipy.sparse
import torch


def cosine_similarity(a):
    # Calculate cosine similarity
    a_norm = a / a.norm(dim=1, keepdim=True)
    a_cos = a_norm @ a_norm.T

    return a_cos


def euclidean_distance(a, scaled=False, norm=False):
    # Calculate euclidean distance
    if a.dtype == torch.float16: a = a.type(torch.float32)
    dist = torch.cdist(a, a, p=2)
    # Scaled makes this equivalent to sqrt(MSE)
    if scaled: dist /= np.sqrt(a.shape[1])
    if norm: dist /= torch.norm(dist)
    return dist


def partition_distance(data, partitions=None, func=euclidean_distance):
    "Calculate distance only within specified partitions"
    # Base case
    if partitions is None: return func(data)

    # Argument handling
    if not isinstance(partitions, torch.Tensor): partitions = torch.Tensor(np.unique(partitions, return_inverse=True)[1]).long()

    # Calculate distance for each partition
    indices = torch.empty([2, 0], dtype=torch.int)
    values = torch.empty([0])
    for stage in partitions.unique():
        # Get indices
        flat_indices = torch.argwhere(partitions==stage).flatten()
        new_indices = torch.stack((
            flat_indices.reshape((-1, 1)).expand((-1, flat_indices.shape[0])).flatten(),
            flat_indices.reshape((1, -1)).expand((flat_indices.shape[0], -1)).flatten(),
        ), dim=0)

        # Get values
        new_values = func(data[partitions==stage]).flatten()

        # Append
        indices = torch.concat((indices, new_indices), dim=-1)
        values = torch.concat((values, new_values), dim=-1)
    
    dist = torch.sparse_coo_tensor(indices, values, 2*data.shape[0:1]).coalesce()
    return scipy.sparse.coo_matrix((dist.values(), dist.indices()), shape=dist.shape).tocsr()

def npolar_to_cartesian(coords):
    "Converts polar `coords` (r, theta_1, ...) to cartesian (x1, x2, ...)"
    # https://stackoverflow.com/a/20133681
    r = coords[..., [0]]
    thetas = torch.concat((torch.zeros((*coords.shape[:-1], 1), device=coords.device), coords[..., 1:]), dim=-1)
    thetas_sin = thetas.sin()
    thetas_sin[..., 0] = 1
    thetas_sin = thetas_sin.cumprod(dim=-1)
    thetas_cos = thetas.cos()
    thetas_cos = thetas_cos.roll(-1, dims=-1)
    return r * thetas_sin * thetas_cos


def vec_to_n1polar(orig):
    r = orig.norm(dim=-1)
    thetas = (orig / r).acos()
    return r, thetas


def n1polar_to_vec(r, thetas):
    return thetas.cos() * r
