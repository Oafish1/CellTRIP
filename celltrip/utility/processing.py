import numpy as np
import scipy.sparse
import sklearn.decomposition
import torch

from .. import utility as _utility


class Preprocessing:
    "Apply modifications to input modalities based on given arguments. Takes np.array as input"
    def __init__(
        self,
        # Standardize
        standardize=True,
        # Filtering
        top_variant=int(4e4),
        # PCA
        pca_dim=512,  # TODO: Add auto-handling for less PCA than samples
        pca_copy=True,  # Set to false if too much memory being used
        # Subsampling
        num_nodes=None,
        num_features=None,
        # End cast
        device=None,
        **kwargs,
    ):
        self.standardize = standardize
        self.top_variant = top_variant
        self.pca_dim = pca_dim
        self.pca_copy = pca_copy  # Unused if sparse
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.device = device

        # Data
        self.is_sparse_transform = None

    
    def fit(self, modalities, *args, total_statistics=False, **kwargs):
        # Parameters
        self.is_sparse_transform = [scipy.sparse.issparse(m) for m in modalities]
        if isinstance(self.top_variant, int): self.top_variant = len(modalities) * [self.top_variant]
        if isinstance(self.pca_dim, int): self.pca_dim = len(modalities) * [self.pca_dim]

        # Standardize
        if self.standardize or self.top_variant is not None:
            self.standardize_mean = [
                np.mean(m, axis=0 if not total_statistics else None, keepdims=True)
                if not m_sparse else
                np.array(np.mean(m, axis=0 if not total_statistics else None).reshape((1, -1)))
                for m, m_sparse in zip(modalities, self.is_sparse_transform)
            ]
            get_standardize_std = lambda total_statistics: [
                np.std(m, axis=0 if not total_statistics else None, keepdims=True)
                if not m_sparse else
                np.array(np.sqrt(m.power(2).mean(axis=0 if not total_statistics else None) - np.square(m.mean(axis=0 if not total_statistics else None))).reshape((1, -1)))
                for m, m_sparse in zip(modalities, self.is_sparse_transform)
            ]
            self.standardize_std = get_standardize_std(total_statistics)

        # Filtering
        if self.top_variant is not None:
            # Calculate per-feature variance if needed
            st_std = self.standardize_std if not total_statistics else get_standardize_std(False)

            # Compute mask
            self.filter_mask = [
                np.argsort(std[0])[:-int(var+1):-1]
                if var is not None else None
                for std, var in zip(st_std, self.top_variant)
            ]
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask)]
            
            # Mask mean and std if needed
            if not total_statistics:
                self.standardize_mean = [st[:, mask] if mask is not None else st for m, st, mask in zip(modalities, self.standardize_mean, self.filter_mask)]
                self.standardize_std = [st[:, mask] if mask is not None else st for m, st, mask in zip(modalities, self.standardize_std, self.filter_mask)]

        # PCA
        if self.pca_dim is not None:
            self.pca_class = [
                sklearn.decomposition.PCA(
                    n_components=dim,
                    svd_solver='auto' if not m_sparse else 'arpack',
                    copy=self.pca_copy,
                ).fit(m)
                # sklearn.decomposition.TruncatedSVD(n_components=dim).fit(m)
                if dim is not None else None
                for m, m_sparse, dim in zip(modalities, self.is_sparse_transform, self.pca_dim)]
            
        return self

    def transform(self, modalities, features=None, **kwargs):
        # Filtering
        # NOTE: Determines if filtering is already done by shape checking the main `modalities` input
        if self.top_variant is not None and np.array([m.shape[1] != mask.shape[0] for m, mask in zip(modalities, self.filter_mask) if mask is not None]).any():
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask)]
            if features is not None: features = [fs[mask] if mask is not None else fs for fs, mask in zip(features, self.filter_mask)]

        # Standardize
        # NOTE: Not mean-centered for sparse matrices
        # TODO: Maybe allow for only one dataset to be standardized?
        if self.standardize:
            modalities = [
                (m - m_mean) / np.where(m_std == 0, 1, m_std) if not m_sparse else
                (m / np.where(m_std == 0, 1, m_std)).tocsr() if scipy.sparse.issparse(m) else
                (m / np.where(m_std == 0, 1, m_std))
                for m, m_mean, m_std, m_sparse in zip(modalities, self.standardize_mean, self.standardize_std, self.is_sparse_transform)]

        # PCA
        if self.pca_dim is not None:
            modalities = [p.transform(m) if p is not None else m for m, p in zip(modalities, self.pca_class)]

        ret = (modalities,)
        if features is not None: ret += (features,)
        return _utility.general.clean_return(ret)


    def inverse_transform(self, modalities, **kwargs):
        # NOTE: Does not reverse top variant filtering or feature sampling, also always dense output
        # PCA
        if self.pca_dim is not None:
            modalities = [p.inverse_transform(m) for m, p in zip(modalities, self.pca_class)]

        # Standardize
        if self.standardize:
            modalities = [
                (m_std * m + m_mean)
                if not m_sparse else
                (m_std * m)
                for m, m_mean, m_std, m_sparse in zip(modalities, self.standardize_mean, self.standardize_std, self.is_sparse_transform)]

        return modalities

    
    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    

    def cast(self, modalities, device=None, copy=False):
        if copy: modalities = modalities.copy()
        if device is None:
            assert self.device is not None, '`device` must be set to call `self.cast`'
            device = self.device

        # Cast types
        # NOTE: Always dense
        modalities = [
            torch.tensor(m if not scipy.sparse.issparse(m) else m.todense(), dtype=torch.float32, device=device)
            for m in modalities
        ]

        return modalities
    

    def inverse_cast(self, modalities, copy=False):
        if copy: modalities = modalities.copy()

        # Inverse cast
        modalities = [m.detach().cpu().numpy() for m in modalities]

        return modalities
    
    
    def subsample(self, modalities, types=None, partition=None, return_idx=False, **kwargs):
        # Subsample features
        # NOTE: Incompatible with inverse transform
        if self.num_features is not None:
            feature_idx = [np.random.choice(m.shape[1], nf, replace=False) for m, nf in zip(modalities, self.num_features)]
            modalities = [m[:, idx] for m, idx in zip(modalities, feature_idx)]

        node_idx = np.arange(modalities[0].shape[0])
        # Partition
        if partition is not None:
            partition_choice = np.random.choice(np.unique(partition), 1)[0]
            node_idx = node_idx[partition==partition_choice]
        
        # Subsample nodes
        if self.num_nodes is not None:
            assert np.array([m.shape[0] for m in modalities]).var() == 0, 'Nodes in all modalities must be equal to use node subsampling'
            if len(node_idx) > self.num_nodes: node_idx = np.random.choice(node_idx, self.num_nodes, replace=False)
            else: print(f'Skipping subsampling, only {len(node_idx)} nodes present.')
            
        # Apply subsampling
        modalities = [m[node_idx] for m in modalities]
        if types is not None: types = [t[node_idx] for t in types]

        ret = (modalities,)
        if types is not None: ret += (types,)
        if return_idx: ret += (node_idx,)
        # if self.num_features is not None: ret += (feature_idx,)
        return _utility.general.clean_return(ret)


class LazyComputation:
    "Lazily compute specified function once on first call"
    def __init__(self, init_func, *args, **kwargs):
        self.init_func = init_func
        self.args, self.kwargs = args, kwargs
        self.func = None

    def __call__(self, x):
        if self.func is None:
            self.func = self.init_func(*self.args, **self.kwargs)
        return self.func(x)


def split_state(
    state,
    idx=None,
    max_nodes=None,
    sample_strategy='random-proximity',
    reproducible_strategy='mean',
    sample_dim=None,  # Should be the dim of the env
    return_mask=False,
):
    "Split full state matrix into individual inputs, self_idx is an optional array"
    # Parameters
    if idx is None: idx = np.arange(state.shape[0]).tolist()
    if not _utility.general.is_list_like(idx): idx = [idx]
    self_idx = idx
    del idx

    # Get self features for each node
    self_entity = state[self_idx]

    # Get node features for each state
    node_mask = torch.eye(state.shape[0], dtype=torch.bool)
    node_mask = ~node_mask

    # Enforce reproducibility
    if reproducible_strategy is not None: generator = torch.Generator(device=state.device)
    else: generator = None

    # Hashing method
    if reproducible_strategy is None:
        pass
    # Hashing method
    # NOTE: Tensors are hashed by object, so would be unreliable to directly hash tensor
    elif reproducible_strategy == 'hash':
        generator.manual_seed(hash(str(state.detach().numpy())))
    # First number
    elif reproducible_strategy == 'first':
        generator.manual_seed((2**16*state.flatten()[0]).to(torch.long).item())
    # Mean value
    elif reproducible_strategy == 'mean':
        generator.manual_seed((2**16*state.mean()).to(torch.long).item())
    # Set seed (not recommended)
    elif type(reproducible_strategy) != str:
        generator.manual_seed(reproducible_strategy)
    else:
        raise ValueError(f'Reproducible strategy \'{reproducible_strategy}\' not found.')

    # Enforce max nodes
    num_nodes = state.shape[0] - 1
    use_mask = max_nodes is not None and max_nodes < num_nodes
    if use_mask:
        # Set new num_nodes
        num_nodes = max_nodes - 1

        # Random sample `num_nodes` to `max_nodes`
        if sample_strategy == 'random':
            # Filter nodes to `max_nodes` per idx
            probs = torch.rand_like(node_mask, dtype=torch.get_default_dtype(), generator=generator)
            probs[~node_mask] = 0
            selected_idx = probs.argsort(dim=-1)[..., -num_nodes:]  # Take `num_nodes` highest values

            # Create new mask
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool)
            node_mask[torch.arange(node_mask.shape[0]).unsqueeze(-1).expand(node_mask.shape[0], num_nodes), selected_idx] = True

        # Sample closest nodes
        elif sample_strategy == 'proximity':
            # Check for dim pass
            assert sample_dim is not None, (
                f'`sample_dim` argument must be passed if `sample_strategy` is \'{sample_strategy}\'')

            # Get inter-node distances
            dist = _utility.distance.euclidean_distance(state[..., :sample_dim])
            dist[~node_mask] = -1  # Set self-dist lowest for case of ties

            # Select `max_nodes` closest
            selected_idx = dist.argsort(dim=-1)[..., 1:num_nodes+1]
            
            # Create new mask
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool)
            node_mask[torch.arange(node_mask.shape[0]).unsqueeze(-1).expand(node_mask.shape[0], num_nodes), selected_idx] = True

        # Randomly sample from a distribution of node distance
        elif sample_strategy == 'random-proximity':
            # Check for dim pass
            assert sample_dim is not None, (
                f'`sample_dim` argument must be passed if `sample_strategy` is \'{sample_strategy}\'')

            # Get inter-node distances
            dist = _utility.distance.euclidean_distance(state[..., :sample_dim])
            prob = 1 / (dist+1)
            prob[~node_mask] = 0  # Remove self

            # Randomly sample
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool)
            idx = prob.multinomial(num_nodes, replacement=False, generator=generator)
            node_mask[torch.arange(node_mask.shape[0]).unsqueeze(-1).expand(node_mask.shape[0], num_nodes), idx] = True
        else:
            # TODO: Verify works
            raise ValueError(f'Sample strategy \'{sample_strategy}\' not found.')
        
    # Shrink mask to appropriate size
    # NOTE: Randomization needs to be done on all samples for reproducibility
    # print(node_mask)
    node_mask = node_mask[self_idx]
    
    # Final formation
    node_entities = state.unsqueeze(0).expand(len(self_idx), *state.shape)
    node_entities = node_entities[node_mask].reshape(len(self_idx), num_nodes, state.shape[1])

    # Return
    ret = (self_entity, node_entities)
    if return_mask: ret += (node_mask,)
    return ret


def recursive_tensor_func(input, func):
    "Applies function to tensors recursively"
    if type(input) == torch.Tensor:
        return func(input)
    
    return [recursive_tensor_func(member, func) for member in input]


def dict_map(dict, func, inplace=False):
    "Takes input dict `dict` and applies function to all members"
    if inplace:
        for k in dict: dict[k] = func(dict[k])
        return dict
    else:
        return {k: func(v) for k, v in dict.items()}
    

def dict_map_recursive_tensor_idx_to(dict, idx, device):
    "Take `idx` and cast to `device` from tensors inside of a dict, recursively"
    # NOTE: List indices into tensors create a copy
    # TODO: Add slice compatibility
    # Define function for each member
    def subfunc(x):
        if idx is not None: x = x[idx]
        if device is not None: x = x.to(device)
        return x
    
    # Apply and return (if needed)
    dict = dict_map(dict, lambda x: recursive_tensor_func(x, subfunc))
    return dict