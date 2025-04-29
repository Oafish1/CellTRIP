import functools as ft
import os
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
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
        top_variant=int(5e3),
        # PCA
        pca_dim=2**8,  # TODO: Add auto-handling for less PCA than samples
        # Fitting
        seed=None,
        # Subsampling
        num_nodes=int(5e3),
        subsample_seed=None,
        # End cast
        device=None,
        **kwargs,
    ):
        self.standardize = standardize
        self.top_variant = top_variant
        self.pca_dim = pca_dim
        self.num_nodes = num_nodes
        self.seed = seed
        self.subsample_seed = subsample_seed
        self.device = device

        # Subsample random
        if subsample_seed is not None: self.subsample_rng = np.random.default_rng(subsample_seed)
        else: self.subsample_rng = np.random

        # Data
        self.is_sparse_transform = None

    def set_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes
    
    def fit(self, modalities, *, total_statistics=False, **kwargs):
        # Parameters
        self.is_sparse_transform = [scipy.sparse.issparse(m) for m in modalities]
        if isinstance(self.top_variant, int): self.top_variant = len(modalities) * [self.top_variant]
        if isinstance(self.pca_dim, int): self.pca_dim = len(modalities) * [self.pca_dim]
        modal_feature_nums = [m.shape[1] for m in modalities]

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
        # NOTE: Currently calculates even if not needed and `top_variant` is set
        # NOTE: Currently requires standardization
        if self.top_variant is not None:
            # Calculate per-feature variance if needed
            st_std = self.standardize_std if not total_statistics else get_standardize_std(False)

            # Compute mask
            self.filter_mask = [
                np.argsort(std[0])[:-int(var+1):-1]
                if var is not None and std[0].shape[0] > var else None
                for std, var in zip(st_std, self.top_variant)
            ]
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask)]
            
            # Mask mean and std if needed
            if not total_statistics:
                self.standardize_mean = [st[:, mask] if mask is not None else st for m, st, mask in zip(modalities, self.standardize_mean, self.filter_mask)]
                self.standardize_std = [st[:, mask] if mask is not None else st for m, st, mask in zip(modalities, self.standardize_std, self.filter_mask)]

        # Apply standardization
        if self.standardize:
            modalities = [
                (m - m_mean) / np.where(m_std == 0, 1, m_std) if not m_sparse else
                (m / np.where(m_std == 0, 1, m_std)).tocsr() if scipy.sparse.issparse(m) else
                (m / np.where(m_std == 0, 1, m_std))
                for m, m_mean, m_std, m_sparse in zip(modalities, self.standardize_mean, self.standardize_std, self.is_sparse_transform)]

        # PCA
        if self.pca_dim is not None:
            for i in range(len(modalities)):
                if self.pca_dim[i] is not None and modal_feature_nums[i] <= self.pca_dim[i]:
                    warnings.warn(
                        f'Modality {i} too small for PCA ({modal_feature_nums[i]} features), skipping',
                        RuntimeWarning)
                    self.pca_dim[i] = None
            self.pca_class = [
                None if dim is None else
                sklearn.decomposition.PCA(
                    n_components=dim,
                    svd_solver='auto',
                    random_state=self.seed).fit(m) if not m_sparse else
                sklearn.decomposition.TruncatedSVD(
                    n_components=dim,
                    random_state=self.seed).fit(m)
                for m, m_sparse, dim in zip(modalities, self.is_sparse_transform, self.pca_dim)]
            
        # Records
        self.modal_dims = [
            modalities[i].shape[1]
            if self.pca_dim[i] is None else
            self.pca_dim[i]
            for i in range(len(modalities))]
            
        return self

    def transform(self, modalities, adata_vars=None, force_filter=True, subset_features=None, subset_modality=None, *args, **kwargs):
        # Default
        # NOTE: `subset_modality` currently incompatible with list arguments, kind of hacky
        if subset_features is not None:
            assert subset_modality is not None, (
                'Argument `subset_features` may not be used without `subset_modality`')
            if not _utility.general.is_list_like(subset_features): subset_features = [subset_features]
        if subset_modality is None: subset_modality = slice(len(self.is_sparse_transform))
        else:
            # Convert to slice
            subset_modality = slice(subset_modality, subset_modality+1)
            # Sanitize inputs
            modalities = [modalities]
            if adata_vars is not None: adata_vars = [adata_vars]
        sm = subset_modality
        
        # Filtering
        # NOTE: Determines if filtering is already done by shape checking the main `modalities` input
        already_applied = not np.array([m.shape[1] != mask.shape[0] for m, mask in zip(modalities, self.filter_mask[sm]) if mask is not None]).any()
        if force_filter is None and already_applied:
            warnings.warn('`force_filter` not assigned, assuming data has already been filtered')
        if self.top_variant is not None and (force_filter or not already_applied):
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask[sm])]
            if adata_vars is not None:
                # NOTE: Creates a view of the object
                adata_vars = [adata_var.iloc[mask] if mask is not None else adata_var for adata_var, mask in zip(adata_vars, self.filter_mask[sm])]

        # Standardize
        # NOTE: Not mean-centered for sparse matrices
        # TODO: Maybe allow for only one dataset to be standardized?
        if self.standardize:
            modalities = [
                (m - m_mean) / np.where(m_std == 0, 1, m_std) if not m_sparse else
                (m / np.where(m_std == 0, 1, m_std)).tocsr() if scipy.sparse.issparse(m) else
                (m / np.where(m_std == 0, 1, m_std))
                for m, m_mean, m_std, m_sparse in zip(modalities, self.standardize_mean[sm], self.standardize_std[sm], self.is_sparse_transform[sm])]

        # Subset features
        if subset_features is not None:
            m_idx = list(range(*subset_modality.indices(len(self.is_sparse_transform))))
            assert len(m_idx) == 1, 'Currently, `subset_features` only compatible with one modality'
            if self.top_variant is not None and self.filter_mask[sm][0] is not None:
                if np.intersect1d(subset_features, self.filter_mask[sm][0]).shape[0] != len(subset_features):
                    warnings.warn('Some `subset_features` elements not included due to filtering', RuntimeWarning)
                idx_loc = [np.argwhere(self.filter_mask[sm][0] == sf).flatten()[0] for sf in subset_features]
            else: idx_loc = subset_features
            # Set all features but requested to zero/center
            all_but_needed = np.setxor1d(np.arange(modalities[0].shape[1]), np.array(idx_loc))
            # PCA compatibility
            if self.pca_class[sm][0] is not None and not self.is_sparse_transform[sm][0]:
                center = self.pca_class[sm][0].mean_
                center = center[all_but_needed]
            else: center = 0
            modalities[0][:, all_but_needed] = center
            
        # PCA
        if self.pca_dim is not None:
            modalities = [p.transform(m) if p is not None else m for m, p in zip(modalities, self.pca_class[sm])]
        
        # Return
        # NOTE: Returns features before PCA transformation, also always dense
        modalities = [m if not scipy.sparse.issparse(m) else m.toarray() for m in modalities]
        modalities = list(map(lambda m: m.astype(np.float32), modalities))
        ret = (modalities,)
        if adata_vars is not None: ret += (adata_vars,)
        return _utility.general.clean_return(ret)


    def inverse_transform(self, modalities, subset_modality=None, *args, **kwargs):
        # Defaults
        if subset_modality is None: subset_modality = slice(len(self.is_sparse_transform))
        else:
            # Convert to slice
            subset_modality = slice(subset_modality, subset_modality+1)
            # Sanitize inputs
            modalities = [modalities]
        sm = subset_modality

        # NOTE: Does not reverse top variant filtering
        # PCA
        if self.pca_dim is not None:
            modalities = [p.inverse_transform(m) if p is not None else m for m, p in zip(modalities, self.pca_class[sm])]

        # Standardize
        if self.standardize:
            modalities = [
                (m_std * m + m_mean)
                if not m_sparse else
                (m_std * m)
                for m, m_mean, m_std, m_sparse in zip(modalities, self.standardize_mean[sm], self.standardize_std[sm], self.is_sparse_transform[sm])]

        return modalities

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    
    def subsample(
        self,
        modalities=None,
        adata_obs=None,
        partition_cols=None,
        mask=None,
        return_partition=False,
        **kwargs,
    ):
        # Defaults
        assert not np.array([v is None for v in (modalities, adata_obs)]).all(), (
            'At least one of `modalities` or `adata_obs` must be provided')
        if adata_obs is None:
            assert np.array([m.shape[0] == modalities[0].shape[0] for m in modalities]).all(), (
                'No `adata_obs` provided but datasets are not aligned')
        modal_sizes = (
            [m.shape[0] for m in modalities] if modalities is not None else
            [adata_ob.shape[0] for adata_ob in adata_obs])
        if mask is not None: sample_mask = np.array(mask)
        else: sample_mask = None
            
        # Align datasets
        # TODO: Add as auxiliary function
        # if adata_obs is not None:
        #     # Get idx for alignment
        #     obs_names = [adata_ob.index.to_list() for adata_ob in adata_obs]
        #     all_names = np.unique(sum(obs_names, []))
        #     intersecting_names = ft.reduce(lambda l, r: np.intersect1d(l, r), obs_names)
        #     non_intersecting_names = np.array(list(set(all_names) - set(intersecting_names[:-2])))
        #     obs_order = np.concatenate([intersecting_names, non_intersecting_names])
        #     # There has to be a better way, right?
        #     series = [pd.Series(np.arange(adata_ob.shape[0]), adata_ob.index) for adata_ob in adata_obs]
        #     series = [s[[idx for idx in obs_order if idx in s.index]].to_numpy() for s in series]
        #     # Apply
        #     if modalities is not None:
        #         modalities = [m[s] for m, s in zip(modalities, series)]
        #     adata_obs = [adata_ob.iloc[s] for adata_ob, s in zip(adata_obs, series)]

        # Apply mask
        if sample_mask is not None:
            if modalities is not None:
                modalities = [m[sample_mask] for m in modalities]
            if adata_obs is not None:
                adata_obs = [adata_ob[sample_mask] for adata_ob in adata_obs]
        
        # Partition
        selected_partition = None
        if partition_cols is not None:
            if adata_obs is None: raise AttributeError('Cannot compute partitions without `adata_obs`.')
            # unique_partition_vals = [np.unique([adata_ob[col].unique() for adata_ob in adata_obs]) for col in partition_cols]
            adata_ob = adata_obs[0]  # Just use first modality to determine, TODO: will need to revise for data with unmatched partitions
            unique_partition_vals = (
                adata_ob.drop_duplicates(partition_cols)[partition_cols]
                .apply(lambda r: tuple(r[col] for col in r.index), axis=1).to_numpy())
            # Continually sample until non-zero combination is found
            while True:
                # selected_partition = _utility.general.rolled_index(
                #     unique_partition_vals, np.random.choice(np.prod(list(map(len, unique_partition_vals)))))
                selected_partition = np.random.choice(unique_partition_vals)
                masks = [
                    np.prod([
                        adata_ob[col] == val
                        for col, val in zip(partition_cols, selected_partition)], axis=0).astype(bool)
                    for adata_ob in adata_obs]
                modal_sizes = [mask.sum() for mask in masks]
                if sum(modal_sizes) > 0: break

            # Apply
            if modalities is not None:
                modalities = [m[mask] for m, mask in zip(modalities, masks)]
            adata_obs = [adata_ob[mask] for adata_ob, mask in zip(adata_obs, masks)]
        
        # Subsample nodes
        if self.num_nodes is not None and np.array([ms > self.num_nodes for ms in modal_sizes]).all():
            if adata_obs is not None:
                # Get all sample ids and choose some
                sample_ids = np.unique(sum([adata_ob.index.to_list() for adata_ob in adata_obs], []))
                choice = self.subsample_rng.choice(sample_ids, self.num_nodes, replace=False)
                # Get numerical index
                # There has to be a better way, right?
                series = [pd.Series(np.arange(adata_ob.shape[0]), adata_ob.index) for adata_ob in adata_obs]
                node_idxs = [s[[idx for idx in choice if idx in s.index]].to_numpy() for s in series]
            else:
                # Assume all samples are aligned
                assert np.array([ms == modal_sizes[0] for ms in modal_sizes]), (
                    'Modalities must be aligned if `adata_obs` is not provided and '
                    '`num_samples` is not `None`')
                node_idx = self.subsample_rng.choice(modal_sizes[0], self.num_nodes, replace=False)
                node_idxs = [node_idx for _ in range(len(modal_sizes))]

            # Sort (required for on-disk AnnData objects)
            node_idxs = [np.sort(node_idx) for node_idx in node_idxs]

            # Apply random selection
            if modalities is not None: modalities = [m[node_idx] for m, node_idx in zip(modalities, node_idxs)]
            if adata_obs is not None: adata_obs = [adata_ob.iloc[node_idx] for adata_ob, node_idx in zip(adata_obs, node_idxs)]

        ret = ()
        if modalities is not None: ret += (modalities,)
        if adata_obs is not None: ret += (adata_obs,)
        if return_partition: ret += (selected_partition,)
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
    

def read_adatas(*fnames, backed=False):
    # Params
    backed_arg = 'r' if backed else None

    # Read adatas
    adatas = []
    for fname in fnames:
        # s3 handling
        if fname.startswith('s3://'):
            # Get file handle
            s3 = _utility.general.get_s3_handler_with_access(fname)
            f = s3.open(fname, 'rb')  # NOTE: Never closed

            # Read from s3
            if backed:
                # Download
                # fname_local = os.path.join(download_dir, fname.split('/')[-1])
                # if not os.path.exists(fname_local):
                #     # A bit redundant and not really thread-safe
                #     s3.download(fname, fname_local)
                # handle = fname_local
                # Backed with s3
                file_h5 = h5py.File(f, 'r')
                if file_h5['X'].attrs['encoding-type'] == 'array':  # Dense data
                    X = file_h5['X']  # TODO
                else: X = ad.io.sparse_dataset(file_h5['X'])  # Sparse data
                adata = ad.AnnData(
                    X=X, **{
                        k: ad.io.read_elem(file_h5[k]) if k in file_h5 else {}
                        for k in ['layers', 'obs', 'var', 'obsm', 'varm', 'uns', 'obsp', 'varp']})
            
            # Read into memory
            else:
                adata = sc.read_h5ad(f, backed=backed_arg)

        # Local handling
        else: adata = sc.read_h5ad(fname, backed=backed_arg)

        # Append
        adatas.append(adata)

    return adatas


def merge_adatas(*adatas, backed=False):
    if not backed:
        adatas = ad.concat(adatas)  # TODO: Test
    else:
        adatas = [ad.experimental.AnnCollection(adatas)]
        adatas[0].var = adatas[0].adatas[0].var
        
    return adatas


def test_adatas(*adatas, partition_cols=None):
    # Parameters
    if partition_cols is not None and not _utility.general.is_list_like(partition_cols):
        partition_cols = [partition_cols]
        
    def red_func(l, r):
        # Join on sample and partition columns
        index_name = '_index' if l.index.name is None else l.index.name
        req_cols = [index_name]
        if partition_cols is not None: req_cols += partition_cols
        else: req_cols += ['_count']; l['_count'] = 0; r['_count'] = 0
        df = l.reset_index(names=index_name)[req_cols].merge(r.reset_index(names=index_name)[req_cols], how='outer', on=req_cols, suffixes=(None, None))
        assert df.groupby(index_name).count().max().max() < 2, 'Duplicate samples with non-equivalent partition metadata found'
        return df.set_index(index_name)
    try: merged_obs = ft.reduce(red_func, [adata.obs for adata in adatas])  # Test for conflicting metadata
    except: raise IndexError('Conflicting metadata found for corresponding sample IDs.')


class PreprocessFromAnnData:
    def __init__(
        self,
        *adatas,
        memory_efficient='auto',  # Also works on in-memory datasets
        partition_cols=None,
        mask=None,  # List or int indicating pct of keepable samples
        fit_sample='auto',
        seed=None,
        **kwargs
    ):
        self.adatas = adatas
        self.preprocessing = Preprocessing(**kwargs, seed=seed)
        self.seed = seed

        # RNG
        self.rng = np.random.default_rng(self.seed)

        # Parameters
        if partition_cols is not None and not _utility.general.is_list_like(partition_cols):
            partition_cols = [partition_cols]
        self.partition_cols = partition_cols
        if isinstance(mask, float):
            self.mask = self.rng.random(adatas[0].shape[0]) < mask
        else: self.mask = mask
        if memory_efficient == 'auto':
            # Only activate if on disk
            self.memory_efficient = np.array([
                isinstance(adata, ad.experimental.AnnCollection) or isinstance(adata.X, ad.abc.CSRDataset)
                for adata in adatas]).any()
        if fit_sample == 'auto':
            fit_sample = int(1e4) if memory_efficient else None
        self.fit_sample = fit_sample

        # Public parameters
        self.num_modalities = len(adatas)
        self.raw_modal_dims = [adata.var.shape[0] for adata in adatas]

        # Functions
        if memory_efficient:
            self.fit = self._fit_disk
            self.transform = self._transform_disk
        else:
            self.fit = self._fit_memory
            self.transform = self._transform_memory
        self.sample = self.transform
        self.fit()
        self.modal_dims = self.preprocessing.modal_dims

    def get_transformables(self):
        adata_obs = [adata.obs for adata in self.adatas]
        adata_vars = [adata.var for adata in self.adatas]
        return self.adatas, adata_obs, adata_vars
        # return self.processed_modalities, self.processed_adata_obs, self.processed_adata_vars

    def _fit_memory(self):
        if self.fit_sample:
            modalities = [
                adata[self.rng.choice(adata.shape[0], self.fit_sample)].X
                if adata.shape[0] > self.fit_sample else
                adata.X
                for adata in self.adatas]
        else: modalities = [adata.X for adata in self.adatas]
        adata_vars = [adata.var for adata in self.adatas]
        adata_obs = [adata.obs for adata in self.adatas]

        # Perform preprocessing
        # self.preprocessing.fit([adata.X for adata in self.adatas])
        self.processed_adata_obs = adata_obs
        self.processed_modalities, self.processed_adata_vars = self.preprocessing.fit_transform(
            modalities, adata_vars=adata_vars, force_filter=True)
        
    def _transform_memory(self, return_partition=False):
        # Perform sampling
        sampled_adata_vars = self.processed_adata_vars
        sampled_modalities, sampled_adata_obs = self.preprocessing.subsample(
            self.processed_modalities,
            adata_obs=self.processed_adata_obs,
            partition_cols=self.partition_cols,
            mask=self.mask,
            return_partition=return_partition)
        if return_partition: sampled_adata_obs, partition = sampled_adata_obs
        
        ret = (sampled_modalities, sampled_adata_obs, sampled_adata_vars)
        if return_partition: ret += (partition,)
        return ret
        
    def _fit_disk(self):
        if self.fit_sample:
            modalities = [
                adata[self.rng.choice(adata.shape[0], self.fit_sample)].X
                if adata.shape[0] > self.fit_sample else
                adata[:].X[:]
                for adata in self.adatas]
        else: modalities = [adata[:].X[:] for adata in self.adatas]

        # Fit preprocessing
        self.preprocessing.fit(modalities)
        # print(self.preprocessing.transform([adata[adata.obs.index[[0]]].X for adata in self.adatas], force_filter=True))

    def _transform_disk(self, return_partition=False):
        adata_obs = [adata.obs for adata in self.adatas]
        adata_vars = [adata.var for adata in self.adatas]

        # Perform sampling
        sampled_adata_obs = self.preprocessing.subsample(
            adata_obs=adata_obs,
            partition_cols=self.partition_cols,
            mask=self.mask,
            return_partition=return_partition)
        if return_partition: sampled_adata_obs, partition = sampled_adata_obs
        sampled_modalities = [adata[adata_ob.index.to_numpy()].X for adata, adata_ob in zip(self.adatas, sampled_adata_obs)]
        processed_adata_obs = sampled_adata_obs
        processed_modalities, processed_adata_vars = self.preprocessing.transform(
            sampled_modalities, adata_vars, force_filter=True)

        ret = (processed_modalities, processed_adata_obs, processed_adata_vars)
        if return_partition: ret += (partition,)
        return ret


def split_state(
    state,
    idx=None,
    sample_strategy='random-proximity',
    # Strategy kwargs
    max_nodes=None,
    reproducible_strategy='mean',
    sample_dim=None,  # Should be the dim of the env
    return_mask=False,
):
    "Split full state matrix into individual inputs, self_idx is an optional array"
    # Skip if indicated 

    # Parameters
    if idx is None: idx = np.arange(state.shape[0]).tolist()
    if not _utility.general.is_list_like(idx): idx = [idx]
    self_idx = idx
    del idx
    device = state.device

    # Batch input for Lite model
    if sample_strategy is None:
        # All processing case
        if len(self_idx) == state.shape[0]:
            # This optimization saves a lot of time
            if (self_idx[:-1] < self_idx[1:]).all(): return state,
            elif (np.unique(self_idx) == np.arange(state.shape[0])).all(): return state[self_idx],

        # Subset case
        self_entity = state[self_idx]
        node_entities = state  # Could remove the self_idx in the case len == 1, but doesn't really matter
        mask = torch.eye(state.shape[0], dtype=torch.bool, device=device)[self_idx]
        return self_entity, node_entities, mask

    # Get self features for each node
    self_entity = state[self_idx]

    # Get node features for each state
    node_mask = torch.eye(state.shape[0], dtype=torch.bool, device=device)
    node_mask = ~node_mask

    # Enforce reproducibility
    if reproducible_strategy is not None: generator = torch.Generator(device=device)
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
            probs = torch.empty_like(node_mask, dtype=torch.get_default_dtype()).normal_(generator=generator)
            probs[~node_mask] = 0
            selected_idx = probs.topk(num_nodes, dim=-1)[1]  # Take `num_nodes` highest values

            # Create new mask
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool, device=device)
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
            selected_idx = dist.topk(num_nodes+1, largest=False, dim=-1)[1][..., 1:]
            
            # Create new mask
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool, device=device)
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
            # NOTE: If you get an error `_assert_async_cuda_kernel`, weights probably exploded making `actions = [nan]`
            node_mask = torch.zeros((state.shape[0], state.shape[0]), dtype=torch.bool, device=device)
            idx = prob.multinomial(num_nodes, replacement=False, generator=generator)

            # Apply sampling
            mat = torch.arange(node_mask.shape[0], device=device).unsqueeze(-1).expand(node_mask.shape[0], num_nodes)
            node_mask[mat, idx] = True
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


def sample_and_cast(
    memory, larger_data, larger_size, smaller_size,
    *, current_level, load_level, cast_level,
    device, sequential_num=None, clip_sequential=False,
    **kwargs):
    """
    Sample and/or cast based on load level and cast level
    NOTE: Ignores sequential num if random sampling
    """
    smaller_data = None
    if larger_size is not None:  #  and larger_data is not None
        if sequential_num is not None:
            min_idx = sequential_num * smaller_size
            max_idx = min(larger_size-1, (sequential_num+1)*smaller_size)
            if clip_sequential: smaller_size = max_idx - min_idx + 1
            # smaller_idx = np.arange(min_idx, min(max_idx, larger_size))
            smaller_idx = slice(min_idx, max_idx)
        else: smaller_idx = np.random.choice(larger_size, smaller_size, replace=False)
    if load_level == current_level:
        smaller_data = memory.fast_sample(smaller_size, **kwargs)
    elif load_level < current_level:
        smaller_data = _utility.processing.dict_map_recursive_tensor_idx_to(larger_data, smaller_idx, None)
    if cast_level == current_level:
        smaller_data = _utility.processing.dict_map_recursive_tensor_idx_to(smaller_data, None, device)

    ret = smaller_data,
    if clip_sequential: ret += smaller_size,
    return _utility.general.clean_return(ret)
