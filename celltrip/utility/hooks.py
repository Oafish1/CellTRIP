import warnings

import numpy as np
import torch


def _move_toward_targets(env, actions, pinning, feature_idx, feature_targets=0., pca_components=None, pca_means=0., factor=1., eps=1e-8):
    # Move toward areas where imputed features match targets, GPT-Assisted
    # Nothing case
    if len(feature_idx) == 0: return

    # Get next positions
    next_pos = env.pos + env.vel * env.delta + actions * (env.delta ** 2)

    # Get gradients before PCA
    with torch.enable_grad():
        next_pos = next_pos.detach().requires_grad_(True)
        imputed_next_pos = pinning(next_pos)
        if pca_components is not None:
            inverse_imputed_next_pos = imputed_next_pos @ pca_components
            inverse_imputed_next_pos = inverse_imputed_next_pos + pca_means
        else: inverse_imputed_next_pos = imputed_next_pos
        # aggregate_effects = inverse_imputed_next_pos[..., feature_idx].sum()
        mse = (inverse_imputed_next_pos[..., feature_idx] - feature_targets).square().mean(dim=-1)
        mse.mean().backward()
        grad = next_pos.grad.clone()
        next_pos.grad.zero_()
    mse = mse.detach()

    # Compute new actions
    # # lambda_reg=5e-5, eps=1e-8
    # numerator = 0.5 * (env.delta ** 2) / (lambda_reg + eps)  # 100 default
    # denominator = 1.0 + numerator * (grad.pow(2).sum(dim=-1, keepdim=True) + eps)
    # actions_delta = - (numerator / denominator) * grad
    # new_actions = actions + actions_delta

    # Compute new actions
    # actions_delta = - factor * grad
    grad_norm = grad.square().sum(keepdim=True, dim=-1) + eps
    actions_delta = - factor * env.delta * grad / grad_norm  # mse.unsqueeze(1)
    new_actions = actions + actions_delta

    return new_actions
    

def move_toward_targets_hook(feature_idx, preprocessing=None, feature_targets=0., modality_idx=0, device='cpu', **kwargs):
    # Preprocessing
    extra_kwargs = _filter_extract_preprocess(feature_idx, preprocessing=preprocessing, feature_targets=feature_targets, modality_idx=modality_idx)
    extra_kwargs['pca_components'] = torch.tensor(extra_kwargs['pca_components'], dtype=torch.get_default_dtype(), device=device)
    if 'pca_means' in extra_kwargs: extra_kwargs['pca_means'] = torch.tensor(extra_kwargs['pca_means'], dtype=torch.get_default_dtype(), device=device)
    extra_kwargs['feature_targets'] = torch.tensor(extra_kwargs['feature_targets'], dtype=torch.get_default_dtype(), device=device)
    extra_kwargs.pop('modality_idx')

    # Generate hook
    return lambda env, actions: _move_toward_targets(env, actions, **extra_kwargs, **kwargs)


def _clamp_input_features(env, feature_idx, feature_targets, pca_components=None, pca_means=None, modality_idx=0):
    # Clamp certain modal expressions to specified values in the input space
    # Nothing case
    if len(feature_idx) == 0: return

    # Non-PCA case
    if pca_components is None:
        env.modality_offsets[modality_idx][:, feature_idx] = feature_targets
        return
    
    # Parameters
    if pca_means is None: pca_means = torch.zeros((1, pca_components.shape[1]), device=pca_components.device)
    
    # Invert current representation
    current_modality = env.get_modalities(noise=False, _indices=[modality_idx])[0]
    current_features = current_modality @ pca_components[:, feature_idx] + pca_means[:, feature_idx]
    current_diff = current_features - feature_targets
    # Transform difference
    transformed_diff = current_diff @ pca_components[:, feature_idx].T

    # Modify offset
    env.modality_offsets[modality_idx] = env.modality_offsets[modality_idx] - transformed_diff


def clamp_input_features_hook(feature_idx, preprocessing=None, feature_targets=0., modality_idx=0, device='cpu'):
    # Preprocess features
    extra_kwargs = _filter_extract_preprocess(feature_idx, preprocessing=preprocessing, feature_targets=feature_targets, modality_idx=modality_idx)
    extra_kwargs['pca_components'] = torch.tensor(extra_kwargs['pca_components'], dtype=torch.get_default_dtype(), device=device)
    if 'pca_means' in extra_kwargs: extra_kwargs['pca_means'] = torch.tensor(extra_kwargs['pca_means'], dtype=torch.get_default_dtype(), device=device)
    extra_kwargs['feature_targets'] = torch.tensor(extra_kwargs['feature_targets'], dtype=torch.get_default_dtype(), device=device)

    # Generate hook
    return lambda env: _clamp_input_features(env, **extra_kwargs)


def _clamp_inverted_features(env, feature_idx, feature_targets, pca_components=None, pca_means=0., modality_idx=0):
    # Clamp certain modal expressions to specified values in the inverse imputed space, GPT-Assisted
    # Nothing case
    if len(feature_idx) == 0: return

    # Non-PCA case
    if pca_components is None:
        env.modality_offsets[modality_idx][:, feature_idx] = feature_targets
        return

    # Get current modalities
    Z = env.get_modalities(noise=False, _indices=[modality_idx])[0].detach().cpu().numpy()
    A = pca_components[:, feature_idx].T

    # Center targets
    b = feature_targets - pca_means

    # Compute pseudoinverse of A A^T
    AAT_pinv = np.linalg.pinv(A @ A.T)

    # Project
    residual = (Z @ A.T - b)
    correction = residual @ AAT_pinv @ A
    Z_new = Z - correction

    # Adjust offsets
    env.modality_offsets[modality_idx] = torch.tensor(Z_new, dtype=torch.get_default_dtype()).cuda() - env.modalities[0]


def clamp_inverted_features_hook(feature_idx, preprocessing=None, feature_targets=0., modality_idx=0):
    # Preprocess features
    extra_kwargs = _filter_extract_preprocess(feature_idx, preprocessing=preprocessing, feature_targets=feature_targets, modality_idx=modality_idx)

    # Generate hook
    return lambda env: _clamp_inverted_features(env, **extra_kwargs)


def _filter_extract_preprocess(feature_idx, preprocessing=None, feature_targets=0., modality_idx=0):
    # Parameters
    kwargs = {
        'feature_idx': feature_idx,
        'feature_targets': feature_targets,
        'modality_idx': modality_idx}

    # Preprocessing
    if preprocessing is not None:
        # Filter if needed
        if preprocessing.filter_mask[modality_idx] is not None:
                feature_idx_intersection = np.intersect1d(feature_idx, preprocessing.filter_mask[modality_idx])
                if feature_idx_intersection.shape[0] != len(feature_idx):
                    warnings.warn(
                        f'Some elements not included due to filtering, '
                        f'{np.setxor1d(feature_idx, feature_idx_intersection)}.', RuntimeWarning)
                kwargs['feature_idx'] = [np.argwhere(preprocessing.filter_mask[modality_idx] == sf).flatten()[0] for sf in feature_idx_intersection]

        # Extract components
        kwargs['pca_components'] = preprocessing.pca_class[modality_idx].components_
        try: kwargs['pca_means'] = np.expand_dims(preprocessing.pca_class[modality_idx].means_, axis=0)
        except: pass

        # Preprocess targets
        if modality_idx is not None:
            kwargs['feature_targets'] = preprocessing.transform_select_features(feature_idx, feature_targets, modality_idx)

    return kwargs


def continuous_feature_targets(hook_func, target_func, *args, **kwargs):
    # `target_func` takes time and outputs targets
    def hook(env, *hook_args, **hook_kwargs):
        hook_func(*args, feature_targets=target_func(env.time), **kwargs)(env, *hook_args, **hook_kwargs)
    return hook