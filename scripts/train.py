import os
# os.environ['RAY_DEDUP_LOGS'] = '0'
# os.environ['NCCL_DEBUG'] = 'WARN'
# os.environ['RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING'] = '1'
# export RAY_BACKEND_LOG_LEVEL=debug
import functools as ft
import time

import numpy as np
import ray
import torch
import tqdm.auto
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.auto.tqdm  # Enable text output in notebooks

import celltrip

# Detect Cython
CYTHON_ACTIVE = os.path.splitext(celltrip.utility.general.__file__)[1] in ('.c', '.so')
print(f'Cython is{" not" if not CYTHON_ACTIVE else ""} active')


# %% [markdown]
# - High priority
#   - Implement stages
#   - Allow memory to pre-process keys and persistent storage
#   - Add checkpoints
#   - Add model loading
#   - Add train/val to dataloader
#   - Add state manager to env and then parallelize in analysis, maybe make `analyze` function
#   - Script arguments, including address for ray
#   - Partition detection in `train_policy`
#   - Seed policy initialization, add reproducibility tag to wait for all rollouts before updating
#   - Add metric returns for updates
#   - Add hook for wandb, etc.
#   - Local data loading per worker
#   - Subtract working memory on host node
#   - Verify worker timeout

# %%
# # Arguments
# import argparse
# parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# # TODO: Figure out how to format arguments for appending h5ad files
# parser.add_argument('datasets', type=str, required=False, help='.h5ad files to use for data')
# parser.add_argument('--concatenate', type=str, required=False, help=
#     '.h5ad files to concatenate as a single modality, may be used multiple times')

# group = parser.add_argument_group('General')
# group.add_argument('--seed', default=42, type=int, help='**Seed for random calls during training')


# %%
# Read data
fnames = ['../data/MERFISH/expression.h5ad', '../data/MERFISH/spatial.h5ad']
partition_cols = None
adatas = celltrip.utility.processing.read_adatas(*fnames, on_disk=False)
celltrip.utility.processing.test_adatas(*adatas, partition_cols=partition_cols)

# Construct dataloader
dataloader = celltrip.utility.processing.PreprocessFromAnnData( # num_nodes=200
    *adatas, partition_cols=partition_cols, num_nodes=200, pca_dim=128, seed=42)
modalities, adata_obs, adata_vars = dataloader.sample()

# Initialize Ray
ray.shutdown()
# ray.init(
#     resources={'VRAM': torch.cuda.get_device_properties(0).total_memory},
#     dashboard_host='0.0.0.0')
ray.init(
    address='ray://127.0.0.1:10001',
    runtime_env={
        'env_vars': {
            # NOTE: Important, NCCL will timeout if network device is non-standard
            'NCCL_SOCKET_IFNAME': 'tailscale',
            # 'NCCL_DEBUG': 'WARN',
            'RAY_DEDUP_LOGS': '0',
        }})

# Initialize distributed manager
policy_init, memory_init = celltrip.train.get_train_initializers(
    3, [m.shape[1] for m in modalities])
distributed_manager = celltrip.train.DistributedManager(
    # modalities=modalities, env_init=env_init,
    policy_init=policy_init,
    memory_init=memory_init)

# Available resources
import json
print(json.dumps(ray.available_resources(), indent=2, sort_keys=False))

# %%
# Perform training
celltrip.train.train_policy(distributed_manager, dataloader)

# %%
# distributed_manager.policy_manager.release_locks.remote()
# ray.get(distributed_manager.update())

# %%
# # Cancel
# # dm.cancel()
# # dm.clean()
# # dm.rollout(dummy=True)
# # dm.wait()

# # Clear locks
# distributed_manager.policy_manager.release_locks.remote()

# # Get policy
# device = 'cuda'
# policy = policy_init().to(device)
# celltrip.train.set_policy_state(policy, ray.get(distributed_manager.policy_manager.get_policy_state.remote()))

# # Get memory
# memory = memory_init(policy)
# memory.append_memory(
#     *ray.get(distributed_manager.policy_manager.get_memory_storage.remote()))


# %%
# # Get state of job from ObjectRef
# import ray.util.state
# object_id = dm.futures['simulation'][0].hex()
# object_state = ray.util.state.get_objects(object_id)[0]
# object_state.task_status



