import os
import time

import ray

import celltrip

# Detect Cython
CYTHON_ACTIVE = os.path.splitext(celltrip.utility.general.__file__)[1] in ('.c', '.so')
print(f'Cython is{" not" if not CYTHON_ACTIVE else ""} active')


# %% [markdown]
# - High priority
#   - Auto set max GPUs for update (?)
#   - Add node to event log
#   - Implement stages
#   - Add checkpoints
#   - Add model loading
#   - Add train/val to dataloader
#   - Partition detection in `train_policy`
#   - Script arguments, including address for ray
# - Medium priority
#   - Eliminate passing of persistent storage for memory objects
#   - Add hook for wandb, etc.
#   - Add state manager to env and then parallelize in analysis, maybe make `analyze` function
# - Low priority
#   - Seed policy initialization and unseed update, add reproducibility tag to wait for all rollouts before updating
#   - Verify worker timeout
#   - Subtract working memory on host node
#   - Local data loading per worker

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
# Start timer
start_time = time.perf_counter()


# %%
# Read data
fnames = ['../data/MERFISH/expression.h5ad', '../data/MERFISH/spatial.h5ad']
partition_cols = None
adatas = celltrip.utility.processing.read_adatas(*fnames, on_disk=False)
celltrip.utility.processing.test_adatas(*adatas, partition_cols=partition_cols)

# Construct dataloader
dataloader = celltrip.utility.processing.PreprocessFromAnnData(
    *adatas, partition_cols=partition_cols, num_nodes=200, pca_dim=128, seed=42)
modalities, adata_obs, adata_vars = dataloader.sample()

# Initialize Ray
ray.shutdown(); ray.init(
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
    policy_init=policy_init, memory_init=memory_init,
    max_jobs_per_gpu=2, update_gpus=1)

# Perform training
celltrip.train.train_policy(distributed_manager, dataloader, rollout_kwargs={'dummy': False})


# %%
# End timer
print(f'Ran for {time.perf_counter() - start_time:.0f} seconds')


# %%
# env_init = lambda policy, modalities: celltrip.environment.EnvironmentBase(
#     *modalities,
#     dim=policy.output_dim,
#     # max_timesteps=1e2,
#     penalty_bound=1,
#     device=policy.device)
# ray.get(distributed_manager.rollout(
#     modalities=modalities,
#     keys=adata_obs[0].index.to_numpy(),
#     env_init=env_init,
#     dummy=False))


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

# # Update remote policy
# distributed_manager.policy_manager.release_locks.remote()
# ray.get(distributed_manager.update())

# # Get state of job from ObjectRef
# import ray.util.state
# object_id = dm.futures['simulation'][0].hex()
# object_state = ray.util.state.get_objects(object_id)[0]
# object_state.task_status
