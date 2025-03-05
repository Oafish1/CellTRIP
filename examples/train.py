import os
import functools as ft
import time

# Set env vars
os.environ['RAY_DEDUP_LOGS'] = '0'

import numpy as np
import ray
import torch

# Enable text output in notebooks
import tqdm.auto
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.auto.tqdm

import celltrip
import data

# Detect Cython
CYTHON_ACTIVE = os.path.splitext(celltrip.utility.general.__file__)[1] in ('.c', '.so')
print(f'Cython is{" not" if not CYTHON_ACTIVE else ""} active')

# Set params
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_FOLDER = os.path.abspath('')
DATA_FOLDER = os.path.join(BASE_FOLDER, '../data/')
MODEL_FOLDER = os.path.join(BASE_FOLDER, 'models/')


# %% [markdown]
# - High priority
#   - Add metric returns for updates
#   - Optimize cancels to only cancel non-running
#   - Implement stages
#   - Add partitioning
#   - Make ray init command for bash and add to README
# - Medium Priority
#   - Add seeding
#   - Add state manager to env and then parallelize in analysis, maybe make `analyze` function
#   - Add parallelism on max_batch and update. With update, encase whole epoch as ray function so splitting occurs within ray function, using ray.remote inline API to allow for non-ray usage. Then, adjustable policy weight sync (i.e. 1 epoch, 10 epochs)
# - Low Priority
#   - Allow memory to pre-process keys and persistent storage
#   - Add hook for wandb, ex.
#   - Move preprocessing to manager
#   - Figure out why sometimes just throws CUDA not available errors
#   - Better split_state reproducibility

# %%
# Read
fnames = ['../data/MERFISH/expression.h5ad', '../data/MERFISH/spatial.h5ad']
partition_cols = 'layer'
adatas = celltrip.utility.processing.read_adatas(*fnames, on_disk=False)
celltrip.utility.processing.test_adatas(*adatas, partition_cols=partition_cols)

# Dataloader
dataloader = celltrip.utility.processing.PreprocessFromAnnData(
    *adatas, partition_cols=partition_cols, pca_dim=128)
modalities, adata_obs, adata_vars = dataloader.sample()


# %%
# modalities, types, features = data.load_data('MMD-MA', DATA_FOLDER)
# ppc = celltrip.utility.processing.Preprocessing(pca_dim=128, device=DEVICE)
# processed_modalities, features = ppc.fit_transform(modalities, features)
# # modalities = ppc.cast(processed_modalities)
# modalities = [m.astype(np.float32) for m in processed_modalities]
# # modalities = [np.concatenate([m for _ in range(10000)], axis=0) for m in modalities]
# # modalities = [m[:100] for m in modalities]


# %%
# Behavioral functions
dim = 3
modal_dims = [m.shape[1] for m in modalities]
policy_init = lambda: celltrip.policy.PPO(
    positional_dim=2*dim,
    modal_dims=modal_dims,
    output_dim=dim,
    # BACKWARDS
    # epochs=5,
    # memory_prune=0,
    update_load_level='batch',
    update_cast_level='minibatch',
    update_batch=1e4,
    update_minibatch=3e3,
    # SAMPLING
    # max_batch=100,
    max_nodes=100,
    # DEVICE
    device='cpu')
# policy = policy_init(modalities)
# policy_init = lambda _: policy
env_init = lambda policy, modalities: celltrip.environment.EnvironmentBase(
    *modalities,
    dim=dim,
    # max_timesteps=1e2,
    penalty_bound=1,
    device=policy.device)
memory_init = lambda policy: celltrip.memory.AdvancedMemoryBuffer(
    sum(policy.modal_dims),
    split_args=policy.split_args)

# Initialize ray and distributed
ray.shutdown()
ray.init(
    resources={'VRAM': torch.cuda.get_device_properties(0).total_memory},
    dashboard_host='0.0.0.0')
dm = celltrip.train.DistributedManager(
    modalities,
    policy_init=policy_init,
    env_init=env_init,
    memory_init=memory_init,
    max_jobs_per_gpu=2)


# %%
from datetime import datetime
print(datetime.now())


# %%
# Train loop iter
max_rollout_futures = 1; num_updates = 0; calibrated = False
while True:
    # Retrieve active futures
    futures = dm.get_futures()
    num_futures = len(dm.get_all_futures())

    # Compute feasibility of futures
    available_resources = ray.available_resources()
    rollout_feasible = np.array([
        v <= available_resources[k]
        if k in available_resources else False
        for k, v in dm.get_requirements('rollout').items()]).all()
    # update_feasible = np.array([
    #     v <= available_resources[k]
    #     for k, v in dm.get_requirements('update').items()]).all()

    # CLI
    # print('; '.join([f'{k} ({len(v)})' for k, v in futures.items()]))
    # print(ray.available_resources())

    ## Check for futures to add
    # Check memory and apply update if needed 
    if len(futures['update']) == 0 and dm.get_memory_len() >= int(1e6):
        # assert False
        # print(f'Queueing policy update {num_updates+1}')
        # dm.cancel()  # Cancel all non-running (TODO)
        dm.update(meta={'Policy Iteration': num_updates+1})

    # Add rollouts if no update future and below max queued futures
    elif rollout_feasible and len(futures['update']) == 0 and num_futures < max_rollout_futures:
        # print(f'Queueing {num_new_rollouts} rollouts')
        # dm.rollout(num_new_rollouts, dummy=False)
        modalities, adata_obs, adata_vars, partition = dataloader.sample(return_partition=True)
        dm.rollout(
            modalities=modalities,
            keys=adata_obs[0].index.to_numpy(),
            meta={
                'Policy Iteration': num_updates,
                'Partition': partition},
            dummy=False)

    ## Check for completed futures
    # Completed rollouts
    if len(ray.wait(futures['rollout'], timeout=0)[0]) > 0:
        # Calibrate if needed
        all_variants_run = True  # TODO: Set to true if all partitions have been run
        if dm.resources['rollout']['core']['memory'] == 0 and all_variants_run:
            dm.calibrate()
            max_rollout_futures = 20
            # print(
            #     f'Calibrated rollout'
            #     f' memory ({dm.resources["rollout"]["core"]["memory"] / 2**30:.2f} GiB)'
            #     f' and VRAM ({dm.resources["rollout"]["custom"]["VRAM"] / 2**30:.2f} GiB)')
            # dm.cancel(); time.sleep(1)  # Cancel all non-running (TODO)
            # dm.policy_manager.release_locks.remote()
        # Clean if calibrated
        if dm.resources['rollout']['core']['memory'] != 0: dm.clean('rollout')

    # Completed updates
    if len(ray.wait(futures['update'], timeout=0)[0]) > 0:
        num_updates += 1
        # Calibrate if needed
        if dm.resources['update']['core']['memory'] == 0:
            dm.calibrate()
            # print(
            #     f'Calibrated update'
            #     f' memory ({dm.resources["update"]["core"]["memory"] / 2**30:.3f} GiB)'
            #     f' and VRAM ({dm.resources["update"]["custom"]["VRAM"] / 2**30:.3f} GiB)')
        dm.clean('update')

    # Wait to scan again
    # num_futures = len(dm.get_all_futures())
    # if num_futures > 0:
    #     num_completed_futures = len(dm.wait(num_returns=num_futures, timeout=0))
    #     if num_completed_futures != num_futures: dm.wait(num_returns=num_completed_futures+1)
    # else: time.sleep(1)
    time.sleep(1)

    # Escape
    if num_updates >= 50: break


# %%
from datetime import datetime
print(datetime.now())


# %%
# # Cancel
# # dm.cancel()
# # dm.clean()
# # dm.rollout(dummy=True)
# # dm.wait()

# # Clear locks
# dm.policy_manager.release_locks.remote()

# # Get policy
# device = DEVICE
# policy = policy_init().to(device)
# celltrip.train.set_policy_state(policy, ray.get(dm.policy_manager.get_policy_state.remote()))

# # Get memory
# memory = memory_init(policy)
# memory.append_memory(
#     *ray.get(dm.policy_manager.get_memory_storage.remote()))


# %%
# policy.update(memory)


# %%
# # Get state of job from ObjectRef
# import ray.util.state
# object_id = dm.futures['simulation'][0].hex()
# object_state = ray.util.state.get_objects(object_id)[0]
# object_state.task_status
