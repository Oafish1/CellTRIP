# %%
# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# - Save preprocessing
# - Early stopping for `train_celltrip` based on action_std and/or KL
# - Maybe [this](https://arxiv.org/abs/2102.09430) but probably not
# - [EFS on clusters maybe](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html#start-ray-with-the-ray-cluster-launcher)

# %%
import argparse
import os
import random

import ray

import celltrip

# Detect Cython
CYTHON_ACTIVE = os.path.splitext(celltrip.utility.general.__file__)[1] in ('.c', '.so')
print(f'Cython is{" not" if not CYTHON_ACTIVE else ""} active')


# %% [markdown]
# # Arguments

# %%
# Arguments
# NOTE: It is not recommended to use s3 with credentials unless the creds are permanent, the bucket is public, or this is run on AWS
parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reading
group = parser.add_argument_group('Input')
group.add_argument('input_files', type=str, nargs='*', help='h5ad files to be used for input')
group.add_argument('--merge_files', type=str, action='append', nargs='+', help='h5ad files to merge as input')
group.add_argument('--partition_cols', type=str, nargs='+', help='Columns for data partitioning, found in `adata.obs` DataFrame')
group.add_argument('--backed', action='store_true', help='Read data directly from disk or s3, saving memory at the cost of time')
group.add_argument('--input_modalities', type=int, nargs='+', help='Input modalities to give to CellTRIP')
group.add_argument('--target_modalities', type=int, nargs='+', help='Target modalities to emulate, dictates environment reward')
# Algorithm
group = parser.add_argument_group('Algorithm')
group.add_argument('--dim', type=int, default=16, help='Dimensions in the output latent space')
group.add_argument('--train_split', type=float, default=1., help='Fraction of input data to use as training')
# Computation
group = parser.add_argument_group('Computation')
group.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use during computation')
group.add_argument('--num_learners', type=int, default=1, help='Number of learners used in backward computation, cannot exceed GPUs')
group.add_argument('--num_runners', type=int, default=1, help='Number of workers for environment simulation')
# Training
group = parser.add_argument_group('Training')
group.add_argument('--update_timesteps', type=int, default=int(1e6), help='Number of timesteps recorded before each update')
group.add_argument('--max_timesteps', type=int, default=int(2e9), help='Maximum number of timesteps to compute before exiting')
group.add_argument('--dont_sync_across_nodes', action='store_true', help='Avoid memory sync across nodes, saving overhead time at the cost of stability')
# File saves
group = parser.add_argument_group('Logging')
group.add_argument('--logfile', type=str, default='cli', help='Location for log file, can be `cli`, `<local_file>`, or `<s3 location>`')
group.add_argument('--flush_iterations', default=25, type=int, help='Number of iterations to wait before flushing logs')
group.add_argument('--checkpoint', type=str, help='Checkpoint to use for initializing model')
group.add_argument('--checkpoint_iterations', type=int, default=100, help='Number of updates to wait before recording checkpoints')
group.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
group.add_argument('--checkpoint_name', type=str, help='Run name, for checkpointing')

# Notebook defaults and script handling
if not celltrip.utility.notebook.is_notebook():
    # ray job submit -- python train.py...
    config = parser.parse_args()
else:
    # experiment_name = 'scglue-both-new'
    experiment_name = 'flysta3d'
    command = (
        # MERFISH
        # f's3://nkalafut-celltrip/MERFISH/expression.h5ad s3://nkalafut-celltrip/MERFISH/spatial.h5ad --target_modalities 1 '
        # scGLUE
        # f's3://nkalafut-celltrip/scGLUE/Chen-2019-RNA.h5ad s3://nkalafut-celltrip/scGLUE/Chen-2019-ATAC.h5ad '
        # f's3://nkalafut-celltrip/scGLUE/Chen-2019-RNA.h5ad s3://nkalafut-celltrip/scGLUE/Chen-2019-ATAC.h5ad --input_modalities 0 --target_modalities 0 '
        # f'../data/scglue/Chen-2019-RNA.h5ad ../data/scglue/Chen-2019-ATAC.h5ad --input_modalities 0 --target_modalities 0 '
        # Flysta3D
        ' '.join([f'--merge_files ' + ' ' .join([f's3://nkalafut-celltrip/Flysta3D/{p}_{m}.h5ad' for p in ('E14-16h_a', 'E16-18h_a', 'L1_a', 'L2_a', 'L3_b')]) for m in ('expression', 'spatial')]) + ' '
        + '--target_modalities 1 '
        f'--partition_cols slice_ID '
        # Tahoe-100M
        # '--merge_files ' + ' '.join('[f's3://nkalafut-celltrip/Tahoe/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad' for i in range(1, 15)]) + ' '
        # f'--partition_cols sample '
        # scMultiSim
        # f's3://nkalafut-celltrip/scMultiSim/expression.h5ad s3://nkalafut-celltrip/scMultiSim/peaks.h5ad '
        # TemporalBrain
        # f's3://nkalafut-celltrip/TemporalBrain/expression.h5ad s3://nkalafut-celltrip/TemporalBrain/peaks.h5ad '
        # f'--partition_cols "Donor ID" '

        f'--backed '
        f'--dim 2 '
        f'--train_split .8 '
        f'--num_gpus 2 --num_learners 2 --num_runners 2 '
        # f'--update_timesteps 1_000_000 '
        # f'--max_timesteps 250_000_000 '
        f'--update_timesteps 1_000_000 '
        f'--max_timesteps 200_000_000 '
        # f'--update_timesteps 100_000 '
        # f'--max_timesteps 100_000_000 '
        f'--dont_sync_across_nodes '
        f'--logfile s3://nkalafut-celltrip/logs/{experiment_name}.log '
        f'--flush_iterations 1 '
        # f'--checkpoint s3://nkalafut-celltrip/checkpoints/merfish-new-0200.weights '
        f'--checkpoint_iterations 25 '
        f'--checkpoint_dir s3://nkalafut-celltrip/checkpoints '
        f'--checkpoint_name {experiment_name}')
    config = parser.parse_args(command.split(' '))
    print(f'python train.py {command}')
    
# Defaults
if config.checkpoint_name is None:
    config.checkpoint_name = f'RUN_{random.randint(0, 2**32):0>10}'
    print(f'Run Name: {config.checkpoint_name}')
# print(config)  # CLI


# %% [markdown]
# # Deploy Remotely

# %%
# Start Ray
ray.shutdown()
a = ray.init(
    # address='ray://100.85.187.118:10001',
    address='ray://localhost:10001',
    runtime_env={
        'py_modules': [celltrip],
        'pip': '../requirements.txt',
        'env_vars': {
            # **access_keys,
            'RAY_DEDUP_LOGS': '0'}},
        # 'NCCL_SOCKET_IFNAME': 'tailscale',  # lo,en,wls,docker,tailscale
    _system_config={'enable_worker_prestart': True})  # Doesn't really work for scripts


# %%
@ray.remote(num_cpus=1e-4)
def train(config):
    import celltrip

    # Initialization
    dataloader_kwargs = {'num_nodes': [2**9, 2**11], 'mask': config.train_split}  # {'num_nodes': 20, 'pca_dim': 128}
    environment_kwargs = {
        'input_modalities': config.input_modalities,
        'target_modalities': config.target_modalities, 'dim': config.dim}
    initializers = celltrip.train.get_initializers(
        input_files=config.input_files, merge_files=config.merge_files,
        backed=config.backed, dataloader_kwargs=dataloader_kwargs,
        partition_cols=config.partition_cols,
        memory_kwargs={'device': 'cuda:0'},  # Skips casting, cutting time significantly for relatively small batch sizes
        environment_kwargs=environment_kwargs)

    # Stages
    stage_functions = []

    # Run function
    celltrip.train.train_celltrip(
        initializers=initializers,
        num_gpus=config.num_gpus, num_learners=config.num_learners,
        num_runners=config.num_runners, max_timesteps=config.max_timesteps,
        update_timesteps=config.update_timesteps, sync_across_nodes=not config.dont_sync_across_nodes,
        flush_iterations=config.flush_iterations,
        checkpoint_iterations=config.checkpoint_iterations, checkpoint_dir=config.checkpoint_dir,
        checkpoint=config.checkpoint, checkpoint_name=config.checkpoint_name,
        stage_functions=stage_functions, logfile=config.logfile)

ray.get(train.remote(config))


# %% [markdown]
# # Run Locally

# %%
# import numpy as np
# import torch
# torch.random.manual_seed(42)
# np.random.seed(42)

# # Initialize locally
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# # config.update_timesteps = 100_000
# # config.max_timesteps = 30_000_000

# dataloader_kwargs = {'num_nodes': [2**9, 2**11], 'mask': config.train_split}  # {'num_nodes': [2**9, 2**11], 'pca_dim': 128}
# environment_kwargs = {
#     'input_modalities': config.input_modalities,
#     'target_modalities': config.target_modalities, 'dim': config.dim}
# env_init, policy_init, memory_init = celltrip.train.get_initializers(
#     input_files=config.input_files, merge_files=config.merge_files,
#     partition_cols=config.partition_cols,
#     backed=config.backed, dataloader_kwargs=dataloader_kwargs,
#     policy_kwargs={'minibatch_size': 10_000},
#     # memory_kwargs={'device': 'cuda:0'},  # Skips casting, cutting time significantly for relatively small batch sizes
#     environment_kwargs=environment_kwargs)

# # Environment
# # os.environ['CUDA_LAUNCH_BLOCKING']='1'
# try: env
# except: env = env_init().to('cuda')

# # Policy
# policy = policy_init(env).to('cuda')

# # Memory
# memory = memory_init(policy)


# %%
# import numpy as np
# import torch

# X = 5*torch.rand(1000, 8)
# Y = 5*torch.rand(1000, 256)
# X_norm_scaled = torch.linalg.matrix_norm(X) * np.sqrt(Y.shape[1]/X.shape[1])
# Y_norm = torch.linalg.matrix_norm(Y)
# # X_norm_scaled = Y_norm = 1

# # Distance method
# # dist = ((torch.cdist(X, X) - torch.cdist(Y, Y))/np.sqrt(X.shape[0])).square().mean(dim=-1)

# # Least squares method
# C = torch.linalg.lstsq(X, Y / Y_norm).solution
# (torch.matmul(X, C) * X_norm_scaled - Y).square().mean(dim=-1)

# %%
# # Forward
# import line_profiler
# memory.mark_sampled()
# memory.cleanup()
# prof = line_profiler.LineProfiler(
#     celltrip.train.simulate_until_completion,
#     celltrip.policy.PPO.forward,
#     celltrip.policy.EntitySelfAttentionLite.forward,
#     celltrip.policy.ResidualAttention.forward,
#     celltrip.environment.EnvironmentBase.step)
# ret = prof.runcall(celltrip.train.simulate_until_completion, env, policy, memory, max_memories=config.update_timesteps, reset_on_finish=True)
# print('ROLLOUT: ' + f'total: {ret[2]:.3f}, ' + ', '.join([f'{k}: {v:.3f}' for k, v in ret[3].items()]))
# # memory.feed_new(policy.reward_standardization)
# memory.compute_advantages()  # moving_standardization=policy.reward_standardization
# prof.print_stats(output_unit=1)


# %%
# idx = np.sort(np.random.choice(len(memory), 10_000, replace=False))
# a = memory[idx]

# list_num = idx // a['states'][1].shape[1]
# in_list_num = np.mod(idx, a['states'][1].shape[1])
# check_idx = 2

# a['states'][0]
# memory.storage['states'][list_num[check_idx]][in_list_num[check_idx]]

# a['states'][1]
# memory.storage['states'][list_num[check_idx]]

# a['advantages'][check_idx]
# memory.storage['advantages'][list_num[check_idx]][in_list_num[check_idx]]

# a['states'][2]


# %%
# # Memory pull
# memory[0]
# import line_profiler
# prof = line_profiler.LineProfiler(
#     celltrip.memory.AdvancedMemoryBuffer.fast_sample)
# ret = prof.runcall(memory.__getitem__, np.random.choice(len(memory), 10_000, replace=False))
# memory.compute_advantages()  # moving_standardization=policy.reward_standardizatio
# prof.print_stats(output_unit=1)


# %%
# # Updates
# import line_profiler
# prof = line_profiler.LineProfiler(
#     # memory.fast_sample, policy.actor_critic.forward,
#     celltrip.policy.ResidualAttentionBlock.forward,
#     policy.calculate_losses, policy.update,
#     celltrip.memory.AdvancedMemoryBuffer.__getitem__)
# ret = prof.runcall(policy.update, memory, verbose=True)
# print('UPDATE: ' + ', '.join([f'{k}: {v:.3f}' for ret_dict in ret[1:] for k, v in ret_dict.items()]))
# prof.print_stats(output_unit=1)


# %%
# for _ in range(int(config.max_timesteps / config.update_timesteps)):
#     # Forward
#     memory.mark_sampled()
#     memory.cleanup()
#     ret = celltrip.train.simulate_until_completion(
#         env, policy, memory,
#         max_memories=config.update_timesteps,
#         # max_timesteps=100,
#         reset_on_finish=True)
#     print('ROLLOUT: ' + f'iterations: {ret[0]: 5.0f}, ' + f'total: {ret[2]: 5.3f}, ' + ', '.join([f'{k}: {v: 5.3f}' for k, v in ret[3].items()]))
#     memory.compute_advantages()

#     # Update
#     # NOTE: Training often only improves when PopArt and actual distribution match
#     ret = policy.update(memory, verbose=False)
#     print('UPDATE: ' + ', '.join([f'{k}: {v: 5.3f}' for ret_dict in ret[1:] for k, v in ret_dict.items()]))


# %%
# # env.max_timesteps = 1_000
# env.reset()
# memory = memory_init(policy)
# ret = celltrip.train.simulate_until_completion(env, policy, memory, store_states=True)
# steady_states = ret[-1].cpu()
# ret

# env.get_distance_match().mean()

# steady_states[-1, :, :env.dim]

# steady_states[-1, :, env.dim:]

# np.linalg.norm(memory.storage['actions'][-1]-1, axis=-1)


# import matplotlib.pyplot as plt
# import seaborn as sns
# import umap

# # Get data and types
# # raw = env.dataloader.adatas[0][env.keys].X
# raw = steady_states[-1, :, :env.dim]
# reducer = umap.UMAP(n_neighbors=100)

# # Transform
# transformed = reducer.fit_transform(raw)
# # transformed = raw[:, :2]
# # transformed = raw[:, 1:3]
# # types = env.dataloader.adatas[0].obs.loc[env.keys, 'layer'].to_numpy()
# types = env.dataloader.adatas[0].obs.loc[env.keys, 'cell_type'].to_numpy()

# # Plot
# fig, ax = plt.subplots(figsize=(4, 4))
# for t in np.unique(types):
#     ax.scatter(*transformed[types==t].T, s=2)
# sns.despine(left=True, bottom=True, ax=ax)
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.add_patch(plt.Circle((0, 0), env.pos_bound, facecolor='none', edgecolor='black'))
# fig.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import umap

# # Get data and types
# # raw = env.dataloader.adatas[0][env.keys].X
# raw = modalities[1].cpu()
# reducer = umap.UMAP()

# # Transform
# transformed = reducer.fit_transform(raw)
# types = env.dataloader.adatas[0].obs.loc[env.keys, 'cell_type'].to_numpy()

# # Plot
# fig, ax = plt.subplots(figsize=(4, 4))
# for t in np.unique(types):
#     ax.scatter(*transformed[types==t].T, s=2)
# sns.despine(left=True, bottom=True, ax=ax)
# fig.show()



