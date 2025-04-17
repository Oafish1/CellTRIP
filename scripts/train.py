# %%
# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# - Sometimes model will be chaotic if it finds how to early-finish early on, if so, it waits until action_std is lower to implement
# - Train/test
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


# %%
# Arguments
# NOTE: It is not recommended to use s3 with credentials unless the creds are permanent, the bucket is public, or this is run on AWS
parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reading
group = parser.add_argument_group('Input')
group.add_argument('input_files', type=str, nargs='*', help='h5ad files to be used for input')
group.add_argument('--merge_files', type=str, action='append', nargs='+', help='h5ad files to merge as input')
group.add_argument('--partition_cols', type=str, action='append', nargs='+', help='Columns for data partitioning, found in `adata.obs` DataFrame')
group.add_argument('--backed', action='store_true', help='Read data directly from disk or s3, saving memory at the cost of time')
group.add_argument('--input_modalities', type=int, nargs='+', help='Input modalities to give to CellTRIP')
group.add_argument('--target_modalities', type=int, nargs='+', help='Target modalities to emulate, dictates environment reward')
# Computation
group = parser.add_argument_group('Computation')
group.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use during computation')
group.add_argument('--num_learners', type=int, default=1, help='Number of learners used in backward computation, cannot exceed GPUs')
group.add_argument('--num_runners', type=int, default=1, help='Number of workers for environment simulation')
# Training
group = parser.add_argument_group('Training')
group.add_argument('--steps', type=int, default=int(5e3), help='Number of steps recorded before each update')
group.add_argument('--max_updates', type=int, default=200, help='Maximum number of policy updates to compute before exiting')
group.add_argument('--dont_sync_across_nodes', action='store_true', help='Avoid memory sync across nodes, saving overhead time at the cost of stability')
# File saves
group = parser.add_argument_group('Logging')
group.add_argument('--logfile', type=str, default='cli', help='Location for log file, can be `cli`, `<local_file>`, or `<s3 location>`')
group.add_argument('--flush_iterations', type=int, help='Number of iterations to wait before flushing logs')
group.add_argument('--checkpoint', type=str, help='Checkpoint to use for initializing model')
group.add_argument('--checkpoint_iterations', type=int, default=20, help='Number of iterations to wait before recording checkpoints')
group.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
group.add_argument('--checkpoint_name', type=str, help='Run name, for checkpointing')

# Notebook defaults and script handling
if not celltrip.utility.notebook.is_notebook():
    # ray job submit -- python train.py...
    config = parser.parse_args()
else:
    experiment_name = '3gpu-9k-ExtraKL-DontSync'
    command = (
        f's3://nkalafut-celltrip/MERFISH/expression.h5ad s3://nkalafut-celltrip/MERFISH/spatial.h5ad '
        # f'/home/nck/repos/INEPT/data/MERFISH/expression.h5ad /home/nck/repos/INEPT/data/MERFISH/spatial.h5ad '
        f'--backed '
        f'--target_modalities 1 '
        f'--num_gpus 3 --num_learners 2 --num_runners 9 '
        f'--steps 9000 '
        f'--max_updates 50 '
        f'--dont_sync_across_nodes '
        f'--logfile s3://nkalafut-celltrip/logs/{experiment_name}.log '
        f'--checkpoint_dir s3://nkalafut-celltrip/checkpoints '
        f'--checkpoint_name {experiment_name}')
    config = parser.parse_args(command.split(' '))
    print(f'python train.py {command}')
    
# Defaults
if config.checkpoint_name is None:
    config.checkpoint_name = f'RUN_{random.randint(0, 2**32):0>10}'
    print(f'Run Name: {config.checkpoint_name}')
# print(config)  # CLI


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
    dataloader_kwargs = {'num_nodes': 200, 'pca_dim': 128}
    environment_kwargs = {
        'input_modalities': config.input_modalities,
        'target_modalities': config.target_modalities, 'dim': 3}
    initializers = celltrip.train.get_initializers(
        input_files=config.input_files, merge_files=config.merge_files,
        backed=config.backed, dataloader_kwargs=dataloader_kwargs,
        environment_kwargs=environment_kwargs)

    stage_functions = [
        # lambda w: w.env.set_rewards(penalty_velocity=1, penalty_action=1),
        # lambda w: w.env.set_rewards(reward_origin=1),
        # lambda w: w.env.set_rewards(reward_origin=0, reward_distance=1),
    ]

    # Run function
    celltrip.train.train_celltrip(
        initializers=initializers,
        num_gpus=config.num_gpus, num_learners=config.num_learners,
        num_runners=config.num_runners, max_updates=config.max_updates,
        sync_across_nodes=not config.dont_sync_across_nodes,
        checkpoint_iterations=config.checkpoint_iterations, checkpoint_dir=config.checkpoint_dir,
        checkpoint=config.checkpoint, checkpoint_name=config.checkpoint_name,
        stage_functions=stage_functions, logfile=config.logfile)

ray.get(train.remote(config))


# %%
# # Get AWS keys
# import boto3
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# session = boto3.Session()
# creds = session.get_credentials()
# access_keys = {
#     'AWS_ACCESS_KEY_ID': creds.access_key,
#     'AWS_SECRET_ACCESS_KEY': creds.secret_key,
#     'AWS_DEFAULT_REGION': 'us-east-2'}

# # Check s3
# import os
# import s3fs
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# s3 = s3fs.S3FileSystem(skip_instance_cache=True)
# s3.ls('s3://nkalafut-celltrip')


# %%
# # Initialize locally
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# dataloader_kwargs = {'num_nodes': 20, 'pca_dim': 128}
# environment_kwargs = {
#     'input_modalities': config.input_modalities,
#     'target_modalities': config.target_modalities, 'dim': 3}
# env_init, policy_init, memory_init = celltrip.train.get_initializers(
#     input_files=config.input_files, merge_files=config.merge_files,
#     backed=config.backed, policy_kwargs={'minibatch_size': 2_000},
#     dataloader_kwargs=dataloader_kwargs,
#     environment_kwargs=environment_kwargs)
# env = env_init().to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)
# # celltrip.train.simulate_until_completion(env, policy, memory)
# # memory.compute_advantages()
# # policy.update(memory)


# %%
# OPTIMIZING
# Removing layer norms helps but doesn't actually reduce time???

# # Single iteration
# import torch
# env.max_timesteps = 1

# # Time profile
# import time
# start = time.perf_counter()
# for _ in range(10):
#     celltrip.train.simulate_until_completion(env, policy, memory)
#     torch.cuda.synchronize()
# duration = time.perf_counter()-start
# print(f'{int(1000*duration/10)}')

# # Torch profile
# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     with record_function("model_inference"):
#         for _ in range(10):
#             celltrip.train.simulate_until_completion(env, policy, memory)
#             torch.cuda.synchronize()

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=1000))



