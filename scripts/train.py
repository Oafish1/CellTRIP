# %%
# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# TODO
# 
# On-disk reads from s3 (?)
# 
# Look into g4dn.12xlarge if quota not approved
# 
# Fix: Failed to launch 1 node(s) of type ray.worker.default. (VcpuLimitExceeded): You have requested more vCPU capacity than your current vCPU limit of 32 allows for the instance bucket that the specified instance type belongs to. Please visit http://aws.amazon.com/contact-us/ec2-request to request an adjustment to this limit.
# 
# [EFS on clusters maybe](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html#start-ray-with-the-ray-cluster-launcher)

# %%
import os
import random

import ray

import celltrip

os.environ['AWS_PROFILE'] = 'waisman-admin'


# %%
# Arguments
# NOTE: It is not recommended to use s3 with credentials unless the creds are permanent, the bucket is public, or this is run on AWS
import argparse
parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Reading
# group = parser.add_argument_group('Input Data')
parser.add_argument('input_files', type=str, nargs='*', help='h5ad files to be used for input')
parser.add_argument('--merge_files', type=str, action='append', nargs='+', help='h5ad files to merge as input')
parser.add_argument('--partition_cols', type=str, action='append', nargs='+', help='Columns for data partitioning, found in `adata.obs` DataFrame')
parser.add_argument('--data_on_disk', action='store_true', help='Read data from disk, saving memory')
# File saves
parser.add_argument('--download_dir', type=str, default='./downloads', help='Location for data download if needed')
parser.add_argument('--logfile', type=str, default='cli', help='Location for log file, can be `cli`, `<local_file>`, or `<s3 location>`')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to use for initializing model')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
parser.add_argument('--checkpoint_name', type=str, help='Run name, for checkpointing')

# Notebook defaults and script handling
if not celltrip.utility.notebook.is_notebook():
    # ray job submit -- python train.py...
    config = parser.parse_args()
else:
    experiment_name = '3gpu-2learn-3run-20iter'
    command = (
        f's3://nkalafut-celltrip/MERFISH/expression.h5ad s3://nkalafut-celltrip/MERFISH/spatial.h5ad '
        f'--data_on_disk '
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
# Get AWS keys
# import boto3
# session = boto3.Session()
# creds = session.get_credentials()

# Reset Ray
ray.shutdown()


# %%
# Start Ray
ray.init(
    # address='ray://100.85.187.118:10001',
    address='ray://localhost:10001',
    runtime_env={
        'py_modules': [celltrip],
        'pip': '../requirements.txt',
        'env_vars': {
            # Logging
            'RAY_DEDUP_LOGS': '0',
            # Networking
            # 'NCCL_SOCKET_IFNAME': 'tailscale',  # lo,en,wls,docker,tailscale
            # Keys
            # 'AWS_ACCESS_KEY_ID': creds.access_key,
            # 'AWS_SECRET_ACCESS_KEY': creds.secret_key,
            # 'AWS_DEFAULT_REGION': 'us-east-2'
        }})


# %%
@ray.remote(num_cpus=1e-4)
def train(config):
    import celltrip

    # Initialization
    dataloader_kwargs = {'num_nodes': 200, 'pca_dim': 128}
    environment_kwargs = {'dim': 3}
    initializers = celltrip.train.get_initializers(
        input_files=config.input_files, merge_files=config.merge_files,
        data_on_disk=config.data_on_disk, download_dir=config.download_dir,
        dataloader_kwargs=dataloader_kwargs, environment_kwargs=environment_kwargs)

    stage_functions = [
        # lambda w: w.env.set_rewards(penalty_velocity=1, penalty_action=1),
        # lambda w: w.env.set_rewards(reward_origin=1),
        # lambda w: w.env.set_rewards(reward_origin=0, reward_distance=1),
    ]

    # Run function
    celltrip.train.train_celltrip(
        initializers=initializers,
        num_gpus=3, num_learners=2, num_runners=6,
        updates=20, flush_iterations=5, checkpoint_iterations=10,
        checkpoint_dir=config.checkpoint_dir, checkpoint=config.checkpoint,
        checkpoint_name=config.checkpoint_name, stage_functions=stage_functions,
        logfile=config.logfile)

ray.get(train.remote(config))


# %%
# import os
# import s3fs
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# s3 = s3fs.S3FileSystem(skip_instance_cache=True)
# s3.ls('s3://nkalafut-celltrip')


# %%
# # Initialize
# dataloader_kwargs = {'num_nodes': 200, 'pca_dim': 128}
# environment_kwargs = {'dim': 3}
# env_init, policy_init, memory_init = celltrip.train.get_initializers(
#     input_files=config.input_files, merge_files=config.merge_files,
#     data_on_disk=config.data_on_disk, download_dir=config.download_dir,
#     dataloader_kwargs=dataloader_kwargs, environment_kwargs=environment_kwargs)
# env = env_init().to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)

# %%
# env = env_init(parent_dir=True).to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)
# celltrip.train.simulate_until_completion(env, policy, memory)
# memory.propagate_rewards()
# memory.normalize_rewards()
# # memory.fast_sample(10_000, shuffle=False)
# len(memory)

# memory.mark_sampled()
# env.reset()
# celltrip.train.simulate_until_completion(env, policy, memory)
# memory.propagate_rewards()
# memory.adjust_rewards()



