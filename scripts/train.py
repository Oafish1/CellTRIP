# %%
# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# TODO
# 
# Format command with ray submit
# 
# Checkpoints
# 
# On-disk reads from s3 (?)
# 
# [EFS on clusters maybe](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html#start-ray-with-the-ray-cluster-launcher)

# %%
import os

import ray

import celltrip


# %%
# Arguments
# NOTE: It is not recommended to use s3 with credentials unless the creds are permanent, the bucket is public, or this is run on AWS
import argparse
parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Important parameters
# group = parser.add_argument_group('Input Data')
parser.add_argument('input_files', type=str, nargs='*', help='h5ad files to be used for input')
parser.add_argument('--merge_files', type=str, action='append', nargs='+', help='h5ad files to merge as input')
parser.add_argument('--partition_cols', type=str, action='append', nargs='+', help='Columns for data partitioning, found in `adata.obs` DataFrame')
parser.add_argument('--data_in_memory', action='store_true', help='Load data into memory for sampling')
parser.add_argument('--download_dir', type=str, default='./downloads', help='Location for data download if needed')
parser.add_argument('--logfile', type=str, default='cli', help='Location for log file, can be `cli`, `<local_file>`, or `<s3 location>`')

# Notebook defaults and script handling
if not celltrip.utility.notebook.is_notebook():
    config = parser.parse_args()
else:
    # python train.py s3://nkalafut-celltrip/MERFISH/expression.h5ad s3://nkalafut-celltrip/MERFISH/spatial.h5ad --logfile s3://nkalafut-celltrip/log.txt
    config = parser.parse_args((
        's3://nkalafut-celltrip/MERFISH/expression.h5ad s3://nkalafut-celltrip/MERFISH/spatial.h5ad '
        '--logfile s3://nkalafut-celltrip/log.txt'
        ).split(' '))

print(config)


# %%
# Get AWS keys
# import boto3
# os.environ['AWS_PROFILE'] = 'waisman-admin'
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
    def env_init():
        # Create dataloader
        adatas = celltrip.utility.processing.read_adatas(*config.input_files, on_disk=(not config.data_in_memory), download_dir=config.download_dir)
        if config.merge_files is not None:
            for merge_files in config.merge_files:
                merge_adatas = celltrip.utility.processing.read_adatas(*merge_files, on_disk=(not config.data_in_memory))
                adatas += celltrip.utility.processing.merge_adatas(*merge_adatas, on_disk=(not config.data_in_memory))
        celltrip.utility.processing.test_adatas(*adatas, partition_cols=config.partition_cols)
        dataloader = celltrip.utility.processing.PreprocessFromAnnData(
            *adatas, partition_cols=config.partition_cols,  num_nodes=200,
            pca_dim=128, seed=42)
        # modalities, adata_obs, adata_vars = dataloader.sample()
        # Return env
        return celltrip.environment.EnvironmentBase(
            dataloader, dim=3)

    # Default 25Gb Forward, 14Gb Update, at max capacity
    policy_init = lambda env: celltrip.policy.PPO(
        2*env.dim, env.dataloader.modal_dims, env.dim)  # update_iterations=2, minibatch_size=3e3,

    memory_init = lambda policy: celltrip.memory.AdvancedMemoryBuffer(
        sum(policy.modal_dims), split_args=policy.split_args)  # prune=100

    initializers = (env_init, policy_init, memory_init)

    stage_functions = [
        # lambda w: w.env.set_rewards(penalty_velocity=1, penalty_action=1),
        # lambda w: w.env.set_rewards(reward_origin=1),
        # lambda w: w.env.set_rewards(reward_origin=0, reward_distance=1),
    ]

    # Run function
    # rollout_kwargs={'dummy': True}, update_kwargs={'update_iterations': 5}, sync_across_nodes=False
    celltrip.train.train_celltrip(
        initializers=initializers,
        num_gpus=3,
        num_learners=2,
        num_runners=3,
        updates=10,
        stage_functions=stage_functions,
        logfile=config.logfile)

ray.get(train.remote(config))


# %%
# import os
# import s3fs
# os.environ['AWS_PROFILE'] = 'waisman-admin'
# s3 = s3fs.S3FileSystem(skip_instance_cache=True)
# s3.ls('s3://nkalafut-celltrip')
# s3.open('s3://nkalafut-celltrip/test', 'w').close()


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



