# %%
# %load_ext autoreload
# %autoreload 2


# %%
import time

import ray

import celltrip


# %%
ray.shutdown()
ray.init(
    address='ray://127.0.0.1:10001',
    runtime_env={
        'env_vars': {
            'RAY_DEDUP_LOGS': '0',
            # Irregular node fixes
            # NOTE: Important, NCCL will timeout if network device is non-standard
            # 'CUDA_LAUNCH_BLOCKING': '1',  # Slow, only for compatibility with X windows
            # 'NCCL_SOCKET_IFNAME': 'tailscale',  # lo,en,wls,docker,tailscale
            # 'NCCL_IB_DISABLE': '1',
            # 'NCCL_CUMEM_ENABLE': '0',
            # 'NCCL_DEBUG': 'INFO',
        }
    }
)


# %%
def env_init(parent_dir=False):
    # Create dataloader
    fnames = ['./data/MERFISH/expression.h5ad', './data/MERFISH/spatial.h5ad']
    if parent_dir: fnames = ['.' + f for f in fnames]
    partition_cols = None  # 'layer'
    adatas = celltrip.utility.processing.read_adatas(*fnames, on_disk=False)
    celltrip.utility.processing.test_adatas(*adatas, partition_cols=partition_cols)
    dataloader = celltrip.utility.processing.PreprocessFromAnnData(
        *adatas, partition_cols=partition_cols,  num_nodes=200,
        pca_dim=128, seed=42)
    # modalities, adata_obs, adata_vars = dataloader.sample()
    # Return env
    return celltrip.environment.EnvironmentBase(
        dataloader, dim=3,
        reward_distance=0, reward_origin=0, penalty_bound=1, penalty_velocity=1, penalty_action=1)

# Default 25Gb Forward, 14Gb Update, at max capacity
policy_init = lambda env: celltrip.policy.PPO(
    2*env.dim, env.dataloader.modal_dims, env.dim)  # update_iterations=2, minibatch_size=3e3,

memory_init = lambda policy: celltrip.memory.AdvancedMemoryBuffer(
    sum(policy.modal_dims), split_args=policy.split_args)  # prune=100

initializers = (env_init, policy_init, memory_init)

stage_functions = [
    lambda w: w.env.set_rewards(penalty_velocity=1, penalty_action=1),
    lambda w: w.env.set_rewards(reward_origin=1),
    lambda w: w.env.set_rewards(reward_origin=0, reward_distance=1),
]


# %%
start = time.perf_counter()
# rollout_kwargs={'dummy': True}, update_kwargs={'update_iterations': 5}, sync_across_nodes=False
workers = ray.get(celltrip.train.train_celltrip.remote(
    initializers=initializers,
    num_gpus=2,
    num_learners=2,
    num_runners=4,
    stage_functions=stage_functions))
print(time.perf_counter() - start)


# %%
# env = env_init(parent_dir=True).to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)
# celltrip.train.simulate_until_completion(env, policy, memory)
# memory.propagate_rewards()
# memory.normalize_rewards()
# # memory.fast_sample(10_000, shuffle=False)
# len(memory)


# %%
# memory.mark_sampled()
# env.reset()
# celltrip.train.simulate_until_completion(env, policy, memory)
# memory.propagate_rewards()
# memory.adjust_rewards()



