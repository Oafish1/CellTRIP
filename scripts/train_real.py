# %%
# %load_ext autoreload
# %autoreload 2


# %%
import time

import numpy as np
import ray
import ray.util.collective as col
import ray.util.scheduling_strategies
import torch

import celltrip


# %% [markdown]
# # Worker Definition

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
    return celltrip.environment.EnvironmentBase(dataloader, dim=3, penalty_bound=1, reward_origin=0)

# Default 25Gb Forward, 14Gb Update, at max capacity
policy_init = lambda env: celltrip.policy.PPO(
    2*env.dim, env.dataloader.modal_dims, env.dim,
    forward_batch_size=int(5e5)) # update_iterations=2, minibatch_size=3e3,

memory_init = lambda policy: celltrip.memory.AdvancedMemoryBuffer(
    sum(policy.modal_dims), split_args=policy.split_args)


# %%
@ray.remote(num_cpus=1e-4, num_gpus=1e-4)
class Worker:
    def __init__(
        self,
        policy_init,
        env_init,
        memory_init=lambda: None,
        world_size=1,
        rank=0,
        learner=True,
        parent=None,  # Policy parent worker ref
    ):
        # Detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Parameters
        self.env = env_init().to(device)
        self.policy = policy_init(self.env).to(device)
        self.memory = memory_init(self.policy)
        self.rank = rank
        self.learner = learner
        self.parent = parent

        # World initialization
        if learner: col.init_collective_group(world_size, rank, 'nccl', 'learners')
        if parent is None: col.init_collective_group(world_size, rank, 'nccl', 'heads')

        # Policy parameters
        self.policy_iteration = 0
        self.sync_policy(iterate_if_exclusive_runner=False)  # Works because non-heads will wait for head init

        # Memory parameters
        self.memory_buffer = []

    @celltrip.decorator.metrics(append_to_dict=True)
    # @celltrip.decorator.profile(time_annotation=True)
    def rollout(self, **kwargs):
        # Perform rollout
        result = celltrip.train.simulate_until_completion(
            self.policy, self.env, self.memory, **kwargs)
        self.memory.propagate_rewards()
        env_nodes = self.env.num_nodes
        self.env.reset()

        # Clean memory
        self.memory.cleanup()

        # Record
        timestep, reward, itemized_reward = result
        ret = {
            'Event Type': 'Rollout',
            'Policy Iteration': self.policy_iteration,
            'Rank': self.rank,
            'Timesteps': timestep,
            'Memories': timestep*env_nodes,
            'Reward': reward,
            'Itemized Reward': itemized_reward}
        return ret

    def rollout_until_new(self, num_new, condition='steps', **kwargs):
        # Parameters
        if condition == 'memories': measure = self.memory.get_new_len
        elif condition == 'steps': measure = self.memory.get_new_steps
        else: raise ValueError(f'Condition `{condition}` not found.')

        # Compute rollouts
        ret = []
        while measure() < num_new:
            ret.append(self.rollout(**kwargs))
        return ret
    
    @celltrip.decorator.metrics(append_to_dict=True)
    # @celltrip.decorator.profile(time_annotation=True)
    def update(self, **kwargs):
        # Perform update
        self.memory.normalize_rewards()
        self.policy.update(self.memory, **kwargs, verbose=True)

        # Annotate
        self.policy_iteration += 1
        num_new_memories = self.memory.get_new_len()
        num_replay_memories = self.memory.get_replay_len()

        # Clean
        self.memory.mark_sampled()

        # Record
        ret = {
            'Event Type': 'Update',
            'Policy Iteration': self.policy_iteration,
            'Rank': self.rank,
            'New Memories': num_new_memories,
            'Replay Memories': num_replay_memories,
            'Total Memories': len(self.memory)}
        return ret

    @celltrip.decorator.metrics(append_to_dict=True, dict_index=1)
    def send_memory(self, **kwargs):
        # Put in object store
        mem = self.memory.get_storage(**kwargs)
        ref = ray.put(mem)

        # Record
        ret = {
            'Event Type': 'Send Memory',
            'Rank': self.rank,
            'Memories': sum([s.shape[0] for s in mem[0]['states']])}
        return ref, ret
    
    @celltrip.decorator.metrics(append_to_dict=True)
    # @celltrip.decorator.profile(time_annotation=True)
    def recv_memories(self, new_memories):
        # Append memories
        num_memories = 0
        for new_memory in new_memories:
            new_memory = ray.get(new_memory)
            self.memory.append_memory(*new_memory)
            num_memories += sum([s.shape[0] for s in new_memory[0]['states']])

        # Clean memory
        self.memory.cleanup()

        # Record
        ret = {
            'Event Type': 'Receive Memories',
            'Rank': self.rank,
            'Memories': num_memories}
        return ret
    
    def get_policy_state(self):
        return celltrip.train.get_policy_state(self.policy)
    
    @celltrip.decorator.metrics(append_to_dict=True)
    def sync_policy(self, iterate_if_exclusive_runner=True):
        # Copy policy
        if self.parent is not None:
            policy_state = ray.get(self.parent.get_policy_state.remote())
            celltrip.train.set_policy_state(self.policy, policy_state)
        else:
            self.policy.synchronize('heads')

        # Iterate policy
        if not self.learner and iterate_if_exclusive_runner:
            self.policy_iteration += 1
            self.memory.mark_sampled()

        # Record
        ret = {
            'Event Type': 'Synchronize Policy',
            'Policy Iteration': self.policy_iteration,
            'Rank': self.rank,
            'Inherited': self.parent is not None}
        return ret

    def destroy(self):
        col.destroy_collective_group()


# %% [markdown]
# # Runtime

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
@ray.remote
def train(
    num_gpus,
    num_learners,
    num_runners,
    learners_can_be_runners=True,
    sync_across_nodes=True,
    updates=50,
    steps=int(5e3),
    rollout_kwargs={},
    update_kwargs={}):
    # Make placement group for GPUs
    pg_gpu = ray.util.placement_group(num_gpus*[{'CPU': 1, 'GPU': 1}])
    ray.get(pg_gpu.ready(), timeout=10)

    # Assign workers
    num_learner_runners = min(num_learners, num_runners) if learners_can_be_runners else 0
    num_exclusive_learners = num_learners - num_learner_runners if learners_can_be_runners else num_learners
    num_exclusive_runners = num_runners - num_learner_runners
    num_workers = num_exclusive_learners + num_learner_runners + num_exclusive_runners
    num_head_workers = min(num_gpus, num_workers)
    assert num_learners <= num_gpus, '`num_learners` cannot be greater than `num_gpus`.'

    # Create workers
    workers = []
    for i in range(num_workers):
        bundle_idx = i % num_head_workers
        child_num = i // num_head_workers
        rank = float(bundle_idx + child_num * 10**-(np.floor(np.log10((num_workers-1)//num_head_workers))+1))
        parent = workers[bundle_idx] if i >= num_head_workers else None
        w = (
            Worker
                .options(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg_gpu, placement_group_bundle_index=bundle_idx))
                .remote(
                    policy_init, env_init, memory_init,
                    world_size=num_head_workers, rank=rank,
                    learner=i<num_learners, parent=parent))
        workers.append(w)
    learners = workers[:-num_exclusive_runners]
    runners = workers[num_exclusive_learners:]
    heads = workers[:num_head_workers]
    non_heads = workers[num_head_workers:]

    # Run policy updates
    # TODO: Add/try async updates
    # TODO: Maybe 80 epochs
    records = []
    for policy_iteration in range(updates):
        # Rollouts
        num_records = len(records)
        if policy_iteration==0:
            new_records = ray.get([w.rollout.remote(**rollout_kwargs, return_metrics=True) for w in runners])
            records += new_records
            for record in records[-(len(records)-num_records):]: print(record)
        num_records = len(records)
        new_records = ray.get([w.rollout_until_new.remote(num_new=steps/num_workers, **rollout_kwargs) for w in runners])
        records += sum(new_records, [])
        for record in records[-(len(records)-num_records):]: print(record)

        # Collect memories
        num_records = len(records)
        ret = ray.get([w.send_memory.remote(new=True) for w in runners])
        new_memories, new_records = [[r[i] for r in ret] for i in range(2)]
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

        # Broadcast memories
        num_records = len(records)
        new_records = []
        for i, w1 in enumerate(learners):
            if sync_across_nodes:
                new_memories_w1 = [ref for w2, ref in zip(runners, new_memories) if w1!=w2]
            else:
                new_memories_w1 = new_memories[num_head_workers+i::num_head_workers]
            future = w1.recv_memories.remote(new_memories=new_memories_w1)
            new_records.append(future)
        new_records = ray.get(new_records)
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

        # Updates
        num_records = len(records)
        new_records = ray.get([w.update.remote(
            **update_kwargs, return_metrics=policy_iteration==0) for w in learners])
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

        # Synchronize policies
        num_records = len(records)
        new_records = ray.get([w.sync_policy.remote() for w in heads])
        new_records += ray.get([w.sync_policy.remote() for w in non_heads])
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

    # Destroy
    # workers[0].destroy.remote()
    # [ray.kill(w) for w in workers]

    # Return
    return workers

start = time.perf_counter()
# workers = ray.get(train.remote(2, 2, 4, rollout_kwargs={'dummy': True}, update_kwargs={'update_iterations': 5}))
# workers = ray.get(train.remote(2, 2, 4, sync_across_nodes=False))  # sync_across_nodes=False
workers = ray.get(train.remote(2, 1, 4))  # TODO: Adjust lr_gamma
print(time.perf_counter() - start)


# %%
# env = env_init(parent_dir=True).to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)
# celltrip.train.simulate_until_completion(policy, env, memory)
# memory.propagate_rewards()
# memory.normalize_rewards()
# # for _ in range(5):
# #     memory.append_memory(memory)
# len(memory)
# # memory.fast_sample(10_000, shuffle=False)



