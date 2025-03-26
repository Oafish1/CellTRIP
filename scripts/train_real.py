# %%
# %load_ext autoreload
# %autoreload 2


# %%
import ray
import ray.util.collective as col
import torch

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
def env_init(parent=False):
    # Create dataloader
    fnames = ['./data/MERFISH/expression.h5ad', './data/MERFISH/spatial.h5ad']
    if parent: fnames = ['.' + f for f in fnames]
    partition_cols = None  # 'layer'
    adatas = celltrip.utility.processing.read_adatas(*fnames, on_disk=False)
    celltrip.utility.processing.test_adatas(*adatas, partition_cols=partition_cols)
    dataloader = celltrip.utility.processing.PreprocessFromAnnData(
        *adatas, partition_cols=partition_cols, num_nodes=200, pca_dim=128, seed=42)
    # modalities, adata_obs, adata_vars = dataloader.sample()
    # Return env
    return celltrip.environment.EnvironmentBase(dataloader, dim=3, penalty_bound=1)

policy_init = lambda env: celltrip.policy.PPO(
    2*env.dim, env.dataloader.modal_dims, env.dim) # update_iterations=2, minibatch_size=3e3,

memory_init = lambda policy: celltrip.memory.AdvancedMemoryBuffer(
    sum(policy.modal_dims), split_args=policy.split_args)  


# %%
# env = env_init(parent=True).to('cuda')
# policy = policy_init(env).to('cuda')
# memory = memory_init(policy)
# celltrip.train.simulate_until_completion(policy, env, memory)
# memory.propagate_rewards()
# memory.normalize_rewards()
# # for _ in range(5):
# #     memory.append_memory(memory)
# len(memory)
# # memory.fast_sample(10_000, shuffle=False)

# %%
@ray.remote(num_gpus=1)
class Worker:
    def __init__(
        self,
        policy_init,
        env_init,
        memory_init=lambda: None,
        world_size=1,
        rank=0,
    ):
        # Detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Parameters
        self.env = env_init().to(device)
        self.policy = policy_init(self.env).to(device)
        self.memory = memory_init(self.policy)
        self.rank = rank

        # World initialization
        col.init_collective_group(world_size, rank, 'nccl')

        # Policy parameters
        self.sync_policy()
        self.policy_iteration = 0

        # Memory parameters
        self.memory_buffer = []

    @celltrip.decorator.metrics(append_to_dict=True)
    # @celltrip.decorator.profile(time_annotation=True)
    def rollout(self):
        # Perform rollout
        result = celltrip.train.simulate_until_completion(
            self.policy, self.env, self.memory, dummy=False)
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

    def rollout_until_new(self, num_new, condition='steps'):
        # Parameters
        if condition == 'memories': measure = self.memory.get_new_len
        elif condition == 'steps': measure = self.memory.get_new_steps
        else: raise ValueError(f'Condition `{condition}` not found.')

        # Compute rollouts
        ret = []
        while measure() < num_new:
            ret.append(self.rollout())
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
        
    @celltrip.decorator.metrics(append_to_dict=True)
    # @celltrip.decorator.profile(time_annotation=True)
    def update(self):
        # Perform update
        self.memory.normalize_rewards()
        self.policy.update(self.memory, verbose=True)

        # Annotate
        self.policy_iteration += 1
        num_new_memories = self.memory.get_new_len()
        num_replay_memories = self.memory.get_replay_len()

        # Clean
        self.memory.mark_sampled()
        self.memory.cleanup()

        # Record
        # TODO: Fix num_* being incorrect (seems like `get_new_len` does half? Test others too)
        ret = {
            'Event Type': 'Update',
            'Policy Iteration': self.policy_iteration,
            'Rank': self.rank,
            'New Memories': num_new_memories,
            'Replay Memories': num_replay_memories,
            'Total Memories': len(self.memory)}
        return ret
    
    def sync_policy(self):
        world_size = col.get_collective_group_size()
        for k, w in self.policy.state_dict().items():
            col.allreduce(w)
            w /= world_size

    def destroy(self):
        col.destroy_collective_group()


# %%
import time
start = time.perf_counter()


# %%
@ray.remote
def train(num_workers, updates, steps):
    workers = [Worker.remote(
        policy_init, env_init, memory_init,
        world_size=num_workers, rank=i) for i in range(num_workers)]
    # TODO: Learners and workers, maybe multiple workers per learner?

    records = []
    for _ in range(updates):
        # Rollouts
        num_records = len(records)
        new_records = ray.get([w.rollout_until_new.remote(steps/num_workers) for w in workers])
        records += sum(new_records, [])
        for record in records[-(len(records)-num_records):]: print(record)

        # Collect memories
        num_records = len(records)
        ret = ray.get([w.send_memory.remote(new=True) for w in workers])
        new_memories, new_records = [[r[i] for r in ret] for i in range(2)]
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

        # Broadcast memories
        num_records = len(records)
        new_records = []
        for i, w in enumerate(workers):
            new_memories_w = [ref for j, ref in enumerate(new_memories) if i!=j]
            future = w.recv_memories.remote(new_memories=new_memories_w)
            new_records.append(future)
        new_records = ray.get(new_records)
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

        # Updates
        num_records = len(records)
        new_records = ray.get([w.update.remote() for w in workers])
        records += new_records
        for record in records[-(len(records)-num_records):]: print(record)

    # Destroy
    # workers[0].destroy.remote()
    # [ray.kill(w) for w in workers]

    # Return
    return workers

workers = ray.get(train.remote(2, 50, 5e3))
# workers = ray.get(train.remote(2, 10, 9e4))


# %%
# # Parameters
# num_runners = 4
# num_learners = 2

# # Create placement groups
# pg_runners = ray.util.placement_group(num_runners*[{'CPU': 1e-4, 'GPU': 1e-4}], strategy='SPREAD')
# pg_learners = ray.util.placement_group(num_learners*[{'CPU': 1e-4, 'GPU': 1e-4}], strategy='STRICT_SPREAD')
# ray.get([pg_runners.ready(), pg_learners.ready()], timeout=10)


# %%
print(time.perf_counter() - start)



