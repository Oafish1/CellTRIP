from collections import defaultdict

import numpy as np
import ray
import ray.util.collective as col
import torch

from . import decorator as _decorator


def get_policy_state(policy):
    return {
        'policy': policy.state_dict(),
        'optimizer': policy.optimizer.state_dict(),
        'scheduler': policy.scheduler.state_dict(),
    }


def set_policy_state(policy, state_dicts):
    policy.load_state_dict(state_dicts['policy'])
    policy.optimizer.load_state_dict(state_dicts['optimizer'])
    policy.scheduler.load_state_dict(state_dicts['scheduler'])


def set_device_recursive(state_dict, device):
    # https://github.com/pytorch/pytorch/issues/1442#issue-225795077
    for key in state_dict:
        if isinstance(state_dict[key], dict):
            state_dict[key] = set_device_recursive(state_dict[key], device)
        else:
            try:
                state_dict[key] = state_dict[key].to(device)
            except:
                pass
    return state_dict


def simulate_until_completion(policy, env, memory=None, keys=None, dummy=False, verbose=False):
    # Params
    if keys is None: keys = env.get_keys()

    # Simulation
    ep_timestep = 0; ep_reward = 0; ep_itemized_reward = defaultdict(lambda: 0); finished = False
    while not finished:
        with torch.inference_mode():
            # Get current state
            state = env.get_state(include_modalities=True)

            # Get actions from policy
            actions = policy.act_macro(
                state,
                keys=keys,
                memory=memory,
            )

            # Step environment and get reward
            rewards, finished, itemized_rewards = env.step(actions, return_itemized_rewards=True)

            # Record rewards
            memory.record(rewards=rewards, is_terminals=finished)

            # Tracking
            ts_reward = rewards.cpu().mean()
            ep_reward = ep_reward + ts_reward
            for k, v in itemized_rewards.items():
                ep_itemized_reward[k] += v.cpu().mean()
            ep_timestep += 1

            # Dummy return for testing
            if dummy:
                if memory and not finished:
                    # 1000 steps
                    ep_timestep = int(1e3)
                    for _ in range(ep_timestep-1): memory.append_memory({k: v[-1:] for k, v in memory.storage.items()})
                    memory.storage['is_terminals'][-1] = True
                break
        
        # CLI
        if verbose and ((ep_timestep % 200 == 0) or ep_timestep in (100,)):
            print(f'Timestep {ep_timestep:>4} - Reward {ts_reward:.3f}')

    # Summarize and return
    ep_reward = (ep_reward / ep_timestep).item()
    ep_itemized_reward = {k: (v / ep_timestep).item() for k, v in ep_itemized_reward.items()}
    return ep_timestep, ep_reward, ep_itemized_reward


@ray.remote(num_cpus=1e-4, num_gpus=1e-4)
class Worker:
    """
    Learner, runner, or both. Runs given environment and synchronizes
    policy with other workers on updates given behavior implied by
    parent worker.
    """
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

    @_decorator.metrics(append_to_dict=True)
    # @_decorator.profile(time_annotation=True)
    def rollout(self, **kwargs):
        # Perform rollout
        result = simulate_until_completion(
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
    
    @_decorator.metrics(append_to_dict=True)
    # @_decorator.profile(time_annotation=True)
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

    @_decorator.metrics(append_to_dict=True, dict_index=1)
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
    
    @_decorator.metrics(append_to_dict=True)
    # @_decorator.profile(time_annotation=True)
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
        return get_policy_state(self.policy)
    
    @_decorator.metrics(append_to_dict=True)
    def sync_policy(self, iterate_if_exclusive_runner=True):
        # Copy policy
        if self.parent is not None:
            policy_state = ray.get(self.parent.get_policy_state.remote())
            set_policy_state(self.policy, policy_state)
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


@ray.remote
def train_celltrip(
    num_gpus,
    num_learners,
    num_runners,
    initializers,
    learners_can_be_runners=True,
    sync_across_nodes=True,
    updates=50,
    steps=int(5e3),
    rollout_kwargs={},
    update_kwargs={}):
    """
    Train CellTRIP model to completion with given parameters and
    cluster layout.
    """
    # Parameters
    policy_init, env_init, memory_init = initializers

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
    # TODO: Maybe add/try async updates
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
