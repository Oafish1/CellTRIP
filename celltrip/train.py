from collections import defaultdict
import functools as ft
import json
import threading
import warnings

import ray
import torch

from . import decorator as _decorators
from . import utility as _utility


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


class _PolicyManager:
    def __init__(self, policy_init, memory_init):
        self.policy = policy_init()
        self.memory = memory_init(self.policy)
        self.locks = {
            'memory': threading.Lock(),
            'state': threading.Lock()}

    def get_policy_state(self):
        return get_policy_state(self.policy)
    
    def get_memory_storage(self, persistent=True):
        storage = self.memory.get_storage()
        if not persistent: storage = storage[0]
        return storage
    
    def get_memory_len(self):
        return len(self.memory)

    def set_policy_state(self, state_dicts):
        set_policy_state(self.policy, state_dicts)

    def lock(self, lock_id, val):
        # NOTE: These locks are not failure-resilient
        if lock_id not in self.locks:
            raise ValueError(f'Lock `{lock_id}` not found.')

        if val: self.locks[lock_id].acquire()
        else:
            if self.locks[lock_id].locked():
                self.locks[lock_id].release()
            else:
                raise RuntimeWarning('Attempting to release unreserved lock')

        return val

    def get_lock(self, lock_id):
        if lock_id not in self.locks:
            raise ValueError(f'Lock `{lock_id}` not found.')

        return self.locks[lock_id].locked()

    def release_locks(self):
        for lock_id in self.locks:
            if self.locks[lock_id].locked():
                self.locks[lock_id].release()

    def clear_memory(self):
        self.memory.clear()

    def append_memory(self, *args):
        self.memory.append_memory(*args)
PolicyManager = ray.remote(_PolicyManager)


@_decorators.metrics
@_decorators.try_catch(show_traceback=True)
@_decorators.call_on_exit
# @_decorators.profile
def _train_func(
    policy_manager,
    modalities,
    policy_init,
    env_init,
    memory_init,
    meta=None,
    **kwargs,
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # CLI
    # print('Beginning simulation')

    # Load policy
    policy = policy_init().to(device)
    ray.get(policy_manager.lock.remote('state', True))
    set_policy_state(policy, ray.get(policy_manager.get_policy_state.remote()))
    ray.get(policy_manager.lock.remote('state', False))

    # Initialize env/memory
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        modalities = [torch.from_numpy(m).to(device) for m in modalities]  # Must be float32
    env = env_init(policy, modalities)
    memory = memory_init(policy)
    
    # Simulate
    ep_timestep, ep_reward, ep_itemized_reward = simulate_until_completion(policy, env, memory, **kwargs)

    # Append memories
    ray.get(policy_manager.lock.remote('memory', True))
    ray.get(policy_manager.append_memory.remote(*memory.get_storage()))
    ray.get(policy_manager.lock.remote('memory', False))

    # CLI
    # print(f'Simulation finished in {ep_timestep} steps with mean reward {ep_reward:.3f}')

    # Aggregate performance
    ret = {
        'Event Type': 'Rollout',
        'Event Details': {
            'Episode timesteps': ep_timestep,
            'Episode reward': ep_reward,
            'Episode itemized reward': ep_itemized_reward}}
    ret = _utility.general.dict_entry(ret, meta)
    print(json.dumps(ret, indent=2, sort_keys=False))
    return ret
train_func = ray.remote(max_calls=1)(_train_func)


@_decorators.metrics
@_decorators.try_catch(show_traceback=True)
@_decorators.call_on_exit
# @_decorators.profile
def _update_func(
    policy_manager,
    policy_init,
    memory_init,
    meta=None,
    **kwargs,
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # CLI
    # print('Beginning policy update')

    # Load policy
    policy = policy_init().to(device)
    ray.get(policy_manager.lock.remote('state', True))
    set_policy_state(policy, ray.get(policy_manager.get_policy_state.remote()))
    ray.get(policy_manager.lock.remote('state', False))

    # Load memory
    memory = memory_init(policy)
    ray.get(policy_manager.lock.remote('memory', True))
    memory.append_memory(
        *ray.get(policy_manager.get_memory_storage.remote()))
    # NOTE: Keep locked during update to prevent adding memories and rerun on crash without clearing memories

    # Perform backwards and update actor state
    policy.update(memory, **kwargs)

    ray.get(policy_manager.lock.remote('state', True))
    policy_state = set_device_recursive(get_policy_state(policy), 'cpu')
    ray.get(policy_manager.set_policy_state.remote(policy_state))
    ray.get(policy_manager.lock.remote('state', False))

    # Clear remote memory
    ray.get(policy_manager.clear_memory.remote())
    ray.get(policy_manager.lock.remote('memory', False))

    # CLI
    # print(f'Done updating policy on {len(memory)} memories')

    # Format return
    ret = {
        'Event Type': 'Update',
        'Event Details': {'Memories': len(memory)}}
    ret = _utility.general.dict_entry(ret, meta)
    print(json.dumps(ret, indent=2, sort_keys=False))
    return ret

    
update_func = ray.remote(max_calls=1)(_update_func)


class DistributedManager:
    def __init__(
        self,
        modalities=None,
        policy_init=None,
        env_init=None,
        memory_init=None,
        max_jobs_per_gpu=4,
    ):
        # Record
        self.modalities = modalities
        self.policy_init = policy_init
        self.env_init = env_init
        self.memory_init = memory_init
        self.max_jobs_per_gpu = max_jobs_per_gpu

        # Initialize linalg
        torch.inverse(torch.ones((1, 1), device='cpu'))

        # Initialize ray
        if not ray.is_initialized(): ray.init()

        # Put policy and modalities in object store
        # self.policy_ref = ray.put(zerocopy.extract_tensors(self.policy))
        if modalities is not None: self.modalities = ray.put(modalities)  # Zero copy
        self.policy_manager = PolicyManager.options(max_concurrency=int(1e3), num_cpus=1).remote(policy_init, memory_init)

        # Initialize futures list
        self.futures = defaultdict(lambda: [])

        # Initialize requirements
        self.resources = {
            'rollout': {
                'core': {
                    'num_cpus': 1e-4,
                    'memory': 0,
                    'num_gpus': 1*(max_jobs_per_gpu>0)},
                'custom': {
                    'VRAM': 0}},
            'update': {
                'core': {
                    'num_cpus': 1e-4,
                    'memory': 0,
                    'num_gpus': 1*(max_jobs_per_gpu>0)},
                'custom': {
                    'VRAM': 0}}}
        
    def get_requirements(self, key):
        requirements = ft.reduce(lambda l,r: dict(**l, **r), self.resources[key].values(), {})
        requirements['CPU'] = requirements.pop('num_cpus')
        requirements['GPU'] = requirements.pop('num_gpus')

        return requirements

    def get_futures(self):
        return self.futures

    def get_all_futures(self, keys=None):
        if not _utility.general.is_list_like(keys): keys = [keys]
        keys = list(self.futures.keys()) if len(keys) == 1 and keys[0] is None else keys
        return sum([self.futures[k] for k in keys], [])
    
    def get_memory_len(self):
        ray.get(self.policy_manager.lock.remote('memory', True))
        memory_len = ray.get(self.policy_manager.get_memory_len.remote())
        ray.get(self.policy_manager.lock.remote('memory', False))

        return memory_len

    # Futures
    def rollout(self, num_simulations=1, modalities=None, keys=None, env_init=None, **kwargs):
        # Defaults
        if modalities is None: modalities = self.modalities
        if keys is None: keys = torch.arange(modalities[0].shape[0]).tolist()
        if env_init is None: env_init = self.env_init

        # Run forward on each worker to completion until a certain number of steps is reached
        def exit_hook():
            ray.get(self.policy_manager.release_locks.remote())
            print('Escaped running rollout')
        rollouts = [
            train_func
                .options(**self.resources['rollout']['core'], resources=self.resources['rollout']['custom'])
                .remote(
                    self.policy_manager,
                    modalities,
                    self.policy_init,
                    env_init,
                    self.memory_init,
                    keys=keys,
                    **kwargs,
                    return_metrics=self.resources['rollout']['core']['memory']==0,
                    exit_hook=exit_hook)
            for _ in range(num_simulations)]
        self.futures['rollout'] += rollouts

        return rollouts
        
    def update(self, **kwargs):
        # Run update
        def exit_hook():
            ray.get(self.policy_manager.release_locks.remote())
            print('Escaped running update')
        updates = [
            update_func
                .options(**self.resources['update']['core'], resources=self.resources['update']['custom'])
                .remote(
                    self.policy_manager,
                    self.policy_init,
                    self.memory_init,
                    **kwargs,
                    return_metrics=self.resources['update']['core']['memory']==0,
                    exit_hook=exit_hook)]
        self.futures['update'] += updates

        return updates
    
    # Utility
    def cancel(self, keys=None):
        # Cancel futures
        if not _utility.general.is_list_like(keys): keys = [keys]
        keys = list(self.futures.keys()) if len(keys) == 1 and keys[0] is None else keys
        # NOTE: Current escape behavior is to release all locks, regardless of holding
        [ray.cancel(future) for future in self.get_all_futures(keys)]
    
    def calibrate(self, memory_overhead=300*2**20, vram_overhead=300*2**20):
        for k in ('rollout', 'update'):
            completed_rollouts, _ = ray.wait(self.futures[k], num_returns=len(self.futures[k]), timeout=0)
            if len(completed_rollouts) > 0:
                metrics = [m for _, m in ray.get(completed_rollouts) if m is not None]
                assert len(metrics) > 0, f'`{k}` future was run without `return_metrics=True`'
                self.resources[k]['core']['memory'], self.resources[k]['custom']['VRAM'] = max([m[0] for m in metrics]), max([m[1] for m in metrics])
                self.resources[k]['core']['memory'] += memory_overhead
                if self.resources[k]['custom']['VRAM'] != 0:
                    self.resources[k]['custom']['VRAM'] += vram_overhead
                    self.resources[k]['core']['num_gpus'] = max(1e-4, 1./self.max_jobs_per_gpu)

    def wait(self, keys=None, **kwargs):
        if not _utility.general.is_list_like(keys): keys = [keys]
        keys = list(self.futures.keys()) if len(keys) == 1 and keys[0] is None else keys
        return ray.wait(self.get_all_futures(keys), **kwargs)[0]  # Wait for at least one completion, return all completions

    # def wait_all(self, verbose=False):
    #     # Get all futures
    #     all_futures = self.get_all_futures()

    #     # Await completions
    #     if verbose: pbar = tqdm(total=len(all_futures))
    #     for future in all_futures:
    #         ray.get(future)
    #         if verbose: pbar.update()
    #     if verbose: pbar.close()
    #     # TODO: Remove from list
    
    def clean(self, keys=None):
        # Remove finished futures, will be used in the future for metrics
        if not _utility.general.is_list_like(keys): keys = [keys]
        keys = list(self.futures.keys()) if keys[0] is None else keys
        for k in keys:
            complete_futures, self.futures[k] = ray.wait(self.futures[k], num_returns=len(self.futures[k]), timeout=0)


def simulate_until_completion(policy, env, memory=None, keys=None, dummy=False):
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
                if memory:
                    # 1024 dummy memories
                    for _ in range(10): memory.append_memory(memory)
                break
        
        # CLI
        # if ep_timestep % 200 == 0: print(f'Timestep {ep_timestep:03} - Reward {ts_reward:.3f}')

    # Summarize and return
    ep_reward = (ep_reward / ep_timestep).item()
    ep_itemized_reward = {k: (v / ep_timestep).item() for k, v in ep_itemized_reward.items()}
    return ep_timestep, ep_reward, ep_itemized_reward
