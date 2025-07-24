from collections import defaultdict
from datetime import datetime
import json
import warnings

import numpy as np
import ray
import ray.util.collective as col
import torch

from . import decorator as _decorator
from . import environment as _environment
from . import memory as _memory
from . import policy as _policy
from . import utility as _utility


# @_decorator.profile
def simulate_until_completion(
    env, policy, memory=None, keys=None,
    max_timesteps=np.inf, max_memories=np.inf, reset_on_finish=False,
    cache_feature_embeds=True, store_states=False, flush=True,
    dummy=False, verbose=False):
    # NOTE: Does not flush buffer
    # Params
    assert not (keys is not None and reset_on_finish), 'Cannot manually set keys while `reset_on_finish` is `True`'
    if keys is None: keys = env.get_keys()

    # Store states
    if store_states: state_storage = [env.get_state()]

    # Simulation
    ep_timestep = 0; ep_memories = 0; ep_reward = 0; ep_itemized_reward = defaultdict(lambda: 0); finished = False
    feature_embeds = None
    with torch.inference_mode():
        while True:
            # Get current state
            state = env.get_state(include_modalities=True)

            # Get actions from policy
            actions = policy(
                state, keys=keys, memory=memory,
                feature_embeds=feature_embeds, return_feature_embeds=cache_feature_embeds)
            if cache_feature_embeds: actions, feature_embeds = actions

            # Step environment and get reward
            rewards, finished, itemized_rewards = env.step(actions, return_itemized_rewards=True)

            # Store states
            if store_states: state_storage.append(env.get_state())

            # Tracking
            ts_reward = rewards.cpu().mean()
            ep_reward = ep_reward + ts_reward
            for k, v in itemized_rewards.items():
                ep_itemized_reward[k] += v.cpu().mean()
            ep_timestep += 1
            ep_memories += env.num_nodes

            # Record rewards
            continue_condition = ep_timestep < max_timesteps
            if memory is not None: continue_condition *= ep_memories < max_memories
            if memory is not None: memory.record_buffer(rewards=rewards, is_terminals=finished or not continue_condition)

            # Dummy return for testing
            # if dummy:
            #     if not finished:
            #         # Fill all but first and last
            #         ep_timestep = env.max_timesteps
            #         memory.flush_buffer()
            #         for _ in range(ep_timestep-2):
            #             if memory is not None: memory.append_memory({k: v[-1:] for k, v in memory.storage.items()})
            #             if store_states: state_storage.append(env.get_state)
            #         memory.storage['is_terminals'][-1] = True
        
            # CLI
            if verbose and ((ep_timestep % 200 == 0) or ep_timestep in (100,) or finished):
                print(f'Timestep {ep_timestep:>4} - Reward {ts_reward:.3f}')
        
            # Record terminals and reset if needed
            if finished or not continue_condition:
                state = env.get_state(include_modalities=True)
                policy(state, keys=keys, memory=memory, terminal=True, feature_embeds=feature_embeds)
            # Reset if needed
            if finished:
                if reset_on_finish:
                    env.reset()
                    keys = env.get_keys()
                    feature_embeds = None
                else: break
            # Escape
            if not continue_condition: break

    # Flush
    if flush and memory: memory.flush_buffer()

    # Summarize and return
    denominator = 1  # env.max_timesteps if env.max_timesteps is not None else ep_timestep
    ep_reward = (ep_reward / denominator).item()  # Standard mean
    ep_itemized_reward = {k: (v / denominator).item() for k, v in ep_itemized_reward.items()}
    ret = (ep_timestep, ep_memories, ep_reward, ep_itemized_reward)
    if store_states:
        state_storage = torch.stack(state_storage)
        ret += (state_storage,)
    return ret


@ray.remote(num_cpus=1e-4)
class RecordBuffer:
    def __init__(self, logfile='cli', hooks=[], flush_on_record=None):
        # Defaults
        self.flush_on_record = (logfile=='cli') if flush_on_record is None else flush_on_record
        self.hooks = hooks

        # Initialize
        self.buffer = []

        # Choose writing function
        self.logfile = logfile
        if self.logfile is None:
            self.write = lambda _: None
        elif self.logfile == 'cli':
            self.write = self._write_cli
        elif self.logfile.startswith('s3://'):
            # Get s3 handle and generate logfile
            self.s3 = _utility.general.get_s3_handler_with_access(self.logfile)
            self.s3.open(self.logfile, 'w').close()  # Create/erase
            self.write = self._write_s3
        else:
            open(self.logfile, 'w').close()  # Create/erase
            self.write = self._write_disk

    def _write_cli(self):
        for record in self.buffer:
            print(json.dumps(record))

    def _write_s3(self):
        with self.s3.open(self.logfile, 'a') as f:
            for record in self.buffer:
                f.write(json.dumps(record) + '\n')

    def _write_disk(self):
        with open(self.logfile, 'a') as f:
            for record in self.buffer:
                f.write(json.dumps(record) + '\n')

    def _execute_hooks(self):
        for f in self.hooks: f(self.buffer)

    def record(self, *records):
        self.buffer += records
        if self.flush_on_record: self.flush()

    def flush(self):
        self.write()
        self._execute_hooks()
        self.buffer.clear()


@ray.remote(num_cpus=1e-4, num_gpus=1e-4)
class Worker:
    """
    Learner, runner, or both. Runs given environment and synchronizes
    policy with other workers on updates given behavior implied by
    parent worker.
    """
    def __init__(
        self,
        env_init,
        policy_init,
        memory_init,
        num_learners=1,
        num_heads=1,
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
        self.num_learners = num_learners
        self.num_heads = num_heads
        self.rank = rank
        self.learner = learner
        self.parent = parent

        # World initialization
        if learner: col.init_collective_group(num_learners, rank, 'nccl', 'learners')
        if (parent is None) and not (learner and rank!=0): col.init_collective_group(
            num_heads-num_learners+1, rank-num_learners+1 if rank !=0 else rank, 'nccl', 'heads')

        # Policy parameters
        self.sync_policy(mark_if_not_learner=False)  # Works because non-heads will wait for head sync

        # Flags
        self.flags = {}

    def is_ready(self):
        "This method is required to make initialization waitable in Ray"
        return True

    @_decorator.metrics(append_to_dict=True)
    # @_decorator.line_profile(signatures=[
    #     simulate_until_completion,
    #     _environment.EnvironmentBase.reset,
    #     _utility.processing.PreprocessFromAnnData._transform_disk,
    #     _utility.processing.Preprocessing.subsample])
    # @_decorator.profile(time_annotation=True)
    def rollout(self, num_new=None, condition='memories', **kwargs):
        # Arguments
        max_kwargs = {}
        if condition == 'steps': max_kwargs['max_timesteps'] = num_new
        elif condition == 'memories': max_kwargs['max_memories'] = num_new

        # Perform rollout
        result = simulate_until_completion(
            self.env, self.policy, self.memory,
            reset_on_finish=(num_new is not None), **max_kwargs,
            **kwargs)
        
        # Reset and clean
        # self.env.reset()
        self.memory.cleanup()

        # Record
        timestep, memories, reward, itemized_reward = result
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Rollout',
            'Policy Iteration': self.policy.get_policy_iteration(),
            'Rank': self.rank,
            'Timesteps': timestep,
            'Memories': memories,
            'Reward': reward,
            'Itemized Reward': itemized_reward}
        return ret

    # def rollout_until_new(self, num_new, condition='steps', **kwargs):
    #     # Parameters
    #     if condition == 'memories': measure = self.memory.get_new_len
    #     elif condition == 'steps': measure = self.memory.get_new_steps
    #     else: raise ValueError(f'Condition `{condition}` not found.')

    #     # Compute rollouts
    #     ret = []
    #     while measure() < num_new:
    #         ret.append(self.rollout(**kwargs))
    #     return ret
    
    @_decorator.metrics(append_to_dict=True)
    # @_decorator.line_profile(signatures=[
    #     _memory.AdvancedMemoryBuffer.compute_advantages,
    #     _memory.AdvancedMemoryBuffer.fast_sample, _policy.PPO.update])
    # @_decorator.profile(time_annotation=True)
    def update(self, **kwargs):
        # Flush buffer
        self.memory.flush_buffer()

        # Compute GAE
        # self.memory.recompute_state_vals(self.policy)  # Technically better when replay used, but takes a ton of time
        self.memory.compute_advantages()  # moving_standardization=self.policy.reward_standardization

        # Synchronize reward normalization
        # self.memory.feed_new(self.policy.reward_standardization)
        # self.policy.synchronize('learners')

        # Perform update
        # with torch.autograd.detect_anomaly():
        iterations, losses, statistics = self.policy.update(self.memory, **kwargs)

        # Annotate and clean
        num_new_memories = self.memory.get_new_len()
        num_replay_memories = self.memory.get_replay_len()
        self.memory.mark_sampled()

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Update',
            'Policy Iteration': self.policy.get_policy_iteration(),
            'Rank': self.rank,
            'New Memories': num_new_memories,
            'Replay Memories': num_replay_memories,
            'Total Memories': len(self.memory),
            'Iterations': iterations,
            'Losses': losses,
            'Statistics': statistics}
        return ret
    
    def set_flags(self, **new_flag_values):
        assert np.array([k in self.flags for k in new_flag_values]).all(), (
            'Unknown flag found in `set_flag`.')
        self.flags.update(new_flag_values)

    @_decorator.metrics(append_to_dict=True, dict_index=1)
    def send_memory(self, **kwargs):
        # Put in object store
        self.memory.flush_buffer()
        mem = self.memory.get_storage(**kwargs)
        ref = ray.put(mem)

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Send Memory',
            'Rank': self.rank,
            'Memories': sum([s.shape[0] for s in mem[0]['states']])}
        return ref, ret
    
    @_decorator.metrics(append_to_dict=True)
    # @_decorator.profile(time_annotation=True)
    def recv_memories(self, new_memories):
        # Append and clean memories
        num_memories = 0
        for new_memory in new_memories:
            new_memory = ray.get(new_memory)
            self.memory.append_memory(*new_memory)
            num_memories += sum([s.shape[0] for s in new_memory[0]['states']])
        self.memory.cleanup()

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Receive Memories',
            'Rank': self.rank,
            'Memories': num_memories}
        return ret
    
    def get_policy_state(self):
        return _utility.general.get_policy_state(self.policy)
    
    def get_policy_iteration(self):
        return self.policy.get_policy_iteration()
    
    @_decorator.metrics(append_to_dict=True)
    def sync_policy(self, mark_if_not_learner=True):
        # Copy policy
        if self.parent is not None:
            policy_state = ray.get(self.parent.get_policy_state.remote())
            _utility.general.set_policy_state(self.policy, policy_state)
        else:
            # Synchronize learners
            if self.learner: self.policy.synchronize('learners')

            # Synchronize head learner and heads
            if not self.learner or (self.learner and self.rank == 0):
                self.policy.synchronize('heads', broadcast=True)

        # Mark sampled
        if mark_if_not_learner and not self.learner: self.memory.mark_sampled()

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Synchronize Policy',
            'Policy Iteration': self.policy.get_policy_iteration(),
            'Rank': self.rank,
            'Inherited': self.parent is not None}
        return ret
    
    @_decorator.metrics(append_to_dict=True)
    def save_checkpoint(self, directory, name=None):
        self.policy.save_checkpoint(directory, name=name)

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Save Checkpoint',
            'Policy Iteration': self.policy.get_policy_iteration(),
            'Rank': self.rank}
        return ret
    
    @_decorator.metrics(append_to_dict=True)
    def load_checkpoint(self, fname):
        self.policy.load_checkpoint(fname)

        # Record
        ret = {
            'Timestamp': str(datetime.now()),
            'Event Type': 'Load Checkpoint',
            'Policy Iteration': self.policy.get_policy_iteration(),
            'Rank': self.rank}
        return ret
    
    def execute(self, func):
        "Remote execution macro"
        return func(self)

    def destroy(self):
        col.destroy_collective_group()


def get_initializers(
    # Environment
    input_files=None,
    merge_files=None,
    backed=False,
    partition_cols=None,
    dataloader_kwargs={},
    environment_kwargs={},
    policy_kwargs={},
    memory_kwargs={},
):
    # Defaults
    dataloader_kwargs_defaults = {'seed': 42}  # Necessary for preprocessing consistency
    dataloader_kwargs_defaults.update(dataloader_kwargs)
    dataloader_kwargs = dataloader_kwargs_defaults

    def env_init():
        # Create dataloader
        adatas = []
        if input_files is not None:
            adatas += _utility.processing.read_adatas(*input_files, backed=backed)
        if merge_files is not None:
            for merge_files_rec in merge_files:
                merge_adatas = _utility.processing.read_adatas(*merge_files_rec, backed=backed)
                adatas += _utility.processing.merge_adatas(*merge_adatas, backed=backed)
        _utility.processing.test_adatas(*adatas, partition_cols=partition_cols)
        dataloader = _utility.processing.PreprocessFromAnnData(
            *adatas, partition_cols=partition_cols, **dataloader_kwargs)
        # modalities, adata_obs, adata_vars = dataloader.sample()
        # Return env
        return _environment.EnvironmentBase(
            dataloader, **environment_kwargs)

    # Default ~25Gb Forward, ~16Gb Update, at max capacity
    policy_init = lambda env: _policy.PPO(
        2*env.dim, np.array(env.dataloader.modal_dims)[env.input_modalities], env.dim, **policy_kwargs)  # update_iterations=2, minibatch_size=3e3,

    memory_init = lambda policy: _memory.AdvancedMemoryBuffer(
        sum(policy.modal_dims), split_args=policy.split_args, **memory_kwargs)
    
    return (env_init, policy_init, memory_init)


def train_celltrip(
    initializers,
    num_gpus,
    num_learners,
    num_runners,
    learners_can_be_runners=True,
    sync_across_nodes=True,
    max_timesteps=2_000_000_000,
    update_timesteps=100_000,
    delayed_rollout_flush=True,
    flush_iterations=50,
    checkpoint_iterations=50,
    checkpoint_dir='./checkpoints',
    checkpoint_name=None,
    checkpoint=None,
    stage_functions=[],
    logfile='cli',
    rollout_kwargs={},
    update_kwargs={},
    early_stopping_kwargs={},
    record_hooks=[]):
    """
    Train CellTRIP model to completion with given parameters and
    cluster layout.
    """
    # Parameters
    env_init, policy_init, memory_init = initializers

    # Create record buffer
    record_buffer = RecordBuffer.remote(
        logfile=logfile, hooks=record_hooks,
        flush_on_record=(flush_iterations is None))

    # Record
    record_buffer.record.remote({
        'Timestamp': str(datetime.now()),
        'Event Type': 'Begin Training'})

    # Stage functions
    curr_stage = 0

    # Make placement group for GPUs
    # TODO: Maybe tighten the placement group CPU requirements
    pg_gpu = ray.util.placement_group(num_gpus*[{'CPU': 1, 'GPU': 1}])
    ray.get(pg_gpu.ready())  # TODO: Why does reservation take so long?

    # Record
    record_buffer.record.remote({
        'Timestamp': str(datetime.now()),
        'Event Type': 'Register Placement Groups'})

    # Assign workers
    num_learner_runners = min(num_learners, num_runners) if learners_can_be_runners else 0
    num_exclusive_learners = num_learners - num_learner_runners if learners_can_be_runners else num_learners
    num_exclusive_runners = num_runners - num_learner_runners
    num_workers = num_exclusive_learners + num_learner_runners + num_exclusive_runners
    num_heads = min(num_gpus, num_workers)
    
    # Warn
    assert num_learners <= num_gpus, '`num_learners` cannot be greater than `num_gpus`.'
    if not sync_across_nodes and num_learners != num_heads:
        warnings.warn(
            'Runner will be placed on a node with no learner, and will not contribute to learning. '
            'Try synchronizing across nodes or adjusting the number of runners.')

    # Create workers
    workers = []
    for i in range(num_workers):
        bundle_idx = i % num_heads
        child_num = i // num_heads
        rank_decimal = 10**-(np.floor(np.log10((num_workers-1)//num_heads))+1) if num_workers > num_heads else 0
        rank = float(bundle_idx + child_num * rank_decimal)
        parent = workers[bundle_idx] if i >= num_heads else None
        w = (
            Worker
                .options(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg_gpu, placement_group_bundle_index=bundle_idx))
                .remote(
                    env_init=env_init, policy_init=policy_init, memory_init=memory_init,
                    num_learners=num_learners, num_heads=num_heads, rank=rank,
                    learner=i<num_learners, parent=parent))
        workers.append(w)
    ray.get([w.is_ready.remote() for w in workers])  # Wait for initialization
    learners = workers[:len(workers)-num_exclusive_runners]
    runners = workers[num_exclusive_learners:]
    heads = workers[:num_heads]
    non_heads = workers[num_heads:]

    # Load checkpoint
    # NOTE: Could be more efficient with sync
    if checkpoint is not None:
        record_buffer.record.remote(*ray.get([w.load_checkpoint.remote(fname=checkpoint) for w in workers]))
    policy_iteration = ray.get(workers[0].get_policy_iteration.remote())

    # Create early stopping class
    early_stopping = _utility.continual.EarlyStopping(**early_stopping_kwargs)

    # Record
    record_buffer.record.remote({
        'Timestamp': str(datetime.now()),
        'Event Type': 'Register Workers',
        'Policy Iteration': policy_iteration,
        'Stage': curr_stage})

    # Run policy updates
    # TODO: Maybe add/try async updates
    checkpoint_future = None
    for current_iteration in range(max_timesteps//update_timesteps):
        # Rollouts
        rewards = []
        new_records = ray.get([
            w.rollout.remote(num_new=update_timesteps/num_workers, flush=not delayed_rollout_flush, **rollout_kwargs, return_metrics=current_iteration==0) for w in runners])
        rewards += [record['Reward'] for record in new_records if record['Event Type'] == 'Rollout']
        record_buffer.record.remote(*new_records)
        mean_rollout_reward = np.mean(rewards)
        # if current_iteration==0:
        #     new_records = ray.get([w.rollout.remote(**rollout_kwargs, return_metrics=True) for w in runners])
        #     rewards += [record['Reward'] for record in new_records if record['Event Type'] == 'Rollout']
        #     record_buffer.record.remote(*new_records)
        # new_records = sum(ray.get([
        #     w.rollout_until_new.remote(num_new=steps/num_workers, flush=not delayed_rollout_flush, **rollout_kwargs) for w in runners]), [])
        # rewards += [record['Reward'] for record in new_records if record['Event Type'] == 'Rollout']
        # record_buffer.record.remote(*new_records)
        # mean_rollout_reward = np.mean(rewards)

        # Share memories
        if num_exclusive_runners > 0 or (sync_across_nodes and num_learners > 1):  # Not dummy-proof if someone strands a runner
            # Collect memories
            ret = ray.get([w.send_memory.remote(new=True) for w in runners])
            new_memories, new_records = [[r[i] for r in ret] for i in range(2)]
            record_buffer.record.remote(*new_records)

            # Broadcast memories
            futures = []; new_memories_w1 = []
            for i, w1 in enumerate(learners):
                if sync_across_nodes: new_memories_w1 = [ref for w2, ref in zip(runners, new_memories) if w1!=w2]
                else: new_memories_w1 = new_memories[num_heads+i::num_heads]
                future = w1.recv_memories.remote(new_memories=new_memories_w1)
                futures.append(future)
            del new_memories, new_memories_w1
            record_buffer.record.remote(*ray.get(futures))

        # Make sure checkpoint is finished
        # NOTE: Technically could be moved back, but unlikely to make a difference
        if checkpoint_future is not None:
            record_buffer.record.remote(ray.get(checkpoint_future))
            checkpoint_future = None

        # Updates
        record_buffer.record.remote(*ray.get([w.update.remote(
            **update_kwargs, return_metrics=current_iteration==0) for w in learners]))

        # Synchronize policies
        record_buffer.record.remote(*ray.get([w.sync_policy.remote() for w in heads]))
        record_buffer.record.remote(*ray.get([w.sync_policy.remote() for w in non_heads]))
        policy_iteration = ray.get(workers[0].get_policy_iteration.remote())  # A little inefficient

        # Flush records and checkpoint
        if ((current_iteration+1) % checkpoint_iterations == 0) and (checkpoint_dir is not None):
                checkpoint_future = workers[0].save_checkpoint.remote(
                    directory=checkpoint_dir, name=checkpoint_name)
        if flush_iterations is not None and current_iteration % flush_iterations == 0:
            record_buffer.flush.remote()

        # Early stopping
        # TODO: Observe how this interacts with the memory buffer when advancing stages
        # TODO: Maybe will need to normalize only on new memories
        if early_stopping(mean_rollout_reward):
            # Reset
            early_stopping.reset()

            # Escape if no more stages
            # TODO: Maybe do this based on action_std or KL (or both)
            if len(stage_functions) == curr_stage: continue
                # print(current_iteration)
                # break # Escape

            # Advance stage
            else:
                # ray.get([w.set_flags.remote(adjust_rewards_next_update=True) for w in learners])  # Recalibrate replay rewards next update
                ray.get([w.execute.remote(func=stage_functions[curr_stage]) for w in workers])
                # TODO: Clear memory? and/or make checkpoint
                curr_stage += 1

                # Record
                record_buffer.record.remote({
                    'Timestamp': str(datetime.now()),
                    'Event Type': 'Advance Stage',
                    'Policy Iteration': policy_iteration,
                    'Stage': curr_stage})
                
    # Record
    record_buffer.record.remote({
        'Timestamp': str(datetime.now()),
        'Event Type': 'Finish Training',
        'Policy Iteration': policy_iteration,
        'Stage': curr_stage})
    
    # Final flush and checkpoint
    if (checkpoint_dir is not None):
        if checkpoint_future is None:
            checkpoint_future = workers[0].save_checkpoint.remote(
                directory=checkpoint_dir, name=checkpoint_name)
        record_buffer.record.remote(ray.get(checkpoint_future))
        checkpoint_future = None
    ray.get(record_buffer.flush.remote())

    # Destroy
    # workers[0].destroy.remote()
    # [ray.kill(w) for w in workers]

    # Return
    return workers
