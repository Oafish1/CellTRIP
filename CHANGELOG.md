### 1.0.0+2025.6.25
- Additional analysis runs
- Figure updates

### 1.0.0+2025.6.24
- Fix partition selection for `Preprocessing.subsample`

### 1.0.0+2025.6.23
- Figure changes

### 1.0.0+2025.6.22
- `README.md` changes
- Add partition-based train/validation splitting
- Analysis bugfixes with feature indexing and retrieval
- Analysis visualization changes
- Bugfix for caching feature embeddings when `PPO.forward_batch_size` is exceeded

### 1.0.0+2025.6.21
- Environment reward tuning
- More individual runs for `Flysta3D` dataset

### 1.0.0+2025.6.20
- Analysis notebook changes
- Environment reward tuning
- Full dataset runs for `TemporalBrain` dataset
- Various individual runs for `Flysta3D` dataset

### 1.0.0+2025.6.19
- Bugfix for dataloader fit subsampling

### 1.0.0+2025.6.18
- Add functionality for targeted partition sampling
- Add minimum timesteps protection toggle
- Bugfix for spherical unbounded velocity initial state
- Clean and reorganize analysis notebook
- Figures for analysis notebook
- Fix environment behavior when not computing rewards
- Notebook version of public API
- Return `EnvironmentBase.finished` reason
- Tweak policy and environment parameters

### 1.0.0+2025.6.17
- Figure changes

### 1.0.0+2025.6.16
- Add normal distribution option to environment
- PopArt module compatibility with layer groups and custom indexing
- Input normalization using PopArt (novel)
- Testing with continuous output head and unbounded velocity
- Time estimation output to runtime checker

### 1.0.0+2025.6.14
- Add degrees to least square solver

### 1.0.0+2025.6.12
- Figure updates

### 1.0.0+2025.5.28
- Many tuning changes and trial runs

### 1.0.0+2025.5.26
- Use log expvar

### 1.0.0+2025.5.22 (1-2)
- Add animation saving using old high-quality writer
- Add notes and light benchmarks for `ROS3` and `h5Coro`
- Animation exports for Flysta3d
- Benchmark caching methods for `fsspec` and change default from `readforward` to `background` with a `100MB` block size, at `32` maximum blocks, resulting in >100x speedup for some datasets
- Fix bug with shell command parsing in arguments definition, especially relevant for `TemporalBrain` dataset
- Recompile requirements

### 1.0.0+2025.5.21
- Add friction to `EnvironmentBase` and change lstsq normalization again
- Move statistics outside minibatch loop, fixing rare `nan` statistic return
- Tweak environment rewards and params to stop expansion of discretized latent space

### 1.0.0+2025.5.20
- Change environment reward scheme to remove action penalty and unbound lstsq norm
- Change scaling to be more intuitive for cross-dim visualizations
- Fix env reset timing in steady state analysis
- Fix initial env reset in permutation analysis

### 1.0.0+2025.5.19
- Add bias for environment lstsq reward
- Add fix for subsampled mask in `analysis`
- Fix for analysis noise setting

### 1.0.0+2025.5.17
- Try different scaling for environment lstsq

### 1.0.0+2025.5.16
- Add `minibatch_memories` and ministeps to policy update, limiting the amount of data allowed for backward computation
- Add option for node feature padding in `AdvancedMemoryBuffer.__getitem__`, resulting in theoretical 2x speedup (almost 10x in practice)
- Optionally remove pre-appending and pre-cast from `AdvancedMemoryBuffer.__getitem__`, 3x slowdown but tremendous memory efficiency
- Update s3 access to allow for endpoints

### 1.0.0+2025.5.15
- Fix partition argument handling in all relevant notebooks

### 1.0.0+2025.5.14
- Add inner join for h5ad files in `merge_adatas`
- Additional loss for environment based on least-squares predictive capability, bounded by scale
- Many minor changes and additional completed runs
- Upload and generate commands for alternate datasets, and format existing datasets

### 1.0.0+2025.5.13
- Add individual dimension visualization for `analysis` notebook
- Change epsilon for environment and entropy coefficient for policy
- Change log calculation timing for environment distance match, with user-configurable options
- Fix `padded_concat` assigning to improper coordinates while padding
- Fix bug with resetting in `simulate_until_completion` not properly updating keys or caching feature embeddings
- Fix default sampling preference for environment reset
- Fix smoothing calculation in `runtime` notebook

### 1.0.0+2025.5.10
- Add deterministic operation to discrete model
- More tuning runs, better results on scGLUE

### 1.0.0+2025.5.9
- Add random environment truncation for stochasticity in parallel environments
- Add random subsample size for dataloader
- Adjust Adam epsilon
- Change and tune hyperparameters
- Direct indexing change to avoid python looping and excessive indexing, speedup of ~15x (90s to 6s for 2.5M buffer, 30 iterations, 100k epoch, 10k batch)
- Fix bug with improper `pad_indexer` behavior when buffer state 0 is not sampled and add further checks to the related optimization
- Fix improper action normalization for discrete actions
- Prevent processing of small batches
- Remove erroneous comment preventing layer norm for key/values in MHA
- Tune batch size and other hyperparameters

### 1.0.0+2025.5.8
- Add discrete action space
- Add further input sanitization to reduce `fast_sample` indexing time
- Store all buffer memories in numpy arrays for faster indexing

### 1.0.0+2025.5.7
- Add direct indexing to `AdvancedMemoryBuffer`
- All optimizations considered, speeds up memory indexing by >10x and allows for specific indexing, rather than grouped
- Added environment behavior parameters
- Revise `PPO.update` to iterate over all memories (if desired)

### 1.0.0+2025.5.6
- Add extremely rapid sampling, soon to be for uniform samples
- Fix bug with improper caching of noised modalities
- Pre-concatenation of buffer for fast load times in memory

### 1.0.0+2025.5.5
- Add scaling for policy entropy based on output size in combination with magnitude scaling
- More testing and tuning

### 1.0.0+2025.5.4
- Assorted minor policy and environment changes, including reward calculation and increased simulation length
- Much hyperparameter tuning and testing

### 1.0.0+2025.5.3
- Add utility functions
- Change policy output to magnitude-direction format
- Environment reward refinement and scaling
- Hyperparameter tuning

### 1.0.0+2025.5.2
- More testing and tuning
- Readd action reward to default environment and make distance relative
- Smoothing correction for `runtime` notebook

### 1.0.0+2025.5.1
- Fix bug with batch size calculation
- Much hyperparameter tuning
- Revise reward structure and add many alternate configs and stopping criterion
- Simplify model and add alternate output calculation
- Switch environment to spherical bounds - more consistent with reality

### 1.0.0+2025.4.30
- Add flush iterations to config
- Add proper Adam and L2 regularization
- Add s3 upload functionality in data notebook
- Additional record log case handling
- Change distance reward formulation to be proportional to mean origin closeness
- Fix for independent critic
- More hyperparameter tuning

### 1.0.0+2025.4.29
- Add `BufferStandardization` class
- Add `noise_std` parameter to `EnvironmentBase` to add noise to feature observations per episode
- Change default environment weights and handling
- Enhance modularity of `EnvironmentBase` get and set functions
- Extensive hyperparameter tuning
- Move arguments to more intuitive and centralized locations for standardization
- Properly implement PopArt

### 1.0.0+2025.4.28
- Add log probability clipping to avoid `nan` values during training
- Add partial training case to logfile concatenation logic
- Change environment reward scales
- Remove strict cosine similarity from action determination to avoid `nan` values

### 1.0.0+2025.4.27
- Add timestep-based training (as opposed to episode-based) as in traditional PPO (5-6x speedup!)
- Add train/val sampling from dataloader
- Change memory hyperparameters
- Fix various normalization quirks
- Opimize preprocessing sampling for large datasets
- Revise record concatenation strategy

### 1.0.0+2025.4.26
- Add advantage clipping
- Implement and try PopArt
- Synchronized moving standardization for rewards
- Remove unused preprocessing code
- Reorganize policy module

### 1.0.0+2025.4.25
- Add epsilon to cap distance and velocity rewards
- Add explained variance
- Add `norm` option to `euclidean_distance`
- Better return sanitization for `PPO.forward`
- Change environment hyperparameters
- Cleanup
- Figure changes
- New output statistics for policy update
- Normalize environment dist matrix and match initial distribution to velocity
- Optimize policy hyperparameters and test with PCA
- Relative logarithmic velocity penalty
- Shorten environment default `max_timesteps` again

### 1.0.0+2025.4.24
- Decorator
  - Added line-by-line profiling decorator
  - Improved reliability of `metrics` decorator by enforcing CUDA synchronization
- Environment
  - `include_timestep` option for `get_state`
  - A litany of new reward schema in comments
  - Additional return information from `finished`
  - Calculation skipping for distance and origin rewards if unneeded
  - Coefficient changes for all rewards
  - Delta scaling for appropriate rewards
  - Early termination and selectable conditions, including velocity, time without improvement, and boundary
  - Implement additional customizable attributes
  - Inter-cell distance caching
  - Logarithmic distance reward - aiding in model optimization
  - Rename several input parameters
  - Seperate `vel_rand_bound`
- Memory
  - Add bootstrapping capabilities, currently no distinction between truncation and termination, however
  - Add calculation of propagated rewards
  - Bugfix from staleness update
  - Change default hyperparameters
  - Downgrate small sample error to warning
  - Efficient padding and batch calculation minimizing redundancy, compatible with new `Lite` attention model (~10x faster on MERFISH)
  - More returns from `fast_sample`
  - Optimizations to avoid unneeded tensor shuffles and similar
  - Optional memory record buffer to not force GPU synchronization
  - Readd pruning, but default to none
  - Rewards normalization for `compute_advantages`
  - Sample "rounding", i.e. selecting or dropping a sample instead of cutting it when the number of requested nodes is reached
  - Uniform memory sampling technique
- Policy
  - Action computation strategy using cosine similarity with learned embedding
  - Add propagated rewards compatibility, mainly for testing and validation purposes
  - Add shared actor/critic model option with two heads and optional computation for each (2x faster and more reliable)
  - Allow regular (non-self) attention in `ResidualAttention` block
  - Better KL divergence estimation for continuous action space
  - Better checkpoint loading procedures
  - Better synchronization tensor sanitization
  - Change default hidden dimensions for `EntitySelfAttention` architecture (original)
  - Class renaming and parameter changes
  - Clean and generalize `PPO.forward` formatting, including calculation for terminal states (later used in bootstrapping)
  - Different default learning rate decay, comments for linear version as well
  - Gradient clipping (disabled by default)
  - Merge actor and critic into one class
  - Minimize stored input parameters in model classes
  - Move KL targeting to main `update` function
  - Move main computation in model and policy to `forward` for ease of use
  - Moving statistics class for any size data
  - New loss recording strategy for `PPO.update`
  - Optimization runs with many combinations of network architectures
  - Optimize previous parameters
  - Optional frozen updates for critic and policy
  - Optional return standardization (off by default)
  - Orthogonal weight initialization and utility function
  - Sampling size consideration even before `load_level`
  - Separate and cachable feature embeddings, cutting down on computational/time cost when a large number of features are used
  - Separation of actor, critic, and log_std learning rates
  - Switch `action_std` for `log_std`
  - Update returns from `PPO.update`
  - Use Smooth L1 loss instead of MSE for critic loss
  - Use `scale_tril` wherever possible, as covariance matrix is equivalent when diagonal
  - Utility world size function for `PPO`
  - `Dummy` network which only takes positional attributes
  - `EntitySelfAttentionLite` class which avoids concatenation of self embeddings onto node embeddings, allowing for ~(#Nodes)x speedup
  - `EntitySelfAttention` better compatibility with batched input
  - `ResidualAttentionBlock` class for more typical pre-ln residual attention
  - `fit_and_strip` option for `EntitySelfAttentionLite` which flattens sample-node dimensions into batches after computation
- Remote Interface
  - Additional returns for `Worker.update`
  - Bugfix with loaded model not performing benchmark on the first update
  - Feature embedding caching capability for `simulate_until_completion`
  - Fix bug with `load_checkpoint`
  - Fix checkpoint save spacing and activation bugs
  - Optional delayed flush for rollout memory
  - Optional state returns for `simulate_until_completion`
  - Reward is now output as the cumulative (by time) mean (by sample) rewards
  - Smarter memory sending and receiving logic and culling
  - Terminal state critic evaluation for later bootstrapping
  - Updated `dummy` handling
- Processing
  - Avoid unnecessary masking
  - Public function for changing sampling size
  - Safer data handling, at the risk of having some user-unfriendliness deep in the API
  - `Lite`-model specific optimized `split_state` handling, constructing (a) nothing or (b) self, node, and attention mask matrices
  - `sample_and_cast` formatting and argument passing to memory
- Notebooks
  - Analyses for trained model, including gene set perturbation, knockdown, trajectory preview, pinning, and prioritization
  - Begin implementing `Fastplotlib` for GPU-accelerated 3d visualization
  - Better runtime visualizations
- Miscellaneous
  - Benchmarking and manual tuning on numerous policies, reward schema, and architectures
  - Distance explanation and unused moving statistic class
  - Recompile requirements
  - Script revisions
  - Trials with training stages for increasing numbers of nodes
  - `Vectorized` notebook containing a minimal implementation of PPO with vectorized environments, for benchmarking and comparison

### 1.0.0+2025.4.18
- Allow state output for `simulate_until_completion`
- Bugfixes for `simulate_until_completion` argument combinations
- Device policy storage bugfix
- Fix log concatenation

### 1.0.0+2025.4.17
- Add total time to time visualization

### 1.0.0+2025.4.16
- Add and tune automatic KL target weighting
- Add and tune KL targeting
- Add stale record culling and staleness tracking
- Add total loss return from `PPO.update`
- Bugfix for whole episode culling
- Environment parameter renaming
- Extensive model tuning
- Wall and compute time visualizations
- Train script imputation support among many new arguments
- Update parameter and loss visualizations

### 1.0.0+2025.4.15
- Add bool return case to `AdvancedMemoryBuffer`
- Add compatibility for multiple logfiles in runtime analysis
- Add confidence interval and various other enhancements to loss plot
- Add gap capability for runtime plots
- Add option for whole episode culling to `AdvancedMemoryBuffer`
- Add separate lr for `action_std`
- Allow non-remote policy update
- Change default PPO parameters to use KL early stopping rather than clipping
- Correct KL approximation technique
- Extensive tuning
- Fix computation of KL divergence to only consider new samples
- Generalize memory culling code
- Make early-stopped simulation rewards comparable
- Recompute state_vals and advantages for each policy iteration
- Remove zero values from runtime visualization
- Runtime visualization helpers and QOL
- Use L2 regularization

### 1.0.0+2025.4.14
- AWS cluster config optimizations
- Add KL divergence target and rollback to policy for update early stopping
- Add s3-backed AnnData reads
- Adjust default record flush behavior
- Better runtime visualization and processing
- Comment cleanup
- Enhanced capability for `PPO.synchronize` function to act on tensor lists
- Enhanced logging for worker updates
- Find workaround for egregious worker initialization times when timeout hadn't been reached
- More arguments for training script
- Rearrange policy arguments
- Tweak policy hyperparameters

### 1.0.0+2025.4.13
- Minor changes

### 1.0.0+2025.4.10
- Comment cleanup
- Remove errant debugging CLI

### 1.0.0+2025.4.9
- Add Ray Job CLI call for `train` script
- Add checkpoint saving and loading functionality
- Add hooks to `RecordBuffer` and `train_celltrip`
- Add policy iteration parameter for easier saving
- Add separate function for generating initializers
- Change `datetime` import to be more concise
- Change default record flush behavior and frequency
- Fix bug with `merge_files` logic in initializers
- More log parsing functionality
- Move utility functions from `train` to `utility.general`
- Synchronize only gradient where possible, which also keeps optimizers synchronized

### 1.0.0+2025.4.8.4
- Additional log reading functionality
- Fix learners array being empty when `num_exclusive_runners` is 0
- JSON record formatting

### 1.0.0+2025.4.8.3
- Fix bug with deleting empty memory array before instantiation
- Wait for `RecordBuffer` flush before exiting `train_celltrip`

### 1.0.0+2025.4.8.2
- Fix parser requirements

### 1.0.0+2025.4.8.1
- Add log reading notebook `runtime`
- Fix broadcast behavior for head worker policy sync
- Fix rank number calculation log contingency
- Rearrange `train` module
- Relax input file requirements to allow only merge files
- Revise `train_celltrip` native record keeping
- Update VRAM estimates

### 1.0.0+2025.4.8
- Add launcher and s3 policies
- Additional command reference
- Additional records for `train_celltrip`
- AWS IAM, policy, role configuration
- Change s3 cred detection method
- Cluster revisions and testing
- Fix `read_adatas` multiple concurrent read bug
- Ray cluster working on AWS
- Temporary credential generation for s3 testing

### 1.0.0+2025.4.7
- Automatic download/streaming from s3 for `read_adatas`
- Correct AWS credential fallback
- Logfile write testing
- Reconfigure AWS credentials
- Revise AWS cluster configuration

### 1.0.0+2025.4.6
- Add `merge_adatas` utility function and add capability to training script
- Add `RecordBuffer` class for storing records from training
- Add s3 access framework for logfiles
- Begin adding arguments to training script
- Bugfixes for improper on-disk index ordering
- Change node subsample behavior for `obs` in `Preprocessing` to not allow repeats
- Change ray image version for Docker and Ray Cluster
- Flexible Cython usage

### 1.0.0+2025.4.4
- Ray cluster config for AWS
- Remove stages for now

### 1.0.0+2025.4.3
- Change default worker update verbosity
- Clean CLI code and more returns for policy update
- More returns for worker update event log
- Readd full stages to training with argument for `train_celltrip`

### 1.0.0+2025.4.2
- Add clipped value loss
- Add generalized advantage estimation
- Additional debug CLI output for policy update
- Fix bug with head synchronization averaging across all heads
- Fix bug with world size detection multiplying model weights unilaterally
- Make `action_std` a trainable parameter
- Make critic and entropy weights `PPO` initialization arguments
- Remove critic `action_std` update as it now does nothing

### 1.0.0+2025.4.1
- Add advantage normalization
- Add early stopping to `train_celltrip` function
- Add layer normalization to feature embeddings
- Add remote execution macro function to `Worker`
- Adjust replay reward mean to match new upon environment reward update
- Change `Adam` optimizer to `AdamW` with weight decay
- Change init argument order in `train_celltrip` function
- Change policy to use ReLU and pre-ln residual attention
- Clean some environment code
- Convert all networks to have one hidden layer and activation
- Fix `action_std` not updating on critic
- Move `sync_iterations` argument (formerly `sync_epochs`) to `PPO` initialization
- Move layer normalization and activations into policy modules
- Remove final activation function for critic
- Rename `utility.train` to `utility.continual`
- Tuning to avoid critic collapse

### 1.0.0+2025.3.31.2
- Add ready function to worker, indicating the instance is initialized and waiting
- Change environment, policy, memory argument ordering
- Fix race condition with initial sync and rollout

### 1.0.0+2025.3.31.1
- Notebook import cleanup and formatting
- Fix bug with improper collection world size assignment when `num_learners` less than `num_head_workers`
- QOL changes

### 1.0.0+2025.3.31
- General
  - Additional profiling handling
  - Cleanup and file organization
  - Command hints in developer notes
  - Cross-node synchronization commands using `rsync`
  - Gitignore additions
  - Profiling and optimization all around
- Advanced memory buffer
  - Add `clean_keys` function to remove unused keys - useful for large datasets
  - Add `mark_sampled` function to mark all present states as sampled
  - Add `normalize_rewards` function, replacing running normalization for compatibility with distributed implementation
  - Add shuffling to `fast_sample`
  - Aggressive culling optimization and replacement of naïve approaches
  - Attribute computation and retrieval functions for new/replay steps/memories
  - Cache cleanup wherever optimal
  - Change default `prune` value to 0 states
  - Fixes for fp-resistant equivalence testing for key suffix storage
  - Internal function rearrangement for readability
  - Memory appending checks and fixes
  - Mixed new/replay sampling for `fast_sample`
  - New storage values to reduce redundant computations
  - Optimization and storage update for `propagate_rewards`, including avoidance of recalculation
  - Optimization of `_append_suffix` function
  - Optimization of sampling and sampling size - 10,000 appears near-optimal
  - QOL default parameter rearrangement to initialization command
  - Rolling memory buffer with reservoir sampling for replacing replay memories
  - Statistic and variable caching to speed up common calls
  - Storage for precomputed propagated and normalized values, as well as pruned states
  - Various cleanup functions for reduction of memory footprint
  - Warning for large amounts of unique key combinations in the buffer
  - Warnings and errors for inappropriate sampling, pruning, and culling
- Dataloader and Preprocessing
  - Additional public variables for easier compatibility testing
  - Seeding fix for `Preprocessing` when initialized in `PreprocessFromAnnData`
  - Separate seeding for sampling and fitting
- Decorators
  - Add time annotation to `profile` output
  - Bugfix for `profile` naming with variable fnames
  - Extended return format handling for `metrics`
- Distributed training
  - Better handling of memory addition for `train.simulate_until_completion`
  - Changes to `start_node` script, including many compatibility arguments, GPU selection, non-separation of GPUs on the same machine, and reliable timeouts and blocking
  - Entirely rewrite distributed implementation to use RLLib-esque learner-runner architecture
  - Greater than 5-fold reduction in time spent training through optimization and locality
  - Hyperparameter tuning for policy and memory for most efficient training
  - Mandatory local data loading for each node
  - Modular number of learners, learner-runners, and runners
  - Move to synchronous update architecture - pending revision
  - Placement groups to ensure locality to head workers, which synchronize policy weights between nodes and GPUs
- Environment
  - Casting capability
  - Dataloader compatibility
  - Key storage
- Policy
  - Add `update_iterations='auto'` (formerly `epochs`)
  - Add helper functions `synchronize`, `backward`, and `utility.processing.sample_and_cast`
  - Adjust `lr_gamma` from `1` to `.99`
  - Adjust default load level
  - Readability cleanup of forward functions
  - Remove collective initialization from `update` function
  - Rename several initialization arguments for more explicit descriptions
  - Research into entropy function - has no current backward as it is computed on old policy
  - Revise sampling strategy to `pool, epoch, batch, minibatch` with more consistent terminology (formerly `maxbatch, batch, minibatch`)

### 1.0.0+2025.3.21
- Change default caching for `AdvancedMemoryBuffer._append_suffix` back to True
- Change default prune value for `AdvancedMemoryBuffer` to 0
- Memory buffer reservoir sampling for buffer overflow
- Persistent reward propagation
- Proportional replay sampling to combat catastrophic forgetting
- Script cleanup

### 1.0.0+2025.3.20
- Add debugging notes in relevant code locations
- Add fallback return values to `try_catch` decorator
- Change calibration print order to more accurately reflect causality by adding message queue
- Change terminal printing behavior
- Cluster resource update record added
- Fix stall detection in `train_policy`
- Ignore logs
- Inplace division for `allreduce`, fixing gradient explosion with `world_size > 1`
- Match device in `split_state`
- Move caching to init arg for `AdvancedMemoryBuffer` and default to `False`
- Testing runs
- QOL for update event records
- Revise dev notes

### 1.0.0+2025.3.19
- Additional Docker commands
- Additional update timing logic for `train.train_policy` function
- Amend `start_node` behavior to work with multi-GPU nodes
- Decorator function name tagging, for ease of debugging
- Downgrade to python 3.10 for Docker CUDA compatibility
- Figure updates
- File reorganization and cleanup
- Fix extra unnecessary calibration rollout
- Parallelization of update using NCCL
- Replace inline `ray.remote` definitions with decorators
- Script updates
- Upgrade `decorator.metrics` return handling
- Use Rsync for remote repo sync
- Various bugfixes and QOL

### 1.0.0+2025.3.18
- Add command to run attachable remote cluster with no script
- Add step number to `AdvancedMemoryBuffer` class
- Add time logging to `decorator.metrics`
- Clean training loop
- Iterate model by step number in `train_policy`
- Output printing decorator
- Test worker node behavior after disconnection (10 minutes)
- Track policy updates in `policy_manager`

### 1.0.0+2025.3.17
- Add Docker compatibility, running scripts, and image
- Revise package versioning strategy

### 1.0.0+2025.3.7
- Read full Tahoe-100M dataset in `data.ipynb`

### 1.0.0+2025.3.6
- Figure revisions

### 1.0.0+2025.3.5.1
- Additional metric returns
- Change `policy.PPO` defaults
- Map training revisions
- Move training loop to `train` module
- Rearrange functions

### 1.0.0+2025.3.5
- Add `keys` input to rollouts
- Add resource checking for distributed loop and remove cancellations
- CUDA tensor storage bugfix through memory sanitization
- Device choice for `AdvancedMemoryBuffer` class
- Finally fix tuple requirement for keys through smarter persistent storage
- JSON formatting for rollout and update metric returns
- Reupdate AnnData
- Various bugfixes for distributed
- Warning for excessive persistent storage usage

### 1.0.0+2025.3.4
- Add partitioning to distributed loop
- Add repeat padding for `AdvancedMemoryBuffer` when memories do not have equivalent dimensions
- Add reporting to rollout and update functions
- AnnData CSRDataset `0.10.x` compatibility
- Enhanced memory profiling for `decorator.metrics`
- Proper typecasting for `Preprocessing.transform`

### 1.0.0+2025.3.3 (1-2)
- Add `get_transformables` utility function to `PreprocessFromAnnData` for external analyses
- Add `rolled_index` utility function
- Add `subset_features` argument to isolate a feature in `Preprocessing.transform`
- Add `subset_modality` argument to run only one modality in `Preprocessing.transform`
- Amend `num_nodes` argument behavior in `Preprocessing.sample()`
- Argument robustness for `test_adatas`
- Capability for gene knockdown in processed data
- Clean up tests for knockdown in `data.ipynb`
- Compatibility with parted h5ad files concatenated by reference using `anndata.experimental.AnnCollection`
- Fix incorrect loading parameter `backed` assignment in `read_adatas`
- Fix standardization timing in `Preprocessing.fit` to be before PCA computation
- Greatly optimize partition filtering for `Preprocessing` class
- Light file reorganization
- Loader for Tahoe 100M dataset working on machine with 16Gb memory
- Migrate `Preprocessing` completely to `TruncatedSVD` from `PCA`
- Parameter `force_filter` added to `Preprocessing.transform` to avoid not shuffling data according to `filter_mask`
- Readd PCA with centering correction in `Preprocessing.transform(..., subset_features=...)`
- Recompile requirements
- Seed random fit for `PreprocessFromAnnData`
- Separate `read_adatas` into `read_adatas` and `test_adatas` for merging compatibility

### 1.0.0+2025.2.29
- Add QOL warning message to dataloader
- Argument sanitization for reading and dataloader functions
- Improved handling of modalities with varying dimensions for `Preprocessing.transform`

### 1.0.0+2025.2.28
- Add data loading and processing functions
- Dataloader and pipeline for h5ad file loading
- In-memory h5ad conversion for MERFISH data
- Revise `Preprocessing` to be more compatible with AnnData
- Separate recompilation into two functions

### 1.0.0+2025.2.27
- Update figure 1/2 logo and equations

### 1.0.0+2025.2.26
- Add `first` method to `reproducible_strategy`
- Add `max_samples_per_state` argument for `fast_sample` to provide more varied states
- Add handling for attempted release of released locks
- Comply with pip versioning
- Cython compilation for all compatible files
- Fix `idx` input sanitization for `split_state`
- Fix `hash` method reproducibility
- Optimize `__getitem__` for `AdvancedMemoryBuffer`
- Optimize alternate `sample_strategy` options for `split_state`
- Recompilation scripts for C files and requirements
- Recompile requirements
- Refactor of all modules, especially utility
- Remove old files
- Sanitize `keys` input to `DistributedManager`
- Updated Conda development environment
- Various bugfixes and testing for boundary penalty compliance

### 1.0.0+2025-02-25
- Add Cython build file
- Add decorator `catch_on_exit` to hook Ray cancel
- Add fast sampling method to `AdvancedMemoryBuffer`
- Add memory usage decorator
- Add profiling decorator
- Approximate `running_statistics` in `AdvancedMemoryBuffer` for efficiency
- Argument standardization and key filtering for `DistributedManager`
- Change behavior of `is_list_like`
- Disable Ray deduplication
- Dynamic worker allocation based on estimated memory and VRAM usage
- Enhanced policy update CLI
- Fix `_flat_index_to_index` from `AdvancedMemoryBuffer` and add inverse
- Fix `propagate_rewards` for `prune=None`
- Fix deadlocks from canceling
- Heavily optimize `split_state`
- New `decorators` module
- Offload major computations to C backend
- Optimize `propagate_rewards` and fix bottlenecks
- Probably 10-100x speedup in total
- Smart rollout and update

### 1.0.0+2025-02-23
- Add `try_catch` decorator utility to print errors and tracebacks for Ray
- Add storage appending to memories
- Fix `proximity` `split_state` `sample_strategy` to sample from position rather than modalities
- Fix for `split_state` reproducibility and compatibility with `max_nodes` and `idx` arguments, very minor performance hit
- Fix snakeviz hosting arguments

### 1.0.0+2025-02-22
- Add basic training loop to `new_train` notebook
- QOL distributed CLI updates
- Utility class functions and logic

### 1.0.0+2025-02-20
- Improve thread-safety with threading locks

### 1.0.0+2025-02-19
- Additional parameter storage for main model
- Ray implementations and locking `PolicyManager` class
- Remove old models from `PPO` class
- Memory appending for `AdvancedMemoryBuffer`
- Updates and more future handling for `DistributedManager` class

### 1.0.0+2025-02-18
- Add early stopping to environment
- Add multithreaded forward and ray outlines

### 1.0.0+2025-02-17
- Figure updates
- Revise TODO lists

### 1.0.0+2025-02-13
- Fix figure 1/2 crop

### 1.0.0+2025-02-12
- Revise TODO organization

### 1.0.0+2025-02-11
- Additional visualizations for perturbation mean velocity plot
- CLI changes
- Extend perturbation mean velocity plot to top genes by effect size
- Perturbation feature data formatting changes
- Plot reruns
- QOL transformation updates
- Standardize transformation functions
- `LazyComputation` utility class

### 1.0.0+2025-02-10
- Add pinning function
- Add perturbation mean velocity plot for chosen gene

### 1.0.0+2025-02-09
- Add visualization for original input modalities
- Figure revisions
- New CellTRIP logo

### 1.0.0+2025-02-07
- Figure revisions

### 1.0.0+2025-02-06 (1-3)
- Additional warnings and checks for perturbation analysis
- Bugfix for dimensionality reduction on video
- Figure exports
- Figure naming corrections
- Figure updates
- Perturbation comparison figure sorting and additional functionality
- Rerun MERFISH perturbation

### 1.0.0+2025-02-05
- Add absolute imputation method involving pinning and least squares
- Add imputation visualization for all methods and seeds
- Add inverse transform to `other_methods` perturbation
- Add performance metrics for CellTRIP imputation pinning
- Add utility script to convert GIF to consituent frames
- Additional CLI preview in `README`
- Allow gaps in loss plot
- Aspect ratio changes for loss plot
- Clean figures after creation in `analysis`
- Extend perturbation analysis to all features by default
- Figure folder reorganization
- Fix variance feature importance calculation in `other_methods`
- Flowchart, schematic, and MERFISH figure updates
- Highlight `CellTRIP` results for integration and imputation
- Increase default GIF resolution
- Reformat notebook markdown header size usage
- Reruns
- Revise stage step and feature effect size CLI display to be more vertically space-efficient
- Undo loss plot resolution changes

### 1.0.0+2025-02-04
- Figure revisions
- Fix `README` comparison image links
- Increase resolution of loss plots

### 1.0.0+2025-02-03
- Add notebook auto detection for `analysis` and `train` scripts
- Add script functionality to `train` notebook
- Change default arguments for a variety of classes and functions
- Clean up notebook requirements
- Figure revisions
- Legacy compatibility with missing default arguments
- Reorganize arguments in `analysis`

### 1.0.0+2025-01-30
- Add `total_statistics` legacy compatibility to `other_methods` notebook
- Additional CLI output and revisions for `analysis`
- Bugfix for integration comparison
- Complete reruns for MMD-MA, MERFISH, and ISS visualizations
- Figure directory cleanup
- Figure updates
- File name change for loss plot from `_performance.pdf` to `_loss.pdf`
- Reruns for `other_methods` on MMD-MA, MERFISH, and ISS

### 1.0.0+2025-01-29
- Add video generation skip argument to `analysis`
- File exports
- Fix label leak for MLP and KNN comparison methods
- Remove notebook seed from `Variation` method
- Training/validation for other imputation methods

### 1.0.0+2025-01-27
- Add automatic method selection in `other_methods` notebook
- Add CLI output for `other_methods` notebook
- Add imputation comparison visualization
- Add preprocessing to other method runs
- Additional CLI output for integration comparison
- Apply skip before projection for `analysis`
- Change title and name capitalization
- Fix `ViewTemporalDiscrepancy` x-label positioning
- Fix arrow formatting for comparison
- Fix for `jamie_helper` imputing wrong modality
- Multi-run validation capability for alternate methods
- Revise `analysis` comparison plot
- Revise return behavior for `Preprocessing` class
- Summary statistics and visualization for other methods in comparison
- Use lazy loading for comparison plot
- Use WandB run id for other method preprocessing

### 1.0.0+2025-01-24 (1-2)
- Add memory saving/loading for multiple analysis keys in `analysis`
- Automatic overwriting for `convert_video_to_gif` script
- Capability to run multiple analysis keys in the same call for `analysis`
- Change memory save location for analyses
- Correct stage printout for simulation
- Fix CLI `analysis` output indentation (x2)
- Fix in `Preprocessing` std for sparse data with `total_statistics` enabled
- General reruns for MMD-MA, MERFISH, and ISS
- Improve GIF quality
- Move memory readouts after compression and saving/loading
- README additions and revisions
- Script for running other scripts on files of all matching extensions, primarily for `plots` folder
- `TemporalBrain` preliminary video output

### 1.0.0+2025-01-23 (1-3)
- Add UMAP and PCA options for high-dimensional projections in `analysis`
- Bugfix for infinitely scaling discrepancy in `ViewTemporalDiscrepancy`
- Compatibility with reduced states in `View3D`
- Fix incorrect identification of `modal_targets` using `base_env` in analysis
- Fix minor progress bar formatting issue with description not propagating until first completed iteration
- Formatting for `tqdm` in `analysis`, especially for notebooks
- Lower tensor precision inplace for saving memories
- Properly assign number of cell pairings to temporal analysis
- Refresh runs
- Rerender gifs with consistent skip and framerate
- Undo model stage indexing change

### 1.0.0+2025-01-22 (1-3)
- Add `View3D` compatibility with 4+ dimensional state input
- Add alternative name to `MMD-MA` dataset, `MMDMA`
- Add arguments to analysis notebook for command-line running capability
- Add empty `runs` directory handling to `analysis` comparison
- Add progress bars to analysis notebook
- Additional testing functionality in analysis
- Aggressive memory optimization
- Allow memray script to pass arguments to source python script
- Bugfix for handling fp16 while calculating euclidean distance
- Bugfix for `jamie_helper` script in `other_methods`
- Bugfixes for `View3D` with new function and casting structure
- Change header formatting in all major notebooks
- Change stage indexing in `train` to start at 1 and add compatibility filter in `analysis`
- Clean data files from remote repository
- Enhanced CLI stage output while running simulation in `analysis`
- Figure updates
- Fix `.gitignore` for `other_methods` folder and `.h5` files
- Fix `README` image links
- Fix `TemporalBrain` temporal stages in analysis
- Folder creation capability for `train` notebook
- Memray profiler preference revisions
- Modify `analysis` output file names
- Recompile requirements
- Reduce redundancy in `View` classes
- Refactor `View` class shared functions to be more computationally efficient
- Rename `inept` to `cellTRIP` in all cases
- Rename integration to convergence in `analysis` notebook
- Revise CLI output for analysis
- Revise main `README`
- Revise TODO lists in all major notebooks
- Utility `convert_video_to_gif` and `convert_pdf_to_png` scripts added to `plots/`

### 1.0.0+2025-01-21
- Add scripts for enhanced memory profiling
- Additional input checking for model `act_macro`
- Begin adding SHAP to comparison methods
- Bugfix for `other_methods` data formatting with numerical feature names
- Change global header formatting to be consistent with `README`
- Change memory profiling behavior
- Fixes for `partition_distance` with string partition labels
- Fully implement BABEL for imputation
- Implement baseline perturbation comparison method, `variance`
- Move `h5_tree` function to `utilities`
- Profiling for analysis notebook on `TemporalBrain` dataset
- Remove dynamic analysis reliance on env
- Rename profile files
- Updates to `other_methods` README

### 1.0.0+2025-01-20.2
- Compression algorithms for analysis memory storage
- Move additional tensors to CPU for analysis memory
- Revise changelog multi-commit structure (not propagated)

### 1.0.0+2025-01-20.1
- `analysis` notebook reorganization
- Add temporary state saving for analysis, currently uncategorized
- Bugfix for `analysis` `use_modalities` within-epoch processing
- Bugfix for state manager class detection in `analysis` notebook
- Fix dynamic y-scaling for `ViewTemporalDiscrepancy`
- Fix state CLI output during analysis memory generation
- Implement `partition_distance` function for memory saving in cases where only subsets of distance matrices are needed
- More closely emulate intra-modal distance scaling behavior for `ViewModalDistBase` and all child views
- Partition compatibility for `Views`, including much sparse handling

### 1.0.0+2025-01-20
- Add tentative PCA alternative for sparse data to `Preprocessing`
- Fix inverse standardization for sparse data in `Preprocessing`
- Fix mix sparse/dense handling for `Preprocessing` class
- Memory profiling, output, and debugging for sparse and large datasets
- Offload state storage to CPU memory
- Recompile requirements
- Remove unnecessary modality transforming when perturbation is unused (extreme speedup for all PCA datasets, especially `TemporalBrain`)

### 1.0.0+2025-01-17
- Add adata output for other methods notebook
- Add hdf5 output for other methods notebook
- Add methods to other methods notebook, including BABEL
- Fix instance detection for analysis
- Reformat other methods folder, add descriptive `README`

### 1.0.0+2025-01-16
- Add JAMIE to integration and imputation methods
- Add KNN and MLP imputation methods from sklearn
- Fix directory creation for alternate methods

### 1.0.0+2025-01-15
- Add memory optimization option for analysis when present is not full
- Add memory profiling from PyTorch
- Add method comparison figure to analysis with several metrics
- Add other integration methods with standardized formatting and a variety of envs, compatible with any python version
- Adjust velocity threshold for Temporal analysis
- Change bounding behavior for `ViewTemporalDiscrepancy`, adding new parameters
- Dynamically adjust `num_lines` for `ViewLinesBase` based on `analysis_key` in `analysis` notebook
- Fix notebook running script inplace writing
- Fix rewards for analyses with present modifications
- Implement device override to `Preprocessing` `cast` function
- Many reruns
- More descriptive CLI output for analysis
- Move raw data to cpu to aid with GPU memory optimization
- Recompile requirements
- Reorganize analysis notebook static visualizations
- Small annotation in `README` for requirement compilation
- Small bugfix for standard environment class

### 1.0.0+2025-01-14
- Bugfix for temporal analysis using `labels` rather than `times`
- Verbosity updates for analysis

### 1.0.0+2025-01-09
- Add `TemporalBrain` performance visualization
- Figure updates - main figure drafts

### 1.0.0+2025-01-08
- Add loss history plot to `analysis`
- Add unloaded option for `analysis`
- Compressed runs for simulation and spatial data
- Figure updates
- Raise `action_std_min` due to behavior collapse in a few models
- Raise detection threshold for `action_std` due to the appearance of extra stages
- Run corrections for MMD-MA and MERFISH using new parameter `total_statistics` in preprocessing class

### 1.0.0+2025-01-07 (1-3)
- Add labels to perturbation view
- Add present consideration to `ViewPerturbationEffect`
- Additional CLI filtering for analysis
- Adjust copy behavior in `Preprocessing` class
- Adjust formatting for `ViewPerturbationEffect`
- Adjust `ViewPerturbationEffect` pad to properly align feature names
- Adjust syntax of perturbation feature names to match perturbation features
- Data loading fixes and feature format standardization
- Feature filtering according to top variant for `Preprocessing` class
- Fix bug in model loading causing `wgt` initializations in `analysis` to be entirely random
- Fix bug in preprocessing standardization incorrectly taking overall mean and std for dense modalities
- Fix bug in `top_variant` filtering for `Preprocessing` class
- Fix bug with processing sparse data using `top_variant` filter
- Fix perturbation state manager behavior for present modal targets
- Optimize modality modification during analysis main run (3x speedup!)
- Remove mdl file saving from `train` notebook
- Switch knockdown method to mean reversion

### 1.0.0+2025-01-06
- Add perturbation state manager class
- Add perturbation visualization class
- Amend behavior of `run_notebook` script for bash arguments
- Fix initial stage of temporal analysis being labeled `-1`
- Fix lingering stage at end of main run for analysis
- Implement dynamic thresholding for perturbation state manager

### 1.0.0+2025-01-05
- Finish state manager implementation
- Fix bug for state manager not cloning tensors, thereby improperly updating `present`
- Replace `get_present` function with state manager classes
- Start framework for perturbation analysis

### 1.0.0+2024-12-30
- Add framework for state manager in analysis runs

### 1.0.0+2024-12-29
- Change location of `present` modifications to match stages exactly in analysis

### 1.0.0+2024-12-27
- Add torch determinism function
- Fixes for diffusion randomization
- More intuitive indices for `ViewLinesBase` visualization class
- Opacity scaling based on actual distance for `ViewLinesBase` derivatives
- Remove outline from points all views
- Reruns

### 1.0.0+2024-12-23
- Add sample strategies `proximity` and `random-proximity`
- Add reproducibility for `split_state` sampling with multiple strategies
  - `hash` and set seed methods
- Additional CLI information for partitioned training

### 1.0.0+2024-12-18
- Change default argument for y-limit in `ViewTemporalDiscrepancy`
- Reruns for various datasets

### 1.0.0+2024-12-17
- Additional runs
- Additional visualizations

### 1.0.0+2024-12-16
- Add explicit `integration` option for analysis
- Add y-limitation option for `ViewTemporalDiscrepancy` class
- Compatibility with custom environment weight stages
- Fix bug with `propagate_rewards` function in `AdvancedMemoryBuffer` with Tuple handling
- Integration runs
- Remove `None` options for `discovery` and `temporal`, as they would then default to `integration` in function
- Reorganize analysis notebook to aggregate run arguments and make names more clear
- Reruns for various analyses

### 1.0.0+2024-12-15 (1-3)
- Add `sample_strategy` argument to `PPO` class
- Add coloration to temporal scatter
- Add customizable alpha limits to `ViewLinesBase` and tweak opacity in `ViewTemporalScatter`
- Add labels to applicable animated plots
- Add legends to temporal scatter
- Additional runs
- Change tick formatting for `ViewTemporalDiscrepancies`
- Delete and revise old scripts
- Fix inversed color for temporal scatter plot
- Implement plot scaling/cutoff approach for `ViewTemporalScatter`
- Include multimodal compatibility with temporal scatter
- Optimize `get_distance_discrepancy` function to run ~20x faster, main bottleneck of animation generation with many cells
- Rearrange and condense analysis figures

### 1.0.0+2024-12-14
- Add scatter visualization for temporal discrepancies
- Add `max_batch` override for visualizations
- Compatibility revisions for `View3D` on modalities of differing sizes
- Implement `ViewTemporalBase` class for scatter distance visualization
- Re-remove auxiliary vars at beginning of analysis notebook
- Rearrange temporal figure

### 1.0.0+2024-12-13
- Add `[None, ...]` compatibility for `pca_dim` and `top_variant` arguments within `Preprocessing` class
- Refactor analysis plot code with base classes and consistent structure
- Separate analysis plots into individual classes
- Standardize analysis notebook for a variety of plot types

### 1.0.0+2024-12-11
- Add CLI output functionality to `run_notebook` script
- Addition of several performance arguments to analysis notebook
- Clean up several unnecessary segments in analysis and training notebooks
- Formatting changes for main training notebook
- Run analysis for MERFISH data with full, unfiltered episodes and 100 max nodes
- Update of analysis notebook for new arguments

### 1.0.0+2024-12-10
- Recompile requirements
- Run partial temporal data

### 1.0.0+2024-12-09
- Fix various bugs with ragged lists pertaining to environments of varying size within `AdvancedMemoryBuffer` class

### 1.0.0+2024-12-08
- Add type partitioning to preprocessing utility
- Clean up preprocessing implementation
- Fix handling of different-sized environments in memory module and training loop
- Implement sparse PCA and standardization handling for preprocessing class

### 1.0.0+2024-12-07
- Compatibility changes for `analysis` notebook
- Reformat `types` output from `data` module to allow for multiple outputs
- Update requirements

### 1.0.0+2024-12-06
- Add temporal brain dataset
- Annotate temporal brain data
- Clean training notebook
- Fix RDS reading techniques

### 1.0.0+2024-12-04
- Add various strategies to `split_state` node sampling
- Additional runs for spatial data with new random node sampling
- Fix standardization bug when std equals zero
- Make forward node sampling compatible with backwards computation

### 1.0.0+2024-11-26
- Add frame skipping to analysis output
- Add time-dependent analysis and discrepancy evaluation
- Begin implementing new dataset
- Generalize deployment and temporal configuration(s)
- Many additional runs
- Scripts for running many analyses at a time
- Standardize running format for analysis notebook

### 1.0.0+2024-11-20
- Extra default arguments

### 1.0.0+2024-11-18
- Add default `num_nodes` to analysis notebook
- Many additional runs

### 1.0.0+2024-11-15
- Additional runs and tweaked hyperparameters
- Updated bugfix for episode random node sampling

### 1.0.0+2024-11-13 (1-3)
- Bugfix for disabled episode random node sampling
- Bugfixes and generality improvements for dataloading
- Datasets for `ExSeq`, `MERFISH`, and `smFISH`
- Figure and data reorganization
- New figures
- Runs for all spatial datasets aside from `BARISTASeq`

### 1.0.0+2024-11-07
- Additional runs
- Bugfix for unmatched dimensions in env and data when using PCA
- New `ISS` spatial data

### 1.0.0+2024-11-06
- Flowchart updates

### 1.0.0+2024-11-05
- Figure updates

### 1.0.0+2024-10-31
- Add random node generation to analysis
- Additional runs for spatial and scGLUE data
- Additional state setting functionality for environment class
- Figure changes including model flowchart
- File cleanup
- Improve movement detection and false early stopping for integration analysis
- Name change
- New analyses for spatial data
- New visualizations

### 1.0.0+2024-10-23
- More runs on spatial data
- Naming and file structure changes for plot outputs
- QOL for model arguments and utility functions

### 1.0.0+2024-10-22 (1-2)
- Add camera rotation during visualization
- Add stage detection and loading for analysis
- Argument naming consistency for policy class
- Fix incorrect coloration for reward preview lines in analysis
- Fix zorder configuration to better emulate 3D
- README update

### 1.0.0+2024-10-21
- Add inverse to preprocessing methods
- Bugfix for non-sequential or non-list `keys` inputs causing slowdowns
- Clean file storage
- Enhance and generalize running/profiling scripts
- Increase reliability of environment modality changes
- Many additional runs
- Randomize nodes in each training episode
- README update
- Replace `modify_data` function with `Preprocessing` class
- Runtime profiling and optimization

### 1.0.0+2024-10-20
- Add silhouette coefficient per cell type
- Add simulation optimal length detection
- Additional runs
- More TODOs for training algorithm

### 1.0.0+2024-10-19
- Add legends to visualization
- Add modal-relationship lines to new animation
- Add velocity arrows to new animation
- Change environment return argument key
- New runs
- Reformat visualization

### 1.0.0+2024-10-18
- Add data loading module
- Change environment defaults
- Change normalize to standardize for accuracy
- Change plot file naming scheme
- Clean formatting in training notebook(s)
- Cleaner utility functions
- Data adjustment utility functions
- Moved temp files to OS temp directory
- New animation notebook
- Remove unnecessary/unchanged arguments from parameters

### 1.0.0+2024-10-17
- Additional arguments for normalization/standardization
- Adjust stages to not have penalty explosion
- Apply standardization to running datasets
- Bugfix for default env weights
- Bugfix for reward calculation flipped sign
- Concurrent runs
- Fix standardization issue

### 1.0.0+2024-10-16
- Add visual cortex dataset for spatial data
- Rerun requirements
- Revise `trajectory` environment to support imputation

### 1.0.0+2024-09-30
- Poster thumbnail updates

### 1.0.0+2024-09-27 (1-4)
- Analysis tuning and exploration
- File structure cleanup
- Final poster visualizations and layout
- More analysis configurability
- New thumbnail/logo
- Poster updates
- Results for BrainChromatin dataset
- Results for scGLUE dataset

### 1.0.0+2024-09-26
- Add legend to analysis
- Better analysis step configurability
- Enhance CLI reporting for analysis progress
- Fix dependencies
- Fix trajectory centering behavior
- Poster updates
- Various bugfixes

### 1.0.0+2024-09-25
- Add `copy` argument to PCA utility function for memory saving
- Add scGLUE dataset
- Increast leniency of early stopping, fixing non-convergence for high dimensionality
- Poster updates
- Reduce required user input for analysis

### 1.0.0+2024-09-24
- Additional runs
- Figure revisions
- Many additions to analysis functionality, including several arguments and tuning
- Poster revisions

### 1.0.0+2024-09-23
- Change `max_nodes` argument to be more intuitive

### 1.0.0+2024-09-20
- Additional runs
- Enhanced plotting functionality for analyses
- Script organization

### 1.0.0+2024-09-18 (1-3)
- Additional runs
- Major figure revisions
- Poster revisions

### 1.0.0+2024-09-17
- Additional runs
- Figure changes

### 1.0.0+2024-09-15
- Substantial figure changes
- Tuning and additional runs

### 1.0.0+2024-09-13
- Additional runs
- Animation optimizations
- Animation UMAP implementation
- Clean code and sanitize user arguments
- Optimize batch loading and casting across all configurations
- Parameter tweaks
- Processing fixes

### 1.0.0+2024-09-11
- Additional runs
- Figure updates

### 1.0.0+2024-09-10
- Additional runs
- Figure updates

### 1.0.0+2024-09-04
- Add poster
- Figure changes
- Runs for scNMT and developing brain datasets

### 1.0.0+2024-08-28
- Figure changes

### 1.0.0+2024-08-21
- Figure changes

### 1.0.0+2024-08-07
- Figure folder reorganization
- Major figure updates

### 1.0.0+2024-07-18 (1-2)
- Figure updates
- Requirement recompile for profilers

### 1.0.0+2024-07-16
- Add `scNMT` dataset
- Additional reporting for trajectory analysis
- Customize `clean_return` behavior
- Extend data loading capabilities to any number of modalities
- Modify feature perturbation analysis to randomize more features
- Modify feature perturbation analysis to start from steady state
- Remove pruned rewards from running statistics calculation
- Use all features with PCA for `BrainChromatin` dataset

### 1.0.0+2024-07-15
- Add batch acting and sampling
- Add memory profiling script and instructions
- Add option to disable smart memory loading for backward subsampling
- Add pruning to training states based on number of future states
- Add running statistics to rewards
- Extensive testing and bugfixes
- Further time optimizations
- Move reward normalization inside memory class
- Sanitize hyperparameters

### 1.0.0+2024-07-12
- Fix memory optimization with sliced tensors remaining unsliced in memory
- State subsampling implemented for forward computation

### 1.0.0+2024-06-25
- Intense memory optimizations, mainly with policy update
- Optional `Sampler` class for varied memory and GPU management
- Timing optimizations, mainly with policy update

### 1.0.0+2024-06-19
- Figure updates

### 1.0.0+2024-06-12
- Figure and schematic updates

### 1.0.0+2024-06-05
- Figure and schematic updates

### 1.0.0+2024-05-30 (1-2)
- Add `average` early stopping method with sliding window
- Add batch calculation to `act_macro`
- Add checkpoint model saves
- Add function comments for ease of use
- Add safety check to environment step for velocity shape
- Add safety check to model forward for number of modalities
- Add stage logging to wandb
- Add stages to training
- Add verbosity to animation creation
- Change default weight scaling for environments
- Figure updates and add schematic
- Fix activations for feature embedding
- Fix `finish` variable not being properly recorded
- Fix memory treating state vectors as `modal-pos-vel` rather than `pos-vel-modal`
- Fix palette for `analysis`
- Full runs for simulation data
- Many small tweaks and optimizations
- Plotting style revisions
- Re-reverse reward sublists for `AdvancedMemoryBuffer`
- Readd origin reward
- Runs for developing brain data
- Tune parameters

### 1.0.0+2024-05-22
- Add caching for static prefix re-appending
- Add environment argument for position randomization range
- Add key argument in forward which is required for memories
- Add normalization for euclidean distance calculation
- Add static prefix for memory based on key
- Figure updates
- Ran full MMD-MA data
- Reduce redundancy in state storage (also added required macro function in `PPO`) and add indexing

### 1.0.0+2024-05-15
- Figure updates
- More runs on real data
- Utility functions

### 1.0.0+2024-05-08
- Start performing memorybuffer optimizations
- Runs on real data

### 1.0.0+2024-04-17.2
- Change formatting for trajectory analysis

### 1.0.0+2024-04-17.1
- Change animations to include first environment state

### 1.0.0+2024-04-17
- Add feature randomization analysis
- Add trajectory analysis

### 1.0.0+2024-04-14.1
- Further generalization for MMD-MA

### 1.0.0+2024-04-14
- Add metadata coloring to animation
- Add more environment hyperparameters
- Add origin penalty to trajectory environment
- Add itemized reward logging
- Change animation export format to mp4
- Logging changes
- Optimize animation generation
- Optimize memory usage in policy update
- Runs on MMD-MA data
- Runs on new hyperparameters

### 1.0.0+2024-04-11.1
- Add file integration for weights and biases
- Recompile requirements
- Revise changelog headers
- Separate analysis and animation into new notebook
- Several tuning runs

### 1.0.0+2024-04-11
- Add early decay and early stopping
- Add GPU compatibility
- Add gradient accumulation to policy update
- Add Monte Carlo sampling to policy update
- Add scheduler for actor and critic lr
- Add timer utility
- Add weights and biases compatibility
- Basic parameter searching
- Cholesky and sampling optimizations, resulting in 5x speedup for action selection
- Fix grad applications in state computation
- Optimize GPU utilization and fix increasing memory usage
- Recalculate environments
- Run on larger data
- Various bugfixes
- Various optimizations

### 1.0.0+2024-02-20
- Figure 1 updates

### 1.0.0+2024-02-14
- Figure 1 initial version

### 1.0.0+2024-02-11
- Add real data implementation

### 1.0.0+2024-02-08
- Environment optimizations
- Monte carlo memory sampling
- Testing on limited real data

### 1.0.0+2023-12-19
- Add basic saving and loading
- Add layer normalization
- README updates
- Remove `selfish` debugging tool
- Tweak hyperparameters

### 1.0.0+2023-12-18
- Additional visualizations and statistics
- Fix several reward bugs
- Hyperparameter tuning
- More advanced rewards
- Utility module
- Working distance-based alignment

### 1.0.0+2023-12-17
- Bugfix for attentions across batches
- Implement residual self attention
- Integrate reward calculation into environment
- Several new rewards for environment
- Successful runs with large modalities and complex objective
- Training progress visualizations

### 1.0.0+2023-12-16
- Add PPO
- Add sample to README
- Add test reward function
- Add update function
- Add variance decay
- Environment updates
- Model updates
- Standardize training layout
- Working simulation

### 1.0.0+2023-12-11
- Add animations
- Add `trajectory` environment
- Develop PPO runtime flow
- Implement memory
- Implement modality embeddings
- Revise argument structure

### 1.0.0+2023-11-04 (1-2)
- Implement centralized policy
- Self attention and embedding framework
