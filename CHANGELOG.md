### 1.0.0+2025-01-07
- Add labels to perturbation view
- Adjust copy behavior in `Preprocessing` class
- Adjust formatting for `ViewPerturbationEffect`
- Data loading fixes and feature format standardization
- Feature filtering according to top variant for `Preprocessing` class
- Optimize modality modification during analysis main run (3x speedup!)
- Remove mdl file saving from `train` notebook

### 1.0.0+2025-01-06
- Add perturbation state manager class
- Add perturbation visualization class
- Amend behavior of `run_notebook` script for bash arguments
- Fix initial stage of temporal analysis being labeled `-1`
- Fix lingering stage at end of main run for analysis
- Implement dynamic thresholding for perturbation state manager

### 1.0.0+2025-01-05
- Finish state manager impelementation
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
