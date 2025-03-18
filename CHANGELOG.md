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
