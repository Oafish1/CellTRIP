# CellTRIP, a Multi-Agent Reinforcement Learning Approach for Cell Trajectory Recovery, Cross-Modal Imputation, and Perturbation in Time and Space

Recent techniques enable functional characterization of single-cells, which allows for the study of cellular and molecular mechanisms in complex biological processes, including cell development. Many methods have been developed to utilize single-cell datasets to reveal cell developmental trajectories, such as dimensionality reduction and pseduotime. However, these methods generally produce static data snapshots, challenging a deeper understanding of the mechanistic dynamics underlying cell development. To address this, we have developed CellTRIP, a multi-agent reinforcement learning model to recapitulate the dynamic progression and interaction of cells during development or progression. CellTRIP takes single or bulk cell data, single or multimodality, and trains a collaborative reinforcement learning model that governs the dynamic interactions between cells that drive development. In particular, it models single cells as individual agents that coordinate progression in a latent space by interacting with neighboring cells. The trained model can further prioritize and impute cellular features and in-silico predict the dependencies of cell development from feature perturbations (e.g., gene knockdown). We apply CellTRIP to both simulation and real-word single-cell multiomics datasets including brain development and spatial-omics, revealing potential novel mechanistic insights on gene expression and regulation in complex developmental processes.

## Why use CellTRIP?

CellTRIP confers several unique advantages over comparable methods:
- Take any number or type of modalities as input
- Variable cell input quantity,
- Interactive latent space,
- Great generalizability.

| Application | Preview | Description | Performance |
| --- | --- | --- | --- |
| Multimodal integration | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/rypltvk5_MMD-MA_convergence.gif" width="300"> | Policy trained on 300 simulated single-cells with multimodal input applied to an integration environment | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/rypltvk5_MMD-MA_comparison_integration.png" width="200"> |
| Cross-modal imputation | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/32jqyk54_MERFISH_convergence.gif" width="300"> | Imputation of spatial data from gene expression, simulated until convergence using a CellTRIP trained model | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/32jqyk54_MERFISH_comparison_imputation.png" width="200"> |
| Development simulation | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/32jqyk54_MERFISH_discovery.gif" width="300"> | Cell differentiation simulation on single-cell brain data, with CellTRIP agents controlling cell movement in the previously unseed environment |  |
| Trajectory recovery, inference, and prediction | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/brf6n6sn_TemporalBrain_temporal.gif" width="300"> | Trajectory estimation on single-cell multimodal human brain data across several age groups | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/rypltvk5_MMD-MA_comparison_integration.png" width="200"> |
| Perturbation analysis | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/c8zsunc9_ISS_perturbation.gif" width="300"> | Estimated effect size calculation of randomly selected genes from a CellTRIP imputation model on spatial data | <img src="https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots/c8zsunc9_ISS_comparison_imputation.png" width="200"> |


## Installation Instructions (~7 minutes)

To install CellTRIP, first clone and navigate to the repository,

```bash
git clone https://github.com/Oafish1/CellTRIP
cd CellTRIP

# CellTRIP may also be installed directly from GitHub without cloning, but does not have version controlled dependencies, and is therefore not recommended
pip install celltrip@git+https://git@github.com/Oafish1/CellTRIP
```

Create and activate a `conda` virtual environment using Python 3.10,

```bash
conda create -n celltrip python=3.10
conda activate celltrip
```

Install dependencies using `pip`

```bash
# Development install (Recommended)
pip install -r requirements-dev.txt
# For full development capabilities, also install ffmpeg and poppler-utils
sudo apt-get install ffmpeg poppler-utils

# Base install
pip install -r requirements.txt
pip install -e .
```


## Usage

CellTRIP can be used either from the command line,

```bash
python train.py ...
# usage: train.py [-h] [--seed SEED] [--gpu GPU] --dataset DATASET
#                 [--no_standardize] [--top_variant [TOP_VARIANT ...]]
#                 [--pca_dim [PCA_DIM ...]] [--num_nodes [NUM_NODES ...]]
#                 [--dim DIM]
#                 [--reward_distance_target [REWARD_DISTANCE_TARGET ...]]
#                 [--env_stages [ENV_STAGES ...]] [--max_nodes MAX_NODES]
#                 [--sample_strategy {random,proximity,random-proximity}]
#                 [--reproducible_strategy REPRODUCIBLE_STRATEGY]
#                 [--update_maxbatch UPDATE_MAXBATCH]
#                 [--update_batch UPDATE_BATCH]
#                 [--update_minibatch UPDATE_MINIBATCH]
#                 [--update_load_level {maxbatch,batch,minibatch}]
#                 [--update_cast_level {maxbatch,batch,minibatch}]
#                 [--feature_embed_dim FEATURE_EMBED_DIM]
#                 [--embed_dim EMBED_DIM] [--action_std_init ACTION_STD_INIT]
#                 [--action_std_decay ACTION_STD_DECAY]
#                 [--action_std_min ACTION_STD_MIN]
#                 [--memory_prune MEMORY_PRUNE]
#                 [--max_ep_timesteps MAX_EP_TIMESTEPS]
#                 [--max_timesteps MAX_TIMESTEPS]
#                 [--update_timesteps UPDATE_TIMESTEPS] [--max_batch MAX_BATCH]
#                 [--no_episode_random_samples]
#                 [--episode_partitioning_feature EPISODE_PARTITIONING_FEATURE]
#                 [--use_wandb] [--buffer BUFFER] [--window_size WINDOW_SIZE]

# Train CellTRIP model

# options:
#   -h, --help            show this help message and exit

# General:
#   --seed SEED           **Seed for random calls during training (default: 42)
#   --gpu GPU             **GPU to use for computation (default: 0)

# Data:
#   --dataset DATASET     Dataset to use (default: None)
#   --no_standardize      Don't standardize data (default: False)
#   --top_variant [TOP_VARIANT ...]
#                         Top variant features to filter for each modality
#                         (default: None)
#   --pca_dim [PCA_DIM ...]
#                         PCA features to generate for each modality (default:
#                         [512, 512])
#   --num_nodes [NUM_NODES ...]
#                         Nodes to sample from data for each episode (default:
#                         None)

# Environment:
#   --dim DIM             CellTRIP output latent space dimension (default: 16)
#   --reward_distance_target [REWARD_DISTANCE_TARGET ...]
#                         Target modalities for imputation, leave empty for
#                         imputation (default: None)
#   --env_stages [ENV_STAGES ...]
#                         Environment stages for training, as input, integers
#                         are expected with each binary digit acting as a flag
#                         for `penalty_bound`, `penalty_velocity`,
#                         `penalty_action`, `reward_distance`, `reward_origin`,
#                         respectively (default: [1, 17, 23, 15])

# Policy:
#   --max_nodes MAX_NODES
#                         **Max number of nodes to include in a single
#                         computation (i.e. 100 = 1 self node, 99 neighbor
#                         nodes) (default: None)
#   --sample_strategy {random,proximity,random-proximity}
#                         Neighbor sampling strategy to use if `max_nodes` is
#                         fewer than in state (default: random-proximity)
#   --reproducible_strategy REPRODUCIBLE_STRATEGY
#                         Method to enforce reproducible sampling between
#                         forward and backward, may be `hash` or int (default:
#                         hash)
#   --update_maxbatch UPDATE_MAXBATCH
#                         **Total number of memories to sample from during
#                         backprop (default: None)
#   --update_batch UPDATE_BATCH
#                         **Number of memories to sample from during each
#                         backprop epoch (default: 10000)
#   --update_minibatch UPDATE_MINIBATCH
#                         **Max memories to backprop at a time (default: 10000)
#   --update_load_level {maxbatch,batch,minibatch}
#                         **What stage to reconstruct memories from compressed
#                         form (default: minibatch)
#   --update_cast_level {maxbatch,batch,minibatch}
#                         **What stage to cast to GPU memory (default:
#                         minibatch)
#   --feature_embed_dim FEATURE_EMBED_DIM
#                         Dimension of modal embedding (default: 32)
#   --embed_dim EMBED_DIM
#                         Internal dimension of state representation (default:
#                         64)
#   --action_std_init ACTION_STD_INIT
#                         Initial policy randomness, in std (default: 0.6)
#   --action_std_decay ACTION_STD_DECAY
#                         Policy randomness decrease per stage iteration
#                         (default: 0.05)
#   --action_std_min ACTION_STD_MIN
#                         Final policy randomness (default: 0.15)
#   --memory_prune MEMORY_PRUNE
#                         How many memories to prune from the end of the data
#                         (default: 100)

# Training:
#   --max_ep_timesteps MAX_EP_TIMESTEPS
#                         Number of timesteps per episode (default: 1000)
#   --max_timesteps MAX_TIMESTEPS
#                         Absolute max timesteps (default: 5000000)
#   --update_timesteps UPDATE_TIMESTEPS
#                         Number of timesteps per policy update (default: 5000)
#   --max_batch MAX_BATCH
#                         **Max number of nodes to calculate actions for at a
#                         time (default: None)
#   --no_episode_random_samples
#                         Don't refresh episode each epoch (default: False)
#   --episode_partitioning_feature EPISODE_PARTITIONING_FEATURE
#                         Type feature to partition by for episode random
#                         samples (default: None)
#   --use_wandb           **Record performance to wandb (default: False)

# Early Stopping:
#   --buffer BUFFER       Leniency for early stopping criterion, in updates
#                         (default: 6)
#   --window_size WINDOW_SIZE
#                         Window size for early stopping criterion evaluation,
#                         in updates (default: 3)
```

or as a python package,

```python
# TODO
```

After training the model, analysis can be performed using the `examples/analysis.py` CLI interface,

```bash
python analysis.py ...
# usage: analysis.py [-h] [-S SEED] [--gpu GPU] [-b MAX_BATCH] [--num NUM_NODES] [--nodes NUM_NEIGHBORS] [--stage STAGE] [--discovery_key DISCOVERY_KEY] [--temporal_key TEMPORAL_KEY] [--force] [--novid] [-g] [-s SKIP]
#                    [--reduction {umap,pca,none}] [--force_reduction] [--reduction_batch REDUCTION_BATCH] [--total_statistics]
#                    run_id {convergence,discovery,temporal,perturbation} [{convergence,discovery,temporal,perturbation} ...]

# Create a video of the specified CellTRIP model

# options:
#   -h, --help            show this help message and exit

# General:
#   run_id                Run ID from WandB to use for processing
#   {convergence,discovery,temporal,perturbation}
#                         Type of analyses to perform (one or more)
#   -S SEED, --seed SEED  Override simulation seed
#   --gpu GPU             GPU(s) to use

# Simulation:
#   -b MAX_BATCH, --batch MAX_BATCH
#                         Override number of nodes which can calculate actions simultaneously
#   --num NUM_NODES       Override number of nodes to take from data
#   --nodes NUM_NEIGHBORS 
#                         Override neighbors considered by each node
#   --stage STAGE         Override model stage to use. 0 is random initialization

# Analysis:
#   --discovery_key DISCOVERY_KEY
#                         Type of discovery analysis (0: Auto)
#   --temporal_key TEMPORAL_KEY
#                         Type of temporal analysis (0: Auto, 1: TemporalBrain)
#   --force               Rerun analysis even if already stored in memory

# Video:
#   --novid               Skip video generation
#   -g, --gif             Output as a GIF rather than MP4
#   -s SKIP, --skip SKIP  Number of steps to advance each frame
#   --reduction {umap,pca,none}
#                         Reduction type to use for high-dimensional projections in 3D visualization
#   --force_reduction     Force reduction, even if unnecessary
#   --reduction_batch REDUCTION_BATCH
#                         Max number of states to reduce in one computation

# Legacy Compatiiblity:
#   --total_statistics    Compatibility argument to compute mean and variance over all samples
```

this script will generate...


<!-- DEV NOTES -->
<!-- Compiling requirements: python -m piptools compile  # Also, pip freeze will always be one git commit behind -->
<!-- Find https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots, Replace ./plots to test images -->
