# Cell Trajectory Recovery, Imputation, and Perturbation to Uncover Temporal and Spatial Dynamics using Multi-Agent Reinforcement Learning (cellTRIP)

Recent techniques enable functional characterization of single-cells, which allows for the study of cellular and molecular mechanisms in complex biological processes, including cell development. Many methods have been developed to utilize single-cell datasets to reveal cell developmental trajectories, such as dimensionality reduction and pseduotime. However, these methods generally produce static data snapshots, challenging a deeper understanding of the mechanistic dynamics underlying cell development. To address this, we have developed cellTRIP, a multi-agent reinforcement learning model to recapitulate the dynamic progression and interaction of cells during development or progression. cellTRIP takes single or bulk cell data, single or multimodality, and trains a collaborative reinforcement learning model that governs the dynamic interactions between cells that drive development. In particular, it models single cells as individual agents that coordinate progression in a latent space by interacting with neighboring cells. The trained model can further prioritize and impute cellular features and in-silico predict the dependencies of cell development from feature perturbations (e.g., gene knockdown). We apply cellTRIP to both simulation and real-word single-cell multiomics datasets including brain development and spatial-omics, revealing potential novel mechanistic insights on gene expression and regulation in complex developmental processes.

## Why use cellTRIP?

cellTRIP confers several unique advantages over comparable methods:
- Take any number or type of modalities as input
- Variable cell input quantity,
- Interactive latent space,
- Great generalizability.

| Application | Preview | Description | Performance |
| --- | --- | --- | --- |
| Multimodal integration | <img src="./plots/MMD-MA_convergence.gif" width="400"> | Policy trained on 300 simulated single-cells with multimodal input applied to an integration environment | <img src="./plots/MMD-MA_comparison.png" width="200"> |
| Cross-modal imputation | <img src="./plots/MERFISH_convergence.gif" width="400"> | Imputation of spatial data from gene expression, simulated until convergence using a cellTRIP trained model | <img src="./plots/MMD-MA_comparison.png" width="200"> |
| Development simulation | <img src="./plots/MERFISH_discovery.gif" width="400"> | Cell differentiation simulation on single-cell brain data, with cellTRIP agents controlling cell movement in the previously unseed environment |  |
| Trajectory recovery, inference, and prediction | <img src="./plots/TemporalBrain_temporal.gif" width="400"> | Trajectory estimation on single-cell multimodal human brain data across several age groups | <img src="./plots/MMD-MA_comparison.png" width="200"> |
| Perturbation analysis | <img src="./plots/ISS_perturbation.gif" width="400"> | Estimated effect size calculation of randomly selected genes from a cellTRIP imputation model on spatial data | <img src="./plots/MMD-MA_comparison.png" width="200"> |


## Installation Instructions (~7 minutes)

To install cellTRIP, first clone and navigate to the repository,

```bash
git clone https://github.com/Oafish1/cellTRIP
cd cellTRIP

# cellTRIP may also be installed directly from GitHub without cloning, but
# does not have version controlled dependencies, and is therefore not
# recommended
pip install celltrip@git+https://git@github.com/Oafish1/cellTRIP
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
<!-- python -m piptools compile -->


## Usage

cellTRIP can be used either from the command line or as a python package.

TODO
