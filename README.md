# CellTRIP, a Multi-Agent Reinforcement Learning Approach for Cell Trajectory Recovery, Cross-Modal Imputation, and Perturbation in Time and Space

Recent techniques enable functional characterization of single-cells, which allows for the study of cellular and molecular mechanisms in complex biological processes, including cell development. Many methods have been developed to utilize single-cell datasets to reveal cell developmental trajectories, such as dimensionality reduction and pseduotime. However, these methods generally produce static data snapshots, challenging a deeper understanding of the mechanistic dynamics underlying cell development. To address this, we have developed CellTRIP, a multi-agent reinforcement learning model to recapitulate the dynamic progression and interaction of cells during development or progression. CellTRIP takes single or bulk cell data, single or multimodality, and trains a collaborative reinforcement learning model that governs the dynamic interactions between cells that drive development. In particular, it models single cells as individual agents that coordinate progression in a latent space by interacting with neighboring cells. The trained model can further prioritize and impute cellular features and in-silico predict the dependencies of cell development from feature perturbations (e.g., gene knockdown). We apply CellTRIP to both simulation and real-word single-cell multiomics datasets including brain development and spatial-omics, revealing potential novel mechanistic insights on gene expression and regulation in complex developmental processes.

## Why use CellTRIP?

CellTRIP confers several unique advantages over comparable methods:
- Take any number or type of modalities as input
- Variable cell input quantity,
- Interactive latent space,
- Great generalizability.

<!-- Process-wise, CellTRIP is also node-failure resistant -->

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
# For full development capabilities, also install ffmpeg, poppler-utils, boto3, cupy, docker, and poppler-utils
pip install -r requirements.txt
pip install -e .

# Base install
pip install -e .
```


## Usage

CellTRIP can be used either from the command line,

```bash
python train.py ...
# TODO
```

or as a python package,

```python
# TODO
```

After training the model, analysis can be performed using the `examples/analysis.ipynb` notebook,

```bash
python analysis.py ...
# TODO
```

this script will generate...


<!--
DEV NOTES
Getting h5py


%%bash
git clone https://github.com/HDFGroup/hdf5.git build/hdf5
mkdir -p build/hdf5/build
cd build/hdf5/build
cmake -DCMAKE_INSTALL_PREFIX=/home/vscode/.local -DHDF5_ENABLE_ROS3_VFD=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=ON -DHDF5_BUILD_UTILS=OFF ../ 2>&1 > /dev/null
make -j 4 2>&1 > /dev/null
make install 2>&1 > /dev/null
cd ../../..

sudo apt update
sudo apt install build-essential git cmake libcurl4-openssl-dev
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5_1_14_3
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/hdf5_ros3 -DHDF5_ENABLE_ROS3_VFD=ON
make -j$(nproc)
make install

export HDF5_DIR=$HOME/hdf5_ros3
export LD_LIBRARY_PATH=$HDF5_DIR/lib:$LD_LIBRARY_PATH
export CPATH=$HDF5_DIR/include:$CPATH
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  # FIX FOR "error: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0"
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 ~/miniconda3/envs/ct/lib/libffi.so.7  # BETTER FIX FOR PREVIOUS
python -m pip install --no-binary=h5py h5py

Compiling requirements
Make sure Ray is 2.43.0!
python -m pip freeze -r requirements.in | sed -e '/@/d' \
    -e 's/+.*//' \
    -e '/nvidia-.*/d' \
    -e '/manim*/d' \
    -e '/pycairo*/d' \
    -e '/ManimPango*/d' \
    -e '/mayavi*/d' \
    > requirements.txt
# python -m pip freeze | grep -f requirements.in > requirements.txt
# python -m pip freeze -q -r requirements.in | sed '/freeze/,$ d' > requirements.txt

Run local
ray start --disable-usage-stats --node-ip-address 100.85.187.118 --head --port=6379 --dashboard-host=0.0.0.0

Run remote
ray start --disable-usage-stats --address 100.64.246.20:6379 --node-ip-address 100.85.187.118 --head --dashboard-host=0.0.0.0

Run train script
python -u train.py &> train_log.txt

Sync with remote (doesn't delete C code)
rsync -v ~/repos/inept/!(*.tar) precision:~/repos/INEPT && \
rsync -v ~/repos/inept/celltrip/* precision:~/repos/INEPT/celltrip && \
rsync -v ~/repos/inept/celltrip/utility/* precision:~/repos/INEPT/celltrip/utility && \
rsync -v ~/repos/inept/scripts/!(*.gzip) precision:~/repos/INEPT/scripts

Profiling
watch -d -n 0.5 nvidia-smi

Add kernel
python -m ipykernel install --user --name celltrip

Environment
conda activate ct

AWS Profile
export AWS_PROFILE=waisman-admin
aws sso login --profile waisman-admin

Ray cluster
ray up -y aws_config.yaml && ray attach aws_config.yaml -p 10001
ray dashboard aws_config.yaml
ray down -y aws_config.yaml

Submit job
cp ./train.py ./submit
export RAY_ADDRESS=http://localhost:8265
ray job submit --no-wait --working-dir ./submit/ --runtime-env-json '{"py_modules": ["../celltrip"], "pip": "../requirements.txt", "env_vars": {"RAY_DEDUP_LOGS": "0"}}' -- <script>

Convert
sh ../run_on_all_ext.sh ../convert_pdf_to_png.sh pdf

TODO
Find https://raw.githubusercontent.com/Oafish1/CellTRIP/refs/heads/main/plots, Replace ./plots to test images
-->
