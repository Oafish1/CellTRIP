# Method Comparisons
This directory houses comparison methods for various aspects of cellTRIP's functionality. Below are instructions for installing each alternate method, to be run in `other_methods.ipynb` above. In general, when creating environments, the following command may be used:
```
conda create -n <name> python=<version>
conda activate <name>
python -m pip install -r <requirements.in file>
```

## Integration Methods
### JAMIE Download
Clone [this repository](https://github.com/Oafish1/JAMIE) to this directory. Follow the installation directions on the GitHub to create a suitable environment.

### ManiNetCluster Download
For CCA, LMA, and NLMA, clone [this repository](https://github.com/namtk/ManiNetCluster) to this directory. Create an environment according to the annotations in `requirements-ManiNetCluster.in`.

### MMD-MA Download
Download [this script](https://bitbucket.org/noblelab/2019_mmd_wabi/src/master/manifoldAlignDistortionPen_mmd_multipleStarts.py) and extract to this directory. Alternatively, clone [this repository](https://bitbucket.org/noblelab/2019_mmd_wabi/src/master/) to this directory. Create an environment according to the annotations in `requirements-MMD-MA.in`.


## Imputation Methods
### BABEL Download
Clone [this repository](https://github.com/wukevin/babel) to this directory. Follow the installation directions on the GitHub to create a suitable environment. Within the same environment, also install the additional requirements in `requirements-babel.in`.

### JAMIE Download
See instructions above.
