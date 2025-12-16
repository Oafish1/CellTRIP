import argparse
import os.path
from setuptools import find_packages, setup
from distutils.extension import Extension

# TODO: Fix this and don't autorun on install
try:
    import Cython
    USE_CYTHON = True
except: USE_CYTHON = False
CYTHON_COMPILED = os.path.exists('celltrip/environment.c')


# Get metadata
with open('celltrip/version.py') as f:
    exec(f.read())

with open('README.md') as f:
    readme = f.read()

# Initialize extensions and cmdclass
cmdclass = {}
ext_modules = []

# Format cython extensions
cython_extensions = [
    # Extension('celltrip.decorators', ['celltrip/decorator.py']),
    Extension('celltrip.environment', ['celltrip/environment.py']),
    Extension('celltrip.manager', ['celltrip/manager.py']),
    Extension('celltrip.memory', ['celltrip/memory.py']),
    Extension('celltrip.policy', ['celltrip/policy.py']),
    # Extension('celltrip.train', ['celltrip/train.py']),
    Extension('celltrip.utility.distance', ['celltrip/utility/distance.py']),
    Extension('celltrip.utility.general', ['celltrip/utility/general.py']),
    Extension('celltrip.utility.hooks', ['celltrip/utility/hooks.py']),
    # Extension('celltrip.utility.notebook', ['celltrip/utility/notebook.py']),
    Extension('celltrip.utility.processing', ['celltrip/utility/processing.py']),
    Extension('celltrip.utility.state_manager', ['celltrip/utility/state_manager.py']),
    Extension('celltrip.utility.continual', ['celltrip/utility/continual.py']),
    Extension('celltrip.utility.view', ['celltrip/utility/view.py']),
]

# Implement cython
def compiled(extensions):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules += cythonize(cython_extensions, compiler_directives={'language_level' : '3'})
elif CYTHON_COMPILED: ext_modules += compiled(cython_extensions)
# Build files: python setup.py build_ext -if
# Remove files: rm -r celltrip/{**,.}/{*.c,*.so}
# Do all: rm -r celltrip/{**,.}/{*.c,*.so} && python setup.py build_ext -if

# Perform setup
setup(
    name='celltrip',
    author='Noah Cohen Kalafut',
    description=
        'CellTRIP, Inferring virtual cell environments using '
        'multi-agent reinforcement learning for spatiotemporal '
        'trajectory interpolation, imputation, and perturbation',
    long_description=readme,
    long_description_content_type="text/markdown",
    version=__version__,
    packages=find_packages(),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        # Python 3.10.16 normally
        'numpy',
        'ray[client,default]',
        'scipy>=1.13.0',  # For sparse, ~1.13.0
        'torch',
    ],
    extras_require={
        # NOTE: Currently requires all to run
        # Additional requirements:
        #   `boto3` for s3
        #   `cupy` for parallel update
        #   `docker` for containers
        #   `ffmpeg` for video
        #   `poppler-utils`
        'compile': [
            'Cython'],
        'dev': [
            # 'jupyter_rfb',  # simplejpeg, pillow
            'line_profiler',
            'memory-profiler',
            'memray',
            'pip-tools',
            'snakeviz'],
        'extras': [
            # 'adjustText',
            'aiobotocore[boto3]',  # Compatibility with boto3 and s3fs
            'anndata',
            # 'fastplotlib[notebook,imgui]',  # imgui needed `sudo apt install xorg-dev cmake`
            'h5py',
            'ipympl',
            # 'manim',  # sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev
            'matplotlib',
            'nbconvert',
            'pandas',
            'POT',
            'rds2py',
            's3fs',
            'scanpy',
            'scikit-learn>=1.4.2',  # Needs ~1.4.2
            'scipy',
            'seaborn',
            'tqdm',
            'umap-learn',  # [parametric_umap]
            'wandb[importers]'],
    },
)
