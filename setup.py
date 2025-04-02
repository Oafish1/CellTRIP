import argparse
import os.path
from setuptools import find_packages, setup
from distutils.extension import Extension

# TODO: Fix this and don't autorun on install
USE_CYTHON = True
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
    Extension('celltrip.memory', ['celltrip/memory.py']),
    Extension('celltrip.policy', ['celltrip/policy.py']),
    # Extension('celltrip.train', ['celltrip/train.py']),
    Extension('celltrip.utility.distance', ['celltrip/utility/distance.py']),
    Extension('celltrip.utility.general', ['celltrip/utility/general.py']),
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
    description='CellTRIP, a Multi-Agent Reinforcement Learning Approach for Cell Trajectory Recovery, Cross-Modal Imputation, and Perturbation in Time and Space',
    long_description=readme,
    long_description_content_type="text/markdown",
    version=__version__,
    packages=find_packages(),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        # Python >=3.11 if Cython is used with Ray, otherwise 3.10 works
        'numpy',
        'ray[client,default]',
        'scipy>=1.13.0',  # For sparse, ~1.13.0
        'torch',
    ],
    extras_require={
        # NOTE: Currently requires all to run
        # Additional requirements:
        #   `cupy` for parallel update
        #   `docker` for containers
        #   `ffmpeg` for video
        #   `poppler-utils`
        'compile': [
            'Cython'],
        'dev': [
            'memory-profiler',
            'memray',
            'pip-tools',
            'snakeviz'],
        'examples': [
            'adjustText',
            'anndata',
            'h5py',
            'ipympl',
            'matplotlib',
            'nbconvert',
            'pandas',
            'rds2py',
            'scanpy',
            'scikit-learn>=1.4.2',  # Needs ~1.4.2
            'seaborn',
            'tqdm',
            'umap-learn',
            'wandb[importers]'],
    },
)
