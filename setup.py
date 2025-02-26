from setuptools import find_packages, setup

with open('celltrip/version.py') as f:
    exec(f.read())

with open('README.md') as f:
    readme = f.read()

setup(
    name='celltrip',
    author='Noah Cohen Kalafut',
    description='CellTRIP, a Multi-Agent Reinforcement Learning Approach for Cell Trajectory Recovery, Cross-Modal Imputation, and Perturbation in Time and Space',
    long_description=readme,
    long_description_content_type="text/markdown",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ray[default]',
        'scipy>=1.13.0',  # For sparse, ~1.13.0
        'torch',
    ],
    extras_require={
        'dev': [
            'memory-profiler',
            'memray',
            'pip-tools',
            'snakeviz',
        ],
        'examples': [
            'adjustText',
            'h5py',
            'ipympl',
            'matplotlib',
            'pandas',
            'rds2py',
            'scanpy',
            'scikit-learn>=1.4.2',  # Needs ~1.4.2
            'seaborn',
            'tqdm',
            'umap-learn',
            'wandb[importers]',
            # Additional CLI requirements: ffmpeg, poppler-utils
        ],
    },
)
