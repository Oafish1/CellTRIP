from setuptools import find_packages, setup

with open('inept/version.py') as f:
    exec(f.read())

with open('README.md') as f:
    readme = f.read()

setup(
    name='inept',
    author='Noah Cohen Kalafut',
    description='Independent Node Exploration and Probabilistic Tracing',
    long_description=readme,
    long_description_content_type="text/markdown",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'scipy>=1.13.0',  # For sparse, ~1.13.0
        'torch',
        'numpy',
    ],
    extras_require={
        'dev': [
            'memory-profiler',
            'pip-tools',
            'snakeviz',
        ],
        'examples': [
            'adjustText',
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
        ],
    },
)
