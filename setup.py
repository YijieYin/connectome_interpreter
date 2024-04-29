from setuptools import setup, find_packages
import pkg_resources

# Function to check if PyTorch is already installed


def get_torch_version():
    try:
        return pkg_resources.get_distribution('torch').version
    except pkg_resources.DistributionNotFound:
        return None


# Specify a PyTorch version requirement only if it's not installed
torch_version = get_torch_version()
if torch_version is None:
    torch_requirement = 'torch>=1.7.1'
else:
    torch_requirement = f'torch=={torch_version}'


setup(
    name='connectome_interpreter',
    # If you're making a patch or a minor bug fix, increment the patch version, e.g., from 0.1.0 to 0.1.1.
    # If you're adding functionality in a backwards-compatible manner, increment the minor version, e.g., from 0.1.0 to 0.2.0.
    # If you're making incompatible API changes, increment the major version, e.g., from 0.1.0 to 1.0.0.
    version='1.6.2',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        'numpy',
        'pandas',
        'scipy',
        # 'torch>=1.7.1',
        torch_requirement,
        'tqdm',
        'plotly',
        'matplotlib',
        'networkx',
        'seaborn',
        'ipywidgets',
        'IPython',
    ],
    extras_require={
        'get_ngl_link': ['nglscenes'],
        'wandb': ['wandb'],
    },
    # Optional metadata
    author='Yijie Yin',
    author_email='yy432@cam.ac.uk',
    description='A tool for connectomics data interpretation',
    keywords=['connectomics', 'neural network',
              'mechanistic interpretability'],
    project_urls={
        'Source': 'https: // github.com/YijieYin/connectome_interpreter',
        'Documentation': 'https://connectome-interpreter.readthedocs.io/en/latest/',
        'Larva connectome example': 'https://colab.research.google.com/drive/1VIMNFBp7dCgN5XOQ9vvzPaqb80BGPZx4?usp=sharing',
        'Adult connectome example (FAFB)': 'https://colab.research.google.com/drive/1ECUagwN-r2rnKyfcYgtR1oG8Lox8m8BW?usp=sharing',
    },
    # url='https://connectome-interpreter.readthedocs.io/en/latest/',
    test_suite='tests'  # Project home page
)
