from setuptools import setup, find_packages
import pkg_resources

# Function to check if PyTorch is already installed


def is_torch_installed():
    try:
        pkg_resources.get_distribution('torch')
        return True
    except pkg_resources.DistributionNotFound:
        return False


install_requires_list = [
    # List your package dependencies here
    'numpy',
    'pandas',
    'scipy',
    'tqdm',
    'plotly',
    'matplotlib',
    'networkx',
    'seaborn',
    'ipywidgets',
    'IPython',
]
# Add torch only if it's not already installed
if not is_torch_installed():
    install_requires_list.append('torch>=1.7.1')


setup(
    name='connectome_interpreter',
    # If you're making a patch or a minor bug fix, increment the patch version, e.g., from 0.1.0 to 0.1.1.
    # If you're adding functionality in a backwards-compatible manner, increment the minor version, e.g., from 0.1.0 to 0.2.0.
    # If you're making incompatible API changes, increment the major version, e.g., from 0.1.0 to 1.0.0.
    version='1.11.0',
    packages=find_packages(),
    install_requires=install_requires_list,
    extras_require={
        'get_ngl_link': ['nglscenes'],
        'map_to_experiment': ['scikit-learn'],
        'wandb': ['wandb'],
    },
    include_package_data=True,
    package_data={
        'connectome_interpreter': ['data/*/*'],
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
        'Adult connectome example (FAFB)': 'https://colab.research.google.com/drive/1ECUagwN-r2rnKyfcYgtR1oG8Lox8m8BW?usp=sharing',
        'Larva connectome example': 'https://colab.research.google.com/drive/1VIMNFBp7dCgN5XOQ9vvzPaqb80BGPZx4?usp=sharing',
    },
    # url='https://connectome-interpreter.readthedocs.io/en/latest/',
    test_suite='tests'  # Project home page
)
