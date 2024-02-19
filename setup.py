from setuptools import setup, find_packages

setup(
    name='connectome_interpreter',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'tqdm',
    ],
    # Optional metadata
    author='Yijie Yin',
    author_email='yy432@cam.ac.uk',
    description='A tool for connectomics data interpretation',
    keywords=['connectomics', 'neural network',
              'mechanistic interpretability'],
    url='http://example.com/MyPackage',  # Project home page
)
