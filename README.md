This package is intended to be used for interpreting connectomics data. 

To install: 
```
pip install connectome-interpreter
```

Documentation [here](https://connectome-interpreter.readthedocs.io/en/latest/) (with some text snippets explaining various things). 

# Example notebooks 
## Full Adult Fly Brain 
Data obtained from [Dorkenwald et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546656v2) and [Schlegel et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2). To visualise the neurons, you can use this url: [https://tinyurl.com/flywire783](https://tinyurl.com/flywire783). By using the connectivity information, you agree to follow the [FlyWire citation guidelines and principles](https://codex.flywire.ai/api/download).
  - [central brain, cell type level](https://colab.research.google.com/drive/1ECUagwN-r2rnKyfcYgtR1oG8Lox8m8BW?usp=sharing)
  - [central brain, single-neuron level](https://colab.research.google.com/drive/1_beqiKPX8pC7---DWepKO8dEv1sJ2vA4?usp=sharing)
  - [right hemisphere optic lobe, single-neuron level](https://colab.research.google.com/drive/1SHMZ3DUTeakdh0znMmXu5g2qffx6rFGV?usp=sharing)

## MaleCNS 
Data obtained from [neuPrint](https://neuprint.janelia.org/?dataset=optic-lobe%3Av1.0&qt=findneurons), with the help of [neuprint-python](https://connectome-neuprint.github.io/neuprint-python/docs/). 
- [Optic lobe, single-neuron level](https://colab.research.google.com/drive/1qEmO1tOOjSksa41OZ4_mX7KnJ8vBsvLU?usp=sharing)

## Larva 
Data from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). You can also e.g. visualise the neurons in 3D in [catmaid](https://catmaid.virtualflybrain.org/).
- [single-neuron level](https://colab.research.google.com/drive/1VIMNFBp7dCgN5XOQ9vvzPaqb80BGPZx4?usp=sharing) 

# Notes 
- Pre-processed connectomics data (and scripts used for pre-processing) are [here](https://anonymous.4open.science/r/interpret_connectome-68F4/README.md), in `scipy.sparse.matrix` (`.npz`) format for the adjacency matrices; and in `.csv` for the metadata.
- For dataset requests / feature requests / feedback, please make an issue or email me at `yy432`at`cam.ac.uk` :). 
