This package is intended to be used for interpreting connectomics data. 

To install: 
```
pip install connectome-interpreter
```

Documentation [here](https://connectome-interpreter.readthedocs.io/en/latest/) (with some text snippets explaining various things). 

# Example notebooks 
## Full Adult Fly Brain 
Data obtained from [Dorkenwald et al. 2024](https://www.nature.com/articles/s41586-024-07558-y), [Schlegel et al. 2024](https://www.nature.com/articles/s41586-024-07686-5), and [Matsliah et al. 2024](https://www.nature.com/articles/s41586-024-07981-1). To visualise the neurons, you can use this url: [https://tinyurl.com/flywire783](https://tinyurl.com/flywire783). By using the connectivity information, you agree to follow the [FlyWire citation guidelines and principles](https://codex.flywire.ai/api/download).
  - [central brain, single-neuron level](https://colab.research.google.com/drive/1_beqiKPX8pC7---DWepKO8dEv1sJ2vA4?usp=sharing) (recommended. Shows a variety of capabilities)
  - [central brain, cell type level](https://colab.research.google.com/drive/1ECUagwN-r2rnKyfcYgtR1oG8Lox8m8BW?usp=sharing)
  - [right hemisphere optic lobe, single-neuron level](https://colab.research.google.com/drive/1SHMZ3DUTeakdh0znMmXu5g2qffx6rFGV?usp=sharing)

## MaleCNS 
Data obtained from [neuPrint](https://neuprint.janelia.org/?dataset=optic-lobe%3Av1.0&qt=findneurons) and [Nern et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589741v2), with the help of [neuprint-python](https://connectome-neuprint.github.io/neuprint-python/docs/). 
- [Optic lobe, single-neuron level](https://colab.research.google.com/drive/1qEmO1tOOjSksa41OZ4_mX7KnJ8vBsvLU?usp=sharing)

## Larva 
Data from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). You can also e.g. visualise the neurons in 3D in [catmaid](https://catmaid.virtualflybrain.org/).
- [single-neuron level](https://colab.research.google.com/drive/1VIMNFBp7dCgN5XOQ9vvzPaqb80BGPZx4?usp=sharing) 

# Mapping known to unknown 
To facilitate neural circuit interpretation, we compile a [list](https://docs.google.com/spreadsheets/d/1VHCEnurOdb4FDC_NUKZX_BpBckQ9LpKxv0CsK_ObVok/edit?usp=sharing) of cell types with known, *experimentally tested*, functions. [This example notebook](https://colab.research.google.com/drive/1oETJthJbdLEBhzApEbRynGxTMrOcwsf-?usp=sharing) uses this list for query of neuron receptive field. The list aims to serve as a quick look-up of literature, instead of a stipulation of neural function. 
- **Everyone is given edit access, to help make the list more comprehensive and correct, and to make sure the publications you care about are cited correctly. Your contributions would be much appreciated. Please handle with care.**
- When multiple entries are to be added in the same cell (e.g. when multiple publications are related to the same cell type), please separate the entries with `; ` (semicolon + space), to facilitate programmatic access. 

# Structure-function relationship 
Using `connectome_interpreter`, we compare the published connectomes against published experimental papers: 
- [Taisz et al. 2023](https://colab.research.google.com/drive/1WNNnNCjTey-iSlHPkxMlr_EaLsRMs9iX?usp=drive_link): Generating parallel representations of position and identity in the olfactory system ([paper](https://www.cell.com/cell/abstract/S0092-8674(23)00472-5))
- [Huaviala et al. 2020](https://colab.research.google.com/drive/1EyrGWO7MqpCZLvT2h4RyT4SaQy2fwYQT?usp=sharing): Neural circuit basis of aversive odour processing in Drosophila from sensory input to descending output ([paper](https://www.biorxiv.org/content/10.1101/394403v2))
- [Frechter et al. 2019](https://colab.research.google.com/drive/1cSWNUdaU8Pll77eh4kOEz-NmrKHLnj-K?usp=sharing): Functional and anatomical specificity in a higher olfactory centre ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6550879/))
- [Olsen et al. 2010](https://colab.research.google.com/drive/1dA5GTHg25S3Mc9CBtexplfjk1z1kM04V?usp=sharing): Divisive normalization in olfactory population codes ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2866644/))

# Notes 
- Pre-processed connectomics data (and scripts used for pre-processing) are [here](https://github.com/YijieYin/connectome_data_prep/tree/main), in `scipy.sparse.matrix` (`.npz`) format for the adjacency matrices; and in `.csv` for the metadata.
- For dataset requests / feature requests / feedback, please make an issue or email me at `yy432`at`cam.ac.uk` :). 
