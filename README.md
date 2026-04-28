![Build](https://img.shields.io/github/actions/workflow/status/YijieYin/connectome_interpreter/python-app.yml?branch=main)&emsp;
![PyPI](https://img.shields.io/pypi/v/connectome_interpreter)&emsp;
![Docs](https://readthedocs.org/projects/connectome-interpreter/badge/?version=latest)&emsp;
[![PyPI Downloads](https://static.pepy.tech/badge/connectome-interpreter)](https://pepy.tech/projects/connectome-interpreter)

`connectome_interpreter` turns synaptic wiring diagrams into testable hypotheses about circuit function. It runs comfortably on a laptop or in Colab, including for whole-brain analyses on connectomes of ~140,000 neurons.

```
pip install connectome-interpreter
```

For the bleeding-edge version: 
```
pip install git+https://github.com/YijieYin/connectome_interpreter.git
```

# What's inside

- **[Effective connectivity](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/matmul.html)** — polysynaptic connectivity strength (same metric as in [Seung 2024](https://www.nature.com/articles/s41586-024-07953-5), [Li et al. 2020](https://elifesciences.org/articles/62576), [Hulse et al. 2021](https://doi.org/10.7554/eLife.66039), [Eschbach et al. 2020](https://www.nature.com/articles/s41593-020-0607-9), [Hoeller et al. 2025](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2) etc.) between any two groups of neurons, with or without excitation/inhibition (`compress_paths`, `compress_paths_signed`, `effective_conn_from_paths`, `signed_effective_conn_from_paths`).
- **[Path-finding](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/path_finding.html)** — retrieve the actual subcircuit between sources and targets, group, filter, and plot (`find_paths_of_length`, `el_within_n_steps`, `group_paths`, `filter_paths`, `plot_paths`).
- **[Differentiable connectome model](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/simple_model.html)** — a simple firing-rate model layered onto the connectome, with per-cell-type slopes, biases, time constants, and divisive normalisation (`MultilayeredNetwork`).
- **Gradient methods** — activation maximisation (find the input that activates a target), saliency and `get_gradients` (find which inputs or intermediates matter for a stimulus), and `train_model` (fit parameters to data).
- **Visualisation** — interactive wiring diagrams, hex-eye and Mollweide receptive fields ([Zhao et al. 2025](https://www.nature.com/articles/s41586-025-09276-5)), [neuroglancer](https://github.com/google/neuroglancer) scenes (`plot_paths`, `hex_heatmap`, `plot_mollweide_projection`, `get_ngl_link`).
- **Resources** — a community-curated [table of fly cell types with experimentally tested functions](https://tinyurl.com/known-neuron-function), and bundled published odour-response datasets (`load_dataset`).

Full documentation: [connectome-interpreter.readthedocs.io](https://connectome-interpreter.readthedocs.io/en/latest/).

# Tutorial notebooks

A three-notebook tour using the FAFB/FlyWire connectome ([Dorkenwald et al. 2024](https://www.nature.com/articles/s41586-024-07558-y), [Schlegel et al. 2024](https://www.nature.com/articles/s41586-024-07686-5)) — open in Colab, no local install needed:

1. [**Linear methods**](https://colab.research.google.com/drive/145Td8_fFTPwTsDdEkQGAgdZ7CFQ8qXnr) — effective connectivity, path-finding, visual receptive fields.
2. [**Modelling**](https://colab.research.google.com/drive/1PcwEBaqwtak1YkhHdlrhkmFyMmMwJIVF) — running the differentiable model upon odour input.
3. [**Gradient methods**](https://colab.research.google.com/drive/1mmgW5Od2aB6MylfABs5sEZNLrcUK3Ws1) — activation maximisation, saliency, and training to experimental data.

Each notebook also loads BANC ([Bates et al. 2025](https://www.biorxiv.org/content/10.1101/2025.07.31.667571v1)), maleCNS ([Berg et al. 2025](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2)), hemibrain ([Scheffer et al. 2020](https://elifesciences.org/articles/57443)), and MANC ([Cheong et al. 2025](https://www.biorxiv.org/content/10.1101/2023.06.07.543976v3); [Marin et al. 2024](https://www.biorxiv.org/content/10.1101/2023.06.05.543407v2); [Takemura et al. 2024](https://elifesciences.org/reviewed-preprints/97769)) so you can swap connectomes by changing one variable.

A separate notebook covers the [larval connectome](https://colab.research.google.com/drive/1VIMNFBp7dCgN5XOQ9vvzPaqb80BGPZx4?usp=sharing) ([Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330); 3D viewer: [catmaid](https://catmaid.virtualflybrain.org/)).

# Mapping known to unknown

To facilitate neural circuit interpretation, we maintain a [list of cell types with known, *experimentally tested* functions](https://tinyurl.com/known-neuron-function). [This example notebook](https://colab.research.google.com/drive/1oETJthJbdLEBhzApEbRynGxTMrOcwsf-?usp=sharing) uses the list to query receptive fields. The list is a quick look-up of literature, not a stipulation of neural function.

- **Everyone has edit access** — please help make the list more comprehensive and correct, and check that the publications you care about are cited correctly. Handle with care.
- When adding multiple entries to one cell (e.g. several papers per cell type), separate them with `; ` (semicolon + space) for programmatic access.

<!--
# Structure-function relationships

Notebooks comparing published connectomes against published experimental papers:

- [Taisz et al. 2023](https://colab.research.google.com/drive/1WNNnNCjTey-iSlHPkxMlr_EaLsRMs9iX?usp=drive_link) — parallel representations of position and identity in the olfactory system ([paper](https://www.cell.com/cell/abstract/S0092-8674(23)00472-5))
- [Huoviala et al. 2020](https://colab.research.google.com/drive/1EyrGWO7MqpCZLvT2h4RyT4SaQy2fwYQT?usp=sharing) — neural circuit basis of aversive odour processing ([paper](https://www.biorxiv.org/content/10.1101/394403v2))
- [Frechter et al. 2019](https://colab.research.google.com/drive/1cSWNUdaU8Pll77eh4kOEz-NmrKHLnj-K?usp=sharing) — functional and anatomical specificity in a higher olfactory centre ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6550879/))
- [Olsen et al. 2010](https://colab.research.google.com/drive/1dA5GTHg25S3Mc9CBtexplfjk1z1kM04V?usp=sharing) — divisive normalization in olfactory population codes ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2866644/))
-->

# Notes

- Pre-processed connectomics data and pre-processing scripts are at [connectome_data_prep](https://github.com/YijieYin/connectome_data_prep/tree/main): adjacency matrices in `scipy.sparse` (`.npz`), metadata in `.csv`.
- Dataset requests, feature requests, feedback: open an issue, or email `yy432`at`cam.ac.uk` :).
