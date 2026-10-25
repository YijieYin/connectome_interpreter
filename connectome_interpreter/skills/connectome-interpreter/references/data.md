# Data: connectomes, metadata, known function, stimuli

## Preprocessed connectomes

Hosted at https://github.com/YijieYin/connectome_data_prep (folder `data/`). Raw base URL:
`https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/`

Each connectome = sparse `.npz` matrices (pre in rows, post in columns) + a metadata `.csv` whose `idx` column is the row/column index. All matrices of one dataset share the same `idx`.

| Dataset | Folder | Metadata | Full connectome | Axon-dendrite only |
|---|---|---|---|---|
| FAFB/FlyWire (female brain, ~139k neurons) | `fafb_all_neuron/` | `fafb_all_neuron_meta.csv` | `fafb_inprop_all_neuron.npz` | `fafb_ad_inprop_all_neuron.npz` |
| maleCNS (male brain + VNC, ~161k) | `maleCNS/` | `mcns_all_neuron_meta.csv` | `mcns_inprop_all_neuron.npz` | `mcns_ad_inprop_all_neuron.npz` |
| BANC (female brain + VNC, ~113k) | `BANC/` | `banc_meta_all_neuron.csv` | `banc_inprop_all_neuron.npz` | not available |

Also in each folder: `*_outprop_*` (normalised by the pre neuron's total output), `*_syncount_*` (raw synapse counts; `fafb_ad_syncount_*`, `mcns_ad_syncount_*` for the split). Other datasets in the repo: hemibrain (`hemibrain/`), MANC (`MANC/`), larva (`larva/`), right optic lobe subsets.

Naming: `inprop` = input proportion (synapses pre->post / all input synapses of post; columns sum to <= 1), `outprop` = output proportion, `syncount` = synapse count, `ad` = axon-dendrite, `cb` = central brain.

**Which matrix**: `inprop` for effective connectivity, path-finding and the model (default). `outprop` for "where does this neuron's output go". `syncount` for filtering by absolute synapse number (e.g. remove 1-synapse connections, see below).

### Axon-dendrite split
Neurons split into axon / dendrite with flow centrality (Schneider-Mizell et al. 2016) plus manual curation; ~20k neurons with low segregation index are left unsplit ("both"). The axon-dendrite connectome keeps axon->dendrite, axon->both, both->dendrite, both->both, i.e. drops axo-axonic and dendro-dendritic connections, so signal flow follows the canonical direction. Compartment-level edgelists (`*_ad_split/{aa,ab,ad,ba,bb,bd,da,db,dd}_el.csv.gz`, columns `pre_root_id`/`post_*`, `syn_count`; `bb` is available on request) let users build custom combinations.

### Loading

```python
import numpy as np, pandas as pd, scipy.sparse as sp, connectome_interpreter as coin
from pathlib import Path
import urllib.request

BASE = "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/"

def fetch(rel, cache="connectome_data"):
    """Download once into a local cache folder; return the local path."""
    p = Path(cache) / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + rel, p)
    return p

inprop = sp.load_npz(fetch("fafb_all_neuron/fafb_inprop_all_neuron.npz"))
# or axon-dendrite: fetch("fafb_all_neuron/fafb_ad_inprop_all_neuron.npz")
meta = pd.read_csv(fetch("fafb_all_neuron/fafb_all_neuron_meta.csv"), low_memory=False)
meta = meta.drop(columns=[c for c in meta.columns if c.startswith("Unnamed")])
```

Files are 15-105 MB each; cache them (on HPC, cache to scratch/project storage and load from there in batch jobs, since compute nodes may lack internet).

### Metadata columns (differ by dataset)

Common: `idx` (matrix index), `cell_type`, `top_nt` (predicted neurotransmitter), `sign` (+1 excitatory / -1 inhibitory, from `top_nt` of the presynaptic neuron).

| | Neuron ID | Side | Coarse class | Finer class |
|---|---|---|---|---|
| FAFB | `root_id` | `side` (left/right/center/na) | `super_class` (optic, central, sensory, visual_projection, ascending, descending, ...) | `cell_class`, `cell_sub_class` |
| maleCNS | `bodyid` | `somaSide` (L/R/M); `rootSide` for sensory neurons | `superclass` (ol_intrinsic, cb_intrinsic, vnc_intrinsic, cb_sensory, descending_neuron, ...) | `class`, `subclass`, `type`, `flywireType`, `instance` |
| BANC | `root_id` | `soma_side` (left/right/midline) | `super_class` (central_brain_intrinsic, sensory, descending, ...) | `class`, `sub_class`, `function` |

Untyped neurons have their neuron ID as `cell_type`; exclude with `~meta.cell_type.astype(str).str.isnumeric()` when sampling types. Inspect `meta.columns` and `value_counts()` before filtering rather than assuming category names.

Standard mappings:

```python
side_col = {"fafb": "side", "mcns": "somaSide", "banc": "soma_side"}[ds]
meta["type_side"] = meta.cell_type.astype(str) + "_" + meta[side_col].astype(str)
idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_type_side = dict(zip(meta.idx, meta.type_side))
idx_to_sign = dict(zip(meta.idx, meta.sign))
type_side_to_sign = dict(zip(meta.type_side, meta.sign))  # for group-level sign arguments
idx_to_root = dict(zip(meta.idx, meta.root_id))  # bodyid for maleCNS
```

Cell type names largely follow FlyWire/hemibrain conventions across datasets (e.g. `MBON01`, `DNa02`, `LPLC1`), but check: maleCNS has `flywireType` for cross-matching; olfactory names look like `ORN_DA1`, `DA1_lPN` / `DA3_adPN` in FAFB.

### Modifying connectivity
`coin.modify_coo_matrix(inprop, input_idx, output_idx, value)` or `updates_df` (columns `input_idx`, `output_idx`, `value`); re-normalises columns by default (`re_normalize=True`). Example, drop single-synapse connections:

```python
sc = sp.load_npz(fetch("fafb_all_neuron/fafb_ad_syncount_all_neuron.npz")).tocoo()
one = pd.DataFrame({"input_idx": sc.row[sc.data == 1], "output_idx": sc.col[sc.data == 1], "value": 0})
ad_inprop = coin.modify_coo_matrix(ad_inprop, updates_df=one)
```

### Own data
Needs a sparse matrix (pre rows, post columns), ideally input-normalised by each neuron's *total* input (including partners outside any subset), and a metadata table mapping `idx` to IDs/types/sign.

## Known-function table

Community-curated, openly editable (https://tinyurl.com/known-neuron-function, a Google Sheet). Read directly:

```python
known = pd.read_csv("https://docs.google.com/spreadsheets/d/1VHCEnurOdb4FDC_NUKZX_BpBckQ9LpKxv0CsK_ObVok/export?format=csv")
# columns: cell_type, known_function, category, experimentally_tested, reference,
#          link_to_paper, notes, contributor, also_known_as
```

Multiple entries per cell are separated by `"; "`. `experimentally_tested == "yes"` marks directly tested functions; other entries are annotations (much from the BANC project). Match on `cell_type` and also check `also_known_as`. When presenting an RFc, name the known partners, their `known_function`, and the `reference`. Users are invited to correct/add entries.

## Stimulus datasets (bundled)

`coin.load_dataset(name)`; names: `DoOR_adult`, `DoOR_adult_sfr_subtracted`, `Dweck_adult_chem`, `Dweck_adult_fruit`, `Dweck_larva_chem`, `Dweck_larva_fruit`, `Hallem2006`, `Hallem2006_dilution`, `Hallem2006_time`, `Badel2016_PN`, `Knaden2012_odour_valence`, plus spatial `Nern2024`, `Matsliah2024`, `Zhao2024`. Adult olfactory tables have glomeruli in rows, odours in columns (see `help(coin.load_dataset)` for units/concentrations).

Odour tuning of a downstream neuron, linearly: build glomerulus -> target effective connectivity (group ORNs/PNs by glomerulus), then `coin.map_to_experiment(df, dataset="DoOR_adult")` (glomeruli in rows, targets in columns). Non-linearly: turn the odour response into model input with `coin.input_from_df` (see model.md).

## Visual space

Columnar coordinates: `Matsliah2024` (FAFB right optic lobe; `root_id`, `type`, `x`, `y`, `p`, `q`, `coords` as `"x,y"` string), `Nern2024` (maleCNS medulla columns), maleCNS metadata `assignedOlHex1/2`. `coin.hex_heatmap(series_indexed_by "x,y", dataset="fafb_right" | "mcns_right")` draws eye maps (DataFrame -> one frame per column); `coin.plot_mollweide_projection(data, dataset="Zhao2024")` projects onto the eye's field of view. `coin.looming_stimulus` / `coin.make_sine_stim` build stimuli.
