# TRIDENT-2 Inference

Inference software for **TRIDENT-2**, a multimodal transformer model that predicts chemical toxicity (EC50, EC10, LOEC, NOEC) across ~6,700 species.

## Model

TRIDENT-2 is a multimodal model combining a RoBERTa-based chemical/taxonomic encoder with a feed-forward neural network head. It predicts four log₁₀-transformed toxicity endpoints simultaneously:

| Endpoint | Description |
|---|---|
| EC50 | Half-maximal effective concentration |
| EC10 | 10% effect concentration |
| NOEC | No-observed-effect concentration |
| LOEC | Lowest-observed-effect concentration |

Inference is always run in half-precision (`fp16`) on GPU.

### Model inputs

| Field | Description |
|---|---|
| SMILES | Canonical SMILES (isomeric encoding disabled) |
| ncbi_taxid | NCBI species taxonomic ID |
| effect | Toxicological effect type (e.g. `<MOR>` for mortality) |
| organism_lifestage | Life stage category |
| administration_route | Exposure route category |
| duration | log₁₀-transformed exposure duration (hours); filled with `1e-6` when missing |
| onehot | 16-dim vector: 14-dim concentration-unit one-hot + 2-dim duration-missing flag |

The default inference settings (acute exposure simulation) are read from `data/taxa_assigned_units_effects_durations_2026-04-17.csv`, which maps each of the ~6,700 supported species to their species-appropriate effect, concentration unit, and duration.

---

## Repository structure

```
trident2_inference/
├── data/
│   ├── taxa_assigned_units_effects_durations_2026-04-17.csv  # default inference settings (~6,700 species)
│   ├── taxid2commonname.json      # NCBI taxid → common name
│   ├── taxid2embedding_*.json     # pre-computed taxon embeddings
│   ├── taxid2parent.json          # taxonomic parent mapping
│   ├── taxid2rank.json            # taxonomic rank mapping
│   ├── taxid2sciname.json         # NCBI taxid → latin name
│   ├── taxid2spgroup.json         # NCBI taxid → coarse species group
│   └── taxonomy_tree_SK_*.nwk     # Newick species tree
│
├── models/
│   └── *_final_model_epoch_70_hf_trident2/  # model checkpoint
│       ├── config.json
│       ├── configuration_trident2.py
│       ├── modeling_trident2.py
│       ├── tokenizer.json / tokenizer_config.json
│       └── trident2_training_config.json
│
└── src/
    ├── quick_inference.py          # CLI inference → CSV/XLSX/TXT
    ├── db_utils/
    │   └── duckdb_utils.py         # DuckDB predictions storage
    └── trident/
        ├── inference/
        │   ├── model.py            # build_model_and_tokenizer, predict
        │   ├── preprocess_input.py # generate_inference_metadata, one-hot encoding
        │   ├── taxonomy.py         # load_taxonomic_information
        │   ├── torch_dataset.py    # MultiModalDataset, MultiModalCollator
        │   ├── torch_loaders.py    # build_dataloader
        │   └── build_inference_settings.py
        └── utils/
            ├── chem_utils.py       # canonicalize_smiles
            └── setup_logger.py     # loguru setup
```

---

## Installation

Requires Python 3.11–3.13. Uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv sync
```

---

## Usage

### Quick inference

Runs inference on a CSV/XLSX/TXT file containing SMILES and taxid columns.

```bash
cd src
python quick_inference.py \
  --data_pth /path/to/input.csv \
  --model models/sxno3jhi_final_model_epoch_70_hf_trident2 \
  --output_pth /path/to/results.csv.zip \
  --inference_mapping_df_path data/taxa_assigned_units_effects_durations_2026-04-17.csv \
  --taxid2speciesname_dict_path data/taxid2sciname.json \
  --taxid2speciesgroup_dict_path data/taxid2spgroup.json \
  --batch_size 256 \
  --mixed_precision
```

Or via a JSON config file:

```bash
python quick_inference.py --config_file_path config.json
```

**Key CLI options:**

| Argument | Default | Description |
|---|---|---|
| `--data_pth` | — | Input CSV/XLSX/TXT with SMILES and taxid columns |
| `--output_pth` | derived from input | Output path (CSV, XLSX, TXT; `.zip` suffix compresses) |
| `--model` | local checkpoint | HuggingFace model ID or local path |
| `--batch_size` | 64 | Inference batch size |
| `--mixed_precision` | off | Use fp16 autocast (strongly recommended on GPU) |
| `--return_CLS_embeddings` | off | Save 768-dim CLS embeddings to a sidecar `.pkl.zip` |
| `--ncbi_taxid_column` | `ncbi_taxid` | Column name containing species taxids |
| `--SMILES_column` | `SMILES` | Column name containing SMILES strings |
| `--debug` | off | Limit to 100 rows |


## Storage

### DuckDB (`db_utils/duckdb_utils.py`)

Predictions are written to a flat DuckDB table with a `(smiles, ncbi_taxid)` primary key. Resumed runs skip already-written rows via `ON CONFLICT DO NOTHING`. Polars + Arrow bulk inserts are used for throughput.

**Schema:**

| Column | Type | Notes |
|---|---|---|
| smiles | VARCHAR | Canonical SMILES (PK) |
| ncbi_taxid | VARCHAR | NCBI taxid (PK) |
| species_name | VARCHAR | Latin species name |
| species_group | VARCHAR | Coarse group (e.g. `fish`) |
| effect | VARCHAR | Endpoint type (e.g. `<MOR>`) |
| duration | FLOAT | log₁₀ hours |
| conc_unit | VARCHAR | e.g. `mg/l` |
| EC50 / EC10 / NOEC / LOEC | FLOAT | log₁₀ concentration (float32) |

**Example queries:**

```python
import duckdb
con = duckdb.connect("predictions.duckdb", read_only=True)

# Most toxic compounds for zebrafish
con.execute("""
    SELECT smiles, EC50 FROM predictions
    WHERE ncbi_taxid = '7955'
    ORDER BY EC50 ASC LIMIT 10
""").df()

# Species-averaged EC50 for aspirin
con.execute("""
    SELECT species_group, AVG(EC50) as mean_EC50, COUNT(*) as n
    FROM predictions
    WHERE smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    GROUP BY species_group ORDER BY mean_EC50
""").df()
```