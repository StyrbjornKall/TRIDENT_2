# Preprocess Toxicity Data

A data preprocessing pipeline that consolidates and harmonizes toxicological data from multiple sources into a structured dataset enriched with chemical SMILES, NCBI taxonomy, phylogenetic distances, and MDS embeddings.

## Data Sources

| Source | Prefix | Description |
|--------|--------|-------------|
| RTECS | `RTECS` | Registry of Toxic Effects of Chemical Substances |
| REACH | `CAR` / `REPRO` | EU REACH carcinogenicity and reproductive toxicity |
| EFSA | `ACUTE` | European Food Safety Authority acute toxicity data |
| ECOTOX | `AQTER` | EPA ECOTOX aquatic and terrestrial toxicity data |

## Pipeline

The pipeline runs as a sequence of steps, each implemented as a Python script (`src/step{N}.py`) invoked by a shell script (`step{N}.sh`). Configuration is loaded from a `.env` file.

```
Raw Toxicity Datasets (RTECS, REACH, EFSA, ECOTOX)
         │
         ▼ Step 1 — Ingest & concatenate
         │  Reads raw CSVs, adds SK_unique_id, standardizes column names
         │
         ▼ Step 2 — Filter
         │  Filters by concentration sign (=), units, endpoints (noec/loec/ec10/ec50),
         │  and effects (mortality, developmental, behavioral, etc.)
         │  Exports unique CAS list for SMILES lookup
         │
         ├──▶ Step 2.5 — CAS → SMILES (parallel)
         │    Multi-API resolution: local PubChem DB (~95%), Webchem, Cactus, cirpy
         │
         ▼ Step 2_add_smiles — Merge SMILES
         │  Left-joins SMILES onto filtered data by CAS number
         │
         ▼ Step 3 — Standardize
         │  Species name standardization (common → Latin via JSON mappings)
         │  Unit normalization (concentrations → mg/L, durations → days)
         │  SMILES canonicalization via RDKit, molar → mg/L conversion
         │  Lifestage categorization and administration route mapping
         │  Deduplication
         │
         ▼ Step 4 — Taxonomy
         │  NCBI taxonomy lookup (TaxID, lineage across 9 ranks)
         │  Matching: exact → common name → fuzzy → GBIF → partial
         │
         ▼ Step 5 — Phylogenetic tree & distance matrix
         │  Builds Newick tree via ete3, computes pairwise species distances
         │
         ▼ Step 6 — MDS embeddings
            Metric MDS on distance matrix → 768-dimensional embeddings
            Validation via Pearson/Spearman correlation
```

### Step Details

| Step | Script | Input | Output | Key Operations |
|------|--------|-------|--------|----------------|
| 1 | `step1.py` | Raw CSVs | `step1.csv` | Ingest, rename columns, add unique IDs |
| 2 | `step2.py` | Step 1 | `step2.csv` + CAS list | Filter by conc_sign, units, endpoints, effects |
| 2.5 | `get_smiles.py` | CAS list | CAS→SMILES mapping | Multi-API SMILES resolution |
| 2+ | `step2_add_smiles.py` | Step 2 + SMILES | `step2.csv` (updated) | Merge SMILES via left join |
| 3 | `step3.py` | Step 2 | `step3.csv` + taxa list | Species/unit standardization, dedup |
| 4 | `step4.py` | Step 3 | `step4.csv` + taxonomy info | NCBI taxonomy enrichment |
| 5 | `step5.py` | Taxonomy info | Distance matrix + Newick tree | Phylogenetic tree and distances |
| 6 | `step6.py` | Distance matrix | MDS embeddings | 768D metric MDS embeddings |

## Project Structure

```
├── pipeline.sh                  # Orchestrates all steps
├── step{1..6}.sh                # Per-step shell wrappers
├── .env                         # Path configuration (not committed)
├── pyproject.toml               # Dependencies (managed by uv)
├── ruff.toml                    # Linter/formatter configuration
│
├── src/
│   ├── step{1..6}.py            # Pipeline step implementations
│   ├── step2_add_smiles.py      # SMILES merge step
│   ├── preprocess_datasets.py   # Per-source preprocessing functions
│   ├── preprocess_taxonomy.py   # TaxonomyPreProcessor, TaxonomyTreeProcessor
│   ├── chem_utils.py            # SMILES canonicalization, molecular properties
│   ├── utils.py                 # Shared utilities, mapping loaders, unit conversions
│   ├── setup_logger.py          # Loguru logger configuration
│   ├── pickle_files.py          # CSV ↔ pickle serialization
│   ├── get_taxid_translation.py # NCBI TaxID translation helpers
│   │
│   ├── cas_to_smiles/           # Multi-API CAS → SMILES resolution
│   │   ├── get_smiles.py        # Main orchestrator
│   │   ├── pubchem_api.py       # Local PubChem database lookup
│   │   ├── cactus_api.py        # NCI/CADD Cactus API
│   │   ├── cheminformant_api.py # Cheminformant API
│   │   ├── cirpy_api.py         # CIR/cirpy API
│   │   └── generate_db/         # Scripts to build local PubChem lookup tables
│   │
│   └── mdscuda/                 # MDS utility module
│
├── data/                        # Mapping files (JSON/CSV)
│   ├── standardize_common_lower.json
│   ├── translate_common_to_latin_lower.json
│   ├── translate_common_to_group_lower.json
│   ├── ECOTOX_mapping_lifestages.csv
│   ├── mapping_administration_route.csv
│   └── gbif_taxonomy_mapping.csv
│
├── logs/                        # Pipeline run logs
└── notebooks/                   # Exploratory notebooks
```

## Setup

### Prerequisites

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) package manager
- NCBI taxonomy database (for steps 4–5)
- Local PubChem lookup tables (for CAS→SMILES, see `src/cas_to_smiles/generate_db/`)

### Installation

```bash
uv sync
```

### Configuration

Create a `.env` file with paths to raw data, output directories, and database locations. Shell scripts source this file before running each step.

## Usage

Run the full pipeline:

```bash
bash pipeline.sh
```

Run individual steps:

```bash
bash step1.sh
bash step2.sh
bash src/cas_to_smiles/get_smiles_from_cas.sh
bash step2_add_smiles.sh
bash step3.sh
bash step4.sh
bash step5.sh
bash step6.sh
```

## Data Conventions

- **Column names**: snake_case — `cas`, `conc`, `conc_unit`, `endpoint`, `effect`, `species_latin_name`, `species_group`, `duration`, `duration_unit`, `admin_route`
- **NCBI columns**: prefixed with `NCBI_` — `NCBI_rank_phylum`, `NCBI_match`, `NCBI_last_known_rank`
- **Unique IDs**: `SK_unique_id` with dataset-specific prefixes (`RTECS`, `CAR`, `REPRO`, `ACUTE`, `AQTER`)
- **Storage**: Intermediate outputs saved as CSV + pickle, with DuckDB tables for structured queries

## Development

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

Code style: Python 3.12, ruff with 88-char lines, double quotes, 4-space indent. Logging via [loguru](https://github.com/Delgan/loguru).