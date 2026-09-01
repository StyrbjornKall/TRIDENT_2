"""
DuckDB storage utilities for TRIDENT-2 batch inference.

Provides functions to initialize a DuckDB predictions database and write
model outputs efficiently using polars + Arrow bulk inserts.

Database layout
---------------
predictions     — core prediction table (one row per smiles × ncbi_taxid pair)
species         — species metadata (name, common name, species group)
lineage         — full 8-level NCBI taxonomic lineage per species
settings        — per-species inference settings (effect, duration, conc_unit)
chemicals       — per-SMILES RDKit descriptors (canonical_smiles, inchi, inchikey)
chemical_summary — pre-aggregated statistics per (smiles, endpoint, effect)
taxa_summary     — pre-aggregated statistics per (ncbi_taxid, endpoint, effect)

predictions schema
------------------
    smiles     VARCHAR  — canonical SMILES (part of PK)
    ncbi_taxid VARCHAR  — NCBI species taxid (part of PK)
    effect     VARCHAR  — toxicological endpoint type, e.g. 'MOR' (part of PK)
    duration   INTEGER  — exposure duration in whole hours; 0 = unspecified/missing (part of PK)
    conc_unit  VARCHAR  — concentration unit (e.g. 'mg/l') (part of PK)
    EC50       FLOAT    — predicted log10 EC50
    EC10       FLOAT    — predicted log10 EC10
    NOEC       FLOAT    — predicted log10 NOEC
    LOEC       FLOAT    — predicted log10 LOEC

Note: species_name and species_group are stored in the ``species`` taxonomy
table and can be joined via ncbi_taxid. The ``settings`` table holds the
default per-species effect/duration/conc_unit derived from the inference
mapping CSV, which may differ from the effect/duration/conc_unit actually
used for a given predictions row when ``--inference_*`` overrides are passed.

Schema note: the (smiles, ncbi_taxid, effect, duration, conc_unit) primary
key allows multiple rows per (smiles, ncbi_taxid) pair — one per
--inference_* metadata combination. Databases created before this primary
key was introduced use the older (smiles, ncbi_taxid) key and are NOT
compatible with this schema; start a fresh database for cartesian runs.
"""

from __future__ import annotations

from typing import Optional

import duckdb
import pandas as pd
import polars as pl
from loguru import logger

PREDICTIONS_TABLE = "predictions"

# ---------------------------------------------------------------------------
# DDL — core tables
# ---------------------------------------------------------------------------

_CREATE_PREDICTIONS_SQL = """\
CREATE TABLE IF NOT EXISTS predictions (
    smiles     VARCHAR NOT NULL,
    ncbi_taxid VARCHAR NOT NULL,
    effect     VARCHAR NOT NULL,
    duration   INTEGER NOT NULL,
    conc_unit  VARCHAR NOT NULL,
    EC50       FLOAT,
    EC10       FLOAT,
    NOEC       FLOAT,
    LOEC       FLOAT,
    PRIMARY KEY (smiles, ncbi_taxid, effect, duration, conc_unit)
)
"""

_CREATE_SPECIES_SQL = """\
CREATE TABLE IF NOT EXISTS species (
    ncbi_taxid   VARCHAR PRIMARY KEY,
    sciname      VARCHAR,
    common_name  VARCHAR,
    species_group VARCHAR
)
"""

_CREATE_LINEAGE_SQL = """\
CREATE TABLE IF NOT EXISTS lineage (
    ncbi_taxid    VARCHAR PRIMARY KEY,
    superkingdom  VARCHAR,
    kingdom       VARCHAR,
    phylum        VARCHAR,
    subphylum     VARCHAR,
    taxon_class   VARCHAR,
    taxon_order   VARCHAR,
    family        VARCHAR,
    genus         VARCHAR
)
"""

_CREATE_SETTINGS_SQL = """\
CREATE TABLE IF NOT EXISTS settings (
    ncbi_taxid VARCHAR PRIMARY KEY,
    effect     VARCHAR,
    duration   INTEGER,
    conc_unit  VARCHAR
)
"""

_CREATE_CHEMICALS_SQL = """\
CREATE TABLE IF NOT EXISTS chemicals (
    smiles           VARCHAR PRIMARY KEY,
    canonical_smiles VARCHAR,
    inchi            VARCHAR,
    inchikey         VARCHAR
)
"""

_CREATE_CHEMICAL_SUMMARY_SQL = """\
CREATE TABLE IF NOT EXISTS chemical_summary (
    smiles     VARCHAR NOT NULL,
    endpoint   VARCHAR NOT NULL,
    effect     VARCHAR NOT NULL,
    n_species  INTEGER,
    mean       FLOAT,
    median     FLOAT,
    std        FLOAT,
    p10        FLOAT,
    p90        FLOAT,
    PRIMARY KEY (smiles, endpoint, effect)
)
"""

_CREATE_TAXA_SUMMARY_SQL = """\
CREATE TABLE IF NOT EXISTS taxa_summary (
    ncbi_taxid  VARCHAR NOT NULL,
    endpoint    VARCHAR NOT NULL,
    effect      VARCHAR NOT NULL,
    n_chemicals INTEGER,
    mean        FLOAT,
    median      FLOAT,
    std         FLOAT,
    p10         FLOAT,
    p90         FLOAT,
    PRIMARY KEY (ncbi_taxid, endpoint, effect)
)
"""

# Ordered column names expected in the DataFrame passed to write_predictions.
PREDICTION_COLUMNS = [
    "smiles",
    "ncbi_taxid",
    "effect",
    "duration",
    "conc_unit",
    "EC50",
    "EC10",
    "NOEC",
    "LOEC",
]


def init_predictions_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open or create a DuckDB predictions database.

    Creates all schema tables if they do not already exist:
    ``predictions``, ``species``, ``lineage``, ``settings``,
    ``chemical_summary``, ``taxa_summary``.

    The returned connection is persistent — the caller is responsible for
    closing it with ``con.close()``.

    Args:
        db_path: Path to the DuckDB file. Created if it does not exist.

    Returns:
        Open DuckDB connection.
    """
    con = duckdb.connect(db_path)
    for ddl in (
        _CREATE_PREDICTIONS_SQL,
        _CREATE_SPECIES_SQL,
        _CREATE_LINEAGE_SQL,
        _CREATE_SETTINGS_SQL,
        _CREATE_CHEMICALS_SQL,
        _CREATE_CHEMICAL_SUMMARY_SQL,
        _CREATE_TAXA_SUMMARY_SQL,
    ):
        con.execute(ddl)

    n_rows = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    logger.info(
        f"DuckDB opened at {db_path!r} — "
        f"{n_rows:,} existing rows in predictions"
    )
    return con


def write_chemicals(
    con: duckdb.DuckDBPyConnection,
    smiles_list: list[str],
) -> int:
    """
    Compute RDKit descriptors for each SMILES and upsert into the
    ``chemicals`` table.  Rows whose ``smiles`` primary key already
    exists are left unchanged.

    Args:
        con: Open DuckDB connection.
        smiles_list: List of SMILES strings (may include duplicates; they
            are deduplicated before the descriptor calculation).

    Returns:
        Number of newly inserted rows.
    """
    from trident.utils.chem_utils import calculate_rdkit_descriptors

    descriptors = calculate_rdkit_descriptors(smiles_list)
    unique_smiles = descriptors.pop("smiles_input")

    rows = [
        {
            "smiles": smi,
            "canonical_smiles": descriptors[smi]["canonical_smiles"],
            "inchi": descriptors[smi]["inchi"],
            "inchikey": descriptors[smi]["inchikey"],
        }
        for smi in unique_smiles
    ]

    if not rows:
        return 0

    chem_pl = pl.DataFrame(
        {
            "smiles": [r["smiles"] for r in rows],
            "canonical_smiles": [r["canonical_smiles"] for r in rows],
            "inchi": [r["inchi"] for r in rows],
            "inchikey": [r["inchikey"] for r in rows],
        }
    )

    n_before = con.execute("SELECT COUNT(*) FROM chemicals").fetchone()[0]
    con.execute(
        "INSERT INTO chemicals SELECT * FROM chem_pl ON CONFLICT DO NOTHING"
    )
    n_after = con.execute("SELECT COUNT(*) FROM chemicals").fetchone()[0]
    inserted = n_after - n_before
    logger.info(f"chemicals table: +{inserted:,} rows ({n_after:,} total)")
    return inserted


def init_taxonomy_tables(
    con: duckdb.DuckDBPyConnection,
    mapping_df: pd.DataFrame,
    taxid2sciname: Optional[dict] = None,
    taxid2commonname: Optional[dict] = None,
) -> None:
    """
    Populate the ``species``, ``lineage``, and ``settings`` tables from
    the inference mapping DataFrame and optional name lookup dicts.

    Safe to call multiple times — uses ``CREATE OR REPLACE TABLE`` so the
    tables are rebuilt from scratch each run (6,700 rows, <1 s).

    Args:
        con: Open DuckDB connection.
        mapping_df: DataFrame from
            ``taxa_assigned_units_effects_durations_*.csv``.  Expected
            columns: ``NCBI_rank_species``, ``NCBI_rank_superkingdom``,
            ``NCBI_rank_kingdom``, ``NCBI_rank_phylum``,
            ``NCBI_rank_subphylum``, ``NCBI_rank_class``,
            ``NCBI_rank_order``, ``NCBI_rank_family``, ``NCBI_rank_genus``,
            ``species_group_corrected``, ``assigned_acute_effect``,
            ``assigned_acute_duration``, ``assigned_conc_unit``.
        taxid2sciname: Dict mapping ncbi_taxid (str) → latin name.
        taxid2commonname: Dict mapping ncbi_taxid (str) → common name.
    """
    def _taxid_col(series: "pd.Series") -> list:
        """Convert a numeric taxid column to str|None, e.g. 33208.0 → '33208', NaN → None."""
        return [str(int(v)) if pd.notna(v) else None for v in series]

    def _str_col(series: "pd.Series") -> list:
        """Convert a string-like column to str|None, replacing NaN with None."""
        return [str(v) if pd.notna(v) else None for v in series]

    taxids = _taxid_col(mapping_df["NCBI_rank_species"])

    # ---- species table ------------------------------------------------
    species_pl = pl.DataFrame(
        {
            "ncbi_taxid": taxids,
            "sciname": [
                (taxid2sciname or {}).get(t, "") if t is not None else ""
                for t in taxids
            ],
            "common_name": [
                (taxid2commonname or {}).get(t, "") if t is not None else ""
                for t in taxids
            ],
            "species_group": _str_col(mapping_df["species_group_corrected"]),
        }
    )
    con.execute("CREATE OR REPLACE TABLE species AS SELECT * FROM species_pl")

    # ---- lineage table ------------------------------------------------
    lineage_pl = pl.DataFrame(
        {
            "ncbi_taxid": taxids,
            "superkingdom": _taxid_col(mapping_df["NCBI_rank_superkingdom"]),
            "kingdom":      _taxid_col(mapping_df["NCBI_rank_kingdom"]),
            "phylum":       _taxid_col(mapping_df["NCBI_rank_phylum"]),
            "subphylum":    _taxid_col(mapping_df["NCBI_rank_subphylum"]),
            "taxon_class":  _taxid_col(mapping_df["NCBI_rank_class"]),
            "taxon_order":  _taxid_col(mapping_df["NCBI_rank_order"]),
            "family":       _taxid_col(mapping_df["NCBI_rank_family"]),
            "genus":        _taxid_col(mapping_df["NCBI_rank_genus"]),
        }
    )
    con.execute("CREATE OR REPLACE TABLE lineage AS SELECT * FROM lineage_pl")

    # ---- settings table -----------------------------------------------
    # duration in the CSV is a whole-number float (e.g. 96.0 h) or NaN; store as integer hours
    # Use round() before int() to guard against floating-point representation issues
    # (e.g. 48 may be stored as 47.9999... in the CSV).
    raw_duration = [int(round(v)) if pd.notna(v) else None for v in mapping_df["assigned_acute_duration"]]

    settings_pl = pl.DataFrame(
        {
            "ncbi_taxid": taxids,
            "effect":    _str_col(mapping_df["assigned_acute_effect"]),
            "duration":  raw_duration,
            "conc_unit": _str_col(mapping_df["assigned_conc_unit"]),
        }
    ).with_columns(pl.col("duration").cast(pl.Int32))
    con.execute("CREATE OR REPLACE TABLE settings AS SELECT * FROM settings_pl")

    logger.info(
        f"Taxonomy tables initialised — "
        f"{len(species_pl):,} species rows"
    )


def write_predictions(
    con: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
    table_name: str = PREDICTIONS_TABLE,
    overwrite: bool = False,
) -> int:
    """
    Bulk-insert a polars DataFrame into the predictions table.

    Rows whose ``(smiles, ncbi_taxid, effect, duration, conc_unit)`` primary
    key already exists are silently skipped if ``overwrite`` is False, so
    that resumed runs do not produce duplicates.

    Args:
        con: Open DuckDB connection (from :func:`init_predictions_db`).
        df: Polars DataFrame.  Must contain at minimum the columns listed
            in :data:`PREDICTION_COLUMNS`; extra columns are ignored.
        table_name: Target table name.
        overwrite: If True, existing rows with the same primary key will be overwritten.
            If False, existing rows will be left unchanged and the new rows skipped.
    Returns:
        Number of newly inserted rows (duplicates excluded).
    """
    if df.is_empty():
        return 0

    # Select and order columns explicitly so the INSERT column list matches
    df = df.select(PREDICTION_COLUMNS)

    n_before = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if overwrite:
        # DuckDB requires explicit column list for DO UPDATE SET
        pk_cols = ("smiles", "ncbi_taxid", "effect", "duration", "conc_unit")
        update_cols = [c for c in PREDICTION_COLUMNS if c not in pk_cols]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        con.execute(
            f"INSERT INTO {table_name} SELECT * FROM df "
            f"ON CONFLICT DO UPDATE SET {set_clause}"
        )
    else:
        con.execute(
            f"INSERT INTO {table_name} SELECT * FROM df ON CONFLICT DO NOTHING"
        )
    n_after = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return n_after - n_before


def create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build ART indexes on the predictions and taxonomy tables.

    Should be called **after** all prediction rows have been inserted for
    best throughput — building indexes post-load is significantly faster
    than maintaining them during bulk insert.

    Args:
        con: Open DuckDB connection.
    """
    index_stmts = [
        "CREATE INDEX IF NOT EXISTS idx_pred_taxid  ON predictions(ncbi_taxid)",
        "CREATE INDEX IF NOT EXISTS idx_pred_smiles ON predictions(smiles)",
        "CREATE INDEX IF NOT EXISTS idx_lin_class   ON lineage(taxon_class)",
        "CREATE INDEX IF NOT EXISTS idx_lin_order   ON lineage(taxon_order)",
        "CREATE INDEX IF NOT EXISTS idx_lin_family  ON lineage(family)",
        "CREATE INDEX IF NOT EXISTS idx_sp_group    ON species(species_group)",
    ]
    for stmt in index_stmts:
        con.execute(stmt)
    logger.info("DuckDB ART indexes created")


def build_summary_tables(con: duckdb.DuckDBPyConnection) -> None:
    """
    Materialise ``chemical_summary`` and ``taxa_summary`` tables.

    Both tables are computed via DuckDB's UNPIVOT to melt the four
    endpoint columns (EC50, EC10, NOEC, LOEC) into rows, then grouped to
    compute mean, median, std, 10th and 90th percentiles.

    Should be called after all predictions are loaded and indexes created.
    At 67M rows, expect 30–120 s of runtime.

    Args:
        con: Open DuckDB connection.
    """
    logger.info("Building chemical_summary table…")
    con.execute("""\
        CREATE OR REPLACE TABLE chemical_summary AS
        SELECT
            smiles,
            endpoint,
            effect,
            COUNT(*)                          AS n_species,
            AVG(value)                        AS mean,
            MEDIAN(value)                     AS median,
            STDDEV_SAMP(value)                AS std,
            QUANTILE_CONT(value, 0.10)        AS p10,
            QUANTILE_CONT(value, 0.90)        AS p90
        FROM (
            UNPIVOT predictions
            ON EC50, EC10, NOEC, LOEC
            INTO NAME endpoint VALUE value
        )
        WHERE value IS NOT NULL
        GROUP BY smiles, endpoint, effect
    """)
    n_chem = con.execute("SELECT COUNT(*) FROM chemical_summary").fetchone()[0]
    logger.info(f"chemical_summary: {n_chem:,} rows")

    logger.info("Building taxa_summary table…")
    con.execute("""\
        CREATE OR REPLACE TABLE taxa_summary AS
        SELECT
            ncbi_taxid,
            endpoint,
            effect,
            COUNT(*)                          AS n_chemicals,
            AVG(value)                        AS mean,
            MEDIAN(value)                     AS median,
            STDDEV_SAMP(value)                AS std,
            QUANTILE_CONT(value, 0.10)        AS p10,
            QUANTILE_CONT(value, 0.90)        AS p90
        FROM (
            UNPIVOT predictions
            ON EC50, EC10, NOEC, LOEC
            INTO NAME endpoint VALUE value
        )
        WHERE value IS NOT NULL
        GROUP BY ncbi_taxid, endpoint, effect
    """)
    n_taxa = con.execute("SELECT COUNT(*) FROM taxa_summary").fetchone()[0]
    logger.info(f"taxa_summary: {n_taxa:,} rows")


def get_row_count(
    con: duckdb.DuckDBPyConnection,
    table_name: str = PREDICTIONS_TABLE,
) -> int:
    """Return the total number of rows in a table."""
    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
