"""
TRIDENT-2 Predictions Database API.

Provides high-level read access to the DuckDB predictions database and
the MilvusLite embedding database produced by batch inference.

DuckDB API
----------
Open a read-only connection with :func:`open_duckdb` (or as a context
manager via :class:`PredictionsDB`), then call the query functions.
All DuckDB functions return :class:`polars.DataFrame`.

Examples
--------
Quick read::

    import db_api

    con = db_api.open_duckdb("predictions.duckdb")
    df = db_api.query_predictions(
        con, species_groups=["fish", "algae"], effects=["MOR"]
    )
    con.close()

Context manager::

    with db_api.PredictionsDB("predictions.duckdb") as db:
        df = db.query_predictions(smiles=["CC=O", "C1=CC=CC=C1"])

"""

from __future__ import annotations

from typing import Optional

import duckdb
import polars as pl
from loguru import logger

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_StrOrList = str | list[str]

# Valid taxon-rank filter parameters understood by query_predictions / query_taxa.
TAXON_RANK_COLS: dict[str, str] = {
    "superkingdoms": "superkingdom",
    "kingdoms": "kingdom",
    "phyla": "phylum",
    "subphyla": "subphylum",
    "classes": "taxon_class",
    "orders": "taxon_order",
    "families": "family",
    "genera": "genus",
}

ENDPOINTS: list[str] = ["EC50", "EC10", "NOEC", "LOEC"]
EFFECTS: list[str] = ["MOR", "GRO", "ITX", "POP"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_list(value: Optional[_StrOrList]) -> Optional[list[str]]:
    """Normalize a string-or-list parameter to list[str] | None."""
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


def _validate_endpoints(endpoints: Optional[list[str]]) -> list[str]:
    if endpoints is None:
        return ENDPOINTS
    bad = set(endpoints) - set(ENDPOINTS)
    if bad:
        raise ValueError(f"Unknown endpoint(s): {bad}. Valid: {ENDPOINTS}")
    return endpoints


# ---------------------------------------------------------------------------
# DuckDB connection helpers
# ---------------------------------------------------------------------------


def open_duckdb(db_path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB predictions database in read-only (default) or read-write mode.

    Args:
        db_path: Path to the ``.duckdb`` file.
        read_only: If ``True`` (default), open in read-only mode — safe for
            concurrent access and for databases managed by an active
            inference run.

    Returns:
        Open :class:`~duckdb.DuckDBPyConnection`.  Close with
        ``con.close()`` when done.
    """
    con = duckdb.connect(db_path, read_only=read_only)
    n = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    logger.debug(f"Opened DuckDB '{db_path}' — {n:,} prediction rows")
    return con


class PredictionsDB:
    """
    Context-manager wrapper around a DuckDB predictions database.

    All :mod:`db_api` query functions are available as methods on this class.

    Args:
        db_path: Path to the ``.duckdb`` file.
        read_only: Open in read-only mode (default ``True``).

    Example::

        with PredictionsDB("predictions.duckdb") as db:
            fish_df = db.query_predictions(species_groups="fish")
            print(fish_df.shape)
    """

    def __init__(self, db_path: str, read_only: bool = True) -> None:
        self._db_path = db_path
        self._read_only = read_only
        self.con: duckdb.DuckDBPyConnection = open_duckdb(db_path, read_only)

    def __enter__(self) -> "PredictionsDB":
        return self

    def __exit__(self, *_) -> None:
        self.con.close()

    # Delegate all module-level functions as methods
    def query_predictions(self, **kwargs) -> pl.DataFrame:
        return query_predictions(self.con, **kwargs)

    def query_predictions_enriched(self, **kwargs) -> pl.DataFrame:
        return query_predictions_enriched(self.con, **kwargs)

    def get_chemical_summary(self, **kwargs) -> pl.DataFrame:
        return get_chemical_summary(self.con, **kwargs)

    def get_taxa_summary(self, **kwargs) -> pl.DataFrame:
        return get_taxa_summary(self.con, **kwargs)

    def get_species(self, **kwargs) -> pl.DataFrame:
        return get_species(self.con, **kwargs)

    def get_lineage(self, **kwargs) -> pl.DataFrame:
        return get_lineage(self.con, **kwargs)

    def get_chemicals(self, **kwargs) -> pl.DataFrame:
        return get_chemicals(self.con, **kwargs)

    def get_settings(self, **kwargs) -> pl.DataFrame:
        return get_settings(self.con, **kwargs)

    def list_species_groups(self) -> list[str]:
        return list_species_groups(self.con)

    def list_effects(self) -> list[str]:
        return list_effects(self.con)

    def list_conc_units(self) -> list[str]:
        return list_conc_units(self.con)

    def count_predictions(self) -> int:
        return count_predictions(self.con)


# ---------------------------------------------------------------------------
# Internal SQL builder
# ---------------------------------------------------------------------------


def _build_where(clauses: list[str]) -> str:
    """Join a list of SQL boolean fragments into a WHERE clause (or empty string)."""
    active = [c for c in clauses if c]
    return ("WHERE " + " AND ".join(active)) if active else ""


def _in_clause(col: str, values: list[str], params: list) -> str:
    """
    Build a safe ``col = ANY(?)`` clause and append the value list to *params*.

    Using array binding instead of f-string interpolation prevents SQL injection.
    """
    params.append(values)
    return f"{col} = ANY(?)"


# ---------------------------------------------------------------------------
# DuckDB query functions
# ---------------------------------------------------------------------------


def query_predictions(
    con: duckdb.DuckDBPyConnection,
    *,
    smiles: Optional[_StrOrList] = None,
    taxids: Optional[_StrOrList] = None,
    effects: Optional[_StrOrList] = None,
    conc_units: Optional[_StrOrList] = None,
    endpoints: Optional[_StrOrList] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query the raw ``predictions`` table with optional filters.

    This is the lowest-latency query function — it touches only the
    ``predictions`` table.  For enriched results with species/lineage
    metadata attached, use :func:`query_predictions_enriched`.

    By default all four endpoints (EC50, EC10, NOEC, LOEC) are returned as
    separate columns.  Pass ``endpoints`` to select a subset (they will be
    returned as columns, not rows).

    Args:
        con: Open DuckDB connection.
        smiles: One or more SMILES strings to filter on.  Must exactly match
            the canonical SMILES stored in the database.
        taxids: One or more NCBI taxon IDs (as strings or ints) to filter on.
        effects: One or more effect codes, e.g. ``["MOR", "GRO"]``.
        conc_units: One or more concentration units, e.g. ``["mg/l"]``.
        endpoints: Subset of ``["EC50", "EC10", "NOEC", "LOEC"]`` to return.
            ``None`` returns all four.
        limit: Maximum number of rows to return.

    Returns:
        Polars DataFrame with predictions columns.
    """
    smiles_list = _as_list(smiles)
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    effect_list = _as_list(effects)
    unit_list = _as_list(conc_units)
    ep_list = _validate_endpoints(_as_list(endpoints))

    ep_cols = ", ".join(ep_list)
    params: list = []
    clauses: list[str] = []

    if smiles_list:
        clauses.append(_in_clause("smiles", smiles_list, params))
    if taxid_list:
        clauses.append(_in_clause("ncbi_taxid", taxid_list, params))
    if effect_list:
        clauses.append(_in_clause("effect", effect_list, params))
    if unit_list:
        clauses.append(_in_clause("conc_unit", unit_list, params))

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"SELECT smiles, ncbi_taxid, effect, conc_unit, {ep_cols} FROM predictions {where} {limit_clause}"
    return con.execute(sql, params).pl()


def query_predictions_enriched(
    con: duckdb.DuckDBPyConnection,
    *,
    smiles: Optional[_StrOrList] = None,
    taxids: Optional[_StrOrList] = None,
    species_groups: Optional[_StrOrList] = None,
    superkingdoms: Optional[_StrOrList] = None,
    kingdoms: Optional[_StrOrList] = None,
    phyla: Optional[_StrOrList] = None,
    subphyla: Optional[_StrOrList] = None,
    classes: Optional[_StrOrList] = None,
    orders: Optional[_StrOrList] = None,
    families: Optional[_StrOrList] = None,
    genera: Optional[_StrOrList] = None,
    effects: Optional[_StrOrList] = None,
    conc_units: Optional[_StrOrList] = None,
    endpoints: Optional[_StrOrList] = None,
    sciname_like: Optional[str] = None,
    include_lineage: bool = False,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query predictions joined with species and (optionally) lineage metadata.

    Supports all taxonomic filter levels — from superkingdom down to genus —
    as well as species group, effect, concentration unit, and SMILES.

    The result always includes the species metadata columns ``sciname``,
    ``common_name``, and ``species_group``.  Pass ``include_lineage=True``
    to also attach the full 8-level taxonomic lineage.

    Args:
        con: Open DuckDB connection.
        smiles: Filter by one or more canonical SMILES strings.
        taxids: Filter by NCBI taxon ID(s).
        species_groups: Filter by coarse species group(s), e.g.
            ``["fish", "algae", "crustaceans"]``.
        superkingdoms: Filter by superkingdom taxon ID(s).
        kingdoms: Filter by kingdom taxon ID(s).
        phyla: Filter by phylum taxon ID(s).
        subphyla: Filter by subphylum taxon ID(s).
        classes: Filter by class taxon ID(s) (``taxon_class`` column).
        orders: Filter by order taxon ID(s) (``taxon_order`` column).
        families: Filter by family taxon ID(s).
        genera: Filter by genus taxon ID(s).
        effects: Filter by effect code(s), e.g. ``"MOR"``.
        conc_units: Filter by concentration unit(s), e.g. ``"mg/l"``.
        endpoints: Subset of endpoints to include as columns.  Default: all four.
        sciname_like: SQL ``ILIKE`` pattern matched against ``species.sciname``,
            e.g. ``"Daphnia%"``.
        include_lineage: If ``True``, attach all lineage rank columns to the result.
        limit: Maximum number of rows to return.

    Returns:
        Polars DataFrame with prediction columns plus species (and optionally
        lineage) metadata.
    """
    smiles_list = _as_list(smiles)
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    effect_list = _as_list(effects)
    unit_list = _as_list(conc_units)
    sg_list = _as_list(species_groups)
    ep_list = _validate_endpoints(_as_list(endpoints))

    ep_cols = "".join(f", p.{ep}" for ep in ep_list)
    lineage_cols = (
        ", l.superkingdom, l.kingdom, l.phylum, l.subphylum, "
        "l.taxon_class, l.taxon_order, l.family, l.genus"
        if include_lineage
        else ""
    )

    # Build taxon-rank filters that require a lineage JOIN
    taxon_rank_filters: dict[str, list[str]] = {}
    for param_name, col_name in TAXON_RANK_COLS.items():
        val = locals().get(param_name)
        if val is not None:
            taxon_rank_filters[col_name] = _as_list(val)  # type: ignore[arg-type]

    needs_lineage = include_lineage or bool(taxon_rank_filters)
    lineage_join = (
        "JOIN lineage l ON p.ncbi_taxid = l.ncbi_taxid" if needs_lineage else ""
    )

    params: list = []
    clauses: list[str] = []

    if smiles_list:
        clauses.append(_in_clause("p.smiles", smiles_list, params))
    if taxid_list:
        clauses.append(_in_clause("p.ncbi_taxid", taxid_list, params))
    if effect_list:
        clauses.append(_in_clause("p.effect", effect_list, params))
    if unit_list:
        clauses.append(_in_clause("p.conc_unit", unit_list, params))
    if sg_list:
        clauses.append(_in_clause("s.species_group", sg_list, params))
    if sciname_like is not None:
        params.append(sciname_like)
        clauses.append("s.sciname ILIKE ?")

    for col_name, values in taxon_rank_filters.items():
        clauses.append(_in_clause(f"l.{col_name}", values, params))

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""\
SELECT
    p.smiles,
    p.ncbi_taxid,
    s.sciname,
    s.common_name,
    s.species_group,
    p.effect,
    p.conc_unit
    {ep_cols}
    {lineage_cols}
FROM predictions p
JOIN species s ON p.ncbi_taxid = s.ncbi_taxid
{lineage_join}
{where}
{limit_clause}
"""
    return con.execute(sql, params).pl()


def get_chemical_summary(
    con: duckdb.DuckDBPyConnection,
    *,
    smiles: Optional[_StrOrList] = None,
    endpoints: Optional[_StrOrList] = None,
    effects: Optional[_StrOrList] = None,
    inchikey: Optional[_StrOrList] = None,
    min_n_species: Optional[int] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query the pre-aggregated ``chemical_summary`` table.

    Each row contains aggregate statistics (mean, median, std, p10, p90)
    for a single ``(smiles, endpoint, effect)`` combination across all
    species that have predictions for that chemical.

    Args:
        con: Open DuckDB connection.
        smiles: Filter to specific SMILES string(s).
        endpoints: Filter to specific endpoint(s), e.g. ``["EC50", "NOEC"]``.
        effects: Filter to specific effect code(s).
        inchikey: Filter by InChIKey(s) — requires a JOIN with the
            ``chemicals`` table.
        min_n_species: Minimum number of species required (``n_species >=``).
        limit: Maximum rows to return.

    Returns:
        Polars DataFrame with columns: smiles, endpoint, effect, n_species,
        mean, median, std, p10, p90.  If *inchikey* is specified, also
        includes inchikey, inchi, canonical_smiles columns.
    """
    smiles_list = _as_list(smiles)
    ep_list = _validate_endpoints(_as_list(endpoints))
    effect_list = _as_list(effects)
    inchikey_list = _as_list(inchikey)

    needs_chem_join = inchikey_list is not None
    chem_join = "JOIN chemicals c ON cs.smiles = c.smiles" if needs_chem_join else ""
    extra_cols = ", c.inchikey, c.inchi, c.canonical_smiles" if needs_chem_join else ""

    params: list = []
    clauses: list[str] = []

    if smiles_list:
        clauses.append(_in_clause("cs.smiles", smiles_list, params))
    if ep_list != ENDPOINTS:
        clauses.append(_in_clause("cs.endpoint", ep_list, params))
    if effect_list:
        clauses.append(_in_clause("cs.effect", effect_list, params))
    if inchikey_list:
        clauses.append(_in_clause("c.inchikey", inchikey_list, params))
    if min_n_species is not None:
        clauses.append(f"cs.n_species >= {int(min_n_species)}")

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""\
SELECT cs.smiles, cs.endpoint, cs.effect, cs.n_species,
       cs.mean, cs.median, cs.std, cs.p10, cs.p90
       {extra_cols}
FROM chemical_summary cs
{chem_join}
{where}
{limit_clause}
"""
    return con.execute(sql, params).pl()


def get_taxa_summary(
    con: duckdb.DuckDBPyConnection,
    *,
    taxids: Optional[_StrOrList] = None,
    species_groups: Optional[_StrOrList] = None,
    orders: Optional[_StrOrList] = None,
    families: Optional[_StrOrList] = None,
    classes: Optional[_StrOrList] = None,
    endpoints: Optional[_StrOrList] = None,
    effects: Optional[_StrOrList] = None,
    min_n_chemicals: Optional[int] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query the pre-aggregated ``taxa_summary`` table, with optional
    species / lineage filters.

    Each row contains aggregate statistics for a
    ``(ncbi_taxid, endpoint, effect)`` combination across all chemicals
    for that species.

    Args:
        con: Open DuckDB connection.
        taxids: Filter by NCBI taxon ID(s).
        species_groups: Filter by coarse species group(s).
        orders: Filter by order taxon ID(s).
        families: Filter by family taxon ID(s).
        classes: Filter by class taxon ID(s).
        endpoints: Filter by endpoint name(s).
        effects: Filter by effect code(s).
        min_n_chemicals: Minimum ``n_chemicals`` value.
        limit: Maximum rows to return.

    Returns:
        Polars DataFrame with columns: ncbi_taxid, sciname, common_name,
        species_group, endpoint, effect, n_chemicals, mean, median, std,
        p10, p90.
    """
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    sg_list = _as_list(species_groups)
    order_list = _as_list(orders)
    family_list = _as_list(families)
    class_list = _as_list(classes)
    ep_list = _validate_endpoints(_as_list(endpoints))
    effect_list = _as_list(effects)

    needs_lineage = any(x is not None for x in (order_list, family_list, class_list))
    lineage_join = (
        "JOIN lineage l ON ts.ncbi_taxid = l.ncbi_taxid" if needs_lineage else ""
    )

    params: list = []
    clauses: list[str] = []

    if taxid_list:
        clauses.append(_in_clause("ts.ncbi_taxid", taxid_list, params))
    if sg_list:
        clauses.append(_in_clause("s.species_group", sg_list, params))
    if ep_list != ENDPOINTS:
        clauses.append(_in_clause("ts.endpoint", ep_list, params))
    if effect_list:
        clauses.append(_in_clause("ts.effect", effect_list, params))
    if order_list:
        clauses.append(_in_clause("l.taxon_order", order_list, params))
    if family_list:
        clauses.append(_in_clause("l.family", family_list, params))
    if class_list:
        clauses.append(_in_clause("l.taxon_class", class_list, params))
    if min_n_chemicals is not None:
        clauses.append(f"ts.n_chemicals >= {int(min_n_chemicals)}")

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""\
SELECT ts.ncbi_taxid, s.sciname, s.common_name, s.species_group,
       ts.endpoint, ts.effect, ts.n_chemicals,
       ts.mean, ts.median, ts.std, ts.p10, ts.p90
FROM taxa_summary ts
JOIN species s ON ts.ncbi_taxid = s.ncbi_taxid
{lineage_join}
{where}
{limit_clause}
"""
    return con.execute(sql, params).pl()


def get_species(
    con: duckdb.DuckDBPyConnection,
    *,
    taxids: Optional[_StrOrList] = None,
    species_groups: Optional[_StrOrList] = None,
    sciname_like: Optional[str] = None,
    common_name_like: Optional[str] = None,
    orders: Optional[_StrOrList] = None,
    families: Optional[_StrOrList] = None,
    classes: Optional[_StrOrList] = None,
    phyla: Optional[_StrOrList] = None,
    include_lineage: bool = False,
    include_settings: bool = False,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query the ``species`` table with optional taxonomy filters.

    Args:
        con: Open DuckDB connection.
        taxids: Filter by NCBI taxon ID(s).
        species_groups: Filter by coarse species group(s).
        sciname_like: SQL ``ILIKE`` pattern on scientific name, e.g.
            ``"Daphnia%"``.
        common_name_like: SQL ``ILIKE`` pattern on common name.
        orders: Filter by order taxon ID(s) (requires lineage join).
        families: Filter by family taxon ID(s) (requires lineage join).
        classes: Filter by class taxon ID(s) (requires lineage join).
        phyla: Filter by phylum taxon ID(s) (requires lineage join).
        include_lineage: Attach all lineage rank columns to the result.
        include_settings: Attach inference settings (effect, duration,
            conc_unit) to the result.
        limit: Maximum rows to return.

    Returns:
        Polars DataFrame with at minimum: ncbi_taxid, sciname, common_name,
        species_group.
    """
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    sg_list = _as_list(species_groups)
    order_list = _as_list(orders)
    family_list = _as_list(families)
    class_list = _as_list(classes)
    phylum_list = _as_list(phyla)

    needs_lineage = include_lineage or any(
        x is not None for x in (order_list, family_list, class_list, phylum_list)
    )
    lineage_join = (
        "JOIN lineage l ON s.ncbi_taxid = l.ncbi_taxid" if needs_lineage else ""
    )
    settings_join = (
        "JOIN settings st ON s.ncbi_taxid = st.ncbi_taxid" if include_settings else ""
    )

    lineage_cols = (
        ", l.superkingdom, l.kingdom, l.phylum, l.subphylum, "
        "l.taxon_class, l.taxon_order, l.family, l.genus"
        if needs_lineage
        else ""
    )
    settings_cols = (
        ", st.effect AS default_effect, st.duration AS default_duration, "
        "st.conc_unit AS default_conc_unit"
        if include_settings
        else ""
    )

    params: list = []
    clauses: list[str] = []

    if taxid_list:
        clauses.append(_in_clause("s.ncbi_taxid", taxid_list, params))
    if sg_list:
        clauses.append(_in_clause("s.species_group", sg_list, params))
    if sciname_like is not None:
        params.append(sciname_like)
        clauses.append("s.sciname ILIKE ?")
    if common_name_like is not None:
        params.append(common_name_like)
        clauses.append("s.common_name ILIKE ?")
    if order_list:
        clauses.append(_in_clause("l.taxon_order", order_list, params))
    if family_list:
        clauses.append(_in_clause("l.family", family_list, params))
    if class_list:
        clauses.append(_in_clause("l.taxon_class", class_list, params))
    if phylum_list:
        clauses.append(_in_clause("l.phylum", phylum_list, params))

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"""\
SELECT s.ncbi_taxid, s.sciname, s.common_name, s.species_group
       {lineage_cols}
       {settings_cols}
FROM species s
{lineage_join}
{settings_join}
{where}
ORDER BY s.sciname
{limit_clause}
"""
    return con.execute(sql, params).pl()


def get_lineage(
    con: duckdb.DuckDBPyConnection,
    *,
    taxids: Optional[_StrOrList] = None,
    orders: Optional[_StrOrList] = None,
    families: Optional[_StrOrList] = None,
    classes: Optional[_StrOrList] = None,
    phyla: Optional[_StrOrList] = None,
    species_groups: Optional[_StrOrList] = None,
) -> pl.DataFrame:
    """
    Query ``lineage`` joined with ``species`` for a readable taxonomy view.

    Args:
        con: Open DuckDB connection.
        taxids: Filter by NCBI taxon ID(s).
        orders: Filter by order taxon ID(s).
        families: Filter by family taxon ID(s).
        classes: Filter by class taxon ID(s).
        phyla: Filter by phylum taxon ID(s).
        species_groups: Filter by coarse species group(s) from the
            ``species`` table.

    Returns:
        Polars DataFrame with sciname, common_name, species_group, and all
        lineage rank columns.
    """
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    sg_list = _as_list(species_groups)
    order_list = _as_list(orders)
    family_list = _as_list(families)
    class_list = _as_list(classes)
    phylum_list = _as_list(phyla)

    params: list = []
    clauses: list[str] = []

    if taxid_list:
        clauses.append(_in_clause("l.ncbi_taxid", taxid_list, params))
    if sg_list:
        clauses.append(_in_clause("s.species_group", sg_list, params))
    if order_list:
        clauses.append(_in_clause("l.taxon_order", order_list, params))
    if family_list:
        clauses.append(_in_clause("l.family", family_list, params))
    if class_list:
        clauses.append(_in_clause("l.taxon_class", class_list, params))
    if phylum_list:
        clauses.append(_in_clause("l.phylum", phylum_list, params))

    where = _build_where(clauses)

    sql = f"""\
SELECT l.ncbi_taxid, s.sciname, s.common_name, s.species_group,
       l.superkingdom, l.kingdom, l.phylum, l.subphylum,
       l.taxon_class, l.taxon_order, l.family, l.genus
FROM lineage l
JOIN species s ON l.ncbi_taxid = s.ncbi_taxid
{where}
ORDER BY l.phylum, l.taxon_class, l.taxon_order, l.family, s.sciname
"""
    return con.execute(sql, params).pl()


def get_chemicals(
    con: duckdb.DuckDBPyConnection,
    *,
    smiles: Optional[_StrOrList] = None,
    inchikey: Optional[_StrOrList] = None,
    inchi_like: Optional[str] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Query the ``chemicals`` table.

    Args:
        con: Open DuckDB connection.
        smiles: Filter by canonical SMILES string(s).
        inchikey: Filter by InChIKey(s).
        inchi_like: SQL ``ILIKE`` pattern on the InChI string.
        limit: Maximum rows to return.

    Returns:
        Polars DataFrame with columns: smiles, canonical_smiles, inchi,
        inchikey.
    """
    smiles_list = _as_list(smiles)
    inchikey_list = _as_list(inchikey)

    params: list = []
    clauses: list[str] = []

    if smiles_list:
        clauses.append(_in_clause("smiles", smiles_list, params))
    if inchikey_list:
        clauses.append(_in_clause("inchikey", inchikey_list, params))
    if inchi_like is not None:
        params.append(inchi_like)
        clauses.append("inchi ILIKE ?")

    where = _build_where(clauses)
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    sql = f"SELECT smiles, canonical_smiles, inchi, inchikey FROM chemicals {where} {limit_clause}"
    return con.execute(sql, params).pl()


def get_settings(
    con: duckdb.DuckDBPyConnection,
    *,
    taxids: Optional[_StrOrList] = None,
    effects: Optional[_StrOrList] = None,
    conc_units: Optional[_StrOrList] = None,
) -> pl.DataFrame:
    """
    Query the ``settings`` table (default inference settings per species).

    Args:
        con: Open DuckDB connection.
        taxids: Filter by NCBI taxon ID(s).
        effects: Filter by effect code(s).
        conc_units: Filter by concentration unit(s).

    Returns:
        Polars DataFrame with columns: ncbi_taxid, sciname, common_name,
        species_group, effect, duration, conc_unit.
    """
    taxid_list = [str(t) for t in _as_list(taxids)] if taxids is not None else None
    effect_list = _as_list(effects)
    unit_list = _as_list(conc_units)

    params: list = []
    clauses: list[str] = []

    if taxid_list:
        clauses.append(_in_clause("st.ncbi_taxid", taxid_list, params))
    if effect_list:
        clauses.append(_in_clause("st.effect", effect_list, params))
    if unit_list:
        clauses.append(_in_clause("st.conc_unit", unit_list, params))

    where = _build_where(clauses)

    sql = f"""\
SELECT st.ncbi_taxid, s.sciname, s.common_name, s.species_group,
       st.effect, st.duration, st.conc_unit
FROM settings st
JOIN species s ON st.ncbi_taxid = s.ncbi_taxid
{where}
ORDER BY s.sciname
"""
    return con.execute(sql, params).pl()


# ---------------------------------------------------------------------------
# Catalogue / enumeration helpers
# ---------------------------------------------------------------------------


def list_species_groups(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the sorted list of distinct species groups."""
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT species_group FROM species WHERE species_group IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]


def list_effects(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the sorted list of distinct effect codes in the predictions table."""
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT effect FROM predictions WHERE effect IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]


def list_conc_units(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the sorted list of distinct concentration units."""
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT conc_unit FROM predictions WHERE conc_unit IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]


def list_endpoints() -> list[str]:
    """Return the four prediction endpoints (static, no DB access required)."""
    return list(ENDPOINTS)


def count_predictions(con: duckdb.DuckDBPyConnection) -> int:
    """Return the total number of rows in the predictions table."""
    return con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def count_chemicals(con: duckdb.DuckDBPyConnection) -> int:
    """Return the number of unique chemicals."""
    return con.execute("SELECT COUNT(*) FROM chemicals").fetchone()[0]


def count_species(con: duckdb.DuckDBPyConnection) -> int:
    """Return the number of species in the database."""
    return con.execute("SELECT COUNT(*) FROM species").fetchone()[0]


def db_summary(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Return a summary dict with row counts for all tables.

    Useful for a quick sanity-check after opening a database.

    Returns:
        Dict with keys: predictions, chemicals, species, lineage, settings,
        chemical_summary, taxa_summary, effects, conc_units, species_groups.
    """
    tables = [
        "predictions",
        "chemicals",
        "species",
        "lineage",
        "settings",
        "chemical_summary",
        "taxa_summary",
    ]
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    counts["effects"] = list_effects(con)
    counts["conc_units"] = list_conc_units(con)
    counts["species_groups"] = list_species_groups(con)
    return counts
