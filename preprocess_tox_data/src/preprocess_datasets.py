import pandas as pd
from utils import (
    create_species_group_from_common,
    create_species_latin_from_common,
    essential_preprocessing,
    map_effects,
    map_effects_using_endpoint,
    map_endpoints,
    print_nunique_cas,
    standardize_colnames,
    standardize_concentration_units,
    standardize_duration_units,
    standardize_species_common_name,
)


def preprocess_acute_toxicity_data(
    df: pd.DataFrame,
    rename_dict: dict,
):
    df = df.copy()
    df = standardize_colnames(df, rename_dict=rename_dict)
    print_nunique_cas(df, "initial")

    # create species information from common names
    df["species_common_name"] = df.species_common_name.apply(
        standardize_species_common_name
    )
    df["species_latin_name"] = df.species_common_name.apply(
        create_species_latin_from_common
    )
    df["species_group"] = df.species_common_name.apply(create_species_group_from_common)
    df = essential_preprocessing(df.copy())

    # Map endpoints and effects
    df["effect"] = df.apply(
        lambda x: map_effects_using_endpoint(
            endpoint=x["endpoint"], effect=x["effect"]
        ),  # Maps LD and LC to MOR
        axis=1,
    )
    df["endpoint"] = (
        df["endpoint"].copy().apply(map_endpoints)
    )  # Standardizes endpoints
    df["effect"] = df["effect"].copy().apply(map_effects)  # Standardizes effects

    # Column removal
    df = df[
        [
            "SK_unique_id",
            "data_source",
            "species_group",
            "species_common_name",
            "species_latin_name",
            "admin_route1",
            "admin_route2",
            "cas",
            "chemical_name",
            "conc_unit",
            "conc",
            "conc_sign",
            "duration_unit",
            "duration",
            "effect",
            "endpoint",
        ]
    ]

    # Administration route
    df["administration_route"] = df["admin_route2"].combine_first(df["admin_route1"])
    df.drop(columns=["admin_route1", "admin_route2"], inplace=True)
    print_nunique_cas(df, "admin route cleaned")

    # Fix units
    # Concentration
    df = standardize_concentration_units(df)
    # Duration
    df = standardize_duration_units(df)

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "cas",
            "endpoint",
            "effect",
            "species_common_name",
            "species_latin_name",
            "conc",
            "conc_sign",
            "conc_unit",
            "duration",
            "duration_unit",
            "administration_route",
        ]
    )
    print_nunique_cas(df, "removed duplicates")

    return df


def preprocess_carc_toxicity_data(
    df: pd.DataFrame,
    rename_dict: dict,
):
    df = df.copy()
    df = standardize_colnames(df, rename_dict=rename_dict)
    print_nunique_cas(df, "initial")

    # create species information from common names
    df["species_common_name"] = df.species_common_name.apply(
        standardize_species_common_name
    )
    df["species_latin_name"] = df.species_common_name.apply(
        create_species_latin_from_common
    )
    df["species_group"] = df.species_common_name.apply(create_species_group_from_common)
    df = essential_preprocessing(df.copy())

    # Map endpoints and effects
    df["endpoint"] = (
        df["endpoint"].copy().apply(map_endpoints)
    )  # Standardizes endpoints
    df["effect"] = "car"

    # Column removal
    df = df[
        [
            "SK_unique_id",
            "data_source",
            "species_group",
            "species_common_name",
            "species_latin_name",
            "admin_route1",
            "admin_route2",
            "cas",
            "chemical_name",
            "conc_unit",
            "conc",
            "conc_sign",
            "duration_unit",
            "duration",
            "effect",
            "endpoint",
        ]
    ]

    # Admin route
    df["administration_route"] = df["admin_route2"].combine_first(df["admin_route1"])
    df.drop(columns=["admin_route1", "admin_route2"], inplace=True)
    print_nunique_cas(df, "admin route cleaned")

    # Fix units
    # Concentration
    df = standardize_concentration_units(df)
    # Duration
    df = standardize_duration_units(df)

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "cas",
            "endpoint",
            "effect",
            "species_common_name",
            "species_latin_name",
            "conc",
            "conc_sign",
            "conc_unit",
            "duration",
            "duration_unit",
            "administration_route",
        ]
    )
    print_nunique_cas(df, "removed duplicates")

    return df


def preprocess_repro_toxicity_data(df: pd.DataFrame, rename_dict: dict):
    df = df.copy()
    df = standardize_colnames(df, rename_dict=rename_dict)
    print_nunique_cas(df, "initial")

    # create species information from common names
    df["species_common_name"] = df.species_common_name.apply(
        standardize_species_common_name
    )
    df["species_latin_name"] = df.species_common_name.apply(
        create_species_latin_from_common
    )
    df["species_group"] = df.species_common_name.apply(create_species_group_from_common)
    df = essential_preprocessing(df.copy())

    # Endpoint standardization
    df["effect"] = df.apply(
        lambda x: map_effects_using_endpoint(
            endpoint=x["endpoint"], effect=x["effect"]
        ),  # Maps LD and LC to MOR
        axis=1,
    )
    df["endpoint"] = (
        df["endpoint"].copy().apply(map_endpoints)
    )  # Standardizes endpoints
    df["effect"] = df["effect"].copy().apply(map_effects)  # Standardizes effects
    df["effect"] = df["effect"].fillna("REP")  # Fill NaN effects with REP

    # Column removal
    df = df[
        [
            "SK_unique_id",
            "data_source",
            "species_group",
            "species_common_name",
            "species_latin_name",
            "admin_route2",
            "cas",
            "chemical_name",
            "conc_unit",
            "conc",
            "conc_sign",
            "duration_unit",
            "duration",
            "effect",
            "endpoint",
        ]
    ]

    # Admin route
    df["administration_route"] = df["admin_route2"]
    df.drop(columns=["admin_route2"], inplace=True)
    print_nunique_cas(df, "admin route cleaned")

    # Fix units
    # Concentration
    df = standardize_concentration_units(df)
    # Duration
    df = standardize_duration_units(df)

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "cas",
            "endpoint",
            "effect",
            "species_common_name",
            "species_latin_name",
            "conc",
            "conc_sign",
            "conc_unit",
            "duration",
            "duration_unit",
            "administration_route",
        ]
    )
    print_nunique_cas(df, "removed duplicates")

    return df


def preprocess_rtecs_toxicity_data(
    df: pd.DataFrame,
    rename_dict: dict,
):
    df = df.copy()
    df = standardize_colnames(df, rename_dict=rename_dict)
    print_nunique_cas(df, "initial")

    # create species information from common names
    df["species_common_name"] = df.species_common_name.apply(
        standardize_species_common_name
    )
    df["species_latin_name"] = df.species_common_name.apply(
        create_species_latin_from_common
    )
    df["species_group"] = df.species_common_name.apply(create_species_group_from_common)
    df = essential_preprocessing(df.copy())

    # No conc sign here so we set it to '='
    df["conc_sign"] = df.conc_sign.fillna("=")

    # Combine effect columns into one
    df["effect"] = df[["effect1", "effect2"]].apply(
        lambda row: (
            " ".join([str(x) for x in row if pd.notnull(x)])
            if row.notnull().any()
            else None
        ),
        axis=1,
    )
    # Map endpoints and effects
    df["effect"] = df.apply(
        lambda x: map_effects_using_endpoint(
            endpoint=x["endpoint"], effect=x["effect"]
        ),  # Maps LD and LC to MOR
        axis=1,
    )
    df["endpoint"] = (
        df["endpoint"].copy().apply(map_endpoints)
    )  # Standardizes endpoints
    df["effect"] = df["effect"].copy().apply(map_effects)  # Standardizes effects

    # Column removal
    df = df[
        [
            "SK_unique_id",
            "data_source",
            "species_group",
            "species_common_name",
            "species_latin_name",
            "admin_route2",
            "cas",
            "chemical_name",
            "conc_unit",
            "conc",
            "conc_sign",
            "duration_unit",
            "duration",
            "effect",
            "endpoint",
        ]
    ]

    # Admin route
    df["administration_route"] = df["admin_route2"]
    df.drop(columns=["admin_route2"], inplace=True)
    print_nunique_cas(df, "admin route cleaned")

    # Fix units
    # Concentration and duration units
    df = standardize_concentration_units(df)
    df = standardize_duration_units(df)

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "cas",
            "endpoint",
            "effect",
            "species_common_name",
            "species_latin_name",
            "conc",
            "conc_sign",
            "conc_unit",
            "duration",
            "duration_unit",
            "administration_route",
        ]
    )
    print_nunique_cas(df, "removed duplicates")

    return df


def preprocess_aqter_toxicity_data(
    df,
    rename_dict: dict,
):
    df = df.copy()
    df = standardize_colnames(df, rename_dict=rename_dict)
    print_nunique_cas(df, "initial")
    df = essential_preprocessing(df.copy())

    df = df.dropna(
        subset=["duration"]
    )  # We filter on duration for this data set since tests are varied

    # Map endpoints and effects
    df["effect"] = df.apply(
        lambda x: map_effects_using_endpoint(
            endpoint=x["endpoint"], effect=x["effect"]
        ),  # Maps LD and LC to MOR
        axis=1,
    )
    df["endpoint"] = (
        df["endpoint"].copy().apply(map_endpoints)
    )  # Standardizes endpoints
    df["effect"] = df["effect"].copy().apply(map_effects)  # Standardizes effects

    # Column removal
    df = df[
        [
            "SK_unique_id",
            "data_source",
            "species_group",
            "species_common_name",
            "species_latin_name",
            "organism_lifestage",
            "cas",
            "chemical_name",
            "conc_unit",
            "conc",
            "conc_sign",
            "duration_unit",
            "duration",
            "effect",
            "endpoint",
            "administration_route",
            "organism_habitat",
        ]
    ]

    # Fix units
    # Concentration
    df = standardize_concentration_units(df)
    # Duration
    df = standardize_duration_units(df)

    # Drop duplicates
    df = df.drop_duplicates(
        subset=[
            "cas",
            "endpoint",
            "effect",
            "species_common_name",
            "species_latin_name",
            "conc",
            "conc_sign",
            "conc_unit",
            "duration",
            "duration_unit",
            "organism_lifestage",
            "administration_route",
        ]
    )
    print_nunique_cas(df, "removed duplicates")

    return df


def preprocess_envirotox_data(
    df: pd.DataFrame,
    rename_dict: dict,
) -> pd.DataFrame:
    # TODO

    return df
