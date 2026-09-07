import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(
    override=True, interpolate=True
)  # Load environment variables from .env file
DATA_LS_MAPPING = Path(os.getenv("DATA_LS_MAPPING"))
DATA_ADMIN_ROUTE_MAPPING = Path(os.getenv("DATA_ADMIN_ROUTE_MAPPING"))
DATA_STANDARDIZE_COMMON_LOWER = Path(os.getenv("DATA_STANDARDIZE_COMMON_LOWER"))
DATA_TRANSLATE_COMMON_TO_GROUP_LOWER = Path(
    os.getenv("DATA_TRANSLATE_COMMON_TO_GROUP_LOWER")
)
DATA_TRANSLATE_COMMON_TO_LATIN_LOWER = Path(
    os.getenv("DATA_TRANSLATE_COMMON_TO_LATIN_LOWER")
)


def format_cas_number(cas):
    """
    Converts CAS numbers into a standardized format.
    Handles NaN, integers, floats, missing dashes, and faulty inputs.
    """
    # Handle NaN values
    if pd.isna(cas) or cas is None:
        return np.nan  # Preserve NaN values

    # Convert floats to integers first
    try:
        cas = int(cas)
    except (ValueError, TypeError):
        pass

    # Convert everything to a string and remove spaces
    cas = str(cas).strip()

    # Remove any non-numeric characters except dashes
    cas = re.sub(r"[^0-9-]", "", cas)

    # If the CAS number is fully numeric and has 5+ digits but no dashes, try to format it
    if cas.isdigit() and len(cas) >= 5:
        cas = f"{cas[:-3]}-{cas[-3:-1]}-{cas[-1]}"

    # Split into parts to verify structure
    parts = cas.split("-")

    if len(parts) != 3:
        return np.nan  # Invalid CAS format

    first, second, check_digit = parts

    # Ensure correct part lengths (first can be variable length, second = 2, last = 1)
    if not (len(second) == 2 and len(check_digit) == 1):
        return np.nan

    # Validate the check digit
    full_number = first + second
    expected_check_digit = (
        sum(int(digit) * (i + 1) for i, digit in enumerate(reversed(full_number))) % 10
    )

    if int(check_digit) != expected_check_digit:
        return np.nan  # Invalid check digit

    return f"{first}-{second}-{check_digit}"  # Return properly formatted CAS number


def print_nunique_cas(df, filter):
    print("\n")
    print(f"N unique CAS after filter {filter}: {df.cas.nunique()}")
    print(f"N rows after filter {filter}: {len(df)}")


def convert_dtype_to_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def standardize_colnames(df, rename_dict):
    df.rename(columns=rename_dict, inplace=True)
    return df


def standardize_species_common_name(
    common_name,
    path_to_species_mapping: Path = DATA_STANDARDIZE_COMMON_LOWER,
):
    # If common_name is NaN, return None
    if pd.isna(common_name) or common_name is None:
        return None
    # replace repeated pipe names like|mouse|mouse with just mouse
    common_name = re.sub(r"^\|(\w+)\|\1$", r"\1", common_name)
    with open(path_to_species_mapping, "r") as f:
        species_dict = json.load(f)
    # If species name is in the dictionary we map it
    try:
        return species_dict[common_name]
    except KeyError:
        return common_name


def create_species_latin_from_common(
    common_name,
    path_to_species_mapping: Path = DATA_TRANSLATE_COMMON_TO_LATIN_LOWER,
):
    with open(path_to_species_mapping, "r") as f:
        species_dict = json.load(f)
    try:
        return species_dict[common_name]
    except KeyError:
        return None


def create_species_group_from_common(
    common_name,
    path_to_species_mapping: Path = DATA_TRANSLATE_COMMON_TO_GROUP_LOWER,
):
    with open(path_to_species_mapping, "r") as f:
        species_dict = json.load(f)
    try:
        return species_dict[common_name]
    except KeyError:
        return None


def map_effects(effect):
    if pd.isnull(effect) or not isinstance(effect, str):
        return None
    effect = effect.lower().strip()
    # If effect starts with mor or leth, return MOR
    if effect.startswith("mor") or effect.startswith("leth"):
        return "mor"
    # If effect starts with car, tum or mut, return CAR
    if effect.startswith("car") or effect.startswith("tum") or effect.startswith("mut"):
        return "car"
    # If effect starts with beh, fbh or avo, return BEH
    if effect.startswith("beh") or effect.startswith("fbh") or effect.startswith("avo"):
        return "beh"
    # If effect starts with rep, return REP
    if effect.startswith("rep"):
        return "rep"
    # If effect starts with in vitro, return IN VITRO
    if effect.startswith("in vitro"):
        return "in vitro"

    # If no effect has been matched yet make more generous pass
    if "mor" in effect or "leth" in effect or "mort" in effect:
        return "mor"
    if "tumor" in effect or "mutation data" in effect:
        return "car"
    if "beh" in effect or "fbh" in effect or "behaviour" in effect or "avo" in effect:
        return "beh"
    if "rep" in effect:
        return "rep"

    # If no effect has been matched, return effect
    else:
        return effect.lower()


def map_effects_using_endpoint(endpoint, effect):
    if pd.isnull(effect) or not isinstance(effect, str):
        if isinstance(endpoint, str):
            if endpoint.lower().startswith(("lc", "ld", "mor", "leth")):
                return "mor"
    return effect


def map_endpoints(endpoint):
    import re

    if not isinstance(endpoint, str):
        return None
    endpoint = endpoint.lower().strip()
    # Hard match for common endpoints
    if endpoint in ["ec50", "lc50", "ic50", "id50", "ld50", "ed50"]:
        return "ec50"
    if endpoint in ["ec10", "lc10", "ic10", "ec15", "ec05", "ec16", "ec17", "ec12"]:
        return "ec10"
    if endpoint in ["noael", "noaec", "noel", "noec"]:
        return "noec"
    if endpoint in ["loael", "loaec", "loec", "loel"]:
        return "loec"

    # Regex matching
    # Define allowed prefixes in regex
    pattern = re.compile(
        r"^(EC|LC|IC|ID|LD|ED)(\d{2})$", re.IGNORECASE
    )  # Only match if it starts with EC, LC, IC, ID, LT, LD or ED followed by 2 digits
    match = pattern.match(endpoint)
    if match:
        digits = match.group(2)
        if digits[0] == "1":
            return "ec10"
        elif digits[0] == "5":
            return "ec50"
        elif digits[0] == "0" and int(digits[1]) >= 5:
            return "ec10"
        # Lägg till typ
    pattern = re.compile(
        r"^(EC|LC|IC|ID|LD|ED)(\d{1})$", re.IGNORECASE
    )  # Only match if it starts with EC, LC, IC, ID, LT, LD or ED followed by 1 digits
    match = pattern.match(endpoint)
    if match:
        digits = match.group(2)
        if int(digits) >= 5:  # EC5 is the same as EC05
            return "ec10"
    else:
        return endpoint.lower()


def standardize_concentration_units(df):
    def convert(row):
        if pd.isna(row["conc_unit"]) or row["conc_unit"] is None:
            return row["conc"], np.nan
        if not isinstance(row["conc"], (int, float)) or pd.isna(row["conc"]):
            return np.nan, np.nan

        unit = row["conc_unit"].strip().lower().replace("ai ", "")
        value = row["conc"]

        # Liquid concentrations to mg/L
        if unit in [
            "mg/l",
            "ai mg/l",
            "ae mg/l",
            "mg/dm3",
            "mg/l diet",
            "mg/L drinking water",
        ]:
            return value, "mg/l"
        elif unit in ["mg/m3"]:
            return value / 1000, "mg/l"
        elif unit in ["ug/l", "ai ug/l", "ug/dm3", "ae ug/l", "ug/l diet", "Âµg/l"]:
            return value / 1000, "mg/l"
        elif unit in ["ng/l", "ai ng/l"]:
            return value / 1e6, "mg/l"
        elif unit in ["ug/ml", "ai ug/ml"]:
            return value, "mg/l"
        elif unit in ["mg/ml", "ai mg/ml"]:
            return value * 1000, "mg/l"
        elif unit == "g/l":
            return value * 1000, "mg/l"
        elif unit == "ai g/l":
            return value * 1000, "mg/l"
        elif unit in ["g/100 l", "g/100l"]:
            return value * 10, "mg/l"
        elif unit in ["ng/ml"]:
            return value / 1000, "mg/l"
        elif unit == "ug/m3":
            return value / 1e6, "mg/l"
        elif unit == "lb/100 gal":
            return value * 1198.26, "mg/l"

        # Percent
        elif unit in ["%", "ai %", "% v/v", "% w/w", "% diet", "% w/v"]:
            return value, "%"

        # ppm
        elif unit in ["ml/l"]:
            return value * 1000, "ppm"
        elif unit in ["ul/l"]:
            return value, "ppm"
        elif unit in ["ppb", "ppb diet"]:
            return value / 1000, "ppm"
        elif unit in ["ppm", "ai ppm", "ppm diet", "ppmw"]:
            return value, "ppm"
        elif unit == "fl oz/100 gal":
            return value * 0.078125 * 1000, "ppm"
        elif unit == "ml/100 L":
            return value * 10, "ppm"

        # Molar concentrations
        elif unit in ["mmol/l", "mm"]:
            return value / 1000, "m"
        elif unit in ["umol/l", "um"]:
            return value / 1e6, "m"
        elif unit in ["nmol/l", "nm"]:
            return value / 1e9, "m"
        elif unit in ["nm"]:
            return value / 1e9, "m"
        elif unit in ["m", "mol/l"]:
            return value, "m"

        # Area concentrations
        elif unit in ["ai kg/ha", "kg/ha", "ae kg/ha", "l/ha", "ai l/ha"]:
            return value * 0.0001, "kg/m2"
        elif unit in ["mg/ha"]:
            return (value / 1e6) * 0.0001, "kg/m2"
        elif unit in ["ai lb/acre", "lb/acre"]:
            return value * 1.12085 * 0.0001, "kg/m2"
        elif unit in ["ai g/ha", "g/ha", "ae g/ha"]:
            return (value / 1000) * 0.0001, "kg/m2"
        elif unit == "ml/ha":
            return (value / 1000) * 0.0001, "kg/m2"
        elif unit == "ug/cm2":
            return (value / 10) * 0.0001, "kg/m2"
        elif unit in ["g/m2", "ai g/m2"]:
            return value / 1000, "kg/m2"
        elif unit in ["oz/acre", "fl oz/acre"]:
            return value * 0.0730778 * 0.0001, "kg/m2"
        elif unit == "mg/cm2":
            return (value * 100) * 0.0001, "kg/m2"
        elif unit == "ng/cm2":
            return (value / 10000) * 0.0001, "kg/m2"
        elif unit in ["gal/acre"]:
            return value * 9.35396 * 0.0001, "kg/m2"
        elif unit in ["oz/1000 ft2"]:
            return value * 3.18326823 * 0.0001, "kg/m2"
        elif unit == "ml/acre":
            return value * 0.00247105 * 0.0001, "kg/m2"
        elif unit == "mg/m2":
            return value / 1e6, "kg/m2"

        # Solid concentrations
        elif unit in ["g/kg", "ai g/kg sd", "g/kg sd"]:
            return value * 1000, "mg/kg"
        elif (
            unit in ["mg/kg", "gm/kg", "ug/g"]
        ):  # 'gm/kg' is a typo for 'mg/kg' based on the distribution of concentrations for the same chemicals
            return value, "mg/kg"
        elif unit == "ug/kg":
            return value / 1000, "mg/kg"
        elif unit in ["mg/kg sediment dw"]:
            return value, "mg/kg"

        # Soil concentrations
        elif unit in ["ai mg/kg dry soil", "ai mg/kg soil"]:
            return value, "mg/kg soil"
        elif unit in ["mg/kg dry soil", "mg/kg soil"]:
            return value, "mg/kg soil"
        elif unit == "ug/g soil":
            return value, "mg/kg soil"
        elif unit in ["ug/kg soil", "ug/kg dry soil", "ug/g dry soil"]:
            return value / 1000, "mg/kg soil"

        # Biota/body weight
        elif unit in ["mg/kg bdwt", "mg/kg dry wt", "mg/kg bw"]:
            return value, "mg/kg bw"
        elif unit in ["g/kg bw"]:
            return value * 1000, "mg/kg bw"
        elif unit in ["ug/g bdwt"]:
            return value, "mg/kg bw"
        elif unit == "ml/kg bw":
            return value, "ml/kg bw"
        elif unit == "ug/kg bdwt":
            return value / 1000, "mg/kg bw"
        elif unit == "mg/g bdwt":
            return value * 1000, "mg/kg bw"
        # Weight per day
        elif unit in ["mg/kg bdwt/d", "mg/kg/d", "mg/kg bw/day"]:
            return value, "mg/kg bw/d"
        elif unit in ["ug/g bdwt/d"]:
            return value * 1000, "mg/kg bw/d"

        # Organism/bioaccumulation
        elif unit in ["ug/org", "ai ug/org", "ug/g org", "ug/bee"]:
            return value / 1000, "mg/org"
        elif unit in ["mg/kg org"]:
            return value, "mg/kg org"
        elif unit in ["ng/org"]:
            return value / 1e6, "mg/org"
        elif unit in ["mg/org"]:
            return value, "mg/org"
        # Per organism per day
        elif unit in ["mg/org/d"]:
            return value, "mg/org/d"
        elif unit in ["ng/org/d"]:
            return value / 1e6, "mg/org/d"
        elif unit in ["ug/org/d", "ai ug/org/d"]:
            return value / 1000, "mg/org/d"

        # Diet
        elif unit in [
            "mg/kg diet",
            "ai mg/kg diet",
            "ug/g dry diet",
            "ug/g wet wt diet",
            "mg/kg dry diet",
        ]:
            return value, "mg/kg diet"
        elif unit == "ug/g diet":
            return value, "mg/kg diet"
        elif unit in ["mg/g diet"]:
            return value * 1000, "mg/kg diet"
        elif unit == "ug/kg diet":
            return value / 1000, "mg/kg diet"

        # Inhalation
        elif unit in ["mg/m3 air"]:
            return value * 0.001, "mg/l air"
        elif unit == "mg/l air":
            return value, "mg/l air"

        else:
            return value, unit  # Return unchanged if no match

    df["conc_unit"] = df.conc_unit.astype(str)
    # Apply the conversion function to each row
    df[["conc", "conc_unit"]] = df.apply(convert, axis=1, result_type="expand")
    return df


def standardize_duration_units(df):
    def convert(row):
        if pd.isna(row["duration_unit"]) or row["duration_unit"] is None:
            return row["duration"], np.nan
        if not isinstance(row["duration"], (int, float)) or pd.isna(row["duration"]):
            return np.nan, np.nan

        unit = row["duration_unit"].strip().lower()
        value = row["duration"]

        if unit in ["h", "hr", "hrs", "hour", "hours"]:
            return value, "h"
        elif unit in ["mi", "min", "minute", "minutes"]:
            return value / 60, "h"
        elif unit in ["d", "day", "days"]:
            return value * 24, "h"
        elif unit in ["wk", "week", "weeks"]:
            return value * 24 * 7, "h"
        elif unit in ["y", "yr", "year", "years"]:
            return value * 24 * 365, "h"
        elif unit == "mo":
            return value * 24 * 30, "h"
        else:
            return value, unit  # Return unchanged if no match

    df["duration_unit"] = df.duration_unit.astype(str)
    # Apply the conversion function to each row
    df[["duration", "duration_unit"]] = df.apply(convert, axis=1, result_type="expand")
    return df


def standardize_nans(df: pd.DataFrame) -> pd.DataFrame:
    # List of "missing" labels, all lower-case
    missing_values = [
        "nr",
        "na",
        "<na>",
        "n/a",
        "nan",
        "other",
        "not reported",
        "not specified",
        "not applicable",
        "miscellaneous",
    ]

    # Replace any cell that matches (case-insensitive)
    return df.map(
        lambda x: (
            np.nan if isinstance(x, str) and x.strip().lower() in missing_values else x
        )
    )


def essential_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print_nunique_cas(df, "initial")
    df["cas"] = df["cas"].apply(format_cas_number)
    df["conc"] = df["conc"].apply(convert_dtype_to_numeric)
    df["duration"] = df["duration"].apply(convert_dtype_to_numeric)

    # Replace "nans" with NaN in all columns and drop nans where essential
    df = standardize_nans(df.copy())
    df = df[
        ~df[["species_group", "species_common_name", "species_latin_name"]]
        .isnull()
        .all(axis=1)
    ]
    print_nunique_cas(df, "no species information")
    df = df.dropna(subset=["conc"])
    print_nunique_cas(df, "conc_value not NaN")
    df = df.dropna(subset=["cas"])
    print_nunique_cas(df, "CAS not NaN")

    # Make all columns with dtype object or string lowercase and stripped of whitespace
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.lower().str.strip()
    return df
