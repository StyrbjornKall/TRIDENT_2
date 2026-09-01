import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle as pkl
import requests
from itertools import chain
from typing import List, Union, Optional
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
pd.options.mode.chained_assignment = None
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Get data for model contains functions for data preprocessing in order to train a transformer model for ecotoxicity prediction.
# These utils functions are imported and used in the main script.


class PreprocessData:
    """
    Class to preprocess data for ecotoxicity prediction.
    """

    tqdm.pandas()

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def filter(
        self,
        endpoint: Optional[Union[List[str], str]] = None,
        effect: Optional[Union[List[str], str]] = None,
        species_groups: Optional[Union[List[str], str]] = None,
        concentration_sign: Optional[Union[List[str], str]] = None,
        concentration_units: Optional[Union[List[str], str]] = None,
        lifestages: Optional[Union[List[str], str]] = None,
        administration_routes: Optional[Union[List[str], str]] = None,
        drop_unknown_ranks: bool = True,
        drop_columns: bool = True,
    ):
        """
        Function to filter out unwanted data from pandas dataframe. Note, requires specific column names to work.

        Columns names should be as follows:
        endpoint: e.g. 'EC50'
        effect: e.g. 'MOR'
        conc: concentration at endpoint
        species_group: e.g. 'fish', 'crustaceans', 'algae', 'rodents'
        duration: duration (h)
        conc_sign: '=', '>', '<'

        """

        # Ensure typing consistency, turn strings into lists
        if isinstance(endpoint, str):
            endpoint = [endpoint]
        if isinstance(effect, str):
            effect = [effect]
        if isinstance(species_groups, str):
            species_groups = [species_groups]
        if isinstance(lifestages, str):
            lifestages = [lifestages]
        if isinstance(administration_routes, str):
            administration_routes = [administration_routes]
        if isinstance(concentration_units, str):
            concentration_units = [concentration_units]
        if isinstance(concentration_sign, str):
            concentration_sign = [concentration_sign]

        if "internal_id" not in self.dataframe.columns:
            self.dataframe.insert(0, "internal_id", range(len(self.dataframe)))

        # Apply filters to dataframe
        def _filter_data(dataframe, column, filter_values):
            if column in dataframe.columns:
                if filter_values is not None:
                    return dataframe[dataframe[column].isin(filter_values)]
                else:
                    return dataframe
            else:
                logger.warning(f"Column '{column}' does not exist in the dataframe.")
            return dataframe

        self.dataframe = _filter_data(self.dataframe, "endpoint", endpoint)
        self.dataframe = _filter_data(self.dataframe, "effect", effect)
        self.dataframe = _filter_data(
            self.dataframe, "species_group_corrected", species_groups
        )
        self.dataframe = _filter_data(self.dataframe, "organism_lifestage", lifestages)
        self.dataframe = _filter_data(
            self.dataframe, "administration_route", administration_routes
        )
        self.dataframe = _filter_data(self.dataframe, "conc_unit", concentration_units)
        self.dataframe = _filter_data(self.dataframe, "conc_sign", concentration_sign)

        if drop_unknown_ranks:
            self.dataframe = self.dataframe[~self.dataframe.NCBI_last_known_rank.isna()]

        if drop_columns:
            for col in self.dataframe.columns:
                try:
                    if col not in [
                        "SMILES",
                        "SMILES_Canonical_RDKit",
                        "endpoint",
                        "effect",
                        "conc",
                        "duration",
                        "species_group",
                        "species_group_corrected",
                        "conc_sign",
                        "chemical_name",
                        "administration_route",
                        "organism_lifestage",
                        "NCBI_last_known_rank",
                        "taxonomic_embedding",
                        "fingerprint",
                        "featurized_input",
                        "onehot_encoding",
                    ]:
                        self.dataframe = self.dataframe.drop(columns=[col])
                except Exception as e:
                    pass

        self.dataframe = self.dataframe.reset_index(drop=True)
        return self.dataframe

    def preprocess(
        self,
        concentration_thresh: Optional[float] = np.inf,
        duration_thresh: Optional[float] = np.inf,
        log_data: bool = True,
        turn_duration_outliers_to_nan: bool = False,
        turn_duration_units_other_than_hours_to_nan: bool = True,
    ) -> pd.DataFrame:

        if turn_duration_units_other_than_hours_to_nan:
            logger.info(
                f"  Turning {(self.dataframe.duration_unit != 'h').sum()} duration values with units other than hours to NaN"
            )
            self.dataframe.loc[self.dataframe.duration_unit != "h", "duration"] = np.nan

        logger.info(
            f"  Turning {(self.dataframe.duration <= 0).sum()} duration values to NaN because they are <= 0"
        )
        self.dataframe.loc[self.dataframe.duration <= 0, "duration"] = np.nan
        logger.info(
            "  Adding 'duration_missing' column to help neural network learn that these values were originally missing"
        )
        self.dataframe["duration_missing"] = (
            self.dataframe["duration"].isna().astype(int)
        )
        logger.info(
            f"  Filling {self.dataframe.duration.isna().sum()} missing duration values with a small value (1e-6) to avoid issues with log-transforming"
        )
        self.dataframe["duration"] = self.dataframe["duration"].fillna(1e-6)
        if duration_thresh is None:
            duration_thresh = np.inf
        logger.info(
            f"  Dropping {(self.dataframe.duration > duration_thresh).sum()} rows where duration > {duration_thresh}"
        )
        self.dataframe = self.dataframe[self.dataframe.duration < duration_thresh]
        logger.info(
            f"  Dropping {(self.dataframe.conc <= 0).sum()} rows where conc <= 0"
        )
        self.dataframe = self.dataframe[self.dataframe.conc > 0]
        logger.info(
            f"  Dropping {(self.dataframe.conc > concentration_thresh).sum()} rows where conc > {concentration_thresh}"
        )
        self.dataframe = self.dataframe[self.dataframe.conc < concentration_thresh]

        if log_data:
            logger.info("  Log-transforming concentration and duration values")
            self.dataframe.conc = np.log10(self.dataframe.conc)
            self.dataframe.duration = np.log10(self.dataframe.duration)

        if turn_duration_outliers_to_nan:
            logger.info("  Turning duration outliers to NaN")

            # Takes the IQR and turns other durations to None
            Q1 = self.dataframe.duration.quantile(0.2)
            Q3 = self.dataframe.duration.quantile(0.8)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.dataframe.duration = self.dataframe.duration.apply(
                lambda x: np.nan if x < lower_bound or x > upper_bound else x
            )

        return self.dataframe

    def get_taxonomic_embeddings(self, path_to_taxonomic_dict: str):
        """
        Function to retrieve taxonomic embeddings from a dictionary containing taxonomic information.
        """
        taxonomic_dict = pd.read_pickle(path_to_taxonomic_dict)
        self.dataframe = self.dataframe.merge(
            taxonomic_dict, left_on="NCBI_last_known_rank", right_index=True, how="left"
        )

        return self.dataframe

    def get_butina_clusters(
        self, path_to_butina_dict: str, radius: float = 0.2
    ) -> pd.DataFrame:
        """
        Function to retrieve butina clusters from a dictionary containing clusters and SMILES.
        """
        butina_dict = pd.read_csv(
            path_to_butina_dict, usecols=["SMILES", f"Cluster_at_cutoff_{radius}"]
        ).set_index("SMILES")
        butina_dict = butina_dict.rename(
            columns={f"Cluster_at_cutoff_{radius}": "butina_cluster"}
        )
        self.dataframe = self.dataframe.merge(
            butina_dict, left_on="SMILES", right_index=True, how="left"
        )
        # Ensure butina_cluster is of type string
        self.dataframe["butina_cluster"] = self.dataframe["butina_cluster"].astype(str)
        # Warn if any SMILES are missing in the butina_dict
        missing_smiles = set(self.dataframe.SMILES.unique()) - set(butina_dict.index)
        if missing_smiles:
            logger.warning(
                f"The following SMILES are missing in the butina dictionary: {missing_smiles}"
            )

        return self.dataframe

    def get_canonical_smiles(
        self,
        canonical: bool = True,
        isomericSmiles: bool = False,
        drop_errorenous_smiles: bool = False,
    ):
        """
        Retrieves Canonical SMILES using RDKit.
        """
        # Extract unique SMILES strings
        unique_smiles = self.dataframe["SMILES"].unique()

        # Create a dictionary mapping original SMILES to canonical SMILES
        smiles_to_canonical = {
            smiles: self.__canonicalize_rdkit(
                smiles,
                canonical=canonical,
                isomericSmiles=isomericSmiles,
                drop_errorenous_smiles=drop_errorenous_smiles,
            )
            for smiles in unique_smiles
        }

        # Map canonical SMILES back to the DataFrame
        self.dataframe["SMILES_Canonical_RDKit"] = self.dataframe["SMILES"].map(
            smiles_to_canonical
        )
        if drop_errorenous_smiles:
            self.dataframe = self.dataframe.dropna(subset=["SMILES_Canonical_RDKit"])

        return self.dataframe

    def __canonicalize_rdkit(
        self,
        smiles,
        canonical: bool = True,
        isomericSmiles: bool = False,
        drop_errorenous_smiles: bool = False,
    ):
        try:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles),
                canonical=canonical,
                isomericSmiles=isomericSmiles,
            )
        except Exception as e:
            if drop_errorenous_smiles:
                return None
            else:
                return smiles


class OneHotEncoder:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def get_onehot_encoding(
        self,
        columns_to_encode: Union[List[str], None] = None,
        encode_None: bool = False,
    ) -> pd.DataFrame:
        """
        Generate a concatenated one-hot encoding vector for one or several columns, e.g. [endpoints, effects, species_group].

        Args:
            columns_to_encode (Union[List[str], None]): List of columns to one-hot encode. If None, no encoding is performed.
            encode_None (bool): If True, encodes None values as a separate category. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with original columns and an added 'onehot_encoding' column containing list of one-hot values.
        """
        if columns_to_encode is None:
            logger.info("No columns to one-hot encode. Returning original dataframe")
            return self.dataframe

        # Keep a copy of the original dataframe
        original_df = self.dataframe.copy()

        # Verify dtype not int, float or bool, as these cannot be one-hot encoded, if found convert to string and warn user
        for col in columns_to_encode:
            if col in original_df.columns:
                if original_df[col].dtype in [int, float, bool]:
                    logger.warning(
                        f"Column '{col}' has dtype {original_df[col].dtype} which cannot be one-hot encoded. Converting to string."
                    )
                    original_df[col] = original_df[col].astype(str)
            else:
                logger.warning(
                    f"Column '{col}' does not exist in the dataframe. It will be skipped for one-hot encoding."
                )
                columns_to_encode.remove(col)

        # Create dummy variables temporarily
        temp_dummies = pd.get_dummies(
            original_df[columns_to_encode],
            prefix=[f"OneHotEnc_{col}" for col in columns_to_encode],
            dummy_na=encode_None,
            dtype=int,
        )

        # Combine the original dataframe with dummy variables
        combined_df = pd.concat([original_df, temp_dummies], axis=1)

        # Extract one-hot encoded values as a list
        onehot_columns = temp_dummies.columns
        combined_df["onehot_encoding"] = combined_df[onehot_columns].values.tolist()

        # Drop temporary one-hot columns (but retain original columns)
        combined_df.drop(columns=onehot_columns, inplace=True)

        return combined_df


def enumerate_smiles(smi: str) -> str:
    """Return a enumerated SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi  # invalid SMILES, return as is
    return Chem.MolToSmiles(mol, doRandom=True)


def fill_missing_taxonomy(df: pd.DataFrame, backwards: bool = True) -> pd.DataFrame:
    """
    Fill missing taxonomy information in a DataFrame.
    Parameters:
    - df: The input DataFrame containing taxonomy information.
    - backwards: If True, fill missing values by propagating the next valid observation backward. If False, fill missing values by propagating the last valid observation forward.

    Returns:
    - DataFrame: The DataFrame with missing taxonomy information filled.

    Example:
    >>> data = {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": [None],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": [None],
    ...     "NCBI_rank_order": [None],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": [None],
    ...     "NCBI_rank_species": [None],
    ... }
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_taxonomy(df, backwards=True)
    >>> print(filled_df)
    >>> {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": ["Proteobacteria_p"],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": ["Cyanobacteriota_f_o"],
    ...     "NCBI_rank_order": ["Cyanobacteriota_f"],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": [None],
    ...     "NCBI_rank_species": [None],
    ... }
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_taxonomy(df, backwards=False)
    >>> print(filled_df)
    >>> {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": ["Bacteria_sk"],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": ["Proteobacteria_p"],
    ...     "NCBI_rank_order": ["Proteobacteria_p_c"],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": ["Cyanobacteriota_f"],
    ...     "NCBI_rank_species": ["Cyanobacteriota_f_g"],
    ... }
    """
    ranks = [
        "NCBI_rank_superkingdom",
        "NCBI_rank_kingdom",
        "NCBI_rank_phylum",
        "NCBI_rank_subphylum",
        "NCBI_rank_class",
        "NCBI_rank_order",
        "NCBI_rank_family",
        "NCBI_rank_genus",
        "NCBI_rank_species",
    ]
    suffixes = {
        "NCBI_rank_superkingdom": "sk",
        "NCBI_rank_kingdom": "k",
        "NCBI_rank_phylum": "p",
        "NCBI_rank_subphylum": "subp",
        "NCBI_rank_class": "c",
        "NCBI_rank_order": "o",
        "NCBI_rank_family": "f",
        "NCBI_rank_genus": "g",
        "NCBI_rank_species": "sp",
    }
    fill_order = ranks.copy()  # Order in which to fill missing ranks
    if backwards:
        fill_order.reverse()  # Start from species
    # Iterate over rows
    for i, row in df.iterrows():
        last_valid = None
        last_valid_rank = None

        for rank in fill_order:
            val = row[rank]

            if pd.notnull(val):
                # record the most recent non-null value and its rank
                last_valid = str(val)
                last_valid_rank = rank
            else:
                # build synthetic ID based on the most recent valid value's rank (append that rank's suffix)
                if last_valid and last_valid_rank:
                    suf = suffixes.get(last_valid_rank, suffixes.get(rank, ""))
                    new_val = f"{last_valid}_{suf}"
                    df.at[i, rank] = new_val
                    # update last_valid to the newly created synthetic value and mark its rank as the current filled rank
                    last_valid = new_val
                    last_valid_rank = rank
    return df


def preprocess_data(data, config):
    logger.info(f"Total number of samples before preprocess: {len(data)}")

    processor = PreprocessData(data)
    data = explicit_casting(processor.dataframe)
    data = processor.filter(
        concentration_units=getattr(config, "concentration_units", None),
        endpoint=getattr(config, "endpoints", None),
        effect=getattr(config, "effects", None),
        species_groups=getattr(config, "species_groups", None),
        lifestages=getattr(config, "lifestages", None),
        administration_routes=getattr(config, "administration_routes", None),
        drop_columns=False,
        drop_unknown_ranks=True,
    )

    if getattr(config, "preprocess_data", True):
        logger.info("Preprocessing data...")
        data = processor.preprocess(
            concentration_thresh=getattr(config, "concentration_thresh", np.inf),
            duration_thresh=getattr(config, "duration_thresh", np.inf),
            log_data=getattr(config, "log_data", False),
            turn_duration_outliers_to_nan=getattr(
                config, "turn_duration_outliers_to_nan", False
            ),
            turn_duration_units_other_than_hours_to_nan=getattr(
                config, "turn_duration_units_other_than_hours_to_nan", True
            ),
        )

    if getattr(config, "fill_missing_taxonomy", True):
        logger.info("Filling missing taxonomy information...")
        data = fill_missing_taxonomy(data, backwards=True)
        data = fill_missing_taxonomy(data, backwards=False)

    if getattr(config, "precompute_taxonomic_embeddings", False):
        logger.info("Retrieving taxonomic embeddings...")
        data = processor.get_taxonomic_embeddings(
            path_to_taxonomic_dict=config.taxonomic_embedding_dict
        )
    if getattr(config, "butina_dir", None) is not None:
        logger.info("Retrieving butina clusters...")
        data = processor.get_butina_clusters(
            path_to_butina_dict=config.butina_dir,
            radius=getattr(config, "butina_cutoff", 0.2),
        )
    logger.info("Retrieving canonical SMILES...")
    data = processor.get_canonical_smiles(
        canonical=True, isomericSmiles=False, drop_errorenous_smiles=True
    )
    logger.info("Generating onehot encoding...")
    data = OneHotEncoder(data).get_onehot_encoding(
        columns_to_encode=getattr(config, "columns_to_onehot_encode", None)
    )

    if hasattr(config, "embedding_columns") and getattr(config, "shuffle_tree", False):
        logger.info("shuffling tree...")
        data["taxonomic_embedding"] = data.taxonomic_embedding.sample(frac=1).values

    logger.info(
        "Filling missing values for effects, lifestages, and administration routes with '<missing>'..."
    )
    logger.info(
        "Adding <> to categorical variables to help tokenizer distinguish them from other tokens..."
    )
    data["effect"] = [
        "<" + str(x) + ">" if not pd.isna(x) else "<missing_effect>"
        for x in data["effect"]
    ]
    data["species_group_corrected"] = [
        "<" + str(x) + ">" if not pd.isna(x) else "<missing_species_group>"
        for x in data["species_group_corrected"]
    ]
    data["administration_route_categorized"] = [
        "<" + str(x) + ">" if not pd.isna(x) else "<missing_administration_route>"
        for x in data["administration_route_categorized"]
    ]
    data["organism_lifestage_categorized"] = [
        "<" + str(x) + ">" if not pd.isna(x) else "<missing_lifestage>"
        for x in data["organism_lifestage_categorized"]
    ]

    logger.success(f"Total number of samples after preprocess: {len(data)}")

    return data


def explicit_casting(
    df: pd.DataFrame,
) -> pd.DataFrame:
    dtypes = {
        "SK_unique_id": "string",
        "data_source": "string",
        "species_group": "string",
        "species_common_name": "string",
        "species_latin_name": "string",
        "cas": "string",
        "chemical_name": "string",
        "conc_unit": "string",
        "conc": float,
        "conc_sign": "string",
        "duration_unit": "string",
        "duration": float,
        "effect": "string",
        "endpoint": "string",
        "administration_route": "string",
        "organism_lifestage": "string",
        "organism_habitat": "string",
        "SMILES": "string",
        "SMILES_Canonical_RDKit": "string",
        "organism_lifestage_categorized": "string",
        "administration_route_categorized": "string",
        "NCBI_sci_name": "string",
        "NCBI_match": "string",
        "NCBI_rank_superkingdom": "string",
        "NCBI_rank_kingdom": "string",
        "NCBI_rank_phylum": "string",
        "NCBI_rank_subphylum": "string",
        "NCBI_rank_class": "string",
        "NCBI_rank_order": "string",
        "NCBI_rank_family": "string",
        "NCBI_rank_genus": "string",
        "NCBI_rank_species": "string",
        "NCBI_last_known_rank": "string",
        "species_group_corrected": "string",
        "duration_missing": "string",
    }
    for column, dtype in dtypes.items():
        if column in df.columns:
            if ((column.startswith("NCBI")) or (column == "duration_missing")) and df[
                column
            ].dtype in [float, "Float64"]:
                # They are floats since there are NaNs. First convert to Int, then to string, while keeping NaNs as NaNs
                df[column] = df[column].astype("Int64")
            df[column] = df[column].astype(dtype)
        else:
            logger.warning(
                f"Column '{column}' not found in DataFrame. Skipping explicit casting for this column."
            )
    return df
