import json
from ete3 import NCBITaxa, PhyloTree, Tree, TreeStyle, TextFace, add_face_to_node
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
import warnings
import sqlite3
import regex as re
import numpy as np
from rapidfuzz import process, fuzz
import pandas as pd
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import os
from dotenv import load_dotenv

os.environ["QT_QPA_PLATFORM"] = "offscreen"

load_dotenv(override=True, interpolate=True)
DATA_STANDARDIZE_COMMON_LOWER = Path(os.getenv("DATA_STANDARDIZE_COMMON_LOWER"))
DATA_TRANSLATE_COMMON_TO_GROUP_LOWER = Path(
    os.getenv("DATA_TRANSLATE_COMMON_TO_GROUP_LOWER")
)
DATA_TRANSLATE_COMMON_TO_LATIN_LOWER = Path(
    os.getenv("DATA_TRANSLATE_COMMON_TO_LATIN_LOWER")
)
DATA_GBIF_MAPPING = Path(os.getenv("DATA_GBIF_MAPPING"))
NCBI_DB = Path(os.getenv("NCBI_DB"))


class TaxonomyPreProcessor:
    """
    # Example Matching Behavior:
    Given the database contains the following:

    - spname: Homo sapiens, Pan troglodytes, Canis lupus
    - common: human, chimpanzee, dog

    Matching Results:
    - Input: "Homo sapiens" → Exact Match
    - Input: "human" → Common Name Match
    - Input: "Homo" → Fuzzy Match
    - Input: "sapien" → Partial Match (LIKE)
    - Input: "unknown species" → None

    """

    def __init__(self, db_path: Path = NCBI_DB):
        self.db_path = db_path
        self.ncbi = NCBITaxa(dbfile=self.db_path)
        if DATA_GBIF_MAPPING.exists():
            self.gbif_mapping = (
                pd.read_csv(DATA_GBIF_MAPPING, sep=";", encoding="unicode_escape")
                .dropna(subset=["query", "canonicalName"])[["query", "canonicalName"]]
                .set_index("query")
                .to_dict()["canonicalName"]
            )

        tqdm.pandas()

    def update_ncbi(self):
        self.ncbi.update_taxonomy_database()

    def preprocess_species_name_using_gbif(self, name: str) -> str:
        # Search for name in the query column
        if name in self.gbif_mapping:
            return self.gbif_mapping.get(name, name)
        return name

    def preprocess_species_name(
        self,
        name: str,
        path_to_species_mapping: Path = DATA_TRANSLATE_COMMON_TO_LATIN_LOWER,
    ) -> str:
        with open(path_to_species_mapping, "r") as f:
            mapping = json.load(f)
        try:
            name = name.lower().strip()
            # Check mapping FIRST, before destructive transformations
            if name in mapping:
                return mapping[name]
            # Then normalize for NCBI direct lookup
            name = name.split(" species; ")[-1]
            name = name.split(", ")[-1]
            name = re.sub(r"[^a-zA-Z0-9 ]", "", name)
            name = name.replace(" sp", "") if name.endswith(" sp") else name
            name = name.replace("other:", "")
            name = " ".join(name.split()[:2])
            name = name.strip()
            # Try mapping again after normalization
            if name in mapping:
                return mapping[name]
            return name
        except Exception:
            return None

    def perform_search(self, name: str, tax_dict: Dict) -> int:
        if name in tax_dict:
            return tax_dict[name][0]
        if name.split(" ")[0] in tax_dict:
            return tax_dict[name.split(" ")[0]][0]
        return tax_dict.get("unspecified", [1])[0]

    def get_taxid(self, species_name: str) -> str:
        if not isinstance(species_name, str) or pd.isna(species_name):
            return None
        taxid = self.ncbi.get_name_translator([species_name])
        if taxid:
            return str(taxid[species_name][0])
        # Try using the first part of the name
        taxid = self.ncbi.get_name_translator([species_name.split(" ")[0]])
        if taxid:
            return str(taxid[species_name.split(" ")[0]][0])
        return None

    def get_taxids(self, species_list: List[str]) -> List[str]:
        taxids = []
        for species in species_list:
            if species is None:
                taxids.append(None)
            else:
                taxids.append(self.get_taxid(species))
        return taxids

    def fuzzy_match_species(
        self, name: str, reference_list: List[str], threshold: int = 85
    ) -> str:
        """
        Perform fuzzy matching using RapidFuzz's ratio scoring with additional checks.
        """
        if not isinstance(name, str) or pd.isna(name):
            return None
        # Increase threshold for stricter matches
        match, score, _ = process.extractOne(name, reference_list, scorer=fuzz.ratio)
        return match if score >= threshold else None

    def sanity_check_lineage(self, species_name: str, species_group: str) -> str:
        """
        Ensure the matched species belongs to the same group as the input species.
        """
        if species_group is None:
            return species_name
        try:
            species_taxid = self.get_taxid(species_name).get(species_name)[0]
            species_lineage = self.ncbi.get_lineage(species_taxid)

            species_group = self.preprocess_species_name(species_group)
            species_group_lineage = self.ncbi.get_lineage(
                self.get_taxid(species_group).get(species_group)[0]
            )

            # Verify if the species group lineage is a subset of the matched species lineage
            if species_group_lineage[-1] in species_lineage:
                return species_name
            else:
                # If the species is not a member of the group, return the group instead
                return self.preprocess_species_name(species_group)
        except Exception as e:
            return self.preprocess_species_name(species_group)

    def fuzzy_match_species_name(self, species_names, path_to_sql_db: str = None):
        """
        Resolve all unknown species names by batch querying the database and matching in memory,
        including fuzzy and partial (`LIKE`) matching.
        """
        if isinstance(species_names, List):
            species_names = pd.Series(species_names)
        path_to_sql_db = path_to_sql_db or self.db_path
        conn = sqlite3.connect(path_to_sql_db)
        cursor = conn.cursor()

        # Fetch all species names and common names from the database in a single query
        cursor.execute("SELECT spname, common FROM species")
        db_records = cursor.fetchall()
        conn.close()

        # Create dictionaries for exact and common name matches
        spname_dict = {row[0]: row[0] for row in db_records}
        common_name_dict = {
            row[1]: row[0] for row in db_records if row[1]
        }  # Exclude None

        # Fetch all spnames for LIKE matching
        db_spnames = list(spname_dict.keys())

        # Resolve species names
        def resolve_name(name):
            if name == "unspecified" or pd.isna(name):
                return None

            # Step 1: Exact match
            if name in spname_dict:
                return spname_dict[name]

            # Step 2: Common name match
            if name in common_name_dict:
                return common_name_dict[name]

            # Step 3: Fuzzy match
            fuzzy_match = self.fuzzy_match_species(name, db_spnames)
            if fuzzy_match:
                return fuzzy_match

            # Step 4: Partial (`LIKE`) match
            # Emulating SQL's `LIKE` by checking if the name is a substring of any spname
            for spname in db_spnames:
                if name.lower() in spname.lower():  # Case-insensitive match
                    return spname

            # If no match is found, return None
            return None

        # Apply the resolution function to the series
        species_names = species_names.progress_apply(resolve_name)
        return species_names

    def get_NCBI_match(
        self,
        species_names: Union[List[str], pd.Series],
        species_groups: Optional[Union[List[str], pd.Series]] = None,
        return_lineage=False,
        return_species_mapping=True,
        return_TaxID=True,
        verify_taxid_using_species_group: bool = False,
        preprocess_using_gbif: bool = False,
        preprocess_using_local_mapping: bool = True,
    ) -> Dict:

        if isinstance(species_names, List):
            species_names = pd.Series(species_names)
        if isinstance(species_groups, List):
            species_groups = pd.Series(species_groups)

        # If species_groups are provided, fill NaNs with the value in species_group
        if species_groups is not None:
            species_names = species_names.fillna(species_groups)

        returns = {}

        processed_names = species_names.copy().apply(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        if preprocess_using_gbif:
            print("Preprocessing species names using GBIF mapping...")
            processed_names = processed_names.apply(
                self.preprocess_species_name_using_gbif
            )

        if preprocess_using_local_mapping:
            print("Preprocessing species names using local mapping...")
            processed_names = processed_names.apply(self.preprocess_species_name)

        print("Fetching Taxonomic IDs from NCBI...")
        taxids = pd.Series(self.get_taxids(processed_names.tolist()), dtype=object)

        # Step 1: Check for unknown names and fuzzy match them
        unknown_mask = taxids.isna().tolist()
        if any(unknown_mask):
            print("Fetching Missing Taxonomic IDs from NCBI fuzzily...")
            fuzzy_matched = self.fuzzy_match_species_name(processed_names[unknown_mask])
            processed_names[unknown_mask] = fuzzy_matched
            taxids = pd.Series(self.get_taxids(processed_names.tolist()), dtype=object)

        # Step 2: Check for unknown names and replace them with species group
        unknown_mask = taxids.isna().tolist()
        if any(unknown_mask) and species_groups is not None:
            print("Replacing Missing Taxonomic IDs with species group...")
            # If species_groups are provided, match the remaining unknown names with the species group
            processed_names[unknown_mask] = species_groups[unknown_mask].apply(
                self.preprocess_species_name
            )
            taxids = pd.Series(self.get_taxids(processed_names.tolist()), dtype=object)

        if species_groups is not None and verify_taxid_using_species_group:
            # Sanity check lineages
            print("Correcting unknown species lineages using species group...")
            unknown_mask = taxids.isna().tolist()
            corrected = [
                self.sanity_check_lineage(species, group)
                for species, group in zip(
                    processed_names[unknown_mask], species_groups[unknown_mask]
                )
            ]
            processed_names[taxids.isna().tolist()] = corrected

        # Final update
        taxids = pd.Series(self.get_taxids(processed_names.tolist()), dtype=object)

        # Manually remove "no rank"
        taxids = taxids.apply(lambda x: None if x == 1 else x)

        print("Fetching and encoding lineages...")
        # Fetch lineages and translate taxids to names
        lineage_encoded = taxids.apply(
            lambda x: self.ncbi.get_lineage(x) if pd.notna(x) else None
        )

        # Last name in lineage is the latin name
        print("Verifying latin names...")
        processed_names = [
            self.ncbi.get_taxid_translator([lin[-1]])[lin[-1]].lower()
            if lin is not None
            else None
            for lin in tqdm(lineage_encoded.tolist())
        ]

        returns["NCBI_sci_name"] = processed_names
        if return_lineage:
            returns["lineage"] = lineage_encoded.tolist()
            returns["named_lineage"] = lineage_encoded.progress_apply(
                lambda lin: (
                    [self.ncbi.get_taxid_translator([taxid])[taxid] for taxid in lin]
                    if lin is not None
                    else None
                )
            ).tolist()
        if return_species_mapping:
            returns["species_mapping"] = dict(
                zip(species_names.tolist(), processed_names)
            )
        if return_TaxID:
            returns["NCBI_match"] = taxids.tolist()
        return returns

    def filter_lineage_by_ranks(
        self,
        lineage: List[Union[int, str]],
        desired_ranks: List[str],
        return_names: bool = True,
    ) -> List[str]:
        """
        Filters a lineage list to include only the desired taxonomic ranks.

        Args:
        - lineage: List of taxonomic names or taxids representing the lineage.
        - desired_ranks: List of desired ranks to retain (e.g., ['kingdom', 'genus', 'family', 'species']).

        Returns:
        - A list of names corresponding to the desired ranks in the lineage.
        """
        if not lineage:
            return {f"NCBI_rank_{rank}": None for rank in desired_ranks}

        # If lineage is a list of strings, convert to int
        if isinstance(lineage[0], str):
            lineage = [int(taxid) for taxid in lineage]

        # Fetch ranks for all taxids
        ranks = self.ncbi.get_rank(lineage)

        # Filter lineage by desired ranks
        inv_ranks = {v: k for k, v in ranks.items()}

        # Convert taxids back to names if input was names
        if return_names:  # convert to names
            return {
                f"NCBI_rank_{rank}": self.ncbi.get_taxid_translator([inv_ranks[rank]])[
                    inv_ranks[rank]
                ].lower()
                if rank in inv_ranks
                else None
                for rank in desired_ranks
            }
        else:
            return {
                f"NCBI_rank_{rank}": str(inv_ranks[rank]) if rank in inv_ranks else None
                for rank in desired_ranks
            }

    def filter_lineages_by_ranks(
        self,
        lineages: List[List[int]],
        desired_ranks: List[str],
        return_names: bool = True,
    ):
        df = pd.DataFrame(
            [
                self.filter_lineage_by_ranks(lineage, desired_ranks, return_names)
                for lineage in tqdm(lineages)
            ]
        )
        df["NCBI_last_known_rank"] = None
        # Get the last known matched rank
        for i, row in df.iterrows():
            # desired ranks need to be defined in the correct order!
            for rank in desired_ranks[::-1]:  # iterate in reverse order
                if pd.notna(row[f"NCBI_rank_{rank}"]):
                    df.loc[i, "NCBI_last_known_rank"] = row[f"NCBI_rank_{rank}"]
                    break
        return df

    def map_missing_species_group_using_NCBI(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Function to map missing species_group using NCBI_last_known_rank.
        This function assumes that the dataframe has columns 'NCBI_last_known_rank' and 'species_group'.
        It creates a mapping from NCBI_last_known_rank to species_group and fills in missing values in species_group based on this mapping.
        """
        # Count nan values in species_group
        nan_count = df["species_group"].isna().sum()

        # Get unique combinations of species_latin_name and species_group
        df["species_group"] = df["species_group"].fillna(
            "unknown"
        )  # Fill NaN values in species_group with 'unknown'
        unique_combinations = df[
            ["NCBI_last_known_rank", "species_group"]
        ].drop_duplicates()

        # Build a mapping from NCBI_last_known_rank to species_group (dictionary with NCBI_last_known_rank as keys and list of species_group as values)
        species_group_mapping = (
            unique_combinations.groupby("NCBI_last_known_rank")["species_group"]
            .apply(list)
            .apply(lambda x: x if len(x) > 1 and "unknown" in x else None)
            .dropna()
            .apply(
                lambda x: [i for i in x if i != "unknown"][0]
            )  # Remove 'unknown' from the list and return the first element (no way of knowing which one is correct)
            .to_dict()
        )

        # Where the dataframe contains 'unknown' in the species_group column, use NCBI_last_known_rank to get the species_group from the mapping
        df["species_group"] = df.apply(
            lambda row: (
                species_group_mapping.get(
                    row["NCBI_last_known_rank"], row["species_group"]
                )
                if row["species_group"] == "unknown"
                else row["species_group"]
            ),
            axis=1,
        )

        # Map back the "unknown" species_group to None
        df["species_group"] = df["species_group"].replace("unknown", np.nan)

        # Count nan values in species_group after mapping
        nan_count_after = df["species_group"].isna().sum()
        print(
            f"Corrected {nan_count - nan_count_after} missing species_group entries using NCBI_last_known_rank mapping."
        )

        return df, species_group_mapping

    def correct_ambiguous_species_group_using_NCBI(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Function to correct ambiguous species_group entries using NCBI_last_known_rank.
        This function assumes that the dataframe has columns 'NCBI_last_known_rank' and 'species_group'.
        It identifies taxids with multiple species_group entries and resolves them by keeping the most frequent species_group for each taxid.
        """

        taxid_species_group_combinations = (
            df[["NCBI_last_known_rank", "species_group"]]
            .groupby(["NCBI_last_known_rank"])
            .nunique()
        )
        ambiguous_taxids = taxid_species_group_combinations[
            taxid_species_group_combinations.species_group > 1
        ]
        ambiguity_mapping = (
            df.groupby(["NCBI_last_known_rank", "species_group"])
            .size()[ambiguous_taxids.index]
            .sort_values(ascending=False)
            .reset_index()
            .drop_duplicates(
                subset="NCBI_last_known_rank"
            )  # Keep the top species_group per taxid
            .set_index("NCBI_last_known_rank")["species_group"]
            .to_dict()
        )

        print(
            f"Correcting {len(ambiguity_mapping)} ambiguous species_group entries using NCBI_last_known_rank mapping."
        )

        # Correct the species_group for ambiguous taxids
        df["species_group"] = df.apply(
            lambda row: (
                ambiguity_mapping.get(row["NCBI_last_known_rank"], row["species_group"])
                if row["NCBI_last_known_rank"] in ambiguity_mapping
                else row["species_group"]
            ),
            axis=1,
        )

        return df, ambiguity_mapping


class TaxonomyTreeProcessor:
    """
    # Class to work with ete3 toolkit phylogenetic trees

    """

    def __init__(self, db_path: Path = NCBI_DB):
        self.ncbi = NCBITaxa(dbfile=db_path)
        self.db_path = db_path
        tqdm.pandas()

    def update_ncbi(self):
        self.ncbi.update_taxonomy_database()

    def get_tree(self, TaxID: List[int], intermediate_nodes=True):
        self.tree = self.ncbi.get_topology(
            TaxID, intermediate_nodes, collapse_subspecies=True
        )
        self.tree_annotated = False
        return self.tree

    def read_tree(self, pth):
        self.tree = PhyloTree(pth)

    def save_tree(self, pth, annotations: Union[None, List[str]] = None):
        if annotations is not None:
            self.tree.write(format=9, features=annotations, outfile=pth)
        else:
            self.tree.write(format=1, outfile=pth)

    def print_node_attributes(self, node):
        print(f"Node: {node.name}")
        print(f"Rank: {node.rank}")
        print(f"Sci name: {node.sci_name}")
        print()

    def prune_tree(
        self, ranks: Union[str, List[str]], preserve_branch_length: bool = False
    ) -> Tree:
        """
        Prune the tree to retain only nodes with specified ranks and remove intermediate nodes.

        Parameters:
        - ranks: A list of desired taxonomic ranks to retain.
        - preserve_branch_length (bool): If True, preserve branch lengths during pruning. This is not recommended since there are many intermediate nodes that are meaningless.

        Returns:
        - A pruned Tree object containing only the specified ranks. If preserve_branch_length is False, all branch lengths will be 1 in resulting tree.
        """
        # Step 1: Collect nodes with the specified ranks
        nodes_to_keep = []
        for rank in ranks:
            nodes_to_keep.extend(self.tree.search_nodes(rank=rank))

        # Step 2: Prune the tree to retain only the collected nodes
        self.tree.prune(nodes_to_keep, preserve_branch_length=preserve_branch_length)

        # Step 3: Collapse intermediate nodes not matching the desired ranks
        for node in self.tree.traverse("postorder"):
            if not node.is_leaf() and node.rank not in ranks:
                node.delete(preserve_branch_length=preserve_branch_length)

        return self.tree

    def adjust_branch_lengths_for_missing_ranks(
        self, ranks: Union[str, List[str]]
    ) -> Tree:
        """
        Adjust branch lengths for nodes with missing ranks in the tree.
        E.g.
        - if a node has rank 'species' and its parent has rank 'genus', the branch length will remain 1.
        - if the species has no genus but has a family, the branch length will be 2.

        Parameters:
        - ranks: A list of desired taxonomic ranks to retain.

        Returns:
        - A Tree object with adjusted branch lengths based on the number of missing ranks.
        """
        rank_order = {rank: i for i, rank in enumerate(ranks)}

        for node in self.tree.traverse("postorder"):
            if node.up is None:
                continue  # Skip root

            parent = node.up
            node_rank = getattr(node, "rank", None)
            parent_rank = getattr(parent, "rank", None)

            if node_rank not in rank_order or parent_rank not in rank_order:
                continue  # Skip unranked or out-of-scope nodes

            steps_apart = abs(rank_order[node_rank] - rank_order[parent_rank])
            if steps_apart == 0:
                steps_apart = 1  # At least 1 step between different nodes

            # Assign adjusted branch length based on number of missing ranks
            node.dist = float(steps_apart)
        return self.tree

    def plot_tree(self, treestyle: Optional[TreeStyle] = None, inline: bool = False):
        def my_layout(node):
            # Construct the label with node name and rank
            label = f"{node.sci_name} | {node.name} | ({getattr(node, 'rank', 'N/A')})"
            # Create a text face with the label
            face = TextFace(label, fsize=5)
            # Add the face to the node at the desired position
            add_face_to_node(
                face,
                node,
                column=0,
                position="branch-right",
            )

        if treestyle is not None:
            if inline:
                return self.tree.render(
                    "%%inline", w=120, units="mm", tree_style=treestyle
                )
            else:
                self.tree.render("tree.svg", w=500, units="mm", tree_style=treestyle)
        else:
            ts = TreeStyle()
            ts.layout_fn = my_layout
            ts.show_leaf_name = False  # Prevent default leaf name display
            ts.show_branch_length = True  # Display branch lengths
            ts.force_topology = (
                False  # Align based on tree structure, not branch lengths
            )
            if inline:
                return self.tree.render("%%inline", w=120, units="mm", tree_style=ts)
            else:
                self.tree.render("tree.svg", w=500, units="mm", tree_style=ts)

    def get_distance_between_nodes(
        self,
        node1,
        node2,
        topology_only: bool = False,
        use_distance_to_common_ancestor: bool = False,
        use_rank_as_distance: bool = False,
        rank_to_dist: Optional[dict] = None,
        treat_ambiguity: str = "max",
    ) -> float:
        """
        Get the distance between two nodes in the tree.

        Parameters:
        - node1: The first node in the tree.
        - node2: The second node in the tree.
        - topology_only (bool): If True, only consider the topology (ignore branch lengths).
        - use_distance_to_common_ancestor (bool): If True, calculate the distance using the common ancestor.
        - use_rank_as_distance (bool): If True, use rank-based distance for calculation.
        - rank_to_dist (Optional[dict]): A dictionary mapping ranks to distances, required if use_rank_as_distance is True. E.g.:
            ```python
            rank_to_dist = {
                "root": 0,
                "superkingdom": 1,
                "kingdom": 2,
                "phylum": 3,
                "subphylum": 4,
                "class": 5,
                "order": 6,
                "family": 7,
                "genus": 8,
                "species": 9,
            }
            ```
        - treat_ambiguity (str): If 'max', use the maximum distance to the common ancestor. Meaning dist between a family and a species will be the distance to the common ancestor of the species and the family.

        Returns:
        - float: The distance between the two nodes.
        """
        # If use rank as distance is set, rank_to_dist must be provided
        if use_rank_as_distance and rank_to_dist is None:
            raise ValueError(
                "rank_to_dist must be provided when use_rank_as_distance is True"
            )
        # Rank as distance can only be provided if use_distance_to_common_ancestor is set to True
        if use_rank_as_distance and not use_distance_to_common_ancestor:
            raise ValueError(
                "use_distance_to_common_ancestor must be True when use_rank_as_distance is True"
            )
        # Topology only will only take effect if use_rank_as_distance is set to False
        if topology_only and use_distance_to_common_ancestor:
            warnings.warn(
                "topology_only will not be used when use_rank_as_distance is True"
            )

        if use_distance_to_common_ancestor:
            ancestor = self.tree.get_common_ancestor(node1, node2)
            if use_rank_as_distance:
                if ancestor.is_root():
                    return max(rank_to_dist.values()) - rank_to_dist["root"]
                return max(rank_to_dist.values()) - rank_to_dist[ancestor.rank]
            if treat_ambiguity == "max":
                # Get the maximum distance to the common ancestor
                return max(
                    self.tree.get_distance(
                        node1, ancestor, topology_only=topology_only
                    ),
                    self.tree.get_distance(
                        node2, ancestor, topology_only=topology_only
                    ),
                )
            elif treat_ambiguity == "min":
                return min(
                    self.tree.get_distance(
                        node1, ancestor, topology_only=topology_only
                    ),
                    self.tree.get_distance(
                        node2, ancestor, topology_only=topology_only
                    ),
                )
            else:
                # Default behavior: return the distance to the common ancestor
                return self.tree.get_distance(
                    node1, ancestor, topology_only=topology_only
                )
        else:
            return self.tree.get_distance(node1, node2, topology_only=topology_only)

    def get_pairwise_distance_matrix(
        self,
        ranks: Union[List[str], str] = "species",
        return_pandas: bool = True,
        return_node_sci_names: bool = True,
        topology_only: bool = False,
        use_distance_to_common_ancestor: bool = False,
        use_rank_as_distance: bool = False,
        rank_to_dist: Optional[dict] = None,
        treat_ambiguity: str = "max",
    ):
        print("Computing pairwise distance matrix...")
        # Get all nodes matching rank (i.e., species, genus, family etc.)
        if isinstance(ranks, str):
            nodes = self.tree.search_nodes(rank=ranks)
        elif isinstance(ranks, list):
            nodes = []
            for rank in ranks:
                nodes.append(self.tree.search_nodes(rank=rank))
            nodes = [node for sublist in nodes for node in sublist]
            # Add the root node to the list of nodes
            nodes.append(self.tree.get_tree_root())
        else:
            nodes = list(self.tree.traverse("levelorder"))

        # Initialize an empty distance matrix
        n = len(nodes)
        self.distance_matrix = np.zeros((n, n), dtype=np.uint16)

        # Fill the distance matrix, this will be symmetric so need only do half
        for i in tqdm(range(n)):
            for j in range(i + 1, n):
                dist = self.get_distance_between_nodes(
                    nodes[i],
                    nodes[j],
                    topology_only=topology_only,
                    use_distance_to_common_ancestor=use_distance_to_common_ancestor,
                    use_rank_as_distance=use_rank_as_distance,
                    rank_to_dist=rank_to_dist,
                    treat_ambiguity=treat_ambiguity,
                )
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

        # Verify that the distance matrix is symmetric
        assert np.allclose(self.distance_matrix, self.distance_matrix.T), (
            "Distance matrix is not symmetric"
        )

        taxids = [node.name for node in nodes]
        node_sci_names = [node.sci_name.lower() for node in nodes]

        if return_pandas:
            self.distance_matrix = pd.DataFrame(
                self.distance_matrix, columns=taxids, index=taxids
            )

        if return_node_sci_names:
            return self.distance_matrix, node_sci_names
        return self.distance_matrix

    def display_distance_matrix(self):
        if hasattr(self, "distance_matrix") and self.distance_matrix is not None:
            sns.heatmap(normalize(self.distance_matrix))
            plt.show()
        else:
            self.get_pairwise_distance_matrix(return_TaxID=False)
            sns.heatmap(normalize(self.distance_matrix))
            plt.show()


def get_MDS_from_distance_matrix(
    distance_matrix,
    dim: int = 768,
    metric: bool = False,
    random_state: int = 42,
    max_iter: int = 2000,
    eps: float = 1e-9,
    n_init: int = 10,
    n_jobs=-1,
    use_cuda: bool = False,
) -> Tuple[np.ndarray, None]:
    print("Performing MDS...")
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    distance_matrix = distance_matrix.astype(np.float32)

    assert np.isfinite(distance_matrix).all(), "Distance matrix contains NaNs or Infs"

    if use_cuda:
        from mds.mdscuda import MDS as MDS_cuda

        eps = "NaN"
        n_init = 1
        metric = True
        print(
            f"Overriding MDS parameters for CUDA: eps={eps}, n_init={n_init}, metric={metric}"
        )
        mds_model = MDS_cuda(
            n_dims=dim, max_iter=max_iter, n_init=n_init, x_init=None, verbosity=2
        )
        mds_embeddings = mds_model.fit(squareform(distance_matrix), calc_r2=True)
    else:
        mds_model = MDS(
            n_components=dim,
            dissimilarity="precomputed",
            metric=metric,
            n_init=n_init,
            random_state=random_state,
            n_jobs=n_jobs,
            normalized_stress="auto",
            max_iter=max_iter,
            eps=eps,
        )
        mds_embeddings = mds_model.fit_transform(distance_matrix)

    evaluate_mds_embeddings(
        pd.DataFrame(distance_matrix), mds_embeddings, ranks=None, tree=None
    )

    return mds_embeddings, mds_model


def evaluate_mds_embeddings(
    distance_matrix,
    mds_embeddings,
    ranks: Optional[Union[List[str], str]] = None,
    tree: Optional[Union[Tree, PhyloTree]] = None,
):
    """
    Evaluate the MDS embeddings by calculating stress and correlations.

    Parameters:
    - distance_matrix: The original distance matrix.
    - mds_embeddings: MDS embeddings.
    - ranks: Optional list of ranks to evaluate.

    Returns:
    - None
    """
    # If ranks are provided, tree must be provided
    if ranks is not None and tree is None:
        raise ValueError("If ranks are provided, tree must be provided")

    mds_distance_matrix = pd.DataFrame(
        squareform(pdist(mds_embeddings, "euclidean")),
        columns=distance_matrix.columns,
        index=distance_matrix.index,
    )

    # Flatten matrices, excluding diagonal elements
    dm_tmp = distance_matrix.values[np.triu_indices_from(distance_matrix.values, k=1)]
    mds_dm_tmp = mds_distance_matrix.values[
        np.triu_indices_from(mds_distance_matrix.values, k=1)
    ]

    # Calculate stress
    diff_squared = np.sum((dm_tmp - mds_dm_tmp) ** 2)
    original_squared = np.sum(dm_tmp**2)
    stress = np.sqrt(diff_squared / original_squared)
    print(f"    Stress value: {stress:.4f}")

    pearson_corr, _ = pearsonr(dm_tmp, mds_dm_tmp)
    spearman_corr, _ = spearmanr(dm_tmp, mds_dm_tmp)
    print(f"    Total Pearson correlation: {pearson_corr:.4f}")
    print(f"    Total Spearman correlation: {spearman_corr:.4f}")

    if ranks is not None:
        if isinstance(ranks, str):
            ranks = [ranks]
        for rank in ranks:
            dm_tmp = distance_matrix.loc[
                [n.name for n in tree.search_nodes(rank=rank)],
                [n.name for n in tree.search_nodes(rank=rank)],
            ]
            mds_dm_tmp = mds_distance_matrix.loc[
                [n.name for n in tree.search_nodes(rank=rank)],
                [n.name for n in tree.search_nodes(rank=rank)],
            ]

            # If only two nodes present, skip
            if dm_tmp.shape[0] <= 2:
                print(f"Rank: {rank} - Not enough nodes to calculate correlation")
                continue

            # Flatten matrices, excluding diagonal elements
            dm_tmp = dm_tmp.values[np.triu_indices_from(dm_tmp.values, k=1)]
            mds_dm_tmp = mds_dm_tmp.values[np.triu_indices_from(mds_dm_tmp.values, k=1)]

            # Calculate stress
            diff_squared = np.sum((dm_tmp - mds_dm_tmp) ** 2)
            original_squared = np.sum(dm_tmp**2)
            stress = np.sqrt(diff_squared / original_squared)

            # Calculate correlations
            pearson_corr, _ = pearsonr(dm_tmp, mds_dm_tmp)
            spearman_corr, _ = spearmanr(dm_tmp, mds_dm_tmp)
            print(f"{'  Rank':<15}{rank:<15}")
            print(f"{'      Stress value:':<25}{stress:.4f}")
            print(f"{'      Pearson correlation:':<25}{pearson_corr:.4f}")
            print(f"{'      Spearman correlation:':<25}{spearman_corr:.4f}")


def map_lineage_to_species_group(ranked_lineage: dict) -> str:
    # mammals
    if ranked_lineage["class"] == "40674":
        if ranked_lineage["order"] == "9989":
            return "rodents"
        else:
            return "other mammals"
    # fish
    if (ranked_lineage["class"] == "186623") or ranked_lineage["class"] in [
        "7777",
        "2682552",
        "117569",
    ]:
        return "fish"
    # amphibians
    if ranked_lineage["class"] == "8292":
        return "amphibians"
    # anthropods
    if ranked_lineage["phylum"] == "6656":
        if (
            ranked_lineage["subphylum"] == "6657" or ranked_lineage["class"] == "6844"
        ):  # Pure crustaceans or horsehoe crabs
            return "crustaceans"
        elif ranked_lineage["class"] == "50557":  # These are the pure insects
            return "insects"
        else:
            return "insects"  # put other terrestrial anthropods here, like springtails, millipedes, they are basically anthropods that are neither insects nore crustaceans, they make up roughly 3000 datapoints
    # arachnids
    if ranked_lineage["class"] == "6854":
        return "spiders"
    # mollusks
    if ranked_lineage["phylum"] == "6447":  # These are pure mollusca
        return "mollusks"
    if ranked_lineage["phylum"] == "7568":  # These are other shells and snails
        return "mollusks"
    # fungi
    if ranked_lineage["kingdom"] == "4751":  # fungi
        return "fungi"
    # birds
    if ranked_lineage["class"] == "8782":  # aves
        return "birds"
    # plants
    if ranked_lineage["phylum"] in [
        "35493",
    ]:  # green plants
        return "plants"
    if ranked_lineage["class"] in [
        "2201463"
    ]:  # mosses and worts (they also belong to green plants though but for clarity)
        return "plants"
    # algae
    if ranked_lineage["phylum"] in [
        "2836",
        "3041",
        "2830",
        "2763",
    ]:  # diatoms, green algae, haptophytes, red algae,  etc
        return "algae"
    if (
        ranked_lineage["class"]
        in [
            "2870",
            "2864",
            "3035",
            "3027",
            "2825",
            "5747",
            "38410",
            "33859",
            "35675",
            "304573",
        ]
    ):  # brown algae, dinoflagellates, euglenids, cryptomonads, golden algae, Eustigmatophyceae, Raphidophytes, Synurids, Pelagophyceae
        return "algae"
    # cyanobacteria
    if ranked_lineage["phylum"] == "1117":  # apparently this is cyanobacteria
        return "cyanobacteria"
    # rotifers
    if ranked_lineage["phylum"] == "10190":  # rotifera
        return "rotifers"
    # echinoderms
    if ranked_lineage["phylum"] in ["7586"]:  # sea stars, sea urchins, sea cucumbers
        return "echinoderms"
    # worms
    if (
        ranked_lineage["phylum"]
        in ["33310", "6157", "43120", "10229", "6340", "6231", "6217", "6178", "10229"]
    ):  # nematodes, annelids, flatworms, goblet worms, segmnented worms, horse hair worms, arrow worms
        return "worms"
    # cnidarians and bryozoans
    if (
        ranked_lineage["phylum"] == "6073"
    ):  # Pure cnidarians = jellyfish, corals, sea anemones.
        return "cnidarians and bryozoans"
    if (
        ranked_lineage["phylum"] in ["10197", "10205"]
    ):  # ctenophores, bryozoans, (comb jellies, bryozoans here since they are aquatic and look like corals)
        return "cnidarians and bryozoans"
    # reptiles
    if ranked_lineage["class"] in ["8504"]:  # lepidosaurs
        return "reptiles"
    if ranked_lineage["order"] in ["8459", "1294634"]:  # turtles, alligators and others
        return "reptiles"
    # tunicates and sponges
    if ranked_lineage["subphylum"] == "7712":  # tunicates
        return "tunicates and sponges"
    if ranked_lineage["phylum"] == "6040":  # sponges
        return "tunicates and sponges"
    # protozoans
    if any(
        t
        in [
            "2605435",
            "136419",
            "5878",
            "6020",
            "5977",
            "6015",
            "37471",
            "5988",
            "33827",
            "194287",
            "422676",
            "1280412",
            "33829",
            "5653",
            "2779609",
            "6000",
            "1485085",
            "2681632",
            "555280",
            "2497438",
        ]
        for t in ranked_lineage.values()
    ):
        return "protozoans"
    return "unmapped"


# Construct ranked dict for each row in results and apply mapping function
def construct_ranked_lineage(row) -> dict:
    return {
        "kingdom": str(row["NCBI_rank_kingdom"]),
        "phylum": str(row["NCBI_rank_phylum"]),
        "subphylum": str(row["NCBI_rank_subphylum"]),
        "class": str(row["NCBI_rank_class"]),
        "order": str(row["NCBI_rank_order"]),
        "family": str(row["NCBI_rank_family"]),
        "genus": str(row["NCBI_rank_genus"]),
        "species": str(row["NCBI_rank_species"]),
    }


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


def write_filled_newick_tree(taxonomy: pd.DataFrame) -> None:
    # Load taxonomy data
    taxonomy = (
        taxonomy[
            [
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
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Fill missing taxonomy levels with synthetic IDs
    taxonomy = fill_missing_taxonomy(taxonomy, backwards=True)
    taxonomy = fill_missing_taxonomy(taxonomy, backwards=False)

    # Rename columns for easier access
    taxonomy.columns = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    # Build a tree using the filled taxonomy
    # Initialize empty tree with a root
    tree = Tree(name="root")

    # Dictionary to keep track of existing nodes
    node_map = {"root": tree}

    ranks = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    # Build the tree
    for _, row in taxonomy.iterrows():
        parent_name = "root"
        for rank in ranks:
            name = str(row[rank])
            # Skip NaN or None values (shouldn’t happen after fill, but safe)
            if name == "nan" or name == "None":
                continue
            # Add new node if it doesn't exist yet
            if name not in node_map:
                node_map[name] = node_map[parent_name].add_child(name=name)
            parent_name = name  # next level builds under this node

    # Add attributes to the tree nodes
    tax2rank = {}
    for rank in ranks:
        for taxid in taxonomy[rank].unique():
            tax2rank[taxid] = rank
    for node in tree.traverse():
        name = node.name
        if name in tax2rank:
            node.add_feature("rank", tax2rank[name])
        else:
            node.add_feature("rank", "no_rank")

    # Detach all taxids not in our data
    for rank in taxonomy.columns:
        taxids_in_rank = taxonomy[rank].unique().tolist()
        for node in tree.traverse():
            if node.rank == rank and node.name not in taxids_in_rank:
                _ = node.detach()

    # Verify only species in our data are present
    species_in_tree = [node.name for node in tree.traverse() if node.is_leaf()]
    species_in_data = taxonomy["species"].unique().tolist()
    print(
        "Species in tree match species in data:",
        set(species_in_tree).issubset(set(species_in_data)),
    )

    return tree
