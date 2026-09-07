import argparse
import json
import time
from ete3 import NCBITaxa
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_blast_name_via_datasets(taxid):
    # sleep to avoid rate limiting
    time.sleep(0.1)
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/taxon/{taxid}"
    r = requests.get(url)
    # BLAST name is under taxonomy_nodes[0]["blast_name"]
    r.raise_for_status()
    data = r.json()
    try:
        return data["taxonomy_nodes"][0]["taxonomy"].get("blast_name", None)
    except KeyError:
        return None

def main(input_file, output_file):
    ncbi = NCBITaxa()

    with open(input_file, 'r') as f:
        taxids = [line.strip() for line in f.readlines()]
    taxid2name = {}
    missing_names = []
    taxids = [str(taxid) for taxid in taxids]  # Ensure all taxids are strings
    taxid2name.update({str(key): value for key, value in ncbi.get_common_names(taxids).items()})
    missing_names.extend(list(set(taxids) - set(taxid2name.keys())))

    print(f"Found {len(taxids)-len(list(set(taxids) - set(taxid2name.keys())))} common names, {len(missing_names)} still missing name.")
    # Some taxids might not have a common name, so we can use the BLAST name as a fallback
    remove = []
    for taxid in tqdm(missing_names, desc="Fetching BLAST names"):
        name = get_blast_name_via_datasets(taxid)
        if name is not None:
            remove.append(taxid)
            taxid2name[taxid] = name

    print(f"Found {len(remove)} BLAST names, {len(missing_names)-len(remove)} still missing name.")
    # Drop names we found
    for taxid in remove:
        missing_names.remove(taxid)

    print(f"Finally {len(missing_names)} taxids still missing name. Assigning latin names as fallback.")
    # Finally we use latin names as a fallback
    taxid2name.update({str(key): value for key, value in ncbi.get_taxid_translator(missing_names).items()})

    # Save mappings to json
    with open(output_file, 'w') as f:
        json.dump(taxid2name, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate taxids to names using common names, BLAST names, and latin names as fallback.")
    parser.add_argument("input_file", help="Path to the input file containing taxids (one per line).")
    parser.add_argument("output_file", help="Path to the output JSON file to save the taxid-to-name mapping.")
    args = parser.parse_args()

    main(args.input_file, args.output_file)