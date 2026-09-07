import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Process and pickle a file.")
    parser.add_argument("--file", required=True, help="Input file path")

    args = parser.parse_args()

    # If file not exists return error
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File {args.file} does not exist.")

    print("Pickling...")

    # Read the input file
    if args.file.endswith(".zip"):
        df = pd.read_csv(args.file, compression="zip", low_memory=False)
        df.to_pickle(args.file.replace("csv", "pkl"), compression="zip")
    else:
        df = pd.read_csv(args.file, low_memory=False)
        df.to_pickle(args.file.replace("csv", "pkl"))


if __name__ == "__main__":
    main()
