import duckdb
import os
import argparse
import json
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from setup_logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump environment variables to JSON and save to database as CSV"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file path to save the environment variables.",
    )
    parser.add_argument(
        "--database-out",
        required=True,
        help="Output database file path to save the environment variables as a table.",
    )

    load_dotenv()  # Load environment variables from .env file
    setup_logger(level="INFO")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.database_out), exist_ok=True)

    # Dump entire env to json and save to databse as csv
    env_vars = dict(os.environ)
    with open(args.out, "w") as f:
        json.dump(env_vars, f, indent=4)
    logger.success(f"Saved environment variables to {args.out}")

    # Save to database as csv
    env_vars_df = pd.DataFrame(list(env_vars.items()), columns=["var", "value"])
    con = duckdb.connect(args.database_out)
    con.execute("CREATE OR REPLACE TABLE preprocess_config AS SELECT * FROM env_vars_df")
    con.close()
    logger.success(
        f"Saved environment variables to database at {args.database_out} to table 'preprocess_config'"
    )