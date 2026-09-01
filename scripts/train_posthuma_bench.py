import os
import argparse
import json
import pickle
import pandas as pd
import numpy as np
import pandas as pd
import sklearn.model_selection
import csv
import random
import subprocess
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from loguru import logger
import json
import time
from dotenv import load_dotenv

load_dotenv(interpolate=True)

from trident2.preprocessing.preprocess_data import preprocess_data
from trident2.training.train_utils import Dict2Class
from trident2.logger.setup_logger import setup_logger


PATH_LIBFM = os.path.join(
    os.getenv("STORAGE"), "libfm-1.42.src/bin/libFM"
)  # directory where you complied the libFM binaries
PATH_CACHE = os.path.join(
    os.getenv("STORAGE"), "libfm_temp"
)  # directory for saving temporary libFM files


# LibSVM format required by LibFM
def save_libsvm(path, X, y):
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter=" ", lineterminator="\n")
        n = X.shape[0]
        for i in range(n):
            row = X.getrow(i)
            row_libsvm = ["%.4f" % y[i]]  # label
            row_libsvm.extend(
                ["%d:%.4f" % (j, d) for j, d in zip(row.indices, row.data)]
            )  # features
            writer.writerow(row_libsvm)


# LibFM learner
class LibFM(object):
    def __init__(
        self,
        bias=1,
        dim=16,
        epoch=50,
        verbose=False,
        path_libfm=PATH_LIBFM,
        path_cache=PATH_CACHE,
    ):
        self.bias = bias
        self.dim = dim
        self.epoch = epoch
        self.verbose = verbose
        self.path_libfm = path_libfm
        self.path_cache = path_cache

    def fit_predict(self, X_train, y_train, X_test, y_test=None, meta=None, cache=None):
        # Create a temporary file if no cache identifier is given
        clean = False
        if cache is None:
            cache = random.getrandbits(128)
            clean = True

        # Ensure cache directory exists
        if not os.path.exists(self.path_cache):
            os.makedirs(self.path_cache)

        if y_test is None:
            y_test = np.zeros(X_test.shape[0])

        if meta is None:
            meta = np.zeros(X_train.shape[1])

        # Always overwrite train/test/meta files to avoid stale folds
        path_train = os.path.join(self.path_cache, f"train_{cache}.svm")
        path_test = os.path.join(self.path_cache, f"test_{cache}.svm")
        path_meta = os.path.join(self.path_cache, f"meta_{cache}.svm")
        path_pred = os.path.join(self.path_cache, f"pred_{cache}.svm")

        logger.info(f"Creating {path_train}")
        save_libsvm(path_train, X_train, y_train)

        logger.info(f"Creating {path_test}")
        save_libsvm(path_test, X_test, y_test)

        logger.info(f"Creating {path_meta}")
        np.savetxt(path_meta, meta, fmt="%d")

        # Fit model and create predictions (pred)
        command = (
            f"{self.path_libfm} -task r "
            f"-dim '1,{self.bias},{self.dim}' "
            f"-method mcmc -iter {self.epoch} "
            f"-train {path_train} -meta {path_meta} "
            f"-test {path_test} -out {path_pred}"
        )
        logger.info(command)

        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) as execute:
            for line in execute.stdout:
                logger.info(line.decode("UTF-8").strip())

        logger.info(f"Reading {path_pred}")
        y_pred = np.loadtxt(path_pred)

        # Debug check
        if len(y_pred) != X_test.shape[0]:
            logger.warning(
                f"Prediction size mismatch! Expected {X_test.shape[0]}, got {len(y_pred)}"
            )

        # Clean temporary files if necessary
        if clean:
            for path in [path_train, path_test, path_meta, path_pred]:
                if os.path.exists(path):
                    os.remove(path)

        return y_pred


# Get out-of-sample predictions with given cross-validation and groups
def cross_validation_predict(clf, X, y, meta, cv, groups=None):
    y_cv = np.zeros(len(y))
    for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
        y_cv[test] = clf.fit_predict(
            X[train], y[train], X[test], y[test], meta, cache=i
        )
    return y_cv


def main(df, config):

    # Labels (mean centered)
    y = df["conc"].values
    ym = np.mean(y)
    y = y - ym

    # Features into one-hot-indicators
    enc1 = OneHotEncoder()
    enc2 = OneHotEncoder()
    enc3 = OneHotEncoder()
    enc4 = OneHotEncoder()
    Xi = enc1.fit_transform(df[["NCBI_rank_species"]])  # Species dummy
    Xj = enc2.fit_transform(df[["CAS"]])  # Drug dummy
    Xd = enc3.fit_transform(df[["duration"]])  # Duration dummy
    Xu = enc4.fit_transform(df[["conc_unit"]])  # Unit dummy
    Xijdu = hstack([Xi, Xj, Xd, Xu], format="csr")
    # Which group the feature belongs to
    meta = np.concatenate(
        [
            np.repeat(0, Xi.shape[1]),
            np.repeat(1, Xj.shape[1]),
            np.repeat(2, Xd.shape[1]),
        ]
    )

    # Indicator of (species, CAS)
    ij = (
        (df["NCBI_rank_species"].astype("str") + " X " + df["CAS"].astype("str"))
        .astype("category")
        .cat.codes.values
    )

    # Counts of (Species, CAS) pairs
    ns = Xi.shape[1]
    nc = Xj.shape[1]
    npairs = Xi.shape[1] * Xj.shape[1]
    npairs_obs = len(np.unique(ij))
    inputs = Xijdu
    logger.info(
        "{} species, {} CAS, {} possible pairs {} observed pairs".format(
            ns, nc, npairs, npairs_obs
        )
    )

    # 10-Fold cross validation grouped by species x compound
    cv = sklearn.model_selection.GroupKFold(n_splits=config.k_folds)

    # Fit a model: global mean
    clf = LibFM(bias=0, dim=0, epoch=2000)
    y_cv = cross_validation_predict(clf, inputs, y, meta, cv, ij)

    # Out-of-sample predictions for training data
    df["libfm_preds_glob_mean"] = ym + y_cv

    # Fit a model: means and pairwise interactions
    clf = LibFM(bias=1, dim=32, epoch=2000, verbose=True)
    y_cv = cross_validation_predict(clf, inputs, y, meta, cv, ij)

    # Out-of-sample predictions for training data
    df["libfm_preds_pairwise"] = ym + y_cv

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path", required=True, help="Path to config file, JSON or YAML"
    )
    args = parser.parse_args()
    setup_logger(log_file=f"logs/leo_bench_{time.strftime('%Y%m%d_%H%M%S')}.log")

    with open(args.config_file_path) as f:
        config = json.load(f)
    config = Dict2Class(config)

    start = time.time()

    # Load dataframe
    if config.data_dir.endswith("pkl.zip"):
        df = pd.read_pickle(config.data_dir, compression="zip")
    else:
        df = pd.read_csv(config.data_dir)
    df = preprocess_data(df, config)

    df = main(df, config)

    stop = time.time()
    logger.success(f"Total time: {stop - start:.2f} seconds")

    # Save dataframe with predictions
    os.makedirs(config.save_dir, exist_ok=True)
    output_path = os.path.join(config.save_dir, "leo_bench_adore_predictions.pkl.zip")
    df.to_pickle(output_path, compression="zip")
    logger.success(f"Saved predictions to {output_path}")
