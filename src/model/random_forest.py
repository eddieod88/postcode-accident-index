from collections import defaultdict
from pathlib import Path

import joblib
from loguru import logger
import polars as pl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.utils.config import Config


ROOT = Path(__file__).parents[2]
N_SPLITS = 5


if __name__ == "__main__":
    model_dir = ROOT / "models/rf"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = Config(additional_save_paths=[model_dir / "config.yaml"])

    index_label = "postcode"
    df_train = pd.read_parquet(ROOT / "data/processed/df_acc_ind.parquet").set_index(index_label)
    X = df_train.filter(
        regex=r"|".join(config()["accident_features"] + config()["additional_rollup_features"]),
        axis=1,
    ).copy()
    y = df_train[config()["response"]]

    results = defaultdict(list)
    seed = config()["general"]["random_seed"]
    split_dirs = []
    for i, (train_index, test_index) in enumerate(
        KFold(n_splits=N_SPLITS, random_state=seed, shuffle=True).split(X)
    ):
        i = i + 1
        split_dir = model_dir / f"split_{i}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs.append(split_dir)
        # df_train.iloc[train_index].to_parquet(
        #     ROOT / f"results/split_{i}/df_train.parquet"
        # )
        # df_train.iloc[test_index].to_parquet(
        #     ROOT / f"results/split_{i}/df_test.parquet"
        # )
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        logger.info(f"train shape {X_train.shape}, test shape {X_test.shape}")
        logger.info(f"Training {i}...")

        rf = RandomForestRegressor(
            **config()["hyperparams"],
            random_state=seed,
        )
        rf.fit(X_train, y_train)
        joblib.dump(rf, split_dir / "rf.joblib")

        logger.info(f"Predicting {i}")
        y_pred = rf.predict(X_test)

        pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": y_pred,
            },
            index=X.index[test_index],
        ).to_parquet(split_dir / "preds.parquet")

    # Combine the parquet files into out long preds file which only has the test values in.
    # Used polars for speed and efficient memory management
    pl.concat(
        [pl.scan_parquet(s_dir / "preds.parquet") for s_dir in split_dirs],
        rechunk=False,
        parallel=True,
    ).sort(index_label).collect().write_parquet(model_dir / "full_test_preds.parquet")
