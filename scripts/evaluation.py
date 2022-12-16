from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.config import Config
from src.utils.metrics import METRICS
from src.viz import MAPPING as VIZ_MAPPING

ROOT = Path(__file__).parents[1]


def make_metrics(config: Dict, df_preds: pd.DataFrame):
    metrics = defaultdict(list)
    for metric_name in config["evaluation"]["metrics"]:
        metric_func = METRICS[metric_name]
        metrics[metric_name].append(metric_func(df_preds["y_true"], df_preds["y_pred"]))
    return pd.DataFrame(metrics)


def make_visualisations(config: Dict, model_dir: Path, df_preds: pd.DataFrame):
    for viz_name, viz_kwargs in config["evaluation"]["visualisations"].items():
        VIZ_MAPPING[viz_name](df_preds["y_true"], df_preds["y_pred"], **viz_kwargs).savefig(
            model_dir / f"{viz_name}.png"
        )


if __name__ == "__main__":
    model_dir = ROOT / "models/rf"
    config = Config(model_dir / "config.yaml")
    df_preds = pd.read_parquet(model_dir / "full_test_preds.parquet")

    eval_dir = model_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    make_metrics(config(), df_preds).to_csv(eval_dir / "metrics.csv")
    make_visualisations(config(), eval_dir, df_preds)
