from pathlib import Path
from typing import Dict, List

import yaml
import polars as pl


ROOT = Path(__file__).parents[2]


class Config:
    def __init__(self, path: Path = ROOT / "config/model.yaml"):
        self.path = path
        with open(self.path) as config_file:
            self.config_dict = yaml.safe_load(config_file)

    def __call__(self) -> Dict:
        return self.config_dict

    def filter_out_drop_categories(self) -> pl.Expr:
        """
        Join all 'drop_categories' items into one long filter out expression to be used in a `filter` ploars method.
        """
        return eval(
            " & ".join(
                [
                    f"~pl.col('{col}').is_in({feature_values})"
                    for col, feature_values in self.config_dict["drop_categories"].items()
                ]
            )
        )

    def group_together_categories(self) -> List[pl.Expr]:
        """
        For categories with low exposure, group them together into another group.

        This can be used in a `with_columns` method in polars
        """
        return [
            pl.when(pl.col(col).is_in(grouping_details["grouping"]))
            .then(grouping_details["name"])
            .otherwise(pl.col(col))
            .alias(col)
            for col, grouping_details in self.config_dict["group_categories"].items()
        ]
