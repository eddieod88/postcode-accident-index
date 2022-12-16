from pathlib import Path
from typing import Dict, List
import shutil

import yaml
import polars as pl


ROOT = Path(__file__).parents[2]


class Config:
    def __init__(self, path: Path = ROOT / "config/model.yaml", additional_save_paths: List[Path] = None):
        self.path = path
        with open(self.path) as config_file:
            self.config_dict = yaml.safe_load(config_file)
        
        if additional_save_paths is not None:
            for save_path in additional_save_paths:
               shutil.copy(self.path, save_path) 

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
