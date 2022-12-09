from pathlib import Path
from typing import Tuple

import polars as pl


ROOT = Path(__file__).parents[1]
CLEAN_RAW_PATH = ROOT / "data/processed/clean_raw.parquet"


def postcode_element_columns(postcode_col_name: str = "postcode") -> Tuple[pl.Expr]:
    """
    Generic function to extract all postcode levels from a given table with a given postcode column
    """
    postcode_regex = r"^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$"

    is_ok = (
        pl.col(postcode_col_name)
        .str.count_match(postcode_regex)
        .cast(pl.Boolean)
        .alias("postcode_is_ok")
    )
    area = (
        pl.col(postcode_col_name)
        .str.extract(r"(?P<area>^[A-Z]{1,2})[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$")
        .alias("postcode_area")
    )
    district = (
        pl.col(postcode_col_name)
        .str.extract(r"(?P<district>^[A-Z]{1,2}[0-9][A-Z0-9]?) ?[0-9][A-Z]{2}$")
        .alias("postcode_district")
    )
    sector = (
        pl.col(postcode_col_name)
        .str.extract(r"(?P<sector>^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9])[A-Z]{2}$")
        .alias("postcode_sector")
    )

    return is_ok, area, district, sector


def cast_urban_rural_as_str() -> pl.Expr:
    """
    Turn the urban/rural 1/2 column into a string column so it can be caught by the feature selector
    """
    area_col = pl.col("Urban_or_Rural_Area")

    return (
        pl.when(area_col == 1)
        .then("urban")
        .when(area_col == 2)
        .then("rural")
        .otherwise(None)
        .cast(pl.Utf8)
        .alias("Urban_or_Rural_Area")
    )


if __name__ == "__main__":
    # Cleaned training data
    df_accident = (
        pl.scan_csv(ROOT / "data/raw/train.csv")
        .with_columns(
            [
                # Casting Dates and Times
                pl.col("Date").str.strptime(pl.Date, "%d/%m/%y"),
                pl.col("Time").str.strptime(pl.Time, "%R"),
                *postcode_element_columns(),
                cast_urban_rural_as_str(),
            ]
        )
    )
    df_accident.collect().write_parquet(CLEAN_RAW_PATH)
