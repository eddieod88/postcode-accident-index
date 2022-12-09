from datetime import time
from pathlib import Path

import polars as pl

from src.utils.config import Config


ROOT = Path(__file__).parents[1]
JOINED_PATH = ROOT / "data/processed/df_acc_ind.parquet"
CLEAN_RAW_PATH = ROOT / "data/processed/clean_raw.parquet"

DF_POP_COL_MAPPING = {
    "postcode": "postcode_sector",
    "Variable: All usual residents; measures: Value": "population",
    "Variable: Area (Hectares); measures: Value": "area",
    "Variable: Density (number of persons per hectare); measures: Value": "density",
}
DF_ROAD_COL_MAPPING = {
    "postcode": "postcode",
    "WKT": "WKT",
    "formOfWay": "nearest_road_type",
    "length": "nearest_road_length",
    "distance to the nearest point on rd": "nearest_road_distance_to",
}


def pop_gender_ratio_column() -> pl.Expr:
    """Column for the gender ratio within a postcode"""
    return (pl.col("Variable: Males; measures: Value") / pl.col("population")).alias("male_ratio")


def pop_homeless_ratio_column() -> pl.Expr:
    """Column for the ratio of homeless people within a postcode"""
    return (
        pl.col("Variable: Lives in a communal establishment; measures: Value")
        / pl.col("population")
    ).alias("homeless_ratio")


def pop_school_children_away_from_home_ratio_column() -> pl.Expr:
    """
    Column for the ratio of school children living at a different address from their home during term time within a postcode
    IE - how many students are at school a long way from home...
    """
    return (
        pl.col(
            "Variable: Schoolchild or full-time student aged 4 and over at their non term-time address; measures: Value"
        )
        / pl.col("population")
    ).alias("schoolchild_diff_address_ratio")


def time_of_day_bucket(time_col_name: str = "Time") -> pl.Expr:
    """
    Convert the time of day into 4 categories - early_hours, morning, afternoon, evening
    """
    time_col = pl.col(time_col_name)

    return (
        pl.when(time_col > time(18, 0))
        .then("evening")
        .when(time_col > time(12, 0))
        .then("afternoon")
        .when(time_col > time(6, 0))
        .then("morning")
        .when(time_col > time(0, 0))
        .then("early_hours")
        .otherwise(None)
        .alias("time_of_day")
    )


if __name__ == "__main__":
    config = Config()

    # Population data associated with postcode (can join at the end)
    df_pop = (
        pl.scan_csv(ROOT / "data/raw/population.csv")
        .rename(DF_POP_COL_MAPPING)
        .with_columns(
            [
                pop_gender_ratio_column(),
                pop_homeless_ratio_column(),
                pop_school_children_away_from_home_ratio_column(),
            ]
        )
        .select(
            [
                *DF_POP_COL_MAPPING.values(),
                "male_ratio",
                "homeless_ratio",
                "schoolchild_diff_address_ratio",
            ]
        )
    )
    # Nearest road data
    df_road = (
        pl.scan_csv(ROOT / "data/raw/roads_network.csv")
        .select(DF_ROAD_COL_MAPPING.keys())
        .rename(DF_ROAD_COL_MAPPING)
    )

    # Add features at accident level first (before postcode aggregation)
    df_joined = (
        pl.scan_parquet(CLEAN_RAW_PATH)
        .filter(
            config.filter_out_drop_categories()
        )  # filter out rows with sparse categories (2000 rows)
        .with_columns(
            [
                pl.col("Date").dt.month().alias("month"),
                time_of_day_bucket(),
                *config.group_together_categories(),  # group together less sparse rows
            ]
        )
        # Join the location data before aggrgation so that we can select features from all available columns
        .join(df_road, how="left", on="postcode")
        .join(df_pop, how="left", on="postcode_sector")
    )

    cat_features = [
        c for c in df_joined.select(pl.col(pl.Utf8)).columns if c in config()["accident_features"]
    ]
    numeric_features = sorted(list(set(config()["accident_features"]) - set(cat_features)))

    # Select only the columns required for the feature dataset
    # Aggregation by postcode:
    # Categorical:
    # - One hot encoded categorical features turning into counts once aggregated
    # Numeric:
    # - Take the mean of all numeric features
    df_postcode = (
        pl.get_dummies(df_joined.collect(), columns=cat_features)
        .select(["postcode"] + numeric_features + [pl.col(f"^{col}_.*$") for col in cat_features])
        .groupby("postcode")
        .agg(
            [
                pl.count(),
                *[pl.col(f"^{col}_.*$").sum() for col in cat_features],
                *[pl.col(col).mean() for col in numeric_features],
            ]
        )
        .rename(
            {"Number_of_Casualties": config()["response"]}
        )  # As we have taken the mean of this value, this becomes the response (accident risk index)
    )

    df_postcode.write_parquet(JOINED_PATH)
