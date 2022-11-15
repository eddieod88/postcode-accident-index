import asyncio
from pathlib import Path

import geopandas
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.log import log_step
from src.api.clients.postcodes_io import PostcodeClient, PostcodeField, PostcodePayload


ROOT = Path(__file__).parents[1]

ISLANDS = [
    "IM",  # Isle of Man
    "GY",  # Guernsey
]


@log_step
def read_regional_gdp_data(path: Path) -> pd.DataFrame:
    columns_rename = {"ITL code": "itl", "Region name": "region", "2020": "gdp_per_head_2020"}
    return (
        pd.read_excel(path, sheet_name="Table 7", header=1, usecols=columns_rename.keys())
        .rename(columns=columns_rename)
        .set_index("itl")
    )


@log_step
def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df


@log_step
def add_postcode_element_columns(df: pd.DataFrame) -> pd.DataFrame:
    postcode_regex = r"^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$"
    df["is_ok"] = df["postcode"].str.match(postcode_regex)
    df["area"] = df["postcode"].str.extract(r"(?P<area>^[A-Z]{1,2})[0-9][A-Z0-9]? ?[0-9][A-Z]{2}$")
    df["district"] = df["postcode"].str.extract(
        r"(?P<district>^[A-Z]{1,2}[0-9][A-Z0-9]?) ?[0-9][A-Z]{2}$"
    )
    df["sector"] = df["postcode"].str.extract(
        r"(?P<sector>^[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9])[A-Z]{2}$"
    )
    return df


@log_step
def remove_islands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove postcodes which are in islands off the mainland - we do not have data for this.
    """
    df = df[~df["area"].isin(ISLANDS)].copy()
    return df


@log_step
def remove_nan_premiums(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["avgprice1_5"].notna()].copy()


@log_step
def add_locations(df: pd.DataFrame, df_locations: pd.DataFrame, override=False) -> pd.DataFrame:
    """
    Add location information to the postcodes, ie longitudes and latitudes
    """
    # TODO Separate this step into two different steps - one updates the database, another updates the dataset

    if override:
        df_locations = df_locations.head(0).copy()

    # find postcodes that we need fetch locations for
    postcodes_fetch_loc = set(df["postcode"]).difference(set(df_locations.index))
    postcode_api_fields = PostcodePayload(
        fields=[PostcodeField.LONG, PostcodeField.LAT, PostcodeField.ITL_CODE]
    )

    if new_locations := asyncio.run(
        PostcodeClient().fetch_locations(postcodes_fetch_loc, postcode_api_fields)
    ):
        df_new_locations = (
            pd.DataFrame(new_locations)
            .rename(
                columns={
                    str(PostcodeField.LONG.value): "long",
                    str(PostcodeField.LAT.value): "lat",
                    str(PostcodeField.ITL_CODE.value): "itl",
                },
            )
            .set_index("postcode")
        )
        df_new_locations.index = df_new_locations.index.str.replace(" ", "")
        df_locations = pd.concat([df_locations, df_new_locations])
        df_locations.to_parquet(ROOT / "data/database/locations.parquet")  # horrible

    return pd.merge(
        df,
        df_locations,
        how="left",
        left_on="postcode",
        right_index=True,
        validate="m:1",
    )


@log_step
def merge_gdp_data(df: pd.DataFrame, df_gdp: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        df,
        df_gdp,
        how="left",
        left_on="itl",
        right_index=True,
        validate="m:1",
    )


@log_step
def make_geodataframe(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Make a geodataframe with longitudes and latitudes, setting the coord reference system too.
    """
    df["coords"] = geopandas.GeoSeries.from_wkt(
        df.apply(lambda x: f"POINT({x.long} {x.lat})", axis=1)
    )
    return geopandas.GeoDataFrame(df, geometry="coords", crs="EPSG:4326")


@log_step
def save(df: pd.DataFrame) -> pd.DataFrame:
    df.to_parquet(ROOT / "data/processed/df_joined_small.parquet")
    return df


if __name__ == "__main__":
    df_locations = pd.read_parquet(ROOT / "data/database/locations.parquet")
    df_gdp = read_regional_gdp_data(ROOT / "data/raw/ons_regional_stats.xlsx")

    columns = ["postcode", "postcode_group", "avgprice1_5"]
    df = (
        pd.read_parquet(ROOT / "data/raw/preprocessed_copy_small.parquet", columns=columns)
        .pipe(clean)
        .pipe(add_postcode_element_columns)
        .pipe(remove_islands)
        .pipe(remove_nan_premiums)
        .pipe(add_locations, df_locations)
        .pipe(merge_gdp_data, df_gdp)
        .pipe(make_geodataframe)
        .pipe(save)
    )

    logger.info("train validation split")
    df_modelling = df[["long", "lat", "avgprice1_5", "postcode_group"]].copy()
    df_train, df_validation = train_test_split(df_modelling, test_size=0.2, random_state=1)
    df_train.to_parquet(ROOT / "data/modelling/df_train.parquet")
    df_validation.to_parquet(ROOT / "data/modelling/df_validation.parquet")
