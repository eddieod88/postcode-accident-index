import asyncio
from pathlib import Path

import pandas as pd

from src.utils.log import log_step
from src.api.clients.postcodes_io import PostcodeClient


ROOT = Path(__file__).parents[1]


@log_step
def clean(df: pd.DataFrame):
    return df


# change to polars maybe??
@log_step
def add_locations(df: pd.DataFrame, df_locations: pd.DataFrame):
    """
    Add location information to the postcodes, ie longitudes and latitudes
    """
    # TODO Separate this step into two different steps - one updates the database, another updates the dataset
    
    # find postcodes that we need fetch locations for
    postcodes_fetch_loc = set(df["postcode"]).difference(set(df_locations.index))

    if new_locations := asyncio.run(PostcodeClient().fetch_locations(postcodes_fetch_loc)):
        df_new_locations = (
            pd.DataFrame(new_locations)
            .rename(columns={"longitude": "long", "latitude": "lat"})
            .set_index("postcode")
        )
        df_new_locations.index = df_new_locations.index.str.replace(" ", "")
        df_locations = pd.concat([df_locations, df_new_locations])
        df_locations.to_parquet(ROOT / "data/database/locations.parquet")  # horrible

    return pd.merge(
        df, df_locations, how="left", left_on="postcode", right_index=True, validate="m:1"
    )


if __name__ == "__main__":
    columns = ["postcode", "old_postcode_group", "postcode_group"]
    df_locations = pd.read_parquet(ROOT / "data/database/locations.parquet")
    df = (
        pd.read_parquet(ROOT / "data/raw/preprocessed_copy_small.parquet", columns=columns)
        # .sample(1000, random_state=1)
        .pipe(clean)
        .pipe(add_locations, df_locations)
    )
    df.to_parquet(ROOT / "data/processed/df_joined_small.parquet")
    # TODO create districts - need this method for prediction too so make a utility method.
