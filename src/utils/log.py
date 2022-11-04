from loguru import logger
import pandas as pd


def log_step(func):
    def _wrapped(*args, **kwargs):
        logger.info(f"Process step: {func.__name__}")
        df: pd.DataFrame = func(*args, **kwargs)
        logger.info(f"Shape: {df.shape}")
        return df

    return _wrapped
