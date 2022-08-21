from typing import (
    Dict,
    Iterable,
    Union,
)

import numpy as np
import pandas as pd


def info(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed summary of dataframe."""

    def get_unique(ser: pd.Series) -> np.ndarray:
        return ser.unique()

    def value_counts(ser: pd.Series) -> Union[Dict[str, int], str]:
        return ser.value_counts().to_dict()

    return pd.DataFrame(
        {
            "Nunique": df.nunique(),
            "dtypes": df.dtypes,
            "na_count": df.isna().sum(),
            "NA%": df.isna().sum() / df.count(),
            "unique": df.apply(get_unique),
            "value_counts": df.apply(value_counts),
        }
    )


QUANTILE_VALS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def quantiles(
    df: pd.DataFrame, quantiles: Iterable[int] = QUANTILE_VALS
) -> pd.DataFrame:
    """Calculate percentile values of dataframe."""
    return df.agg(pd.Series.quantile, q=quantiles)


def gen_value_counts(df, thresh=10):
    df_dict = {}
    cols = df.columns
    for col in cols:
        value_count_df = df[col].value_counts()
        value_counts = len(value_count_df)
        if value_counts > thresh:
            continue
        # df_dict[col] = (value_counts, value_count_df)
        iterables = [[col], value_count_df.index]
        multi = pd.MultiIndex.from_product(iterables, names=["feature", "count"])
        df_dict[col] = pd.DataFrame(
            value_count_df.values, columns=["values"], index=multi
        )
    concat_df = pd.concat([df_dict[key] for key in df_dict])
    return df_dict


def tril_corr(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:

    cols = df.select_dtypes("number").columns
    corr = pd.DataFrame(
        np.ma.masked_equal(np.tril(df.corr(), k=-1 if drop else 0), 0),
        columns=cols,
        index=cols,
    )
    return corr.iloc[1:, :-1] if drop else corr


def corr_vars(df: pd.DataFrame, thresh: float = 0.8) -> pd.DataFrame:
    return (
        df.pipe(tril_corr)
        .mask(lambda df: df.abs() < thresh)
        .dropna(how="all")
        .dropna(how="all", axis=1)
    )
