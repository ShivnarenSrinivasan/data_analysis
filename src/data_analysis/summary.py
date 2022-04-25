from typing import (
    Iterable,
    Union,
)
from functools import partial

import numpy as np
import pandas as pd


def value_summary(df: pd.DataFrame, unique_thresh: int = 20) -> pd.DataFrame:
    def get_unique(ser: pd.Series) -> Union[np.ndarray, str]:
        unique = ser.unique()
        extra_char = '...'
        return extra_char if len(unique) > unique_thresh else unique

    return pd.DataFrame(
        {
            'Nunique': df.nunique(),
            'dtypes': df.dtypes,
            'na_count': df.isna().sum(),
            'NA%': df.isna().sum() / df.count(),
            'unique': df.apply(get_unique),
        }
    )


def describe_quantiles(
    df: pd.DataFrame, quantiles: Union[Iterable[int], None] = None
) -> pd.DataFrame:
    """Calculate quantile values of dataframe."""
    if quantiles is None:
        quantiles = (1, 5, 10, 25, 50, 75, 90, 95, 99)

    quantile_funcs = []

    for quantile in quantiles:
        func = partial(pd.Series.quantile, q=quantile / 100)
        func.__name__ = f'{quantile}%'
        quantile_funcs.append(func)

    return df.agg(quantile_funcs)


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
