from typing import (
    Callable,
)

import pandas as pd


def build_imputer(
    df: pd.DataFrame, ref: str, targ: str
) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    """Return relative imputer with state."""
    lookup = df.groupby(ref)[targ].describe()

    def impute(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        strategies = {'mean': 'mean', 'median': '50%'}
        col = strategies[strategy]
        imputed = df.set_index(ref)[targ].fillna(lookup[col]).to_numpy()
        return df.assign(**{targ: imputed})

    return impute
