from typing import (
    Collection,
)

import pandas as pd


def invariant_cols(df: pd.DataFrame) -> Collection[str]:
    invariant_cols = df.nunique()[lambda x: x == 1].index
    return invariant_cols
