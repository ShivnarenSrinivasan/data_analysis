import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import (
    axes,
)


def plot_value_counts(df_dict, figsize=(16, 9)):
    # plot_count = len(set(df.index.get_level_values(0)))
    plot_count = len(df_dict)
    cols = 4 if plot_count > 4 else plot_count
    plt.figure(figsize=figsize)
    rows = math.ceil(plot_count / cols)
    print(rows, cols)
    for i, key in enumerate(df_dict):
        X = df_dict[key].index.get_level_values(1)
        plt.subplot(rows * 100 + 41 + i)
        sns.barplot(x=X, y="values", data=df_dict[key], order=X)
        plt.title(key)
    plt.suptitle("Value Counts")


def tril_corr(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    cols = df.columns
    corr = pd.DataFrame(
        np.ma.masked_equal(np.tril(df.corr(), k=-1 if drop else 0), 0),
        columns=cols,
        index=cols,
    )
    return corr.iloc[1:, :-1] if drop else corr


def corr_heatmap(df: pd.DataFrame, drop: bool = True, **kwargs) -> axes.Axes:
    corr = tril_corr(df, drop)
    return sns.heatmap(data=corr, **kwargs)
