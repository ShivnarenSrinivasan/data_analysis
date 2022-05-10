from typing import (
    Collection,
    Iterable,
)

import numpy as np
import pandas as pd
from sklearn import (
    base,
)


def glm_params(
    lin_models: Iterable[base.BaseEstimator], cols: Collection
) -> pd.DataFrame:
    """Compare parameters of GLM models."""
    return pd.concat(
        (_make_series(lin_model, cols) for lin_model in lin_models), axis=1
    )


def _make_series(lin_model, cols: Collection):
    return pd.Series(
        np.append(lin_model.coef_.ravel(), lin_model.intercept_),
        index=list(cols) + ['_COEF_'],
        name=str(lin_model),
    )
