from __future__ import annotations
from dataclasses import (
    dataclass,
    field,
)

from typing import (
    NamedTuple,
    Sequence,
)

import numpy as np
import pandas as pd

from sklearn import (
    base,
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


class Data(NamedTuple):
    X: pd.DataFrame
    y: np.ndarray | pd.Series

    @property
    def df(self) -> pd.DataFrame:
        y = pd.Series(self.y, name='Target') if isinstance(self.y, np.ndarray) else y
        return pd.concat([self.X, y], axis=1)

    @property
    def arr(self) -> np.ndarray:
        y = self.y.to_numpy() if isinstance(self.y, pd.Series) else self.y
        return np.column_stack([self.X.to_numpy(), y])


class TrainTestData(NamedTuple):
    train: Data
    test: Data

    def __repr__(self) -> str:
        return str(self._fields)

    @classmethod
    def from_xy(
        cls, X: pd.DataFrame, y: np.ndarray | pd.Series, **kwargs
    ) -> TrainTestData:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
        return TrainTestData(Data(X_train, y_train), Data(X_test, y_test))


@dataclass(repr=False)
class Dataset:
    data: TrainTestData
    name: str = field(default='')

    def __str__(self) -> str:
        return f'Dataset: `{self.name}`'


# Evaluation
@dataclass(repr=True)
class Report:
    confusion_matrix: pd.DataFrame
    metrics: pd.DataFrame

    @classmethod
    def from_preds(
        cls,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        classes: Sequence[str | int],
    ) -> Report:
        return Report(
            pd.DataFrame(
                confusion_matrix(y_true, y_pred), index=classes, columns=classes
            ),
            pd.DataFrame(
                classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            ),
        )

    @classmethod
    def from_estimator(cls, data: Data, model: base.BaseEstimator) -> cls:
        return cls.from_preds(data.y, model.predict(data.X), model.classes_)
