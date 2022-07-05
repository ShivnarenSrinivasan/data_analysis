# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modifed for use by Shivnaren Srinivasan
# License: BSD 3 clause

from typing import (
    Sequence,
    Collection,
    Mapping,
    Tuple,
)
from types import MappingProxyType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import (
    base,
    datasets,
    ensemble,
    linear_model,
    model_selection,
    neighbors,
    svm,
    tree,
)

from sklearn.preprocessing import StandardScaler

# from sklearn.neural_network import MLPClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


Dataset = Tuple[np.ndarray, np.ndarray]
CM_BRIGHT = ListedColormap(["#FF0000", "#0000FF"])


def main():
    compare_classifiers(make_datasets())


def _gen_classifiers() -> MappingProxyType[str, base.ClassifierMixin]:
    return MappingProxyType(
        {
            "Nearest Neighbors": neighbors.KNeighborsClassifier(5),
            # "Linear SVM": svm.SVC(kernel="linear", C=0.025),
            "Logistic Regression": linear_model.LogisticRegression(),
            # "RBF SVM": svm.SVC(gamma=2, C=1),
            # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
            # "Decision Tree": tree.DecisionTreeClassifier(max_depth=5),
            # "Random forest": ensemble.RandomForestClassifier(
            #     max_depth=5, n_estimators=10, max_features=1
            # ),
            # "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
            # "AdaBoost": ensemble.AdaBoostClassifier(),
            # "Naive Bayes": GaussianNB(),
            # "QDA": QuadraticDiscriminantAnalysis(),
        }
    )


def compare_classifiers(
    datasets: Collection[Dataset],
    classifiers: Mapping[str, base.ClassifierMixin] = _gen_classifiers(),
) -> Sequence[Sequence[base.ClassifierMixin]]:
    """Plot 2-D features."""
    _, axs = plt.subplots(
        len(datasets),
        len(classifiers) + 1,  # To plot input data
        figsize=(min(len(classifiers) * 7 + 7, 27), max(len(datasets) * 3, 6)),
    )
    if axs.ndim == 1:
        axs = [axs]

    trained_clfs: list[Sequence[base.ClassifierMixin]] = []
    for ds_cnt, dataset in enumerate(datasets):
        X, y = dataset
        clf = plot_dataset_row(X, y, axs[ds_cnt], classifiers, ds_cnt)
        trained_clfs.append(clf)

    plt.tight_layout()
    plt.show()
    return trained_clfs


def plot_dataset_row(
    X,
    y,
    axs,
    classifiers: Mapping[str, base.ClassifierMixin],
    ds_cnt: int,
    cm=plt.cm.RdBu,
) -> Sequence[base.ClassifierMixin]:
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    xx, yy = _make_mesh(X)
    i = 0

    ax = axs[i]
    if ds_cnt == 0:
        ax.set_title("Input data")

    plot_train(ax, X_train, y_train)
    plot_test(ax, X_test, y_test)

    set_ax_bound(ax, xx, yy)

    trained_clfs: list[base.ClassifierMixin] = []

    for name, clf in classifiers.items():
        i += 1
        clf = plot_classifier(
            axs[i],
            clf,
            X_train,
            y_train,
            X_test,
            y_test,
            xx,
            yy,
            cm,
            name,
            ds_cnt,
        )
        trained_clfs.append(clf)
    return trained_clfs


def _make_mesh(X: np.ndarray, step: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    return xx, yy


def plot_train(ax, X, y, cmap=CM_BRIGHT):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k")


def plot_test(ax, X, y, cmap=CM_BRIGHT):
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        s=100,
        alpha=0.6,
        edgecolors='w',
    )


def set_ax_bound(ax, xx, yy):
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())


def plot_classifier(
    ax,
    clf: base.ClassifierMixin,
    X_train,
    y_train,
    X_test,
    y_test,
    xx,
    yy,
    cm,
    name,
    ds_cnt,
) -> base.ClassifierMixin:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    _plot_decision_boundary(ax, clf, xx, yy, cm)

    # plot_train(ax, X_test, y_test)
    plot_test(ax, X_test, y_test)
    set_ax_bound(ax, xx, yy)

    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        f"{score:.2f}",
        size=15,
        horizontalalignment="right",
    )

    return clf


def _plot_decision_boundary(ax, clf, xx, yy, cm):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)


def make_datasets(
    types: Sequence[str] = ('linear',), seed: int = 2
) -> Sequence[Dataset]:

    linear = datasets.make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=seed,
        n_clusters_per_class=1,
    )
    linear2 = datasets.make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=1,
        n_clusters_per_class=1,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)

    def _add_noise(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X += 2 * rng.uniform(size=X.shape)
        return X, y

    _generated_data = {
        'linear': _add_noise(*linear),
        'linear2': _add_noise(*linear2),
        'moon': datasets.make_moons(noise=0.3, random_state=seed),
        'circle': datasets.make_circles(noise=0.2, factor=0.5, random_state=seed),
    }
    return tuple(_generated_data[name] for name in types)
