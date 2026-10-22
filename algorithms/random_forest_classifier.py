
"""Random Forest classifier built from scratch using NumPy."""

import numpy as np

from algorithms.decision_tree_classifier import (
    DecisionTreeClassifierScratch
)


class _RandomFeatureTree(DecisionTreeClassifierScratch):
    """Decision tree with random features at each split."""

    def __init__(self, max_depth, max_features, rng):
        super().__init__(max_depth=max_depth)
        self._max_features = max_features
        self._rng = rng

    def _best_split(self, X, y):
        parent_impurity = self._gini(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_mask = None

        features = self._rng.choice(
            X.shape[1],
            size=self._max_features,
            replace=False
        )

        for feature in features:
            values = np.unique(X[:, feature])

            for lower, upper in zip(values[:-1], values[1:]):
                threshold = lower + (upper - lower) / 2.0

                mask = X[:, feature] <= threshold

                n_left = np.count_nonzero(mask)
                n_right = len(y) - n_left

                if n_left == 0 or n_right == 0:
                    continue

                impurity = (
                    n_left * self._gini(y[mask])
                    + n_right * self._gini(y[~mask])
                ) / len(y)

                gain = parent_impurity - impurity

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_mask = mask

        return best_feature, best_threshold, best_mask


class RandomForestClassifierScratch:
    """Ensemble of bootstrapped, randomized decision trees."""

    def __init__(
        self,
        n_estimators=31,
        max_depth=None,
        max_features="sqrt",
        random_state=42
    ):
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, int)
            or n_estimators < 1
        ):
            raise ValueError("Invalid n_estimators")

        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError("Invalid max_depth")

        if not (
            max_features is None
            or max_features in ("sqrt", "log2")
            or (
                isinstance(max_features, int)
                and not isinstance(max_features, bool)
                and max_features >= 1
            )
        ):
            raise ValueError("Invalid max_features")

        if (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
        ):
            raise ValueError("Invalid random_state")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def _feature_count(self, n_features):
        if self.max_features is None:
            return n_features

        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))

        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))

        if self.max_features > n_features:
            raise ValueError(
                "max_features exceeds feature count"
            )

        return self.max_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if (
            len(X) == 0
            or X.shape[1] == 0
            or len(X) != len(y)
        ):
            raise ValueError("Invalid training data")

        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        feature_count = self._feature_count(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []

        for _ in range(self.n_estimators):
            bootstrap = rng.integers(
                0, len(X), size=len(X)
            )

            tree = _RandomFeatureTree(
                max_depth=self.max_depth,
                max_features=feature_count,
                rng=rng
            )

            tree.fit(X[bootstrap], y[bootstrap])
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features_in_
        ):
            raise ValueError("Invalid prediction dimensions")

        if len(X) == 0 or not np.all(np.isfinite(X)):
            raise ValueError("Invalid prediction data")

        votes = np.zeros(
            (len(X), len(self.classes_)),
            dtype=float
        )

        for tree in self.estimators_:
            labels = tree.predict(X)

            indices = np.searchsorted(
                self.classes_,
                labels
            )

            votes[
                np.arange(len(X)),
                indices
            ] += 1

        return votes / self.n_estimators

    def predict(self, X):
        probabilities = self.predict_proba(X)

        return self.classes_[
            np.argmax(probabilities, axis=1)
        ]


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [8, 8],
        [8, 9],
        [9, 8],
        [9, 9]
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = RandomForestClassifierScratch()
    model.fit(X, y)

    X_test = [
        [0.2, 0.2],
        [8.8, 8.8]
    ]

    print("Predictions:", model.predict(X_test))
    print("Probabilities:", model.predict_proba(X_test))
