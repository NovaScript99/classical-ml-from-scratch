
"""A decision tree classifier using Gini impurity."""

import numpy as np


class DecisionTreeClassifierScratch:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError(
                "max_depth must be None or a positive integer"
            )

        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, int)
            or min_samples_split < 2
        ):
            raise ValueError(
                "min_samples_split must be an integer >= 2"
            )

        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError(
                "min_samples_leaf must be a positive integer"
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    @staticmethod
    def _gini(y):
        counts = np.unique(y, return_counts=True)[1]
        probabilities = counts / len(y)

        return 1.0 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        parent_impurity = self._gini(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_mask = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])

            for lower, upper in zip(values[:-1], values[1:]):
                threshold = lower + (upper - lower) / 2.0

                mask = X[:, feature] <= threshold

                n_left = np.count_nonzero(mask)
                n_right = len(y) - n_left

                if min(n_left, n_right) < self.min_samples_leaf:
                    continue

                impurity = (
                    n_left * self._gini(y[mask])
                    + n_right * self._gini(y[~mask])
                ) / len(y)

                gain = parent_impurity - impurity

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_mask = mask

        return best_feature, best_threshold, best_mask

    def _grow(self, X, y, depth):
        classes, counts = np.unique(
            y, return_counts=True
        )

        prediction = classes[np.argmax(counts)]

        node = {"prediction": prediction}

        if len(classes) == 1 or len(y) < self.min_samples_split:
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            return node

        if len(y) < 2 * self.min_samples_leaf:
            return node

        feature, threshold, mask = self._best_split(X, y)

        if feature is None:
            return node

        node["feature"] = feature
        node["threshold"] = threshold

        node["left"] = self._grow(
            X[mask],
            y[mask],
            depth + 1
        )

        node["right"] = self._grow(
            X[~mask],
            y[~mask],
            depth + 1
        )

        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(
                "X must be 2D and y must be 1D"
            )

        if (
            len(X) == 0
            or X.shape[1] == 0
            or len(X) != len(y)
        ):
            raise ValueError(
                "Training data must be nonempty and aligned"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Training features must be finite"
            )

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.tree_ = self._grow(
            X, y, depth=0
        )

        return self

    def predict(self, X):
        if not hasattr(self, "tree_"):
            raise RuntimeError(
                "Call fit before predict"
            )

        X = np.asarray(X, dtype=float)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                "Prediction features have the wrong dimensions"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Prediction features must be finite"
            )

        predictions = []

        for row in X:
            node = self.tree_

            while "feature" in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            predictions.append(node["prediction"])

        return np.asarray(
            predictions,
            dtype=self.classes_.dtype
        )


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [4, 4],
        [4, 5],
        [5, 4]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTreeClassifierScratch(
        max_depth=3
    )

    model.fit(X, y)

    predictions = model.predict([
        [0.2, 0.2],
        [4.5, 4.5]
    ])

    accuracy = np.mean(
        model.predict(X) == y
    )

    print("Predictions:", predictions)
    print("Training accuracy:", accuracy)
