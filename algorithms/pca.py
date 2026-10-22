
"""Principal Component Analysis implemented with NumPy."""

import numpy as np


class PCAScratch:
    """Reduce feature dimensions with principal components."""

    def __init__(self, n_components=None):
        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
            or n_components < 1
        ):
            raise ValueError("n_components must be None or a positive integer")

        self.n_components = n_components

    @staticmethod
    def _validate_array(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D array")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        return X

    def fit(self, X):
        X = self._validate_array(X)

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples")

        max_components = min(n_samples, n_features)
        count = self.n_components or max_components

        if count > max_components:
            raise ValueError("n_components exceeds available dimensions")

        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        _, singular_values, directions = np.linalg.svd(
            centered, full_matrices=False
        )

        # Make component directions reproducible.
        for direction in directions:
            pivot = np.argmax(np.abs(direction))
            if direction[pivot] < 0:
                direction *= -1

        all_variances = singular_values ** 2 / (n_samples - 1)
        total_variance = all_variances.sum()

        self.n_features_in_ = n_features
        self.n_components_ = count
        self.components_ = directions[:count].copy()
        self.singular_values_ = singular_values[:count].copy()
        self.explained_variance_ = all_variances[:count].copy()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
            if total_variance > 0
            else np.zeros(count, dtype=float)
        )

        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before transform")

        X = self._validate_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match training data")

        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before inverse_transform")

        X = self._validate_array(X)

        if X.shape[1] != self.n_components_:
            raise ValueError("Component count does not match fitted PCA")

        return X @ self.components_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    data = np.array([
        [1, 2], [2, 4], [3, 6], [4, 8], [5, 10]
    ], dtype=float)

    model = PCAScratch(n_components=1)
    reduced = model.fit_transform(data)

    print("Original shape:", data.shape)
    print("Reduced shape:", reduced.shape)
    print("Explained variance ratio:", model.explained_variance_ratio_)
    print("Reduced data:\n", reduced)
