
import numpy as np


class KMeansScratch:
    """K-Means clustering implemented using NumPy."""

    def __init__(self, n_clusters=2, max_iter=100,
                 tol=1e-4, random_state=42):

        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")

        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")

        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("random_state must be an integer")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D array")

        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed sample count")

        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(
            len(X),
            size=self.n_clusters,
            replace=False
        )

        centers = X[indices].copy()

        for iteration in range(self.max_iter):
            squared_distances = np.sum(
                (X[:, None, :] - centers[None, :, :]) ** 2,
                axis=2
            )

            labels = np.argmin(squared_distances, axis=1)
            updated = centers.copy()

            for cluster in range(self.n_clusters):
                members = X[labels == cluster]

                if len(members):
                    updated[cluster] = members.mean(axis=0)

            movement = np.max(
                np.linalg.norm(updated - centers, axis=1)
            )

            centers = updated

            if movement <= self.tol:
                break

        self.cluster_centers_ = centers
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = iteration + 1

        self.labels_ = self.predict(X)

        self.inertia_ = float(
            np.sum((X - centers[self.labels_]) ** 2)
        )

        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same feature count as training data"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        squared_distances = np.sum(
            (
                X[:, None, :]
                - self.cluster_centers_[None, :, :]
            ) ** 2,
            axis=2
        )

        return np.argmin(squared_distances, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [9, 9],
        [9, 10],
        [10, 9]
    ])

    model = KMeansScratch(n_clusters=2)
    model.fit(X)

    print("Labels:", model.labels_)
    print("Centers:")
    print(model.cluster_centers_)
    print("Inertia:", round(model.inertia_, 4))
