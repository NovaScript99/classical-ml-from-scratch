
import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier using NumPy."""

    def __init__(self, var_smoothing=1e-9):
        if (
            not isinstance(var_smoothing, (int, float))
            or isinstance(var_smoothing, bool)
            or not np.isfinite(var_smoothing)
            or var_smoothing <= 0
        ):
            raise ValueError(
                "var_smoothing must be positive and finite"
            )

        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths")

        if not np.all(np.isfinite(X)):
            raise ValueError("Training features must be finite")

        self.classes_, counts = np.unique(
            y, return_counts=True
        )

        self.class_prior_ = counts / len(y)

        self.mean_ = np.array([
            X[y == label].mean(axis=0)
            for label in self.classes_
        ])

        self.var_ = np.array([
            np.maximum(
                X[y == label].var(axis=0),
                self.var_smoothing
            )
            for label in self.classes_
        ])

        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "classes_"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Prediction data must be 2D")

        if len(X) == 0:
            raise ValueError("Prediction data cannot be empty")

        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError("Feature dimensions do not match")

        if not np.all(np.isfinite(X)):
            raise ValueError("Prediction features must be finite")

        squared = (
            X[:, None, :] - self.mean_[None, :, :]
        ) ** 2

        log_density = -0.5 * np.sum(
            np.log(
                2.0 * np.pi * self.var_
            )[None, :, :]
            + squared / self.var_[None, :, :],
            axis=2
        )

        return (
            log_density
            + np.log(self.class_prior_)[None, :]
        )

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)

        shifted = scores - scores.max(
            axis=1, keepdims=True
        )

        likelihood = np.exp(shifted)

        return likelihood / likelihood.sum(
            axis=1, keepdims=True
        )

    def predict(self, X):
        scores = self._joint_log_likelihood(X)

        indices = np.argmax(scores, axis=1)

        return self.classes_[indices]


if __name__ == "__main__":
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [4.0, 4.0],
        [4.0, 5.0],
        [5.0, 4.0]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    X_test = np.array([
        [0.2, 0.1],
        [4.5, 4.2]
    ])

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("Predictions:", predictions)
    print("Probabilities:")
    print(probabilities)
