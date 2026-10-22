
"""Linear Support Vector Machine classifier using NumPy."""

import numpy as np


class LinearSVMClassifierScratch:
    """Binary SVM trained with batch subgradient descent."""

    def __init__(self, learning_rate=0.01, C=1.0, max_iter=1000):
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Invalid learning_rate")

        if (
            isinstance(C, bool)
            or not isinstance(C, (int, float))
            or not np.isfinite(C)
            or C <= 0
        ):
            raise ValueError("Invalid C")

        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, int)
            or max_iter < 1
        ):
            raise ValueError("Invalid max_iter")

        self.learning_rate = float(learning_rate)
        self.C = float(C)
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if len(X) == 0 or X.shape[1] == 0 or len(X) != len(y):
            raise ValueError("Invalid training data")

        if not np.all(np.isfinite(X)):
            raise ValueError("Training features must be finite")

        if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
            raise ValueError("Training labels must be finite")

        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("SVM requires exactly two classes")

        y_signed = np.where(y == classes[0], -1.0, 1.0)

        n_samples, n_features = X.shape

        weights = np.zeros(n_features, dtype=float)
        bias = 0.0

        for _ in range(self.max_iter):
            margins = y_signed * (X @ weights + bias)
            violations = margins < 1.0

            gradient_w = weights - (
                self.C / n_samples
            ) * (X[violations].T @ y_signed[violations])

            gradient_b = -(
                self.C / n_samples
            ) * np.sum(y_signed[violations])

            weights -= self.learning_rate * gradient_w
            bias -= self.learning_rate * gradient_b

        self.classes_ = classes
        self.weights_ = weights
        self.bias_ = float(bias)
        self.n_features_in_ = n_features
        self.n_iter_ = self.max_iter

        return self

    def decision_function(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Invalid prediction dimensions")

        if len(X) == 0 or not np.all(np.isfinite(X)):
            raise ValueError("Invalid prediction data")

        return X @ self.weights_ + self.bias_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]


if __name__ == "__main__":
    X = np.array([
        [-3, -2],
        [-2, -3],
        [-2, -2],
        [2, 2],
        [2, 3],
        [3, 2]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LinearSVMClassifierScratch().fit(X, y)

    X_test = np.array([
        [-2.5, -2.0],
        [2.5, 2.0]
    ])

    print("Predictions:", model.predict(X_test))
    print("Decision scores:", model.decision_function(X_test))
    print("Weights:", model.weights_)
    print("Bias:", model.bias_)
