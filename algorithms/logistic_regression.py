
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Inputs must have matching, nonempty shapes")

    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("Labels must be binary")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Probabilities must be finite")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Probabilities must be between 0 and 1")

    p = np.clip(y_prob, 1e-12, 1 - 1e-12)

    return float(-np.mean(
        y_true * np.log(p) +
        (1 - y_true) * np.log(1 - p)
    ))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, epochs=2000):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("Epochs must be a positive integer")

        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input dimensions")

        if len(X) == 0 or X.shape[1] == 0 or len(X) != len(y):
            raise ValueError("Invalid training data")

        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Labels must be 0 or 1")

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        n = len(y)

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            probabilities = sigmoid(z)

            error = probabilities - y

            dw = (X.T @ error) / n
            db = np.mean(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        if not hasattr(self, "weights"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Invalid prediction dimensions")

        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


if __name__ == "__main__":
    X = np.array([
        [-2, -2],
        [-1, -1],
        [-2, -1],
        [1, 1],
        [2, 2],
        [1, 2]
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch()
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print("Predictions:", predictions)
    print("Accuracy:", np.mean(predictions == y))
    print("Loss:", binary_cross_entropy(y, probabilities))
