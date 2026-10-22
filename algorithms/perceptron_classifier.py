
import numpy as np


class PerceptronClassifier:
    """Binary perceptron with mistake-driven weight updates."""

    def __init__(self, learning_rate=0.1, epochs=100):
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")

        self.learning_rate = float(learning_rate)
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if not len(X) or X.shape[1] == 0 or len(X) != len(y):
            raise ValueError("Training data must be nonempty and aligned")

        if not np.all(np.isfinite(X)):
            raise ValueError("Training features must be finite")

        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Labels must be 0 or 1")

        self.weights_ = np.zeros(X.shape[1], dtype=float)
        self.bias_ = 0.0
        self.n_iter_ = 0

        for epoch in range(self.epochs):
            mistakes = 0

            for features, target in zip(X, y):
                score = np.dot(features, self.weights_) + self.bias_
                prediction = int(score >= 0)

                update = self.learning_rate * (int(target) - prediction)

                if update:
                    self.weights_ += update * features
                    self.bias_ += update
                    mistakes += 1

            self.n_iter_ = epoch + 1

            if mistakes == 0:
                break

        return self

    def decision_function(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before predicting")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.weights_):
            raise ValueError("Prediction features have wrong dimensions")

        if not np.all(np.isfinite(X)):
            raise ValueError("Prediction features must be finite")

        return X @ self.weights_ + self.bias_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    model = PerceptronClassifier()
    model.fit(X, y)

    print("Predictions:", model.predict(X))
    print("Weights:", model.weights_)
    print("Bias:", model.bias_)
    print("Epochs:", model.n_iter_)
