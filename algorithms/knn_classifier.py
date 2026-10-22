
import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")

        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input dimensions")

        if len(X) == 0 or len(X) != len(y):
            raise ValueError("Invalid training data")

        if self.k > len(X):
            raise ValueError("k cannot exceed training samples")

        if not np.all(np.isfinite(X)):
            raise ValueError("Training features must be finite")

        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Call fit before predict")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Invalid prediction dimensions")

        if not np.all(np.isfinite(X)):
            raise ValueError("Prediction features must be finite")

        predictions = []

        for point in X:
            distances = np.linalg.norm(
                self.X_train - point,
                axis=1
            )

            nearest_indices = np.argsort(
                distances, kind="stable"
            )[:self.k]

            nearest_labels = self.y_train[nearest_indices]

            labels, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            prediction = labels[np.argmax(counts)]
            predictions.append(prediction)

        return np.asarray(predictions)


if __name__ == "__main__":
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [4, 4],
        [4, 5],
        [5, 4]
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    X_test = np.array([
        [0.2, 0.2],
        [4.5, 4.2]
    ])

    predictions = model.predict(X_test)

    print("Predictions:", predictions)
