
import numpy as np


def predict(x, w, b):
    return w * x + b


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Target and prediction shapes must match")

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")

    return np.mean((y_true - y_pred) ** 2)


def train(x, y, learning_rate=0.05, epochs=2000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be nonempty 1D arrays of equal length")

    if learning_rate <= 0 or epochs < 1:
        raise ValueError("Invalid training parameters")

    w = 0.0
    b = 0.0
    n = len(x)

    for _ in range(epochs):
        y_pred = predict(x, w, b)
        error = y_pred - y

        dw = (2 / n) * np.dot(x, error)
        db = (2 / n) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


if __name__ == "__main__":
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 3, 5, 7])

    w, b = train(x, y)

    predictions = predict(x, w, b)

    print(f"Weight: {w:.4f}")
    print(f"Bias: {b:.4f}")
    print(f"MSE: {mse(y, predictions):.8f}")
