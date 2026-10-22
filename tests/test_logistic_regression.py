
import unittest

import numpy as np

from algorithms.logistic_regression import (
    LogisticRegressionScratch,
    sigmoid,
    binary_cross_entropy,
)


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [-2, -2],
            [-1, -1],
            [-2, -1],
            [1, 1],
            [2, 2],
            [1, 2]
        ])

        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_training(self):
        model = LogisticRegressionScratch()
        model.fit(self.X, self.y)

        predictions = model.predict(self.X)
        np.testing.assert_array_equal(predictions, self.y)

    def test_probabilities(self):
        model = LogisticRegressionScratch()
        model.fit(self.X, self.y)

        probabilities = model.predict_proba(self.X)

        self.assertEqual(probabilities.shape, (6,))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_sigmoid(self):
        result = sigmoid(np.array([-1000, 0, 1000]))

        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(result[1], 0.5)

    def test_binary_cross_entropy(self):
        loss = binary_cross_entropy(
            np.array([0, 1]),
            np.array([0.1, 0.9])
        )

        self.assertAlmostEqual(loss, -np.log(0.9))

        with self.assertRaises(ValueError):
            binary_cross_entropy([0, 1], [[0.1], [0.9]])

    def test_predict_before_fit(self):
        model = LogisticRegressionScratch()

        with self.assertRaises(RuntimeError):
            model.predict([[1, 2]])

    def test_invalid_labels(self):
        labels = np.array([0, 0, 2, 1, 1, 1])

        with self.assertRaises(ValueError):
            LogisticRegressionScratch().fit(self.X, labels)

    def test_mismatched_samples(self):
        with self.assertRaises(ValueError):
            LogisticRegressionScratch().fit(
                self.X, [0, 1]
            )

    def test_invalid_prediction_dimensions(self):
        model = LogisticRegressionScratch()
        model.fit(self.X, self.y)

        with self.assertRaises(ValueError):
            model.predict([[1, 2, 3]])

    def test_non_finite_features(self):
        X = self.X.astype(float).copy()
        X[0, 0] = np.nan

        with self.assertRaises(ValueError):
            LogisticRegressionScratch().fit(X, self.y)

    def test_invalid_hyperparameters(self):
        for rate in [0, -0.1, float("inf")]:
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    LogisticRegressionScratch(learning_rate=rate)

        for epochs in [0, -1, True, 1.5]:
            with self.subTest(epochs=epochs):
                with self.assertRaises(ValueError):
                    LogisticRegressionScratch(epochs=epochs)


if __name__ == "__main__":
    unittest.main()
