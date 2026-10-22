
import unittest

import numpy as np

from algorithms.perceptron_classifier import PerceptronClassifier


class TestPerceptronClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    def test_and_gate(self):
        y = np.array([0, 0, 0, 1])
        model = PerceptronClassifier().fit(self.X, y)

        np.testing.assert_array_equal(
            model.predict(self.X), y
        )

    def test_or_gate(self):
        y = np.array([0, 1, 1, 1])
        model = PerceptronClassifier().fit(self.X, y)

        np.testing.assert_array_equal(
            model.predict(self.X), y
        )

    def test_prediction_shape(self):
        model = PerceptronClassifier().fit(
            self.X, [0, 0, 0, 1]
        )

        result = model.predict([[0, 0], [1, 1]])
        self.assertEqual(result.shape, (2,))

    def test_decision_function(self):
        model = PerceptronClassifier().fit(
            self.X, [0, 0, 0, 1]
        )

        scores = model.decision_function(self.X)

        np.testing.assert_array_equal(
            (scores >= 0).astype(int),
            model.predict(self.X)
        )

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            PerceptronClassifier().predict([[1, 2]])

    def test_mismatched_samples(self):
        with self.assertRaises(ValueError):
            PerceptronClassifier().fit(self.X, [0])

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            PerceptronClassifier().fit(
                self.X, [0, 0, 2, 1]
            )

    def test_invalid_training_dimensions(self):
        cases = [
            (np.array([1, 2]), [0, 1]),
            ([], []),
            ([[], []], [0, 1])
        ]

        for X, y in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    PerceptronClassifier().fit(X, y)

    def test_nonfinite_training_features(self):
        with self.assertRaises(ValueError):
            PerceptronClassifier().fit(
                [[float("nan"), 0]], [1]
            )

    def test_invalid_prediction_data(self):
        model = PerceptronClassifier().fit(
            self.X, [0, 0, 0, 1]
        )

        cases = [
            [[0, 1, 2]],
            [[float("inf"), 1]],
            [0, 1]
        ]

        for X in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_invalid_hyperparameters(self):
        for rate in [0, -1, True, float("inf"), float("nan")]:
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    PerceptronClassifier(learning_rate=rate)

        for epochs in [0, -1, True, 1.5]:
            with self.subTest(epochs=epochs):
                with self.assertRaises(ValueError):
                    PerceptronClassifier(epochs=epochs)

    def test_deterministic(self):
        a = PerceptronClassifier().fit(
            self.X, [0, 0, 0, 1]
        )

        b = PerceptronClassifier().fit(
            self.X, [0, 0, 0, 1]
        )

        np.testing.assert_array_equal(
            a.weights_, b.weights_
        )

        self.assertEqual(a.bias_, b.bias_)
        self.assertEqual(a.n_iter_, b.n_iter_)


if __name__ == "__main__":
    unittest.main()
