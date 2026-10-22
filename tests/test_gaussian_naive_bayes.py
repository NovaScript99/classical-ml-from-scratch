
import unittest

import numpy as np

from algorithms.gaussian_naive_bayes import GaussianNaiveBayes


class TestGaussianNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [4, 4],
            [4, 5],
            [5, 4]
        ])

        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_binary_predictions(self):
        model = GaussianNaiveBayes().fit(self.X, self.y)

        result = model.predict([
            [0.2, 0.1],
            [4.5, 4.2]
        ])

        np.testing.assert_array_equal(result, [0, 1])

    def test_multiclass_predictions(self):
        X = [
            [0], [0.2],
            [5], [5.2],
            [10], [10.2]
        ]

        y = ["a", "a", "b", "b", "c", "c"]

        model = GaussianNaiveBayes().fit(X, y)

        result = model.predict([
            [0.1], [5.1], [10.1]
        ])

        np.testing.assert_array_equal(
            result, ["a", "b", "c"]
        )

    def test_probabilities_sum_to_one(self):
        model = GaussianNaiveBayes().fit(self.X, self.y)

        probabilities = model.predict_proba([
            [0, 0],
            [4, 4]
        ])

        self.assertEqual(probabilities.shape, (2, 2))

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            [1.0, 1.0]
        )

        self.assertTrue(
            np.all(
                (probabilities >= 0) &
                (probabilities <= 1)
            )
        )

    def test_class_priors(self):
        X = [[1], [1], [1], [1]]
        y = [0, 0, 0, 1]

        model = GaussianNaiveBayes().fit(X, y)

        np.testing.assert_allclose(
            model.class_prior_,
            [0.75, 0.25]
        )

        np.testing.assert_allclose(
            model.predict_proba([[1]]),
            [[0.75, 0.25]]
        )

    def test_predict_before_fit(self):
        model = GaussianNaiveBayes()

        with self.assertRaises(RuntimeError):
            model.predict([[1, 2]])

    def test_invalid_training_shapes(self):
        cases = [
            ([1, 2], [0, 1]),
            ([[1], [2]], [0]),
            ([], [])
        ]

        for X, y in cases:
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    GaussianNaiveBayes().fit(X, y)

    def test_non_finite_training_features(self):
        with self.assertRaises(ValueError):
            GaussianNaiveBayes().fit(
                [[np.nan], [1]],
                [0, 1]
            )

    def test_invalid_prediction_dimensions(self):
        model = GaussianNaiveBayes().fit(self.X, self.y)

        with self.assertRaises(ValueError):
            model.predict([[1, 2, 3]])

        with self.assertRaises(ValueError):
            model.predict(np.empty((0, 2)))

    def test_non_finite_prediction_features(self):
        model = GaussianNaiveBayes().fit(self.X, self.y)

        with self.assertRaises(ValueError):
            model.predict_proba([[np.inf, 0]])

    def test_invalid_smoothing(self):
        for value in [0, -1, True, float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GaussianNaiveBayes(
                        var_smoothing=value
                    )


if __name__ == "__main__":
    unittest.main()
