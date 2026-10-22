
import unittest

import numpy as np

from algorithms.knn_classifier import KNNClassifier


class TestKNNClassifier(unittest.TestCase):

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

    def test_predictions(self):
        model = KNNClassifier(k=3)
        model.fit(self.X, self.y)

        result = model.predict([
            [0.2, 0.2],
            [4.5, 4.2]
        ])

        np.testing.assert_array_equal(result, [0, 1])

    def test_single_neighbor(self):
        model = KNNClassifier(k=1)
        model.fit(self.X, self.y)

        result = model.predict([[4.1, 4.1]])

        self.assertEqual(result[0], 1)

    def test_tie_breaking(self):
        model = KNNClassifier(k=2)
        model.fit([[0, 0], [0, 2]], [2, 1])

        result = model.predict([[0, 1]])

        self.assertEqual(result[0], 1)

    def test_predict_before_fit(self):
        model = KNNClassifier()

        with self.assertRaises(RuntimeError):
            model.predict([[1, 2]])

    def test_invalid_k(self):
        for k in [0, -1, True, 1.5]:
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    KNNClassifier(k=k)

    def test_k_exceeds_samples(self):
        model = KNNClassifier(k=10)

        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)

    def test_invalid_training_data(self):
        model = KNNClassifier()

        with self.assertRaises(ValueError):
            model.fit([[1, 2], [3, 4]], [0])

    def test_feature_dimension_mismatch(self):
        model = KNNClassifier(k=3)
        model.fit(self.X, self.y)

        with self.assertRaises(ValueError):
            model.predict([[1, 2, 3]])

    def test_non_finite_features(self):
        model = KNNClassifier(k=1)

        with self.assertRaises(ValueError):
            model.fit([[np.nan, 1]], [0])

        model.fit(self.X, self.y)

        with self.assertRaises(ValueError):
            model.predict([[np.inf, 1]])

    def test_fit_copies_training_data(self):
        X = np.array([
            [0.0, 0.0],
            [10.0, 10.0]
        ])

        y = np.array([0, 1])

        model = KNNClassifier(k=1).fit(X, y)

        # Modify the original arrays after fitting.
        X[0] = [100.0, 100.0]
        y[0] = 9

        # The fitted model should remain unchanged.
        result = model.predict([[0.1, 0.1]])

        np.testing.assert_array_equal(result, [0])


if __name__ == "__main__":
    unittest.main()
