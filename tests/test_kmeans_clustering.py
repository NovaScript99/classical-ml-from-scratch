
import unittest

import numpy as np

from algorithms.kmeans_clustering import KMeansScratch


class TestKMeansScratch(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [9, 9],
            [9, 10],
            [10, 9]
        ])

    def test_clusters_separated_points(self):
        model = KMeansScratch(n_clusters=2).fit(self.X)

        a = model.labels_[:3]
        b = model.labels_[3:]

        self.assertTrue(np.all(a == a[0]))
        self.assertTrue(np.all(b == b[0]))
        self.assertNotEqual(a[0], b[0])

    def test_centers(self):
        model = KMeansScratch().fit(self.X)

        actual = model.cluster_centers_[
            np.argsort(model.cluster_centers_[:, 0])
        ]

        expected = np.array([
            [1 / 3, 1 / 3],
            [28 / 3, 28 / 3]
        ])

        np.testing.assert_allclose(actual, expected)

    def test_predict(self):
        model = KMeansScratch().fit(self.X)

        predicted = model.predict([
            [0.2, 0.3],
            [9.2, 9.3]
        ])

        self.assertEqual(predicted[0], model.labels_[0])
        self.assertEqual(predicted[1], model.labels_[3])

    def test_deterministic(self):
        a = KMeansScratch(random_state=42).fit(self.X)
        b = KMeansScratch(random_state=42).fit(self.X)

        np.testing.assert_array_equal(a.labels_, b.labels_)

        np.testing.assert_allclose(
            a.cluster_centers_,
            b.cluster_centers_
        )

    def test_inertia(self):
        model = KMeansScratch().fit(self.X)

        self.assertAlmostEqual(model.inertia_, 8 / 3)

    def test_fit_predict(self):
        model = KMeansScratch()

        result = model.fit_predict(self.X)

        np.testing.assert_array_equal(
            result,
            model.labels_
        )

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            KMeansScratch().predict([[0, 0]])

    def test_invalid_hyperparameters(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(n_clusters=value):
                with self.assertRaises(ValueError):
                    KMeansScratch(n_clusters=value)

        for value in [0, -1, True, 1.5]:
            with self.subTest(max_iter=value):
                with self.assertRaises(ValueError):
                    KMeansScratch(max_iter=value)

        with self.assertRaises(ValueError):
            KMeansScratch(tol=-1)

    def test_invalid_training_data(self):
        cases = [
            [],
            [0, 1],
            [[np.nan, 0], [1, 2]]
        ]

        for X in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    KMeansScratch().fit(X)

        with self.assertRaises(ValueError):
            KMeansScratch(n_clusters=7).fit(self.X)

    def test_invalid_prediction_data(self):
        model = KMeansScratch().fit(self.X)

        with self.assertRaises(ValueError):
            model.predict([[1, 2, 3]])

        with self.assertRaises(ValueError):
            model.predict([[np.inf, 0]])


if __name__ == "__main__":
    unittest.main()
