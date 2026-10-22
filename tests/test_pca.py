
import unittest

import numpy as np

from algorithms.pca import PCAScratch


class TestPCAScratch(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 2], [2, 4], [3, 6], [4, 8], [5, 10]
        ], dtype=float)

    def test_reduced_dimensions(self):
        model = PCAScratch(n_components=1).fit(self.X)
        self.assertEqual(model.transform(self.X).shape, (5, 1))
        self.assertEqual(model.components_.shape, (1, 2))

    def test_fit_transform_matches_transform(self):
        model = PCAScratch(n_components=1)
        actual = model.fit_transform(self.X)
        np.testing.assert_allclose(actual, model.transform(self.X))

    def test_transformed_data_is_centered(self):
        reduced = PCAScratch(n_components=1).fit_transform(self.X)
        np.testing.assert_allclose(
            reduced.mean(axis=0), [0], atol=1e-12
        )

    def test_first_component_has_maximum_variance(self):
        model = PCAScratch(n_components=1).fit(self.X)
        direction = np.array([1, 2]) / np.sqrt(5)
        np.testing.assert_allclose(
            model.components_[0], direction, atol=1e-12
        )

    def test_components_are_orthonormal(self):
        model = PCAScratch(n_components=2).fit(self.X)
        np.testing.assert_allclose(
            model.components_ @ model.components_.T,
            np.eye(2), atol=1e-12
        )

    def test_full_reconstruction(self):
        model = PCAScratch(n_components=2)
        reduced = model.fit_transform(self.X)
        np.testing.assert_allclose(
            model.inverse_transform(reduced), self.X, atol=1e-12
        )

    def test_explained_variance_is_descending(self):
        model = PCAScratch(n_components=2).fit(self.X)
        self.assertTrue(
            np.all(np.diff(model.explained_variance_) <= 1e-12)
        )

    def test_explained_variance_ratio(self):
        model = PCAScratch(n_components=2).fit(self.X)
        self.assertAlmostEqual(
            model.explained_variance_ratio_.sum(), 1.0
        )
        self.assertAlmostEqual(
            model.explained_variance_ratio_[0], 1.0
        )

    def test_constant_data(self):
        X = np.ones((4, 3))
        model = PCAScratch(n_components=2).fit(X)
        np.testing.assert_array_equal(
            model.explained_variance_ratio_, [0, 0]
        )
        np.testing.assert_allclose(
            model.transform(X), np.zeros((4, 2))
        )

    def test_default_component_count(self):
        model = PCAScratch().fit(self.X)
        self.assertEqual(model.n_components_, 2)
        self.assertEqual(model.transform(self.X).shape, (5, 2))

    def test_reproducible_directions(self):
        a = PCAScratch(n_components=2).fit(self.X)
        b = PCAScratch(n_components=2).fit(self.X)
        np.testing.assert_array_equal(
            a.components_, b.components_
        )

    def test_transform_before_fit(self):
        model = PCAScratch()
        with self.assertRaises(RuntimeError):
            model.transform(self.X)
        with self.assertRaises(RuntimeError):
            model.inverse_transform([[0]])

    def test_invalid_component_count(self):
        for count in [0, -1, True, 1.5, "2"]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    PCAScratch(n_components=count)

        with self.assertRaises(ValueError):
            PCAScratch(n_components=3).fit(self.X)

    def test_invalid_training_and_prediction_data(self):
        cases = [
            [],
            [1, 2],
            [[np.nan, 1], [2, 3]],
            [[1, 2]]
        ]

        for X in cases:
            with self.subTest(training=X):
                with self.assertRaises(ValueError):
                    PCAScratch().fit(X)

        model = PCAScratch(n_components=1).fit(self.X)

        for X in [[1, 2], [[1, 2, 3]], [[np.inf, 1]], []]:
            with self.subTest(transform=X):
                with self.assertRaises(ValueError):
                    model.transform(X)

        for X in [[1, 2], [[1, 2]], [[np.nan]], []]:
            with self.subTest(inverse_transform=X):
                with self.assertRaises(ValueError):
                    model.inverse_transform(X)


if __name__ == "__main__":
    unittest.main()
