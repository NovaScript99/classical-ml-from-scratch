
import unittest

import numpy as np

from algorithms.random_forest_classifier import (
    RandomForestClassifierScratch
)


class TestRandomForestClassifierScratch(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [8, 8],
            [8, 9],
            [9, 8],
            [9, 9]
        ])

        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_binary_predictions(self):
        model = RandomForestClassifierScratch().fit(
            self.X, self.y
        )

        np.testing.assert_array_equal(
            model.predict([
                [0.2, 0.2],
                [8.8, 8.8]
            ]),
            [0, 1]
        )

    def test_training_accuracy(self):
        model = RandomForestClassifierScratch().fit(
            self.X, self.y
        )

        np.testing.assert_array_equal(
            model.predict(self.X),
            self.y
        )

    def test_probability_distribution(self):
        model = RandomForestClassifierScratch().fit(
            self.X, self.y
        )

        probabilities = model.predict_proba([
            [0, 0],
            [8, 8]
        ])

        self.assertEqual(
            probabilities.shape,
            (2, 2)
        )

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            [1, 1]
        )

        self.assertTrue(
            np.all(
                (probabilities >= 0)
                & (probabilities <= 1)
            )
        )

    def test_multiclass(self):
        X = [
            [-10], [-9],
            [0], [1],
            [10], [11]
        ]

        y = ["a", "a", "b", "b", "c", "c"]

        model = RandomForestClassifierScratch(
            n_estimators=51
        ).fit(X, y)

        np.testing.assert_array_equal(
            model.predict([
                [-9.5],
                [0.5],
                [10.5]
            ]),
            ["a", "b", "c"]
        )

    def test_determinism(self):
        a = RandomForestClassifierScratch(
            random_state=7
        ).fit(self.X, self.y)

        b = RandomForestClassifierScratch(
            random_state=7
        ).fit(self.X, self.y)

        np.testing.assert_allclose(
            a.predict_proba(self.X),
            b.predict_proba(self.X)
        )

    def test_fit_resets_estimators(self):
        model = RandomForestClassifierScratch(
            n_estimators=3
        )

        model.fit(self.X, self.y)

        self.assertEqual(
            len(model.estimators_),
            3
        )

        model.fit(self.X, self.y)

        self.assertEqual(
            len(model.estimators_),
            3
        )

    def test_single_class(self):
        model = RandomForestClassifierScratch().fit(
            [[0], [1], [2]],
            [2, 2, 2]
        )

        np.testing.assert_array_equal(
            model.predict([[1], [3]]),
            [2, 2]
        )

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            RandomForestClassifierScratch().predict(
                [[1, 2]]
            )

    def test_invalid_training(self):
        cases = [
            ([1, 2], [0, 1]),
            ([[1], [2]], [0]),
            ([], []),
            ([[], []], [0, 1])
        ]

        for X, y in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    RandomForestClassifierScratch().fit(
                        X, y
                    )

    def test_nonfinite_training(self):
        with self.assertRaises(ValueError):
            RandomForestClassifierScratch().fit(
                [[np.nan], [1]],
                [0, 1]
            )

    def test_invalid_prediction(self):
        model = RandomForestClassifierScratch().fit(
            self.X, self.y
        )

        cases = [
            [[1, 2, 3]],
            [[np.inf, 0]],
            [0, 1],
            []
        ]

        for X in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_invalid_hyperparameters(self):
        for count in [0, -1, True, 1.5]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    RandomForestClassifierScratch(
                        n_estimators=count
                    )

        for depth in [0, True, 1.5]:
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    RandomForestClassifierScratch(
                        max_depth=depth
                    )

        for features in [0, -1, True, 1.5, "invalid"]:
            with self.subTest(features=features):
                with self.assertRaises(ValueError):
                    RandomForestClassifierScratch(
                        max_features=features
                    )

        with self.assertRaises(ValueError):
            RandomForestClassifierScratch(
                random_state=1.5
            )

    def test_max_features_limit(self):
        with self.assertRaises(ValueError):
            RandomForestClassifierScratch(
                max_features=3
            ).fit(self.X, self.y)

    def test_feature_modes(self):
        for features in [None, "log2", "sqrt", 1, 2]:
            with self.subTest(features=features):
                model = RandomForestClassifierScratch(
                    n_estimators=5,
                    max_features=features
                ).fit(self.X, self.y)

                self.assertEqual(
                    len(model.estimators_),
                    5
                )

                self.assertEqual(
                    model.predict([[0.2, 0.2]]).shape,
                    (1,)
                )


if __name__ == "__main__":
    unittest.main()
