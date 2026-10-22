
import unittest

import numpy as np

from algorithms.decision_tree_classifier import (
    DecisionTreeClassifierScratch
)


class TestDecisionTreeClassifierScratch(unittest.TestCase):

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

    def test_binary_classification(self):
        model = DecisionTreeClassifierScratch().fit(
            self.X, self.y
        )

        np.testing.assert_array_equal(
            model.predict(self.X),
            self.y
        )

    def test_unseen_predictions(self):
        model = DecisionTreeClassifierScratch().fit(
            self.X, self.y
        )

        np.testing.assert_array_equal(
            model.predict([
                [0.2, 0.2],
                [4.5, 4.2]
            ]),
            [0, 1]
        )

    def test_multiclass(self):
        X = [
            [0], [1],
            [5], [6],
            [10], [11]
        ]

        y = ["a", "a", "b", "b", "c", "c"]

        model = DecisionTreeClassifierScratch().fit(
            X, y
        )

        np.testing.assert_array_equal(
            model.predict([
                [0.5],
                [5.5],
                [10.5]
            ]),
            ["a", "b", "c"]
        )

    def test_depth_limit(self):
        X = [[0], [1], [2], [3], [4], [5]]
        y = [0, 0, 1, 1, 0, 0]

        model = DecisionTreeClassifierScratch(
            max_depth=1
        ).fit(X, y)

        self.assertFalse(
            "feature" in model.tree_.get("left", {})
        )

        self.assertFalse(
            "feature" in model.tree_.get("right", {})
        )

    def test_min_samples_split_stops_split(self):
        model = DecisionTreeClassifierScratch(
            min_samples_split=7
        ).fit(self.X, self.y)

        self.assertNotIn(
            "feature",
            model.tree_
        )

    def test_min_samples_leaf_stops_split(self):
        model = DecisionTreeClassifierScratch(
            min_samples_leaf=4
        ).fit(self.X, self.y)

        self.assertNotIn(
            "feature",
            model.tree_
        )

    def test_constant_features_majority_vote(self):
        model = DecisionTreeClassifierScratch().fit(
            [[1], [1], [1], [1]],
            [1, 1, 0, 1]
        )

        np.testing.assert_array_equal(
            model.predict([[1], [2]]),
            [1, 1]
        )

    def test_tied_labels_choose_smallest(self):
        model = DecisionTreeClassifierScratch(
            min_samples_split=3
        ).fit(
            [[1], [1]],
            [2, 1]
        )

        self.assertEqual(
            model.predict([[1]])[0],
            1
        )

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifierScratch().predict(
                [[1, 2]]
            )

    def test_invalid_training_shapes(self):
        cases = [
            ([1, 2], [0, 1]),
            ([[1], [2]], [0]),
            ([], [])
        ]

        for X, y in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifierScratch().fit(
                        X, y
                    )

    def test_nonfinite_training_features(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifierScratch().fit(
                [[0], [np.nan]],
                [0, 1]
            )

    def test_invalid_prediction_features(self):
        model = DecisionTreeClassifierScratch().fit(
            self.X, self.y
        )

        cases = [
            [[0, 1, 2]],
            [[np.inf, 1]],
            [0, 1]
        ]

        for X in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_hyperparameters(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(max_depth=value):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifierScratch(
                        max_depth=value
                    )

            with self.subTest(min_samples_leaf=value):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifierScratch(
                        min_samples_leaf=value
                    )

        for value in [0, 1, -1, True, 1.5]:
            with self.subTest(min_samples_split=value):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifierScratch(
                        min_samples_split=value
                    )

    def test_repeatable_training(self):
        a = DecisionTreeClassifierScratch().fit(
            self.X, self.y
        )

        b = DecisionTreeClassifierScratch().fit(
            self.X, self.y
        )

        np.testing.assert_array_equal(
            a.predict(self.X),
            b.predict(self.X)
        )


if __name__ == "__main__":
    unittest.main()
