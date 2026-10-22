
import unittest

import numpy as np

from algorithms.linear_svm_classifier import (
    LinearSVMClassifierScratch as SVM
)


class TestLinearSVMClassifierScratch(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [-3, -2],
            [-2, -3],
            [-2, -2],
            [2, 2],
            [2, 3],
            [3, 2]
        ])

        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_binary_classification(self):
        model = SVM().fit(self.X, self.y)

        np.testing.assert_array_equal(
            model.predict(self.X),
            self.y
        )

    def test_unseen_predictions(self):
        model = SVM().fit(self.X, self.y)

        np.testing.assert_array_equal(
            model.predict([
                [-2.5, -2],
                [2.5, 2]
            ]),
            [0, 1]
        )

    def test_string_class_labels(self):
        y = np.where(self.y == 0, "cat", "dog")

        model = SVM().fit(self.X, y)

        np.testing.assert_array_equal(
            model.predict([[-2, -2], [2, 2]]),
            ["cat", "dog"]
        )

    def test_negative_positive_labels(self):
        labels = np.where(self.y == 0, -1, 1)

        model = SVM().fit(self.X, labels)

        np.testing.assert_array_equal(
            model.predict([[-2, -2], [2, 2]]),
            [-1, 1]
        )

    def test_decision_function_consistency(self):
        model = SVM().fit(self.X, self.y)

        scores = model.decision_function(self.X)

        np.testing.assert_array_equal(
            model.predict(self.X),
            model.classes_[(scores >= 0).astype(int)]
        )

    def test_prediction_shape(self):
        model = SVM().fit(self.X, self.y)

        self.assertEqual(
            model.predict([[0, 1], [1, 2]]).shape,
            (2,)
        )

        self.assertEqual(
            model.decision_function([[0, 1]]).shape,
            (1,)
        )

    def test_learned_parameters(self):
        model = SVM().fit(self.X, self.y)

        self.assertEqual(model.weights_.shape, (2,))
        self.assertGreater(np.linalg.norm(model.weights_), 0)
        self.assertEqual(model.n_features_in_, 2)
        self.assertEqual(model.n_iter_, 1000)

    def test_objective_improves(self):
        model = SVM().fit(self.X, self.y)

        signed = np.where(self.y == 0, -1, 1)
        margins = signed * model.decision_function(self.X)

        objective = (
            0.5 * np.sum(model.weights_ ** 2)
            + model.C * np.maximum(0, 1 - margins).mean()
        )

        self.assertLess(objective, model.C)

    def test_deterministic(self):
        a = SVM().fit(self.X, self.y)
        b = SVM().fit(self.X, self.y)

        np.testing.assert_array_equal(
            a.weights_,
            b.weights_
        )

        self.assertEqual(a.bias_, b.bias_)

    def test_predict_before_fit(self):
        with self.assertRaises(RuntimeError):
            SVM().predict([[1, 2]])

        with self.assertRaises(RuntimeError):
            SVM().decision_function([[1, 2]])

    def test_invalid_training_shapes(self):
        cases = [
            ([1, 2], [0, 1]),
            ([[1], [2]], [0]),
            ([], []),
            ([[], []], [0, 1])
        ]

        for X, y in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    SVM().fit(X, y)

    def test_invalid_class_count(self):
        with self.assertRaises(ValueError):
            SVM().fit(
                [[0], [1]],
                [1, 1]
            )

        with self.assertRaises(ValueError):
            SVM().fit(
                [[0], [1], [2]],
                [0, 1, 2]
            )

    def test_nonfinite_training_data(self):
        with self.assertRaises(ValueError):
            SVM().fit(
                [[np.nan], [1]],
                [0, 1]
            )

        with self.assertRaises(ValueError):
            SVM().fit(
                [[0], [1]],
                [0, np.inf]
            )

    def test_invalid_prediction_data(self):
        model = SVM().fit(self.X, self.y)

        cases = [
            [0, 1],
            [[0, 1, 2]],
            [[np.inf, 0]],
            []
        ]

        for X in cases:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    model.predict(X)

    def test_invalid_hyperparameters(self):
        for value in [0, -1, True, float("nan"), float("inf")]:
            with self.subTest(learning_rate=value):
                with self.assertRaises(ValueError):
                    SVM(learning_rate=value)

            with self.subTest(C=value):
                with self.assertRaises(ValueError):
                    SVM(C=value)

        for value in [0, -1, True, 1.5]:
            with self.subTest(max_iter=value):
                with self.assertRaises(ValueError):
                    SVM(max_iter=value)


if __name__ == "__main__":
    unittest.main()
