
import unittest

import numpy as np

from algorithms.linear_regression import mse, predict, train


class TestLinearRegression(unittest.TestCase):

    def test_prediction(self):
        x = np.array([0, 1, 2])
        np.testing.assert_array_equal(
            predict(x, 2, 1), [1, 3, 5]
        )

    def test_mse(self):
        self.assertAlmostEqual(
            mse(np.array([1, 2]), np.array([1, 3])), 0.5
        )

    def test_training(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([1, 3, 5, 7])

        w, b = train(x, y)

        self.assertAlmostEqual(w, 2, places=3)
        self.assertAlmostEqual(b, 1, places=3)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            train([1, 2], [3])


if __name__ == "__main__":
    unittest.main()
