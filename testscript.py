import unittest
import numpy as np
from simplex_test import *  # Assuming this is where your simplex function is imported


class Test(unittest.TestCase):
    def test_simplex_example1(self):
        A = np.array([[1, 2, 1, 1, 2, 1, 2], [-2, -1, 0, 1, 2, 0, 1], [0, 1, 2, -1, 1, -2, -1]])
        A = np.hstack((A, np.eye(A.shape[0])))  # Add identity matrix to A
        b = np.array([16, 16, 16])
        C = np.array([[1, 2, -1, 3, 2, 0, 1], [0, 1, 1, 2, 3, 1, 0], [1, 0, 1, -1, 0, -1, -1]])
        C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))  # Add zero columns to C

        indices, result = simplex(A, b, C)
        expected_indices = [[2, 3, 8], [0, 2, 8], [3, 8, 9], [2, 4, 8], [0, 8, 9], [4, 8, 9]]
        expected_result = [
            [0, 0, 10.66666667, 5.33333333, 0, 0, 0, 0, 10.66666667, 0],
            [8, 0, 8, 0, 0, 0, 0, 0, 32, 0],
            [0, 0, 0, 16, 0, 0, 0, 0, 0, 32],
            [0, 0, 5.33333333, 0, 5.33333333, 0, 0, 0, 5.33333333, 0],
            [16, 0, 0, 0, 0, 0, 0, 0, 48, 16],
            [0, 0, 0, 0, 8, 0, 0, 0, 0, 8],
        ]

        # Sorting indices and results before comparison
        try:
            # Compare indices (allow unordered)
            np.testing.assert_array_equal(sorted(indices, key=lambda x: x[0]), sorted(expected_indices, key=lambda x: x[0]))
        except AssertionError as e:
            print("Indices mismatch:")
            print("Expected indices:\n", expected_indices)
            print("Actual indices:\n", indices)
            raise e

        try:
            # Compare result arrays (allow unordered)
            np.testing.assert_array_almost_equal(np.array(sorted(result, key=lambda x: x[0])),
                                                 np.array(sorted(expected_result, key=lambda x: x[0])))
        except AssertionError as e:
            print("Result mismatch:")
            print("Expected result:\n", expected_result)
            print("Actual result:\n", result)
            raise e

    def test_simplex_example2(self):
        A = np.array([[1, 1, 2], [1, 2, 1], [2, 1, 1]])
        A = np.hstack((A, np.eye(A.shape[0])))  # Add identity matrix to A
        b = np.array([12, 12, 12])
        C = np.array([[6, 4, 5], [0, 0, 1]])
        C = np.hstack((C, np.zeros((C.shape[0], A.shape[0]))))  # Add zero columns to C

        indices, result = simplex(A, b, C)
        expected_indices = [[2, 4, 5], [0, 2, 4], [0, 1, 2]]
        expected_result = [
            [0.0, 0.0, 6.0, 0.0, 6.0, 6.0],
            [4.0, 0.0, 4.0, 0.0, 4.0, 0.0],
            [3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
        ]

        # Sorting indices and results before comparison
        try:
            # Compare indices (allow unordered)
            np.testing.assert_array_equal(sorted(indices, key=lambda x: x[0]), sorted(expected_indices, key=lambda x: x[0]))
        except AssertionError as e:
            print("Indices mismatch:")
            print("Expected indices:\n", expected_indices)
            print("Actual indices:\n", indices)
            raise e

        try:
            # Compare result arrays (allow unordered)
            np.testing.assert_array_almost_equal(np.array(sorted(result, key=lambda x: x[0])),
                                                 np.array(sorted(expected_result, key=lambda x: x[0])))
        except AssertionError as e:
            print("Result mismatch:")
            print("Expected result:\n", expected_result)
            print("Actual result:\n", result)
            raise e


if __name__ == "__main__":
    unittest.main()
