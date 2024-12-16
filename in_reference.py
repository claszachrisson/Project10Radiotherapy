import numpy as np
from numba import njit

@njit
def is_row_in_reference(matrix_to_check, reference_matrix):
    # Get the dimensions
    m, k = matrix_to_check.shape
    n, s = reference_matrix.shape
    
    # Output array to store results
    result = np.zeros(m, dtype=np.bool_)
    
    for i in range(m):
        # Compare the i-th row of matrix_to_check with all rows of reference_matrix
        for j in range(n):
            if np.array_equal(matrix_to_check[i], reference_matrix[j]):
                result[i] = True
                break  # No need to check further
    return result