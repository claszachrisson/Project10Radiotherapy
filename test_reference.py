from in_reference import is_row_in_reference
import numpy as np

matrix_to_check = np.array([[10, 11, 12],
                            [1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]
                            ])

reference_matrix = np.array([[1, 2, 3],
                             [5, 6, 7],
                             [7, 8, 9],
                             [10, 11, 12]])

res = is_row_in_reference(matrix_to_check,reference_matrix)
print(res)