# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#rc35ed51944ec-2

import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
    cost_matrix = np.array([[7, 6, 2, 9, 2],
                            [6, 8, 5, 8, 6]])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(f"row index is {row_ind}")
    print(f"col index is {col_ind}")
    print(f"total cost is {cost_matrix[row_ind,col_ind].sum()}")
    

if __name__ == '__main__':
	main()