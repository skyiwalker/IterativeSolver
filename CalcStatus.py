import numpy as np

# prints the solution and error
def print_solution_error(A, b, x) :
    print("Solution:")
    print(x)
    error = np.dot(A, x) - b
    print("Error:")
    print(error)

# prints the system
def print_system(A, b) :
    print("System:")
    for i in range(A.shape[0]):
        row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
        print(" + ".join(row), "=", b[i])
    print()