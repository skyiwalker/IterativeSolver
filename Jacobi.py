import numpy as np

def jacobi(A, b, x0, TOLERANCE = 1.0e-10, MAX_ITERATIONS = 100):

    x = np.zeros_like(b)
    for it_count in range(MAX_ITERATIONS):
        print("Current solution:", x)
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, x_new, atol=TOLERANCE):
            break

        x = x_new
    
    return x