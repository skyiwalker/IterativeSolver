import numpy as np
import ConjugateGradient as cg
import GaussSeidel as gs
import Jacobi as jc
import CalcStatus as cs

# initialize the matrix
# A = np.array([[10., -1., 2., 0.],
#               [-1., 11., -1., 3.],
#               [2., -1., 10., -1.],
#               [0.0, 3., -1., 8.]])
A = np.array([[3.,1.],
             [7.,4.]])
# initialize the RHS vector
# b = np.array([6., 25., -11., 15.])
b = np.array([2.,3.])

# initialize unknown vector
x0 = np.zeros_like(b)

cs.print_system(A,b)

# Estimate the solution by Conjugate Gradient
print("-"*20+"Conjugate Gradient"+"-"*20)
x = cg.conjugate_gradient(A,b,x0)
cs.print_solution_error(A,b,x)

# Estimate the solution by Jacobi Method
print("-"*20+"Jacobi"+"-"*20)
x = jc.jacobi(A,b,x0)
cs.print_solution_error(A,b,x)

# Estimate the solution by Gauss-Seidel Method
print("-"*20+"Gauss Seidel"+"-"*20)
x = gs.gauss_seidel(A,b,x0)
cs.print_solution_error(A,b,x)