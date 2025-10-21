'''
Problem 2: Computations (6 points; 3 points per task)
In this problem, perform numerical computations in python using the values provided in the csv file.
• Compute H and C.
• Compute LS solutions  𝑥𝑜𝑝𝑡  and  𝐿𝑜𝑝𝑡  by means of SVD.
'''

import numpy as np
import os
import pandas as pd

#using this function to calculate the gradient of L(x)
def gradientL(x, A, y):
  return 2 * A.T @ (A @ x - y)

#calculating the gradient at the initial point "xint"
gradientAtXinit = gradientL(xInit, A, y)

print("Gradient of L(x) at xInit:")
print(gradientAtXinit)

#making sure A is an numerical array
A = np.asarray(A, dtype=float)

#calculating Hessian matrix H
H = 2 * A.T @ A

print(f"Hessian matrix H is \n {H}")

#computing the SVD of A
U, epsilon, V = np.linalg.svd(A) #U = complex unitary matrix; epsilon = rectangular diagonal matrix; V = conjugate transpose

#setting a tolerance in case of super small singular values (epsilon)
tolerance = 1e-10
epsilonInv = np.zeros_like(s)
epsilonInv[epsilon > tolerance] = 1.0 / epsilon[epsilon > tolerance]

#making the diagonal matrix of singular values
epsilonInv = np.zeros((A.shape[1], A.shape[0]))
epsilonInv[:A.shape[1], :A.shape[1]] = np.diag(epsilonInv)

#computing the pseudo inverse of A using SVD components
pseudoInverseA = Vh.T @ epsilonInv @ U.T

#computing the optimal solution xopt
xopt = pseudoInverseA @ y

#compute the optimal value Lopt
Lopt = np.linalg.norm(A @ xopt - y)**2

print(f"The optimal solution x_opt is \n{x_opt}")
print(f"\nThe optimal value L_opt is {L_opt}")
