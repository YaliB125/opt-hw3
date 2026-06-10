import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [5,  4,  4, -1,  0],
    [3, 12,  4, -5, -5],
    [-4, 2,  6,  0,  3],
    [4,  5, -7, 10,  2],
    [1,  2,  5,  3, 10]
])

b = np.array([1, 1, 1, 1, 1])
x = np.array([0, 0, 0, 0, 0]) 

num_iterations = 50
residual = [] 

for k in range(num_iterations):
    r = b - np.dot(A, x)
    
    Ar = np.dot(A, r)
    
    numerator = np.dot(r, Ar)
    denominator = np.dot(Ar, Ar)
    
    alpha = numerator / denominator
    
    x = x + alpha * r
    
    r_new = b - np.dot(A, x)
    res_norm = np.linalg.norm(r_new)
    residual.append(res_norm)

plt.figure(figsize=(8, 5))
plt.semilogy(range(1, num_iterations + 1), residual, 'b-', linewidth=2, label='GMRES(1) Residual')
plt.xlabel('Iteration (k)')
plt.ylabel('Residual Norm ||r^(k)|| (Log Scale)')
plt.title('Convergence of GMRES(1)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()