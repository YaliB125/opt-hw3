import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [5, 4, 4, -1, 0],
    [3, 12, 4, -5, -5],
    [-4, 2, 6, 0, 3],
    [4, 5, -7, 10, 2],
    [1, 2, 5, 3, 10]
])

b = np.array([1, 1, 1, 1, 1])
x = np.array([0, 0, 0, 0, 0], dtype=float)

iterations = 50
residual_norms = []
r = b - A @ x
for k in range(iterations):
    Ar = A @ r
    
    numerator = np.dot(r, Ar)
    denominator = np.dot(Ar, Ar)
    alpha = numerator / denominator
    
    x = x + alpha * r
    
    r = r - alpha * Ar
    residual_norms.append(np.linalg.norm(r))

plt.semilogy(range(1, iterations + 1), residual_norms, marker='o', markersize=3)
plt.title('GMRES(1) Convergence')
plt.xlabel('Iteration (k)')
plt.ylabel('Residual Norm ||r^(k)||')
plt.grid(True, which="both", ls="--")
plt.show()