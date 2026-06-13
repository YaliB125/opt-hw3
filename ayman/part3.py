import numpy as np
import matplotlib.pyplot as plt

def GMRES_method(A, b, x0, max_iter, epsilon):
    b_norm = np.linalg.norm(b)

    r = b - A @ x0

    x = x0.copy()

    residual_norms = []
    iterations = []

    for k in range(max_iter):
        iterations.append(k)

        residual_norm = np.linalg.norm(r)
        residual_norms.append(residual_norm)

        if  residual_norm / b_norm < epsilon:
            break

        Ar = A @ r
        numerator = np.dot(r, Ar)
        denominator = np.dot(Ar, Ar)
        alpha = numerator / denominator

        new_solution = x + alpha * r
        new_residual = r - alpha * Ar

        x = new_solution
        r = new_residual
    
    return x, residual_norms, iterations

A = np.array([
    [5, 4, 4, -1, 0],
    [3, 12, 4, -5, -5],
    [-4, 2, 6, 0, 3],
    [4, 5, -7, 10, 2],
    [1, 2, 5, 3, 10]
])

b = np.array([1, 1, 1, 1, 1])
x0 = np.array([0, 0, 0, 0, 0], dtype=float)

max_iter = 50
atol = 1e-12

sol, residual_norms, iterations = GMRES_method(A, b, x0, max_iter, atol)
plt.semilogy(np.array(iterations), np.array(residual_norms))
plt.xlabel("iterations")
plt.ylabel("residual norm (||Axk - b||)")
plt.title("GMRES(1) residual norm")
plt.grid(True)
plt.show()