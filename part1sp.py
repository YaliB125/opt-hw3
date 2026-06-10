import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from scipy.sparse import random  

def Jacobi_method(A, b, x0, omega=1.0, max_iterations=100, eps=1e-10):
    diag_A = A.diagonal()
    b = np.array(b, dtype=float).flatten()
    x = np.array(x0, dtype=float).flatten()
    
    residuals = [np.linalg.norm(A.dot(x) - b)]
    
    for k in range(max_iterations):
        r = b - A.dot(x)
        if np.linalg.norm(r) < eps:
            break
            
        x = x + omega * (r / diag_A)
        residuals.append(np.linalg.norm(A.dot(x) - b))
        
    return x, residuals


def gauss_seidel_method(A, b, x0, max_iterations=100, eps=1e-10):
    b = np.array(b, dtype=float).flatten()
    x = np.array(x0, dtype=float).flatten()
    
    L_D = sparse.tril(A).tocsr()
    U = sparse.triu(A, k=1).tocsr()

    residuals = [np.linalg.norm(b - A.dot(x))]
    
    for k in range(max_iterations):
        r = b - A.dot(x)
        if np.linalg.norm(r) < eps:
            break
            
        # Solve (L+D)x^{(k+1)} = b - U x^{(k)}
        rhs = b - U.dot(x)
        x = splinalg.spsolve_triangular(L_D, rhs, lower=True)
        residuals.append(np.linalg.norm(b - A.dot(x)))
        
    return x, residuals


def steepest_descent(A, b, x0, maxIter=100, eps=1e-10):
    b = np.array(b, dtype=float).flatten()
    x = np.array(x0, dtype=float).flatten()
    r = b - A.dot(x)
    
    residuals = [np.linalg.norm(r)]
    
    for k in range(maxIter):
        if np.linalg.norm(r) < eps:
            break

        Ar = A.dot(r)
        alpha = np.dot(r, r) / np.dot(r, Ar)

        x = x + alpha * r
        r = r - alpha * Ar
        residuals.append(np.linalg.norm(r))
        
    return x, residuals



def conjugate_gradient(A, b, x0, maxIter=100, eps=1e-10):
    b = np.array(b, dtype=float).flatten()
    x = np.array(x0, dtype=float).flatten()
    r = b - A.dot(x)
    p = r.copy() 
    
    residuals = [np.linalg.norm(r)]
    
    for k in range(maxIter):
        if np.linalg.norm(r) < eps:
            break
        Ap = A.dot(p)
        r_dot_r_old = np.dot(r, r) 
        denominator_alpha = np.dot(p, Ap)
        
        if denominator_alpha == 0: break
        alpha = r_dot_r_old / denominator_alpha
  
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(np.linalg.norm(r))        
        
        r_dot_r_new = np.dot(r, r)
        beta = r_dot_r_new / r_dot_r_old
        p = r + beta * p
        
    return x, residuals



def compute_factors(res_list):
    factors = []
    for k in range(1, len(res_list)):
        factors.append(res_list[k] / res_list[k-1])
    return factors


n = 256
A = random(n, n, 5 / n, dtype=float)
v = np.random.rand(n)
v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
A = A.transpose() * v * A + 0.1*sparse.eye(n)
b = np.random.randn(n)
x0 = np.zeros(n)
max_iterations = 100

x_jacobi, jacobi_res = Jacobi_method(A, b, x0, omega=1.0, max_iterations=max_iterations)

if jacobi_res[-1] > jacobi_res[0]:
    for w in np.arange(0.1, 1.0, 0.1):
        x_temp, res_temp = Jacobi_method(A, b, x0, omega=w, max_iterations=max_iterations)
        
        if res_temp[-1] < res_temp[0]:
            x_jacobi, jacobi_res = x_temp, res_temp
            jacobi_label = f"Jacobi (Weighted, $\omega$={w:.1f})"
            break

x_gs, gs_res = gauss_seidel_method(A, b, x0, max_iterations=max_iterations)
x_sd, sd_res = steepest_descent(A, b, x0, maxIter=max_iterations)
x_cg, cg_res = conjugate_gradient(A, b, x0, maxIter=max_iterations)

x_sol_part_b = x_cg 

jacobi_fac = compute_factors(jacobi_res)
gs_fac = compute_factors(gs_res)
sd_fac = compute_factors(sd_res)
cg_fac = compute_factors(cg_res)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.semilogy(jacobi_res, label=jacobi_label, color='blue')
plt.semilogy(gs_res, label="Gauss-Seidel", color='green')
plt.semilogy(sd_res, label="Steepest Descent", color='orange')
plt.semilogy(cg_res, label="Conjugate Gradient (CG)", color='red')
plt.title("Convergence Graph: $||Ax^{(k)} - b||_2$")
plt.xlabel("Iteration (k)")
plt.ylabel("Residual Norm (Log Scale)")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(jacobi_fac, label=jacobi_label, color='blue')
plt.plot(gs_fac, label="Gauss-Seidel", color='green')
plt.plot(sd_fac, label="Steepest Descent", color='orange')
plt.plot(cg_fac, label="Conjugate Gradient (CG)", color='red')
plt.title("Residual Convergence Factor")
plt.xlabel("Iteration (k)")
plt.ylabel("Factor Value")
plt.ylim(0, 1.2)
plt.grid(True, ls="--")
plt.legend()

plt.tight_layout()
plt.show()


A_normal = A.transpose().dot(A)
b_normal = A.transpose().dot(b)

jacobi_label_norm = "Jacobi (Standard, $\omega$=1.0)"
x_jacobi_norm, jacobi_res_norm = Jacobi_method(A_normal, b_normal, x0, omega=1.0, max_iterations=max_iterations)

if jacobi_res_norm[-1] > jacobi_res_norm[0]:
    for w in np.arange(0.1, 1.0, 0.1):
        x_temp, res_temp = Jacobi_method(A_normal, b_normal, x0, omega=w, max_iterations=max_iterations)
        
        if res_temp[-1] < res_temp[0]:
            x_jacobi_norm, jacobi_res_norm = x_temp, res_temp
            jacobi_label_norm = f"Jacobi (Weighted, $\omega$={w:.1f})"
            break

x_gs_norm, gs_res_norm = gauss_seidel_method(A_normal, b_normal, x0, max_iterations=max_iterations)
x_sd_norm, sd_res_norm = steepest_descent(A_normal, b_normal, x0, maxIter=max_iterations)
x_cg_norm, cg_res_norm = conjugate_gradient(A_normal, b_normal, x0, maxIter=max_iterations)

# Compute the difference between part B and part C solutions
difference = np.linalg.norm(x_sol_part_b - x_cg_norm)
print(f"Difference between solutions = {difference:.2e}")
print(np.linalg.norm(A @ x_sol_part_b - b), np.linalg.norm(A @ x_cg_norm - b))

jacobi_fac_norm = compute_factors(jacobi_res_norm)
gs_fac_norm = compute_factors(gs_res_norm)
sd_fac_norm = compute_factors(sd_res_norm)
cg_fac_norm = compute_factors(cg_res_norm)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.semilogy(jacobi_res_norm, label=jacobi_label_norm, color='blue')
plt.semilogy(gs_res_norm, label="Gauss-Seidel", color='green')
plt.semilogy(sd_res_norm, label="Steepest Descent", color='orange')
plt.semilogy(cg_res_norm, label="Conjugate Gradient (CG)", color='red')
plt.title("Normal Equations: $||(A^TA)x^{(k)} - A^Tb||_2$")
plt.xlabel("Iteration (k)")
plt.ylabel("Residual Norm (Log Scale)")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(jacobi_fac_norm, label=jacobi_label_norm, color='blue')
plt.plot(gs_fac_norm, label="Gauss Seidel", color='green')
plt.plot(sd_fac_norm, label="Steepest Descent", color='orange')
plt.plot(cg_fac_norm, label="Conjugate Gradient (CG)", color='red')
plt.title("Normal Equations Convergence Factor")
plt.xlabel("Iteration (k)")
plt.ylabel("Factor Value")
plt.ylim(0, 1.2)
plt.grid(True, ls="--")
plt.legend()

plt.tight_layout()
plt.show()