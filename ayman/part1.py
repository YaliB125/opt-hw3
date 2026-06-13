import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve_triangular
import matplotlib.pyplot as plt

def Jacobi_Method(A, b, x0, max_iter, epsilon):
    D_inv = 1.0 / A.diagonal()

    b_norm = np.linalg.norm(b)
    prev_solution = x0.copy()

    for k in range(max_iter):
        residual = b - A @ prev_solution

        if np.linalg.norm(residual) / b_norm < epsilon:
            break

        new_solution = prev_solution + D_inv * residual
        prev_solution = new_solution
    
    return prev_solution

def Weighted_Jacobi_Method(A, b, x0, w, max_iter, epsilon):
    D_inv = 1.0 / A.diagonal()

    b_norm = np.linalg.norm(b)
    prev_solution = x0.copy()

    residual_norms = []
    iterations = []

    for k in range(max_iter):
        iterations.append(k)

        residual = b - A @ prev_solution
        residual_norm = np.linalg.norm(residual)

        residual_norms.append(residual_norm)

        if  residual_norm / b_norm < epsilon:
            break

        new_solution = prev_solution + w * (D_inv * residual)

        prev_solution = new_solution

    return prev_solution, residual_norms, iterations

def Gauss_Seide(A, b, x0, max_iter, epsilon):
    L_pluse_D = sparse.tril(A).tocsc()

    b_norm = np.linalg.norm(b)
    prev_solution = x0.copy()

    residual_norms = []
    iterations = []

    for k in range(max_iter):
        iterations.append(k)

        residual = b - A @ prev_solution
        residual_norm = np.linalg.norm(residual)

        residual_norms.append(residual_norm)

        if  residual_norm / b_norm < epsilon:
            break

        y = spsolve_triangular(L_pluse_D, residual, lower=True)
        new_solution = prev_solution + y
        
        prev_solution = new_solution
    
    return prev_solution, residual_norms, iterations

def Steepest_Descent(A, b, x0, max_iter, epsilon): # A is SPD
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
        alpha = np.dot(r, r)/np.dot(r, Ar)

        new_solution = x + alpha * r
        new_residual = r - alpha * Ar

        x = new_solution
        r = new_residual
    
    return x, residual_norms, iterations

def Conjugate_Gradient(A, b, x0, max_iter, epsilon):
    b_norm = np.linalg.norm(b)

    x = x0.copy()
    r = b - A @ x
    p = r

    residual_norms = []
    iterations = []

    for k in range(max_iter):
        iterations.append(k)

        residual_norm = np.linalg.norm(r)
        residual_norms.append(residual_norm)

        if residual_norm / b_norm < epsilon:
            break

        Ap = A @ p
        rTr = np.dot(r, r)
        alpha = rTr / np.dot(p, Ap)

        x = x + alpha * p
        new_r = r - alpha * Ap

        beta = np.dot(new_r, new_r) / rTr

        p = new_r + beta * p
        r = new_r
    
    return x, residual_norms, iterations

# helper function
def get_residual_convergence_factor(residual_norms):
    residual_convergence_factor = [] 

    for i in range(1, len(residual_norms)):
        residual_convergence_factor.append(residual_norms[i] / residual_norms[i-1])

    return residual_convergence_factor

def Jacobi_Method_Graphs(A, b, x0, max_iter, atol, title_ext):
    sol, jacobi_residual_norms, jacobi_iterations = Weighted_Jacobi_Method(A, b, x0, 0.1, max_iter, atol)

    plt.semilogy(np.array(jacobi_iterations), np.array(jacobi_residual_norms))
    plt.xlabel("iterations")
    plt.ylabel("residual norm (||Axk - b||)")
    plt.title("Jacobi residual norm: " + title_ext)
    plt.grid(True)
    plt.show()
    
    plt.plot(np.array(jacobi_iterations[1:]), np.array(get_residual_convergence_factor(jacobi_residual_norms)))
    plt.xlabel("iterations")
    plt.ylabel("residual convergence factor")
    plt.title("Jacobi residual convergence factor: " + title_ext)
    plt.show()

    return sol

def Gauss_Seide_Graphs(A, b, x0, max_iter, atol, title_ext):
    sol, gauss_seide_residual_norms, gauss_seide_iterations = Gauss_Seide(A, b, x0, max_iter, atol)

    plt.semilogy(np.array(gauss_seide_iterations), np.array(gauss_seide_residual_norms))
    plt.xlabel("iterations")
    plt.ylabel("residual norm (||Axk - b||)")
    plt.title("Gauss Seide residual norm: " + title_ext)
    plt.grid(True)
    plt.show()
    
    plt.plot(np.array(gauss_seide_iterations[1:]), np.array(get_residual_convergence_factor(gauss_seide_residual_norms)))
    plt.xlabel("iterations")
    plt.ylabel("residual convergence factor")
    plt.title("Gauss Seide residual convergence factor: " + title_ext)
    plt.show()

    return sol

def Steepest_Descent_Graphs(A, b, x0, max_iter, atol, title_ext):
    sol, steepest_descent_residual_norms, steepest_descent_iterations = Steepest_Descent(A, b, x0, max_iter, atol)

    plt.semilogy(np.array(steepest_descent_iterations), np.array(steepest_descent_residual_norms))
    plt.xlabel("iterations")
    plt.ylabel("residual norm (||Axk - b||)")
    plt.title("Steepest Descent residual norm: " + title_ext)
    plt.grid(True)
    plt.show()

    plt.plot(np.array(steepest_descent_iterations[1:]), np.array(get_residual_convergence_factor(steepest_descent_residual_norms)))
    plt.xlabel("iterations")
    plt.ylabel("residual convergence factor")
    plt.title("Steepest Descent residual convergence factor: " + title_ext)
    plt.show()

    return sol

def Conjugate_Gradient_Graphs(A, b, x0, max_iter, atol, title_ext):
    sol, conjugate_gradient_residual_norms, conjugate_gradient_iterations = Conjugate_Gradient(A, b, x0, max_iter, atol)

    plt.semilogy(np.array(conjugate_gradient_iterations), np.array(conjugate_gradient_residual_norms))
    plt.xlabel("iterations")
    plt.ylabel("residual norm (||Axk - b||)")
    plt.title("Conjugate Gradient residual norm: " + title_ext)
    plt.grid(True)
    plt.show()
    
    plt.plot(np.array(conjugate_gradient_iterations[1:]), np.array(get_residual_convergence_factor(conjugate_gradient_residual_norms)))
    plt.xlabel("iterations")
    plt.ylabel("residual convergence factor")
    plt.title("Conjugate Gradient residual convergence factor: " + title_ext)
    plt.show()

    return sol


def main():
    n = 256
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    A = A.transpose() * v * A + 0.1 * sparse.eye(n)

    b = np.random.randn(n)
    x0 = np.zeros(n)
    max_iter = 100
    atol = 1e-12
    
    ATA = A.T @ A
    ATb = A.T @ b

    print("Solve the systems Ax = b and A^TAx = A^Tb:")

    ### Jacobi
    J_sol = Jacobi_Method_Graphs(A, b, x0, max_iter, atol, "Ax = b")
    J_sol_normal_equations = Jacobi_Method_Graphs(ATA, ATb, x0, max_iter, atol, "A^TAx = A^Tb")

    if np.allclose(J_sol, J_sol_normal_equations, atol):
        print("Jacobi method returned same solution to both systems")

    ### Gauss Seide
    GS_sol = Gauss_Seide_Graphs(A, b, x0, max_iter, atol, "Ax = b")
    GS_sol_normal_equations = Gauss_Seide_Graphs(ATA, ATb, x0, max_iter, atol, "A^TAx = A^Tb")

    if np.allclose(GS_sol, GS_sol_normal_equations, atol):
        print("Gauss Seide method returned same solution to both systems")

    ### Steepest Descent 
    SD_sol = Steepest_Descent_Graphs(A, b, x0, max_iter, atol, "Ax = b")
    SD_sol_normal_equations = Steepest_Descent_Graphs(ATA, ATb, x0, max_iter, atol, "A^TAx = A^Tb")

    if np.allclose(SD_sol, SD_sol_normal_equations, atol):
        print("Steepest Descent method returned same solution to both systems")

    ### Conjugate Gradient 
    CG_sol = Conjugate_Gradient_Graphs(A, b, x0, max_iter, atol, "Ax = b")
    CG_sol_normal_equations = Conjugate_Gradient_Graphs(ATA, ATb, x0, max_iter, atol, "A^TAx = A^Tb")

    if np.allclose(CG_sol, CG_sol_normal_equations, atol):
        print("Conjugate Gradient method returned same solution to both systems")

    print("Jacobi difference:", np.linalg.norm(J_sol - J_sol_normal_equations))

    print("GS difference:", np.linalg.norm(GS_sol - GS_sol_normal_equations))

    print("SD difference:", np.linalg.norm(SD_sol - SD_sol_normal_equations))

    print("CG difference:", np.linalg.norm(CG_sol - CG_sol_normal_equations))

    print("Jacboi residual on Ax=b:", np.linalg.norm(b - A @ J_sol))

    print("GS residual on Ax=b:", np.linalg.norm(b - A @ GS_sol))

    print("SD residual on Ax=b:", np.linalg.norm(b - A @ SD_sol))

    print("Jacboi normal residual on ATAx=ATb:", np.linalg.norm(ATb - ATA @ J_sol_normal_equations))

    print("GS normal residual on ATAx=ATb:", np.linalg.norm(ATb - ATA @ GS_sol_normal_equations))

    print("SD normal residual on ATAx=ATb:", np.linalg.norm(ATb - ATA @ SD_sol_normal_equations))

    print("CG residual on Ax=b:", np.linalg.norm(b - A @ CG_sol))

    print("CG normal residual on ATAx=ATb:", np.linalg.norm(ATb - ATA @ CG_sol_normal_equations))

main()
