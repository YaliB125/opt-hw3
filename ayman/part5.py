import numpy as np
import matplotlib.pyplot as plt

def parta():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)

    X, Y = np.meshgrid(x, y)

    F1 = X**2 + 2 * Y**2
    F2 = X**2 + 0.5 * Y**2

    plt.contour(X, Y, F1, colors='blue')
    plt.contour(X, Y, F2, colors='red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Level sets of $x^2+2y^2$ (blue) and $x^2+0.5y^2$ (red)")

    # Legend
    plt.plot([], [], color='blue', label=r'$x^2+2y^2$')
    plt.plot([], [], color='red', label=r'$x^2+0.5y^2$')
    plt.legend()

    plt.show()

def partc():
    theta = np.pi / 8

    U = np.array([
        [np.cos(theta),  np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)

    X, Y = np.meshgrid(x, y)

    X_prime = U[0, 0] * X + U[1, 0] * Y
    Y_prime = U[0, 1] * X + U[1, 1] * Y

    F1 = X_prime**2 + 2 * Y_prime**2
    F2 = X_prime**2 + 0.5 * Y_prime**2

    cp = plt.contour(X, Y, F1, 20, colors = 'blue') 
    cp = plt.contour(X, Y, F2, 20, colors = 'red') 
    plt.colorbar(cp) 

    plt.title(r'Rotated Ellipse Contours ($\theta = \pi/8$)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('image')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Legend
    plt.plot([], [], color='blue', label=r"${x'}^2 + 2{y'}^2$")
    plt.plot([], [], color='red', label=r"${x'}^2 + 0.5{y'}^2$")
    plt.legend()

    plt.show()

def main():
    partc()

main()