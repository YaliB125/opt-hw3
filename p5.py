import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) 
F1 = 1 * X**2 + 2 * Y**2
cp1 = plt.contour(X, Y, F1, 20)
plt.colorbar(cp1) 
plt.title('Contours for a=1, b=2\n(Mstretched along X-axis)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('image') 

plt.subplot(1, 2, 2)
F2 = 1 * X**2 + 0.5 * Y**2
cp2 = plt.contour(X, Y, F2, 20)
plt.colorbar(cp2) 
plt.title('Contours for a=1, b=0.5\n(Mstretched along Y-axis)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('image')

plt.tight_layout()
plt.show()


print("part c")

a = 1
b = 2
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

F_rotated = a * (X_prime)**2 + b * (Y_prime)**2

plt.figure(figsize=(6, 6))
cp = plt.contour(X, Y, F_rotated, 20, cmap='viridis') 
plt.colorbar(cp) 

plt.title(r'Rotated Ellipse Contours ($\theta = \pi/8$)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('image')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()