import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
rho_eau = 1025     # kg/m³
g = 9.81           # m/s²
d = 0.02           # diamètre câble (m)
l = 1              # longueur d'un segment (m)
N = 100            # nombre de segments
V = 1.0            # vitesse du courant (m/s)
C_d = 1.2          # coeff. traînée pour cylindre
S = d * l          # surface frontale du segment
m = 0.5            # masse d'un segment (kg)
W = (m - rho_eau * np.pi * (d/2)**2 * l) * g
F_T = 0.5 * rho_eau * C_d * S * V**2

# Initialisation
T = np.zeros(N+1)
theta = np.zeros(N+1)
x = np.zeros(N+1)
y = np.zeros(N+1)

# Conditions initiales au fond
T[-1] = 10  # tension au fond supposée
theta[-1] = np.radians(85)  # presque verticale

# Boucle de récurrence
for i in reversed(range(N)):
    Tx = T[i+1]*np.cos(theta[i+1]) + F_T
    Ty = T[i+1]*np.sin(theta[i+1]) + W
    T[i] = np.hypot(Tx, Ty)
    theta[i] = np.arctan2(Ty, Tx)

# Reconstruction de la forme
for i in range(1, N+1):
    x[i] = x[i-1] + l * np.cos(theta[i-1])
    y[i] = y[i-1] + l * np.sin(theta[i-1])

# Affichage
plt.plot(x, y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Ligne de mouillage")
plt.gca().invert_yaxis()
plt.grid(True)
plt.axis('equal')
plt.show()
