import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def reynolds_number(rho, v, L_char, mu):
    return rho * v * L_char / mu

def find_alpha(Fx_i, Fy_i, g, rho, D_cable, d, deltaL, Cp, V):
    def equation(alpha):
        return (
            Fy_i * np.sin(alpha) -
            Fx_i * np.cos(alpha) +
            0.5 * g * deltaL * ((rho * np.pi * D_cable ** 2) / 4 - d) -
            0.25 * rho * Cp * D_cable * V ** 2 * np.cos(alpha) ** 2
        )
    alpha_guess = np.pi / 4
    alpha_solution = fsolve(equation, alpha_guess)[0]
    return alpha_solution

def calculate_layback(
    rho, v, mu, g, m,
    Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
    Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
    alpha_1, Fx_1, Fy_1
):
    # Reynolds
    Re_sonar = reynolds_number(rho, v, D_sonar, mu)
    print(f"Reynolds du sonar = {Re_sonar:.2e}")
    Re_cable = reynolds_number(rho, v, D_cable, mu)
    print(f"Reynolds du câble = {Re_cable:.2e}")

    deltaL = L_cable / n
    Fx = [Fx_1]
    Fy = [Fy_1]
    alpha = [alpha_1]

    for i in range(n):
        cos_alpha = np.cos(alpha[-1])
        sin_alpha = np.sin(alpha[-1])

        delta_Fx = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * (Cp_cable * cos_alpha ** 3 + Cf_cable * sin_alpha ** 3)

        delta_Fy = (
            0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * cos_alpha * sin_alpha * (-Cp_cable * cos_alpha + Cf_cable * sin_alpha)
            + g * deltaL * ((rho * np.pi * D_cable ** 2) / 4 - d)
        )

        Fx_next =Fx[-1] - delta_Fx
        Fy_next = Fy[-1] - delta_Fy

        Fx.append(Fx_next)
        Fy.append(Fy_next)

        alpha_next = find_alpha(Fx_next, Fy_next, g, rho, D_cable, d, deltaL, Cp_cable, v)
        alpha.append(alpha_next)

    return Fx, Fy, alpha

# === Paramètres ===
rho = 1027  # kg/m³
mu = ((1.89 + 0.8) / 2 ) * 1e-3  # Pa.s
v = (1.54 + 2.06) / 2  # m/s
g = 9.81  # m/s²
m = 6.7  # kg

Cp_cable = 1.1
Cf_cable = 0.004
D_sonar = 0.06  # m
D_cable = 0.00630  # m
L_sonar = 0.850  # m
L_cable = 50  # m
Cp_sphere_sonar = 0.5
Cf_cyl_sonar = 0.002
S_sphere_sonar = np.pi * D_sonar**2
S_cyl_sonar = np.pi * D_sonar * L_sonar
n = 10000
d = 40 / 1000  # kg/m (converti depuis g/m)

# Tensions initiales
Tx = 0.5 * rho * v ** 2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
Ty = m * g
print(f"Tx : {Tx:.2f} N, Ty : {Ty:.2f} N")

alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
Fx_1 = -Tx
Fy_1 = -Ty
print(f"alpha_1 = {np.degrees(alpha_1):.2f}°")

Fx, Fy, alpha = calculate_layback(
    rho, v, mu, g, m,
    Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
    Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
    alpha_1, Fx_1, Fy_1
)

# === Tracé tensions et angle ===
s = np.linspace(0, L_cable, n + 1)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(s, Fx)
plt.ylabel('Fx (N)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(s, Fy, color='orange')
plt.ylabel('Fy (N)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(s, np.degrees(alpha), color='green')
plt.xlabel('Longueur du câble (m)')
plt.ylabel('α (°)')
plt.grid()

plt.tight_layout()
plt.show()

# === Calcul layback ===
x = [0]
y = [0]
deltaL = L_cable / n

for i in range(n):
    dx = deltaL * np.sin(alpha[i])
    dy = deltaL * np.cos(alpha[i])
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Lay back du câble", color="blue")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Profil (lay back) du câble")
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()


print(f"Fx,max = {Fx[-1]:.4g} N, Fy,max = {Fy[-1]:.4g} N, alpha_final = {np.degrees(alpha[-1]):.4g}°" )
print(f"depth = {y[-1]-y[0]:.4g} m, lay-back = {x[-1]-x[0]:.4g}m")