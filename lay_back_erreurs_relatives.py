import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# === Fonctions ===

def reynolds_number(rho, v, L_char, mu):
    return rho * v * L_char / mu

def find_alpha(Fx_i, Fy_i, g, rho, D_cable, d, deltaL, Cp, V):
    def equation(alpha):
        return (
            Fy_i * np.sin(alpha) - Fx_i * np.cos(alpha)
            + 0.5 * g * deltaL * ((rho * np.pi * D_cable**2) / 4 - d)
            - 0.25 * rho * Cp * D_cable * V**2 * np.cos(alpha)**2
        )
    alpha_guess = np.pi / 4
    return fsolve(equation, alpha_guess)[0]

def calculate_layback(rho, v, mu, g, m,
                      Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
                      Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
                      alpha_1, Fx_1, Fy_1):

    deltaL = L_cable / n
    Fx, Fy, alpha = [Fx_1], [Fy_1], [alpha_1]

    for _ in range(n):
        cos_a = np.cos(alpha[-1])
        sin_a = np.sin(alpha[-1])

        delta_Fx = 0.5 * rho * np.pi * D_cable * deltaL * v**2 * (Cp_cable * cos_a**3 + Cf_cable * sin_a**3)
        delta_Fy = (0.5 * rho * np.pi * D_cable * deltaL * v**2 * cos_a * sin_a * (-Cp_cable * cos_a + Cf_cable * sin_a)
                    + g * deltaL * ((rho * np.pi * D_cable**2) / 4 - d))

        Fx_next = Fx[-1] - delta_Fx
        Fy_next = Fy[-1] - delta_Fy

        Fx.append(Fx_next)
        Fy.append(Fy_next)

        alpha_next = find_alpha(Fx_next, Fy_next, g, rho, D_cable, d, deltaL, Cp_cable, v)
        alpha.append(alpha_next)

    # Calcul du profil
    x, y = [0], [0]
    for a in alpha:
        dx = deltaL * np.sin(a)
        dy = deltaL * np.cos(a)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    return x[-1], y[-1]  # layback, depth

# === Paramètres physiques ===

rho = 1027  # kg/m³
mu = ((1.89 + 0.8) / 2 ) * 1e-3  # Pa.s
g = 9.81  # m/s²
m = 6.7  # kg

Cp_cable = 1.1
Cf_cable = 0.004
D_sonar = 0.06
D_cable = 0.00630
L_sonar = 0.850
Cp_sphere_sonar = 0.5
Cf_cyl_sonar = 0.002
S_sphere_sonar = np.pi * D_sonar**2
S_cyl_sonar = np.pi * D_sonar * L_sonar
n = 10000
d = 40 / 1000  # kg/m

# === Données constructeur (copiées depuis l'image) ===

cable_outs = [10, 25, 50, 100, 150, 200]
speeds_knots = [0.5, 1, 1.5, 2, 3, 5]
depth_table = np.array([
    [10, 9, 8, 7, 6, 4],
    [24, 24, 22, 20, 14, 8],
    [50, 45, 40, 33, 22, 12],
    [98, 81, 64, 48, 33, 19],
    [138, 124, 80, 55, 40, 24],
    [186, 124, 80, 55, 40, 24],
])
layback_table = np.array([
    [0, 0, 1, 2, 4, 7],
    [4, 7, 11, 17, 28, 42],
    [15, 27, 35, 42, 56, 77],
    [51, 73, 83, 93, 120, 147],
    [95, 121, 133, 142, 162, 183],
    [141, 171, 183, 192, 192, 197]
])

# === Boucle de vérification ===

print(f"{'Cable':>5} {'Speed':>5} | {'Layback':>8} {'Ref':>6} {'Err%':>6} | {'Depth':>8} {'Ref':>6} {'Err%':>6}")
print("-"*65)

for i, L_cable in enumerate(cable_outs):
    for j, speed_knots in enumerate(speeds_knots):
        v = speed_knots * 0.514444  # converti en m/s

        Tx = 0.5 * rho * v**2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
        Ty = m * g
        alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
        Fx_1 = -Tx
        Fy_1 = -Ty

        try:
            layback_sim, depth_sim = calculate_layback(
                rho, v, mu, g, m,
                Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
                Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
                alpha_1, Fx_1, Fy_1
            )
        except Exception as e:
            print(f"Erreur à {L_cable}m, {speed_knots}kt : {e}")
            continue

        layback_ref = layback_table[i, j]
        depth_ref = depth_table[i, j]

        err_lay = abs(layback_sim - layback_ref) / layback_ref * 100 if layback_ref != 0 else 0
        err_dep = abs(depth_sim - depth_ref) / depth_ref * 100 if depth_ref != 0 else 0

        print(f"{L_cable:5} {speed_knots:5.1f} | {layback_sim:8.2f} {layback_ref:6} {err_lay:6.1f} | {depth_sim:8.2f} {depth_ref:6} {err_dep:6.1f}")
