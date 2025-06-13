import numpy as np
import matplotlib.pyplot as plt


def reynolds_number(rho, v, L_char, mu):
    return rho * v * L_char / mu


def calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
):
    # Nombre de Reynolds sonar
    L_char_sonar = D_sonar
    Re = reynolds_number(rho, v, L_char_sonar, mu)

    # Nombre de Reynolds câble
    L_char_cable = D_cable
    Re = reynolds_number(rho, v, L_char_cable, mu)

    deltaL = L_cable / n

    Fx = [Fx_1]
    Fy = [Fy_1]
    alpha = [alpha_1]

    for i in range(n):
        cos_alpha = np.cos(alpha[-1])
        sin_alpha = np.sin(alpha[-1])

        delta_Fx = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * (
                    Cp_cable * cos_alpha ** 3 + Cf_cable * sin_alpha ** 3)
        delta_Fy = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * cos_alpha * sin_alpha * (
                -Cp_cable * cos_alpha + Cf_cable * sin_alpha) + g * deltaL * ((rho * np.pi * D_cable ** 2) / 4 - d)

        Fx_next = Fx[-1] - delta_Fx
        Fy_next = Fy[-1] - delta_Fy

        Fx.append(Fx_next)
        Fy.append(Fy_next)

        delta_Fx_diff = Fx_next + Fx[-2]
        delta_Fy_diff = Fy_next + Fy[-2]

        alpha_next = np.arctan2(-delta_Fx_diff, delta_Fy_diff)
        alpha.append(alpha_next)

    # Calcul de la trajectoire
    x = [0]
    y = [0]
    for i in range(n):
        dx = deltaL * np.sin(alpha[i])
        dy = deltaL * np.cos(alpha[i])
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    return x, y, Fx, Fy, alpha


# Paramètres de base
rho = 1027  # kg/m^3
mu = ((1.89 + 0.8) / 2) * 10 ** (-3)  # Pa.s
g = 9.81  # m/s^2
m = 6.7  # kg
Cp_cable = 1.1
Cf_cable = 0.004
D_sonar = 0.06  # m
D_cable = 0.00630
L_sonar = 0.00850  # m
Cp_sphere_sonar = 0.5
Cf_cyl_sonar = 0.002
S_sphere_sonar = np.pi * D_sonar ** 2
S_cyl_sonar = np.pi * D_sonar * L_sonar
n = 1000
d = 4.0  # g/m

# ==============================================
# 1. Graphes pour différentes vitesses
# ==============================================
L_cable = 200  # Longueur fixe
speeds = [0.5, 1.0, 1.5, 2.0, 3.0]  # m/s

plt.figure(figsize=(12, 8))
for v in speeds:
    # Calcul des tensions initiales
    Tx = 0.5 * rho * (v ** 2) * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g
    alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
    Fx_1 = -Tx
    Fy_1 = Ty

    # Calcul du layback
    x, y, Fx, Fy, alpha = calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
    )

    plt.plot(x, y, label=f'Vitesse = {v} m/s')

plt.xlabel("Déplacement horizontal (m)")
plt.ylabel("Profondeur (m)")
plt.title(f"Profil du câble pour différentes vitesses (L={L_cable}m)")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

# ==============================================
# 2. Graphes pour différentes longueurs de câble
# ==============================================
v = 1.5  # Vitesse fixe
cable_lengths = [50, 100, 150, 200]  # m

plt.figure(figsize=(12, 8))
for L_cable in cable_lengths:
    # Calcul des tensions initiales
    Tx = 0.5 * rho * (v ** 2) * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g
    alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
    Fx_1 = -Tx
    Fy_1 = Ty

    # Calcul du layback
    x, y, Fx, Fy, alpha = calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
    )

    plt.plot(x, y, label=f'Longueur câble = {L_cable}m')

plt.xlabel("Déplacement horizontal (m)")
plt.ylabel("Profondeur (m)")
plt.title(f"Profil du câble pour différentes longueurs (v={v}m/s)")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

# ==============================================
# 3. Graphes pour différents angles initiaux
# ==============================================
v = 1.5  # Vitesse fixe
L_cable = 150  # Longueur fixe
initial_angles = [30, 45, 60, 75]  # degrés

plt.figure(figsize=(12, 8))
for angle_deg in initial_angles:
    # Calcul des tensions initiales
    Tx = 0.5 * rho * (v ** 2) * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g

    # Forcer l'angle initial
    alpha_1 = np.radians(angle_deg)
    Fx_1 = -Tx * np.cos(alpha_1)
    Fy_1 = Ty * np.sin(alpha_1)

    # Calcul du layback
    x, y, Fx, Fy, alpha = calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
    )

    plt.plot(x, y, label=f'Angle initial = {angle_deg}°')

plt.xlabel("Déplacement horizontal (m)")
plt.ylabel("Profondeur (m)")
plt.title(f"Profil du câble pour différents angles initiaux (v={v}m/s, L={L_cable}m)")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

# ==============================================
# 4. Graphes des tensions et angles le long du câble
# ==============================================
v = 1.5  # Vitesse
L_cable = 200  # Longueur

# Calcul des tensions initiales
Tx = 0.5 * rho * (v ** 2) * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
Ty = m * g
alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
Fx_1 = -Tx
Fy_1 = Ty

# Calcul du layback
x, y, Fx, Fy, alpha = calculate_layback(
    rho, v, mu, g, m,
    Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
    Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
    alpha_1, Fx_1, Fy_1
)

s = np.linspace(0, L_cable, n + 1)

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(s, Fx, label='Fx')
plt.ylabel('Fx (N)')
plt.title('Tensions et angles le long du câble')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(s, Fy, label='Fy', color='orange')
plt.ylabel('Fy (N)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(s, np.degrees(alpha), label='α', color='green')
plt.xlabel('Longueur du câble (m)')
plt.ylabel('α (°)')
plt.grid()

plt.tight_layout()
plt.show()