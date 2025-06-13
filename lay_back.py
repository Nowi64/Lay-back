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
    L_char_sonar = D_sonar  # ou autre caractéristique pertinente
    Re = reynolds_number(rho, v, L_char_sonar, mu)
    print(f"Reynolds du sonar = {Re:.2e}")

    # Nombre de Reynolds câble
    L_char_cable = D_cable  # ou autre caractéristique pertinente
    Re = reynolds_number(rho, v, L_char_cable, mu)
    print(f"Reynolds du cable = {Re:.2e}")

    # Tensions initiales
    Tx = 0.5 * rho * v ** 2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g

    deltaL = L_cable / n

    Fx = [Fx_1]
    Fy = [Fy_1]
    alpha = [alpha_1]

    for i in range(n):
        cos_alpha = np.cos(alpha[-1])
        sin_alpha = np.sin(alpha[-1])

        delta_Fx = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * (Cp_cable * cos_alpha ** 3 + Cf_cable * sin_alpha ** 3)
        delta_Fy = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * cos_alpha * sin_alpha * (
                    -Cp_cable * cos_alpha + Cf_cable * sin_alpha) + g*deltaL*((rho*np.pi* D_cable**2) /4 - d)

        Fx_next = Fx[-1] - delta_Fx
        Fy_next = Fy[-1] - delta_Fy

        Fx.append(Fx_next)
        Fy.append(Fy_next)

        delta_Fx_diff = Fx_next + Fx[-2]
        delta_Fy_diff = Fy_next + Fy[-2]

        alpha_next = np.arctan2(-delta_Fx_diff, delta_Fy_diff)


        alpha.append(alpha_next)

    return Fx, Fy, alpha



# === Paramètres d'exemple ===

rho = 1027  # kg/m^3 (eau salée)
mu = ((1.89 + 0.8) / 2 ) * 10**(-3)   # Pa.s
v = (1.54 + 2.06) / 2  # m/s (vitesse relative)
g = 9.81  # m/s^2
m = 6.7  # kg (masse sonar)
Cp_cable = 1.1
Cf_cable = 0.004
D_sonar = 0.06  # m (diamètre câble)
D_cable = 0.00630
L_sonar = 0.00850  # m (longueur totale)
L_cable = 50 #à modifier
Cp_sphere_sonar = 0.5
Cf_cyl_sonar = 0.002
S_sphere_sonar = np.pi * D_sonar**2  # m²
S_cyl_sonar = np.pi * D_sonar * L_sonar  # m²
S_sphere_cable = np.pi * D_cable**2  # m²
S_cyl_cable = np.pi * D_cable * L_cable  # m²
n = 1000  # Nombre de tronçons
d = 4.0 / 1000 #g/m

# Tensions initiales
Tx = 0.5 * rho * (v ** 2) * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
Ty = m * g

print(f"Tx : {Tx}, Ty : {Ty}")

# Conditions initiales
alpha_1 = np.pi/2 - np.arctan2(Ty,Tx)
Fx_1 = - Tx  # N
Fy_1 =  Ty  # N

print(f"alpha_1 = {alpha_1* 180 / np.pi}")

Fx, Fy, alpha = calculate_layback(
    rho, v, mu, g, m,
    Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
    Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
    alpha_1, Fx_1, Fy_1
)

# === Tracé des résultats ===
s = np.linspace(0, L_cable, n + 1)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(s, Fx, label='Fx')
plt.ylabel('Fx (N)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(s, Fy, label='Fy', color='orange')
plt.ylabel('Fy (N)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(s, np.degrees(alpha), label='alpha', color='green')
plt.xlabel('Longueur du câble (m)')
plt.ylabel('α (°)')
plt.grid()

plt.tight_layout()
plt.show()




# === Calcul de la lay back ===
x = [0]
y = [0]

deltaL = L_cable / n

for i in range(n):
    dx = deltaL * np.sin(alpha[i])
    dy = deltaL * np.cos(alpha[i])
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)

# === Tracé de la lay back ===
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Lay back du câble", color="blue")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Profil (lay back) du câble")
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
