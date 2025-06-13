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
    deltaL = L_cable / n
    Fx = [Fx_1]
    Fy = [Fy_1]
    alpha = [alpha_1]

    for i in range(n):
        cos_alpha = np.cos(alpha[-1])
        sin_alpha = np.sin(alpha[-1])

        delta_Fx = 0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * (
                    Cp_cable * cos_alpha ** 3 + Cf_cable * sin_alpha ** 3)

        delta_Fy = (
                0.5 * rho * np.pi * D_cable * deltaL * v ** 2 * cos_alpha * sin_alpha * (
                    -Cp_cable * cos_alpha + Cf_cable * sin_alpha)
                + g * deltaL * ((rho * np.pi * D_cable ** 2) / 4 - d)
        )

        Fx_next = Fx[-1] - delta_Fx
        Fy_next = Fy[-1] - delta_Fy

        Fx.append(Fx_next)
        Fy.append(Fy_next)

        alpha_next = find_alpha(Fx_next, Fy_next, g, rho, D_cable, d, deltaL, Cp_cable, v)
        alpha.append(alpha_next)

    return Fx, Fy, alpha


def run_simulation(v, L_cable, initial_alpha_deg=None):
    """Run simulation for given parameters"""
    # === Paramètres de base ===
    rho = 1027  # kg/m³
    mu = ((1.89 + 0.8) / 2) * 1e-3  # Pa.s
    g = 9.81  # m/s²
    m = 6.7  # kg

    Cp_cable = 1.1
    Cf_cable = 0.004
    D_sonar = 0.06  # m
    D_cable = 0.00630  # m
    L_sonar = 0.850  # m
    Cp_sphere_sonar = 0.5
    Cf_cyl_sonar = 0.002
    S_sphere_sonar = np.pi * D_sonar ** 2
    S_cyl_sonar = np.pi * D_sonar * L_sonar
    n = 1000  # Reduced for faster computation
    d = 40 / 1000  # kg/m

    # Tensions initiales
    Tx = 0.5 * rho * v ** 2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g

    if initial_alpha_deg is None:
        alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
    else:
        alpha_1 = np.radians(initial_alpha_deg)

    Fx_1 = -Tx
    Fy_1 = -Ty

    Fx, Fy, alpha = calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
    )

    # Calcul des positions
    x = [0]
    y = [0]
    deltaL = L_cable / n

    for i in range(n):
        dx = deltaL * np.sin(alpha[i])
        dy = deltaL * np.cos(alpha[i])
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    s = np.linspace(0, L_cable, n + 1)

    return {
        'x': x, 'y': y, 's': s, 'Fx': Fx, 'Fy': Fy, 'alpha': alpha,
        'layback': x[-1], 'depth': y[-1], 'final_alpha': np.degrees(alpha[-1])
    }


# === ANALYSE MULTI-PARAMÈTRES ===
print("=== ANALYSE DES PARAMÈTRES ===\n")

# 1. Analyse des vitesses
speeds = [1.0, 1.5, 2.0, 2.5, 3.0]  # m/s
cable_length = 50  # m

print("1. EFFET DE LA VITESSE:")
plt.figure(figsize=(15, 10))

# Profils des câbles pour différentes vitesses
plt.subplot(2, 3, 1)
for v in speeds:
    result = run_simulation(v, cable_length)
    plt.plot(result['x'], result['y'], label=f'v = {v} m/s', linewidth=2)
    print(f"   v = {v} m/s: layback = {result['layback']:.1f}m, depth = {result['depth']:.1f}m")

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Profil du câble - Effet vitesse')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Tensions Fx vs vitesse
plt.subplot(2, 3, 2)
for v in speeds:
    result = run_simulation(v, cable_length)
    plt.plot(result['s'], result['Fx'], label=f'v = {v} m/s', linewidth=2)

plt.xlabel('Longueur câble (m)')
plt.ylabel('Fx (N)')
plt.title('Tension Fx - Effet vitesse')
plt.legend()
plt.grid(True)

# Angles vs vitesse
plt.subplot(2, 3, 3)
for v in speeds:
    result = run_simulation(v, cable_length)
    plt.plot(result['s'], np.degrees(result['alpha']), label=f'v = {v} m/s', linewidth=2)

plt.xlabel('Longueur câble (m)')
plt.ylabel('α (°)')
plt.title('Angle - Effet vitesse')
plt.legend()
plt.grid(True)

# 2. Analyse des longueurs de câble
cable_lengths = [25, 40, 50, 75, 100]  # m
speed = 2.0  # m/s

print("\n2. EFFET DE LA LONGUEUR DE CÂBLE:")

# Profils des câbles pour différentes longueurs
plt.subplot(2, 3, 4)
for L in cable_lengths:
    result = run_simulation(speed, L)
    plt.plot(result['x'], result['y'], label=f'L = {L} m', linewidth=2)
    print(f"   L = {L} m: layback = {result['layback']:.1f}m, depth = {result['depth']:.1f}m")

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Profil du câble - Effet longueur')
plt.legend()
plt.grid(True)
plt.axis('equal')

# 3. Analyse des angles initiaux
initial_angles = [0, 10, 20, 30, 45, 60, 75, 90]  # degrés (plage complète 0-90°)
speed = 2.0  # m/s
cable_length = 50  # m

print("\n3. EFFET DE L'ANGLE INITIAL (0° à 90°):")

# Profils des câbles pour différents angles initiaux
plt.subplot(2, 3, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_angles)))  # Palette de couleurs
for i, angle in enumerate(initial_angles):
    result = run_simulation(speed, cable_length, angle)
    plt.plot(result['x'], result['y'], label=f'α₁ = {angle}°', linewidth=2, color=colors[i])
    print(f"   α₁ = {angle}°: layback = {result['layback']:.1f}m, depth = {result['depth']:.1f}m")

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Profil du câble - Effet angle initial (0-90°)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.axis('equal')

# Résumé des effets
plt.subplot(2, 3, 6)
# Layback vs vitesse
laybacks_speed = []
for v in speeds:
    result = run_simulation(v, 50)
    laybacks_speed.append(result['layback'])

plt.plot(speeds, laybacks_speed, 'o-', label='Layback vs Vitesse', linewidth=2, markersize=8)
plt.xlabel('Vitesse (m/s)')
plt.ylabel('Layback (m)')
plt.title('Résumé: Layback vs Paramètres')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# === TABLEAU RÉCAPITULATIF ===
print("\n=== TABLEAU RÉCAPITULATIF ===")
print("VITESSE (L=50m, α₁=auto):")
print("Vitesse | Layback | Profondeur | Angle final")
print("--------|---------|------------|------------")
for v in speeds:
    result = run_simulation(v, 50)
    print(f"{v:7.1f} | {result['layback']:7.1f} | {result['depth']:10.1f} | {result['final_alpha']:11.1f}°")

print("\nLONGUEUR (v=2m/s, α₁=auto):")
print("Longueur | Layback | Profondeur | Angle final")
print("---------|---------|------------|------------")
for L in cable_lengths:
    result = run_simulation(2.0, L)
    print(f"{L:8.0f} | {result['layback']:7.1f} | {result['depth']:10.1f} | {result['final_alpha']:11.1f}°")

print("\nANGLE INITIAL (v=2m/s, L=50m) - Plage complète 0-90°:")
print("Angle ini | Layback | Profondeur | Angle final")
print("----------|---------|------------|------------")
for angle in initial_angles:
    result = run_simulation(2.0, 50, angle)
    print(f"{angle:9.0f}° | {result['layback']:7.1f} | {result['depth']:10.1f} | {result['final_alpha']:11.1f}°")

# === GRAPHIQUE SUPPLÉMENTAIRE: ANALYSE COMPLÈTE DES ANGLES ===
plt.figure(figsize=(15, 10))

# Graphique 1: Profils pour tous les angles 0-90°
plt.subplot(2, 3, 1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_angles)))
for i, angle in enumerate(initial_angles):
    result = run_simulation(2.0, 50, angle)
    plt.plot(result['x'], result['y'], label=f'{angle}°', linewidth=2, color=colors[i])
plt.xlabel('Layback x (m)')
plt.ylabel('Depth y (m)')
plt.title('Profils câble: angles 0° à 90° (v=2m/s, L=50m)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Graphique 2: Layback vs angle initial
plt.subplot(2, 3, 2)
laybacks_all_angles = [run_simulation(2.0, 50, a)['layback'] for a in initial_angles]
plt.plot(initial_angles, laybacks_all_angles, 'o-', linewidth=2, markersize=8, color='red')
plt.xlabel('Angle initial (°)')
plt.ylabel('Layback (m)')
plt.title('Layback en fonction de l\'angle initial')
plt.grid(True)

# Graphique 3: Profondeur vs angle initial
plt.subplot(2, 3, 3)
depths_all_angles = [run_simulation(2.0, 50, a)['depth'] for a in initial_angles]
plt.plot(initial_angles, depths_all_angles, 's-', linewidth=2, markersize=8, color='blue')
plt.xlabel('Angle initial (°)')
plt.ylabel('Profondeur (m)')
plt.title('Profondeur en fonction de l\'angle initial')
plt.grid(True)

# Graphique 4: Angle final vs angle initial
plt.subplot(2, 3, 4)
final_angles_all = [run_simulation(2.0, 50, a)['final_alpha'] for a in initial_angles]
plt.plot(initial_angles, final_angles_all, '^-', linewidth=2, markersize=8, color='green')
plt.plot(initial_angles, initial_angles, '--', color='gray', alpha=0.5, label='Angle initial = Angle final')
plt.xlabel('Angle initial (°)')
plt.ylabel('Angle final (°)')
plt.title('Évolution angle initial → angle final')
plt.legend()
plt.grid(True)

# Graphique 5: Tension finale Fx vs angle initial
plt.subplot(2, 3, 5)
tensions_fx = []
for angle in initial_angles:
    result = run_simulation(2.0, 50, angle)
    tensions_fx.append(result['Fx'][-1])
plt.plot(initial_angles, tensions_fx, 'd-', linewidth=2, markersize=8, color='purple')
plt.xlabel('Angle initial (°)')
plt.ylabel('Tension Fx finale (N)')
plt.title('Tension Fx finale vs angle initial')
plt.grid(True)

# Graphique 6: Synthèse comparative
plt.subplot(2, 3, 6)
# Normalisation pour comparaison
layback_norm = np.array(laybacks_all_angles) / max(laybacks_all_angles)
depth_norm = np.array(depths_all_angles) / max(depths_all_angles)
angle_diff_norm = np.abs(np.array(final_angles_all) - np.array(initial_angles)) / 90

plt.plot(initial_angles, layback_norm, 'o-', label='Layback (norm.)', linewidth=2)
plt.plot(initial_angles, depth_norm, 's-', label='Profondeur (norm.)', linewidth=2)
plt.plot(initial_angles, angle_diff_norm, '^-', label='Diff. angle (norm.)', linewidth=2)
plt.xlabel('Angle initial (°)')
plt.ylabel('Valeur normalisée')
plt.title('Comparaison normalisée des effets')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
for v in speeds:
    result = run_simulation(v, 50)
    plt.plot(result['x'], result['y'], label=f'{v} m/s', linewidth=2)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Effet de la vitesse (L=50m)')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(2, 2, 2)
for L in cable_lengths:
    result = run_simulation(2.0, L)
    plt.plot(result['x'], result['y'], label=f'{L} m', linewidth=2)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Effet de la longueur (v=2m/s)')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(2, 2, 3)
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_angles)))  # Même palette
for i, angle in enumerate(initial_angles):
    result = run_simulation(2.0, 50, angle)
    plt.plot(result['x'], result['y'], label=f'{angle}°', linewidth=2, color=colors[i])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Effet angle initial 0-90° (v=2m/s, L=50m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.axis('equal')

plt.subplot(2, 2, 4)
# Comparaison des laybacks
laybacks_speed = [run_simulation(v, 50)['layback'] for v in speeds]
laybacks_length = [run_simulation(2.0, L)['layback'] for L in cable_lengths]
laybacks_angle = [run_simulation(2.0, 50, a)['layback'] for a in initial_angles]

plt.plot(speeds, laybacks_speed, 'o-', label='Vitesse', linewidth=2, markersize=6)
plt.plot([L / 25 for L in cable_lengths], laybacks_length, 's-', label='Longueur/25', linewidth=2, markersize=6)
plt.plot([a / 45 for a in initial_angles], laybacks_angle, '^-', label='Angle/45', linewidth=2, markersize=6)
plt.xlabel('Paramètre normalisé')
plt.ylabel('Layback (m)')
plt.title('Comparaison des effets sur layback')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== ANALYSE TERMINÉE ===")
print("Les graphiques montrent l'influence de chaque paramètre sur le comportement du câble.")
print("Observations principales:")
print("- La vitesse augmente significativement le layback")
print("- La longueur du câble augmente le layback de façon quasi-linéaire")
print("- L'angle initial a un effet significatif sur le profil: de 0° (horizontal) à 90° (vertical)")
print(f"- Plage de layback: {min(laybacks_all_angles):.1f}m à {max(laybacks_all_angles):.1f}m")
print(f"- Plage de profondeur: {min(depths_all_angles):.1f}m à {max(depths_all_angles):.1f}m")