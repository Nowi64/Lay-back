import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    alpha_solution = np.clip(alpha_solution, 0, np.pi / 2)
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


def simulate_for_speed_and_cable(speed_knots, cable_length):
    """Simule pour une vitesse et longueur de câble données"""
    # Conversion vitesse
    v = speed_knots * 0.514444  # conversion nœuds vers m/s

    # Paramètres constants
    rho = 1027  # kg/m³
    mu = ((1.89 + 0.8) / 2) * 1e-3  # Pa.s
    g = 9.81  # m/s²
    m = 6.7  # kg

    Cp_cable = 1.1
    Cf_cable = 0.004
    D_sonar = 0.06  # m
    D_cable = 0.00630  # m
    L_sonar = 0.850  # m
    L_cable = cable_length  # m
    Cp_sphere_sonar = 0.6
    Cf_cyl_sonar = 0.003
    S_sphere_sonar = np.pi * D_sonar ** 2
    S_cyl_sonar = np.pi * D_sonar * L_sonar
    n = 10000
    d = 40 / 1000  # kg/m

    # Tensions initiales
    Tx = 0.5 * rho * v ** 2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g

    alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
    Fx_1 = -Tx
    Fy_1 = -Ty

    # Calcul
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

    depth = y[-1] - y[0]
    layback = x[-1] - x[0]

    return depth, layback


# Données des tableaux (référence)
speeds = np.array([0.5, 1, 1.5, 2, 3, 5])
cable_lengths = np.array([10, 25, 50, 100, 150, 200])

# Table 2: Depth data
depth_reference = np.array([
    [10, 9, 9, 9, 8, 5],  # 10m cable
    [24, 24, 22, 20, 14, 8],  # 25m cable
    [49, 46, 39, 31, 20, 10],  # 50m cable
    [98, 81, 58, 43, 25, 12],  # 100m cable
    [143, 105, 71, 50, 28, 13],  # 150m cable
    [186, 124, 80, 55, 30, 14]  # 200m cable
])

# Table 3: Layback data
layback_reference = np.array([
    [0, 0, 1, 2, 4, 7],  # 10m cable
    [1, 4, 8, 12, 18, 22],  # 25m cable
    [4, 15, 27, 35, 42, 47],  # 50m cable
    [17, 51, 73, 83, 92, 97],  # 100m cable
    [37, 95, 121, 133, 142, 147],  # 150m cable
    [63, 141, 171, 183, 192, 197]  # 200m cable
])




# Calcul des erreurs relatives
print("Calcul des simulations et erreurs relatives...")
depth_simulated = np.zeros((len(cable_lengths), len(speeds)))

# Utiliser les valeurs layback simulées fournies
layback_simulated = np.array([
    [9.99, 8.99, 7.99, 6.99, 5.99, 3.99],
    [23.99, 23.98, 21.97, 19.97, 13.95, 7.96],
    [49.99, 44.97, 39.96, 32.95, 21.94, 11.93],
    [97.99, 80.96, 63.93, 47.92, 32.91, 18.90],
    [137.98, 123.96, 79.94, 54.92, 39.91, 23.90],
    [185.97, 123.96, 79.94, 54.92, 39.91, 23.90]
])

depth_errors = np.zeros((len(cable_lengths), len(speeds)))
layback_errors = np.zeros((len(cable_lengths), len(speeds)))

for i, cable_length in enumerate(cable_lengths):
    for j, speed in enumerate(speeds):
        print(f"Simulation: Câble {cable_length}m, Vitesse {speed} nœuds")
        depth_sim, _ = simulate_for_speed_and_cable(speed, cable_length)  # On ignore layback_sim

        depth_simulated[i, j] = depth_sim

        # Calcul erreur relative en pourcentage
        depth_ref = depth_reference[i, j]
        layback_ref = layback_reference[i, j]

        if depth_ref != 0:
            depth_errors[i, j] = abs(depth_sim - depth_ref) / depth_ref * 100
        else:
            depth_errors[i, j] = 0

        if layback_ref != 0:
            layback_errors[i, j] = abs(layback_simulated[i, j] - layback_ref) / layback_ref * 100
        else:
            layback_errors[i, j] = 0

# Borner les erreurs entre 0 et 100%
depth_errors = np.clip(depth_errors, 0, 100)
layback_errors = np.clip(layback_errors, 0, 100)

# Création des heat maps
plt.figure(figsize=(15, 6))

# Heat map pour les erreurs de profondeur
plt.subplot(1, 2, 1)
sns.heatmap(depth_errors[::-1],  # Inverser l'ordre des lignes pour depth_errors
            xticklabels=[f'{s} nœuds' for s in speeds],
            yticklabels=[f'{c}m' for c in cable_lengths[::-1]],  # Inverser l'ordre
            annot=True,
            fmt='.1f',
            cmap='Reds',
            cbar_kws={'label': 'Erreur relative (%)'},
            vmin=0,
            vmax=100)  # Borner à 100%
plt.title('Erreurs relatives - Profondeur (Depth)')
plt.xlabel('Vitesse')
plt.ylabel('Longueur de câble')

# Heat map pour les erreurs de layback
plt.subplot(1, 2, 2)
sns.heatmap(layback_errors[::-1],  # Inverser l'ordre des lignes pour layback_errors
            xticklabels=[f'{s} nœuds' for s in speeds],
            yticklabels=[f'{c}m' for c in cable_lengths[::-1]],  # Inverser l'ordre
            annot=True,
            fmt='.1f',
            cmap='Blues',
            cbar_kws={'label': 'Erreur relative (%)'},
            vmin=0,
            vmax=100)  # Borner à 100%
plt.title('Erreurs relatives - Layback')
plt.xlabel('Vitesse')
plt.ylabel('Longueur de câble')

plt.tight_layout()
plt.show()

# Courbes de comparaison en fonction de la vitesse
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparaison Référence vs Modèle en fonction de la vitesse', fontsize=16)

# Couleurs pour chaque longueur de câble
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
cable_labels = [f'{length}m' for length in cable_lengths]

# Graphiques pour DEPTH
for i in range(3):
    ax = axes[0, i]
    cable_idx = i * 2  # indices 0, 2, 4 pour 10m, 50m, 100m
    if cable_idx < len(cable_lengths):
        # Référence
        ax.plot(speeds, depth_reference[cable_idx, :], 'o-',
                color=colors[cable_idx], linewidth=2, markersize=8,
                label=f'Référence {cable_labels[cable_idx]}')
        # Modèle
        ax.plot(speeds, depth_simulated[cable_idx, :], 's--',
                color=colors[cable_idx], linewidth=2, markersize=6, alpha=0.7,
                label=f'Modèle {cable_labels[cable_idx]}')

        ax.set_xlabel('Vitesse (nœuds)')
        ax.set_ylabel('Profondeur (m)')
        ax.set_title(f'Depth - Câble {cable_labels[cable_idx]}')
        ax.grid(True, alpha=0.3)
        ax.legend()

# Graphiques pour LAYBACK
for i in range(3):
    ax = axes[1, i]
    cable_idx = i * 2  # indices 0, 2, 4 pour 10m, 50m, 100m
    if cable_idx < len(cable_lengths):
        # Référence
        ax.plot(speeds, layback_reference[cable_idx, :], 'o-',
                color=colors[cable_idx], linewidth=2, markersize=8,
                label=f'Référence {cable_labels[cable_idx]}')
        # Modèle
        ax.plot(speeds, layback_simulated[cable_idx, :], 's--',
                color=colors[cable_idx], linewidth=2, markersize=6, alpha=0.7,
                label=f'Modèle {cable_labels[cable_idx]}')

        ax.set_xlabel('Vitesse (nœuds)')
        ax.set_ylabel('Layback (m)')
        ax.set_title(f'Layback - Câble {cable_labels[cable_idx]}')
        ax.grid(True, alpha=0.3)
        ax.legend()

plt.tight_layout()
plt.show()

# Graphiques complets avec toutes les longueurs de câble
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# DEPTH - Toutes les longueurs
for i, cable_length in enumerate(cable_lengths):
    ax1.plot(speeds, depth_reference[i, :], 'o-',
             color=colors[i], linewidth=2, markersize=6,
             label=f'Ref {cable_labels[i]}')
    ax1.plot(speeds, depth_simulated[i, :], 's--',
             color=colors[i], linewidth=1.5, markersize=4, alpha=0.7,
             label=f'Mod {cable_labels[i]}')

ax1.set_xlabel('Vitesse (nœuds)')
ax1.set_ylabel('Profondeur (m)')
ax1.set_title('Profondeur vs Vitesse - Toutes longueurs')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# LAYBACK - Toutes les longueurs
for i, cable_length in enumerate(cable_lengths):
    ax2.plot(speeds, layback_reference[i, :], 'o-',
             color=colors[i], linewidth=2, markersize=6,
             label=f'Ref {cable_labels[i]}')
    ax2.plot(speeds, layback_simulated[i, :], 's--',
             color=colors[i], linewidth=1.5, markersize=4, alpha=0.7,
             label=f'Mod {cable_labels[i]}')

ax2.set_xlabel('Vitesse (nœuds)')
ax2.set_ylabel('Layback (m)')
ax2.set_title('Layback vs Vitesse - Toutes longueurs')
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Affichage des statistiques
print("\n=== STATISTIQUES DES ERREURS ===")
print(f"Erreur moyenne depth: {np.mean(depth_errors):.2f}%")
print(f"Erreur max depth: {np.max(depth_errors):.2f}%")
print(f"Erreur moyenne layback: {np.mean(layback_errors):.2f}%")
print(f"Erreur max layback: {np.max(layback_errors):.2f}%")

# Comparaison des valeurs
print("\n=== COMPARAISON VALEURS SIMULÉES vs RÉFÉRENCE ===")
print("DEPTH:")
print("Référence:")
print(depth_reference)
print("Simulé:")
print(np.round(depth_simulated, 1))
print("\nLAYBACK:")
print("Référence:")
print(layback_reference)
print("Simulé:")
print(np.round(layback_simulated, 1))