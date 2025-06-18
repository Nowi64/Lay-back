import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time


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


def run_simulation(v, L_cable, n=1000, initial_alpha_deg=None):
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
        'layback': x[-1], 'depth': y[-1], 'final_alpha': np.degrees(alpha[-1]),
        'deltaL': deltaL
    }


# === ANALYSE DÉTAILLÉE DE LA LONGUEUR DU CÂBLE ===
print("=== ANALYSE DÉTAILLÉE - LONGUEUR DU CÂBLE ===\n")

# Paramètres fixes
speed = 2.0  # m/s
n_segments = 1000

# Gamme étendue de longueurs de câble
cable_lengths = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 200])

# Collecte des données
laybacks = []
depths = []
final_angles = []
computation_times = []

print("Calcul en cours pour différentes longueurs de câble...")
for i, L in enumerate(cable_lengths):
    start_time = time.time()
    result = run_simulation(speed, L, n_segments)
    end_time = time.time()

    laybacks.append(result['layback'])
    depths.append(result['depth'])
    final_angles.append(result['final_alpha'])
    computation_times.append(end_time - start_time)

    print(f"L = {L:3.0f}m: layback = {result['layback']:6.2f}m, depth = {result['depth']:6.2f}m, "
          f"angle final = {result['final_alpha']:5.1f}°, temps = {computation_times[-1] * 1000:.1f}ms")

laybacks = np.array(laybacks)
depths = np.array(depths)
final_angles = np.array(final_angles)

# === GRAPHIQUES DÉTAILLÉS ===
fig = plt.figure(figsize=(16, 12))

# 1. Profils de câbles pour longueurs sélectionnées
plt.subplot(3, 3, 1)
selected_lengths = [25, 50, 75, 100, 150, 200]
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_lengths)))

for L, color in zip(selected_lengths, colors):
    result = run_simulation(speed, L, n_segments)
    plt.plot(result['x'], result['y'], label=f'L = {L}m', linewidth=2, color=color)

plt.xlabel('Layback x (m)')
plt.ylabel('Profondeur y (m)')
plt.title('Profils de câbles - Différentes longueurs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 2. Layback vs Longueur (linéaire)
plt.subplot(3, 3, 2)
plt.plot(cable_lengths, laybacks, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Longueur du câble (m)')
plt.ylabel('Layback (m)')
plt.title('Layback vs Longueur du câble')
plt.grid(True, alpha=0.3)

# Ajout d'une régression linéaire
coeffs = np.polyfit(cable_lengths, laybacks, 1)
poly_line = np.poly1d(coeffs)
plt.plot(cable_lengths, poly_line(cable_lengths), 'r--',
         label=f'Régression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
plt.legend()

# 3. Profondeur vs Longueur
plt.subplot(3, 3, 3)
plt.plot(cable_lengths, depths, 'go-', linewidth=2, markersize=6)
plt.xlabel('Longueur du câble (m)')
plt.ylabel('Profondeur (m)')
plt.title('Profondeur vs Longueur du câble')
plt.grid(True, alpha=0.3)

# 4. Ratio Layback/Longueur
plt.subplot(3, 3, 4)
ratio_layback_length = laybacks / cable_lengths
plt.plot(cable_lengths, ratio_layback_length, 'mo-', linewidth=2, markersize=6)
plt.xlabel('Longueur du câble (m)')
plt.ylabel('Ratio Layback/Longueur')
plt.title('Efficacité du layback')
plt.grid(True, alpha=0.3)

# 5. Angle final vs Longueur
plt.subplot(3, 3, 5)
plt.plot(cable_lengths, final_angles, 'co-', linewidth=2, markersize=6)
plt.xlabel('Longueur du câble (m)')
plt.ylabel('Angle final (°)')
plt.title('Angle final vs Longueur')
plt.grid(True, alpha=0.3)

# 6. Layback vs Longueur (échelle log-log)
plt.subplot(3, 3, 6)
plt.loglog(cable_lengths, laybacks, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Longueur du câble (m) - Log')
plt.ylabel('Layback (m) - Log')
plt.title('Layback vs Longueur (échelle log-log)')
plt.grid(True, alpha=0.3)

# Régression en échelle log
log_coeffs = np.polyfit(np.log(cable_lengths), np.log(laybacks), 1)
plt.loglog(cable_lengths, np.exp(log_coeffs[1]) * cable_lengths ** log_coeffs[0], 'g--',
           label=f'Loi puissance: y = {np.exp(log_coeffs[1]):.3f} × x^{log_coeffs[0]:.3f}')
plt.legend()

# === ANALYSE DU PAS DE DISCRÉTISATION ===
print("\n=== ANALYSE DU PAS DE DISCRÉTISATION ===\n")

# Paramètres fixes pour l'analyse de convergence
L_test = 100.0  # m
discretization_steps = np.array([50, 100, 200, 500, 1000, 2000, 5000])

laybacks_discretization = []
depths_discretization = []
computation_times_discretization = []

print("Calcul en cours pour différents pas de discrétisation...")
for n in discretization_steps:
    start_time = time.time()
    result = run_simulation(speed, L_test, n)
    end_time = time.time()

    laybacks_discretization.append(result['layback'])
    depths_discretization.append(result['depth'])
    computation_times_discretization.append(end_time - start_time)

    print(f"n = {n:4.0f}: layback = {result['layback']:6.3f}m, depth = {result['depth']:6.3f}m, "
          f"Δx = {result['deltaL'] * 1000:.2f}mm, temps = {computation_times_discretization[-1] * 1000:.1f}ms")

laybacks_discretization = np.array(laybacks_discretization)
depths_discretization = np.array(depths_discretization)

# 7. Convergence du layback avec la discrétisation
plt.subplot(3, 3, 7)
plt.semilogx(discretization_steps, laybacks_discretization, 'bs-', linewidth=2, markersize=6)
plt.xlabel('Nombre de segments n')
plt.ylabel('Layback (m)')
plt.title('Convergence - Layback vs Discrétisation')
plt.grid(True, alpha=0.3)

# Ligne de référence (valeur convergée)
layback_ref = laybacks_discretization[-1]
plt.axhline(y=layback_ref, color='r', linestyle='--', alpha=0.7,
            label=f'Référence: {layback_ref:.3f}m')
plt.legend()

# 8. Erreur relative vs discrétisation
plt.subplot(3, 3, 8)
errors = np.abs(laybacks_discretization - layback_ref) / layback_ref * 100
plt.loglog(discretization_steps, errors, 'rs-', linewidth=2, markersize=6)
plt.xlabel('Nombre de segments n')
plt.ylabel('Erreur relative (%)')
plt.title('Erreur de convergence')
plt.grid(True, alpha=0.3)

# 9. Temps de calcul vs discrétisation
plt.subplot(3, 3, 9)
plt.loglog(discretization_steps, np.array(computation_times_discretization) * 1000, 'gs-', linewidth=2, markersize=6)
plt.xlabel('Nombre de segments n')
plt.ylabel('Temps de calcul (ms)')
plt.title('Performance computationnelle')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === TABLEAUX RÉCAPITULATIFS ===
print("\n=== TABLEAUX RÉCAPITULATIFS ===")

print("\n1. EFFET DE LA LONGUEUR DU CÂBLE (v=2m/s, n=1000):")
print("Longueur | Layback | Profondeur | Ratio L/B | Angle final | Temps")
print("---------|---------|------------|-----------|-------------|-------")
for i, L in enumerate(cable_lengths):
    ratio = laybacks[i] / L
    print(
        f"{L:8.0f} | {laybacks[i]:7.2f} | {depths[i]:10.2f} | {ratio:9.3f} | {final_angles[i]:11.1f}° | {computation_times[i] * 1000:5.1f}ms")

print(f"\nRégression linéaire: Layback = {coeffs[0]:.3f} × Longueur + {coeffs[1]:.2f}")
print(f"Coefficient de corrélation R² = {np.corrcoef(cable_lengths, laybacks)[0, 1] ** 2:.6f}")

print("\n2. ANALYSE DE CONVERGENCE (L=100m, v=2m/s):")
print("Segments | Layback  | Profondeur | Δx (mm) | Erreur (%) | Temps")
print("---------|----------|------------|---------|------------|-------")
for i, n in enumerate(discretization_steps):
    delta_x = L_test / n * 1000  # en mm
    error = np.abs(laybacks_discretization[i] - layback_ref) / layback_ref * 100
    print(
        f"{n:8.0f} | {laybacks_discretization[i]:8.3f} | {depths_discretization[i]:10.3f} | {delta_x:7.2f} | {error:10.4f} | {computation_times_discretization[i] * 1000:5.1f}ms")

# === ANALYSE STATISTIQUE ===
print("\n=== ANALYSE STATISTIQUE ===")

print("\nSTATISTIQUES SUR LA RELATION LAYBACK-LONGUEUR:")
print(f"• Pente moyenne: {coeffs[0]:.4f} (layback/longueur)")
print(f"• Layback moyen par mètre de câble: {np.mean(laybacks / cable_lengths):.4f}")
print(f"• Écart-type du ratio: {np.std(laybacks / cable_lengths):.4f}")
print(f"• Coefficient de variation: {np.std(laybacks / cable_lengths) / np.mean(laybacks / cable_lengths) * 100:.2f}%")

print(f"\nCONVERGENCE DE LA DISCRÉTISATION:")
print(f"• Layback de référence (n=5000): {layback_ref:.4f}m")
print(f"• Erreur à n=1000: {errors[4]:.4f}%")
print(f"• Erreur à n=2000: {errors[5]:.4f}%")

# Recommandation sur le nombre de segments
recommended_n = discretization_steps[errors < 0.01][0] if any(errors < 0.01) else discretization_steps[-1]
print(f"• Nombre de segments recommandé (erreur < 0.01%): n = {recommended_n}")

print("\n=== CONCLUSIONS ===")
print("1. RELATION LAYBACK-LONGUEUR:")
print(f"   - Relation quasi-linéaire avec pente {coeffs[0]:.3f}")
print(f"   - Pour chaque mètre de câble, le layback augmente de ~{coeffs[0]:.3f}m")
print(f"   - Très haute corrélation (R² = {np.corrcoef(cable_lengths, laybacks)[0, 1] ** 2:.4f})")

print("\n2. CONVERGENCE NUMÉRIQUE:")
print(f"   - Convergence atteinte avec n ≥ {recommended_n} segments")
print(f"   - Compromis performance/précision optimal: n = 1000-2000")
print(f"   - Temps de calcul augmente linéairement avec n")

print("\n3. RECOMMANDATIONS PRATIQUES:")
print(f"   - Pour des calculs rapides: n = 500-1000")
print(f"   - Pour une précision maximale: n ≥ 2000")
print(f"   - La relation linéaire permet une extrapolation fiable")