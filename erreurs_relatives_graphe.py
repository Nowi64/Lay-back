# Re-import libraries due to kernel reset
import numpy as np
import matplotlib.pyplot as plt

# === Données extraites du tableau (provenant de l'image) ===

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

# Résultats simulés
simulated_laybacks = np.array([
    [0.00, 0.17, 0.54, 1.07, 2.47, 4.76],
    [2.29, 4.13, 6.24, 8.46, 13.80, 21.39],
    [7.78, 13.63, 19.59, 25.39, 36.92, 51.23],
    [22.61, 34.90, 46.72, 57.76, 78.61, 101.55],
    [42.97, 61.79, 79.59, 95.62, 123.51, 150.83],
    [66.38, 91.38, 114.65, 136.21, 168.10, 197.27]
])
simulated_depths = np.array([
    [9.99, 8.99, 7.99, 6.99, 5.99, 3.99],
    [23.99, 23.98, 21.97, 19.97, 13.95, 7.96],
    [49.99, 44.97, 39.96, 32.95, 21.94, 11.93],
    [97.99, 80.96, 63.93, 47.92, 32.91, 18.90],
    [137.98, 123.96, 79.94, 54.92, 39.91, 23.90],
    [185.97, 123.96, 79.94, 54.92, 39.91, 23.90]
])

# === Calcul des erreurs relatives ===

layback_errors = np.abs(simulated_laybacks - layback_table) / np.maximum(layback_table, 1e-3) * 100
depth_errors = np.abs(simulated_depths - depth_table) / np.maximum(depth_table, 1e-3) * 100

# === Tracé des erreurs relatives ===

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphe 1 : Erreurs relatives (incertitudes) en % - heatmaps
im1 = ax[0].imshow(layback_errors, cmap="Reds", aspect='auto', interpolation='nearest')
ax[0].set_title("Erreur relative Layback (%)")
ax[0].set_xlabel("Vitesse (nœuds)")
ax[0].set_ylabel("Cable Out (m)")
ax[0].set_xticks(np.arange(len(speeds_knots)))
ax[0].set_xticklabels(speeds_knots)
ax[0].set_yticks(np.arange(len(cable_outs)))
ax[0].set_yticklabels(cable_outs)
fig.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(depth_errors, cmap="Blues", aspect='auto', interpolation='nearest')
ax[1].set_title("Erreur relative Depth (%)")
ax[1].set_xlabel("Vitesse (nœuds)")
ax[1].set_ylabel("Cable Out (m)")
ax[1].set_xticks(np.arange(len(speeds_knots)))
ax[1].set_xticklabels(speeds_knots)
ax[1].set_yticks(np.arange(len(cable_outs)))
ax[1].set_yticklabels(cable_outs)
fig.colorbar(im2, ax=ax[1])

plt.tight_layout()
plt.show()

# === Graphe 2 : Comparaison valeurs modèle vs constructeur ===

fig2, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for i, L in enumerate(cable_outs):
    axs[0].plot(speeds_knots, layback_table[i], 'o--', label=f"Réf. {L}m")
    axs[0].plot(speeds_knots, simulated_laybacks[i], 'x-', label=f"Modèle {L}m")

for i, L in enumerate(cable_outs):
    axs[1].plot(speeds_knots, depth_table[i], 'o--', label=f"Réf. {L}m")
    axs[1].plot(speeds_knots, simulated_depths[i], 'x-', label=f"Modèle {L}m")

axs[0].set_title("Comparaison Layback (réf vs modèle)")
axs[0].set_ylabel("Layback (m)")
axs[0].legend()

axs[1].set_title("Comparaison Depth (réf vs modèle)")
axs[1].set_ylabel("Depth (m)")
axs[1].set_xlabel("Vitesse (nœuds)")
axs[1].legend()

plt.tight_layout()
plt.show()
