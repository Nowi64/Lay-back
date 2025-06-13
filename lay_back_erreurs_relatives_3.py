import numpy as np
from scipy.optimize import fsolve

# Valeurs de référence de l'image
reference_values = {
    # Tableau 1: Table out meter
    "table_out": {
        10: {0.5: 10, 1: 10, 1.5: 9, 2: 9, 3: 9, 5: 8},
        25: {0.5: 24, 1: 24, 1.5: 22, 2: 20, 3: 14, 5: 14},
        50: {0.5: 49, 1: 46, 1.5: 39, 2: 31, 3: 20, 5: 20},
        100: {0.5: 98, 1: 81, 1.5: 58, 2: 43, 3: 25, 5: 25},
        150: {0.5: 143, 1: 105, 1.5: 71, 2: 50, 3: 28, 5: 28},
        200: {0.5: 186, 1: 124, 1.5: 80, 2: 55, 3: 30, 5: 30}
    },
    # Tableau 2: Speed knots
    "speed_knots": {
        10: {0.5: 0, 1: 0, 1.5: 1, 2: 2, 3: 4, 5: 7},
        25: {0.5: 1, 1: 4, 1.5: 8, 2: 12, 3: 18, 5: 22},
        50: {0.5: 4, 1: 15, 1.5: 27, 2: 35, 3: 42, 5: 47},
        100: {0.5: 17, 1: 51, 1.5: 73, 2: 83, 3: 92, 5: 97},
        150: {0.5: 37, 1: 95, 1.5: 121, 2: 133, 3: 142, 5: 147},
        200: {0.5: 63, 1: 141, 1.5: 171, 2: 183, 3: 192, 5: 197}
    }
}


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

    # Calcul du layback
    x = [0]
    y = [0]
    for i in range(n):
        dx = deltaL * np.sin(alpha[i])
        dy = deltaL * np.cos(alpha[i])
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)

    return x[-1]


def calculate_for_conditions(cable_out, speed_knots):
    # Convertir la vitesse de noeuds en m/s (1 noeud = 0.514444 m/s)
    v = speed_knots * 0.514444

    # Paramètres
    rho = 1027  # kg/m³
    mu = ((1.89 + 0.8) / 2) * 1e-3  # Pa.s
    g = 9.81  # m/s²
    m = 6.7  # kg
    Cp_cable = 1.1
    Cf_cable = 0.004
    D_sonar = 0.06  # m
    D_cable = 0.00630  # m
    L_sonar = 0.850  # m
    L_cable = cable_out  # m
    Cp_sphere_sonar = 0.5
    Cf_cyl_sonar = 0.002
    S_sphere_sonar = np.pi * D_sonar ** 2
    S_cyl_sonar = np.pi * D_sonar * L_sonar
    n = 10000
    d = 40 / 1000  # kg/m (converti depuis g/m)

    # Tensions initiales
    Tx = 0.5 * rho * v ** 2 * (Cp_sphere_sonar * S_sphere_sonar + Cf_cyl_sonar * S_cyl_sonar)
    Ty = m * g

    alpha_1 = np.pi / 2 - np.arctan2(Ty, Tx)
    Fx_1 = -Tx
    Fy_1 = -Ty

    # Calcul du layback
    layback = calculate_layback(
        rho, v, mu, g, m,
        Cp_sphere_sonar, Cf_cyl_sonar, S_sphere_sonar, S_cyl_sonar,
        Cp_cable, Cf_cable, D_sonar, D_cable, L_cable, n, d,
        alpha_1, Fx_1, Fy_1
    )

    return layback


def generate_error_table():
    cable_outs = [10, 25, 50, 100, 150, 200]
    speeds_knots = [0.5, 1, 1.5, 2, 3, 5]

    print("\nTableau d'erreurs relatives (%) pour le layback (comparaison avec les valeurs 'speed knots'):")
    print("Cable out (m) | " + " | ".join(f"{speed:>5} kt" for speed in speeds_knots))
    print("-" * (12 + 6 * len(speeds_knots)))

    for cable_out in cable_outs:
        row = f"{cable_out:>11} | "
        for speed in speeds_knots:
            try:
                # Calculer la valeur avec notre modèle
                calculated = calculate_for_conditions(cable_out, speed)
                # Récupérer la valeur de référence
                reference = reference_values["speed_knots"][cable_out][speed]
                # Calculer l'erreur relative
                if reference != 0:
                    error = (calculated - reference) / reference * 100
                else:
                    error = 0.0 if calculated == 0 else float('inf')
                row += f"{error:>5.1f}% "
            except:
                row += "  ERR "
        print(row)


# Générer le tableau d'erreurs
generate_error_table()


# Tableau supplémentaire pour les valeurs 'table out meter'
def generate_second_error_table():
    cable_outs = [10, 25, 50, 100, 150, 200]
    speeds_knots = [0.5, 1, 1.5, 2, 3, 5]

    print("\nTableau d'erreurs relatives (%) pour le layback (comparaison avec les valeurs 'table out meter'):")
    print("Cable out (m) | " + " | ".join(f"{speed:>5} kt" for speed in speeds_knots))
    print("-" * (12 + 6 * len(speeds_knots)))

    for cable_out in cable_outs:
        row = f"{cable_out:>11} | "
        for speed in speeds_knots:
            try:
                # Calculer la valeur avec notre modèle
                calculated = calculate_for_conditions(cable_out, speed)
                # Récupérer la valeur de référence
                reference = reference_values["table_out"][cable_out][speed]
                # Calculer l'erreur relative
                if reference != 0:
                    error = (calculated - reference) / reference * 100
                else:
                    error = 0.0 if calculated == 0 else float('inf')
                row += f"{error:>5.1f}% "
            except:
                row += "  ERR "
        print(row)


generate_second_error_table()