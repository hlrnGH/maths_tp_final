import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import itertools

# Définition du système proies-prédateurs (Lotka-Volterra)
def lotka_volterra(time, populations, prey_growth_rate, predation_rate, predator_growth_rate, predator_death_rate):
    prey, predator = populations
    d_prey_dt = prey_growth_rate * prey - predation_rate * prey * predator
    d_predator_dt = predator_growth_rate * prey * predator - predator_death_rate * predator
    return [d_prey_dt, d_predator_dt]

# Paramètres du modèle
prey_growth_rate = 1.1         # Taux de croissance des proies
predation_rate = 0.4           # Taux de prédation
predator_growth_rate = 0.1     # Taux de reproduction des prédateurs
predator_death_rate = 0.4      # Taux de mortalité des prédateurs

# Conditions initiales
initial_prey_population = 10
initial_predator_population = 5
initial_conditions = [initial_prey_population, initial_predator_population]

# Intervalle de temps de la simulation
start_time = 0
end_time = 50
time_points = np.linspace(start_time, end_time, 1000)

# Résolution du système avec solve_ivp (pour résoudre des equations differentielles)
solution = solve_ivp(
    fun=lambda t, y: lotka_volterra(
        t, y,
        prey_growth_rate,
        predation_rate,
        predator_growth_rate,
        predator_death_rate
    ),
    t_span=(start_time, end_time),
    y0=initial_conditions,
    t_eval=time_points,
    method='RK45'
)

# Extraction des résultats
time = solution.t
prey_population = solution.y[0]
predator_population = solution.y[1]

# Tracé des populations au cours du temps
plt.figure(figsize=(10, 5))

plt.plot(time, prey_population, label='Proies', color='blue')
plt.plot(time, predator_population, label='Prédateurs', color='red')
plt.title("Évolution des populations")
plt.xlabel("Temps")
plt.ylabel("Population")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#_____________________________

# Chargement du fichier CSV
data = pd.read_csv("datasets/populations_lapins_renards.csv")

# Conversion forcée en float + gestion des erreurs de type
data["lapin"] = pd.to_numeric(data["lapin"], errors="coerce")
data["renard"] = pd.to_numeric(data["renard"], errors="coerce")

# Remplacement des valeurs négatives par 0
data["lapin"] = data["lapin"].clip(lower=0)
data["renard"] = data["renard"].clip(lower=0)

# Suppression stricte de toutes les lignes contenant des NaN
data = data.dropna(subset=["lapin", "renard"]).reset_index(drop=True)

# Renommage cohérent pour le modèle
time = np.arange(len(data))
prey_population_real = data["lapin"].values.astype(np.float64)
predator_population_real = data["renard"].values.astype(np.float64)
population_data_real = np.vstack((prey_population_real, predator_population_real)).T

# --- Simulation du modèle ---
def simulate_model(params):
    prey_growth_rate, predation_rate, predator_growth_rate, predator_death_rate = params

    try:
        solution = solve_ivp(
            fun=lambda t, populations: lotka_volterra(
                t, populations,
                prey_growth_rate,
                predation_rate,
                predator_growth_rate,
                predator_death_rate
            ),
            t_span=(time[0], time[-1]),
            y0=[prey_population_real[0], predator_population_real[0]],
            t_eval=time,
            method="RK45"
        )

        # Vérification de la réussite de l'intégration
        if not solution.success:
            return np.full_like(population_data_real, np.nan)

        predicted = solution.y.T

        # Vérification des valeurs invalides (nan, inf, négatives ou éteintes trop vite)
        if (
            np.isnan(predicted).any()
            or np.isinf(predicted).any()
            or np.min(predicted) < 0
            or np.max(predicted[:, 0]) < 10  # proies très faibles
            or np.max(predicted[:, 1]) < 10  # prédateurs très faibles
        ):
            return np.full_like(population_data_real, np.nan)

        return predicted

    except Exception as e:
        # Erreurs silencieuses : mieux vaut les ignorer proprement
        return np.full_like(population_data_real, np.nan)

# --- Fonction objectif : erreur quadratique moyenne ---
def mean_squared_error_objective(params):
    predicted_population = simulate_model(params)
    if np.isnan(predicted_population).any():
        return np.inf
    return mean_squared_error(population_data_real, predicted_population)

# --- Recherche exhaustive (grid search) des meilleurs paramètres ---
param_values = [1/3, 2/3, 1, 4/3]
param_grid = list(itertools.product(param_values, repeat=4))

best_params = None
best_mse = np.inf

for params in param_grid:
    mse = mean_squared_error_objective(params)
    if mse < best_mse:
        best_mse = mse
        best_params = params

"""

# --- Optimisation avec Newton-CG ---
initial_guess = [1.0, 0.4, 0.1, 0.4]  # alpha, beta, delta, gamma

def gradient_mse(params, h=1e-5):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_forward = params.copy()
        p_backward = params.copy()
        p_forward[i] += h
        p_backward[i] -= h
        grad[i] = (mean_squared_error_objective(p_forward) - mean_squared_error_objective(p_backward)) / (2 * h)
    return grad

result = minimize(
    mean_squared_error_objective,
    x0=initial_guess,
    method='Newton-CG',
    jac=gradient_mse,
    options={'disp': True, 'maxiter': 100}
)

best_params_newton = result.x
mse_newton = result.fun

"""

# --- Affichage des résultats ---
print("✅ Meilleurs paramètres trouvés :")
print(f"alpha = {best_params[0]}, beta = {best_params[1]}, delta = {best_params[2]}, gamma = {best_params[3]}")
print(f"MSE minimale = {best_mse:.2f}")

# --- Simulation avec les meilleurs paramètres avec grid search ---
best_model_prediction = simulate_model(best_params)

# --- Affichage graphique ---
plt.plot(time, prey_population_real, '--', label="Proies (réelles)", color='blue')
plt.plot(time, predator_population_real, '--', label="Prédateurs (réelles)", color='red')
plt.plot(time, best_model_prediction[:, 0], label="Proies (modèle)", color='blue')
plt.plot(time, best_model_prediction[:, 1], label="Prédateurs (modèle)", color='red')
plt.title("Ajustement du modèle Lotka-Volterra aux données réelles")
plt.xlabel("Temps (jours)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""

# --- Résultats optimisés ---
print("\n✅ Paramètres optimisés avec Newton-CG :")
print(f"alpha = {best_params_newton[0]:.4f}")
print(f"beta  = {best_params_newton[1]:.4f}")
print(f"delta = {best_params_newton[2]:.4f}")
print(f"gamma = {best_params_newton[3]:.4f}")
print(f"MSE   = {mse_newton:.2f}")

# --- Simulation avec ces paramètres ---
best_model_prediction = simulate_model(best_params_newton)

# --- Affichage graphique ---
plt.figure(figsize=(14, 6))
plt.plot(time, prey_population_real, '--o', label="Proies (réelles)", color='blue')
plt.plot(time, predator_population_real, '--o', label="Prédateurs (réelles)", color='red')
plt.plot(time, best_model_prediction[:, 0], '-', label="Proies (modèle Newton)", color='blue')
plt.plot(time, best_model_prediction[:, 1], '-', label="Prédateurs (modèle Newton)", color='red')
plt.title("Optimisation du modèle Lotka-Volterra par Newton-CG")
plt.xlabel("Temps (jours)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""