import numpy as np
import matplotlib.pyplot as plt

# Fonction dérivée (dans ce cas dy/dt = y)
def f(t, y):
    return y

# Solution exacte : y(t) = exp(t)
def solution_exacte(t):
    return np.exp(t)

# Méthode d'Euler
def methode_euler(f, y0, t):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + f(t[i-1], y[i-1]) * (t[i] - t[i-1])
    return y

# Calcul de la MSE
def calcul_mse(y_approchee, y_exacte):
    return np.mean((y_approchee - y_exacte)**2)

# Paramètres
y0 = 1
intervalle = [0, 2]
valeurs_h = [0.1, 0.5]

# Comparaison pour chaque valeur de h
for h in valeurs_h:
    t = np.arange(intervalle[0], intervalle[1] + h, h)
    y_euler = methode_euler(f, y0, t)
    y_exact = solution_exacte(t)
    mse = calcul_mse(y_euler, y_exact)
    
    print(f"h = {h}")
    print(f"MSE = {mse:.6f}")
    
    # Tracer les courbes
    plt.figure()
    plt.plot(t, y_exact, label="Solution exacte", linestyle='--')
    plt.plot(t, y_euler, label=f"Euler (h={h})", marker='o')
    plt.title(f"Comparaison des solutions (h = {h})")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
