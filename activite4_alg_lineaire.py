# --- Activité 4 : Résoudre des équations plus grandes ---
# Objectif :
# 1. Écrire les tableaux NumPy correspondants aux matrices A (coefficients) et Y (second membre)
# 2. Déterminer si le système possède une solution
# 3. Résoudre le système si possible

import numpy as np

# --- 1. Matrices A et Y ---
A = np.array([
    [1,  2, -1,  1,  1],
    [2, -1,  3, -1,  4],
    [-1, 3,  2,  1, -1],
    [3, -1,  1,  2,  3],
    [2,  1, -1, -3,  1]
], dtype=float)

Y = np.array([[8], [15], [7], [20], [5]], dtype=float)

# --- 1 bis. Regroupement en un seul tableau [A | Y] ---
system_matrix = np.hstack((A, Y))
print("Matrice augmentée [A | Y] :\n", system_matrix)

# --- 2. Vérifier s'il existe une solution ---
# On teste si le déterminant est non nul (système carré de 5x5)
try:
    det = np.linalg.det(A)
    print("Déterminant :", det)
    if np.isclose(det, 0):
        print("Le système n'a pas de solution unique (peut être indéterminé ou incompatible)")
    else:
        # --- 3. Résoudre le système ---
        solution = np.linalg.solve(A, Y)
        print("Solution du système :")
        print(solution)
except np.linalg.LinAlgError:
    print("Le système est incompatible (matrice non inversible)")

# --- Remarques ---
# Si det(A) ≠ 0, le système a une solution unique.
# Si det(A) = 0, le système peut être indéterminé (infinies solutions) ou incompatible (aucune solution).
# Dans ce cas précis, on utilise np.linalg.solve pour obtenir la solution lorsque c'est possible.
