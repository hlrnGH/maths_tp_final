# --- Question 1 ---
# Générer des systèmes d'équations aléatoires
# Implémentation : génération de coefficients pour un système de 2 équations à 2 inconnues

import numpy as np

def generate_random_system():
    A = np.random.randint(-10, 10, size=(2, 2))
    b = np.random.randint(-10, 10, size=(2,))
    return A, b

# --- Question 2 ---
# Implémenter un algorithme qui détermine si le système a une solution
# Si oui, donner la solution

def solve_linear_system(A, b):
    det = np.linalg.det(A)
    if det == 0:
        return "Pas de solution unique (système indéterminé ou incompatible)", None
    else:
        solution = np.linalg.solve(A, b)
        return "Solution unique", solution

# Exemple d'utilisation
A, b = generate_random_system()
print("Système généré :")
print("A =\n", A)
print("b =", b)

status, solution = solve_linear_system(A, b)
print("Résultat :", status)
if solution is not None:
    print("Solution :", solution)

# --- Question 3 ---
# Travailler sur la modularité dans le code et manipuler les concepts
# Réponse :
# La modularité consiste à structurer le code en fonctions indépendantes et réutilisables.
# Cela permet de clarifier la logique, de faciliter les tests, la maintenance et l'extension du code.
# Ici, la génération du système et sa résolution sont séparées dans deux fonctions distinctes.

# --- Question 4 ---
# Quand est-ce qu’un système est indéterminé ?
# Réponse :
# Un système est dit indéterminé lorsqu’il possède une infinité de solutions.
# Cela se produit lorsque les équations sont dépendantes (par exemple, elles sont multiples l’une de l’autre)
# et que le second membre est compatible avec cette dépendance.

# Quand est-ce qu’un système est incompatible ?
# Réponse :
# Un système est incompatible lorsqu’il ne possède aucune solution.
# Cela se produit lorsque les équations sont contradictoires (ex : deux droites parallèles mais non confondues).
# Mathématiquement, cela se traduit souvent par un déterminant nul (matrice non inversible),
# combiné à un second membre qui n’est pas dans l’image de la matrice des coefficients.
