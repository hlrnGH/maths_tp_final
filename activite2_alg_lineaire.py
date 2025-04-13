# --- Question 1 ---
# Quelle est la fonction NumPy pour faire le produit d'une matrice et d'un vecteur ?
# Réponse : np.dot(A, x) ou A @ x

import numpy as np

A = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
produit_matrice_vecteur = A @ x
print("Produit matrice x vecteur:", produit_matrice_vecteur)

# --- Question 2 ---
# Est-ce que la multiplication fonctionne dans les deux sens ? Pourquoi ?
# Réponse : Non. La multiplication matricielle n'est pas commutative.
# A @ x est valide si les dimensions sont compatibles (matrice (m,n) et vecteur (n,)),
# mais x @ A ne l'est pas forcément sauf si le vecteur est correctement transposé.

try:
    produit_inversé = x @ A
    print("Produit vecteur x matrice:", produit_inversé)
except ValueError as e:
    print("Erreur lors du produit dans l'autre sens:", e)

# --- Question 3 ---
# Quelle est la fonction NumPy qui permet de trouver l'inverse d'une matrice ?
# Réponse : np.linalg.inv(A), où A est une matrice carrée et inversible.

try:
    A_inverse = np.linalg.inv(A)
    print("Inverse de la matrice A:\n", A_inverse)
except np.linalg.LinAlgError:
    print("La matrice A n'est pas inversible.")

# --- Question 4 ---
# Écrire un script qui permet d'obtenir :
# - le nombre d'unités x arrondi à l'entier inférieur
# - la quantité de matière première y arrondie par excès à deux chiffres après la virgule

# Méthode :
# - x → arrondi avec np.floor(x)
# - y → arrondi avec np.ceil(y * 100) / 100

def process_quantities(x, y):
    x_arrondi = int(np.floor(x))
    y_arrondi = np.ceil(y * 100) / 100
    return x_arrondi, y_arrondi

# Exemple d'utilisation
x_val = 7.89
y_val = 3.14159

x_out, y_out = process_quantities(x_val, y_val)
print(f"x arrondi inf : {x_out}")
print(f"y arrondi sup à 2 décimales : {y_out}")
