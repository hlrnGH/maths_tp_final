import numpy as np
import matplotlib.pyplot as plt

# --- Fonction de coût global ---
def global_cost(fixed_cost, unit_cost, quantity):
    return fixed_cost + unit_cost * quantity

# --- Paramètres de base ---
base_fixed_cost = 500
base_unit_cost = 20
base_quantity = 50

# --- 1. Variation du coût global selon la quantité ---
quantities = np.linspace(0, 100, 200)
costs_quantity = global_cost(base_fixed_cost, base_unit_cost, quantities)

# --- 2. Variation du coût global selon le coût unitaire ---
unit_costs = np.linspace(5, 50, 200)
costs_unit = global_cost(base_fixed_cost, unit_costs, base_quantity)

# --- 3. Variation du coût global selon le coût fixe ---
fixed_costs = np.linspace(0, 2000, 200)
costs_fixed = global_cost(fixed_costs, base_unit_cost, base_quantity)

# --- Affichage des graphiques ---
plt.figure(figsize=(15, 10))

# Sous-plot 1 : Quantité
plt.subplot(3, 1, 1)
plt.plot(quantities, costs_quantity, label="Quantité variable", color='blue')
plt.title("Coût global en fonction de la quantité produite")
plt.xlabel("Quantité")
plt.ylabel("Coût global")
plt.grid(True)
plt.legend()

# Sous-plot 2 : Coût unitaire
plt.subplot(3, 1, 2)
plt.plot(unit_costs, costs_unit, label="Coût unitaire variable", color='green')
plt.title("Coût global en fonction du coût unitaire")
plt.xlabel("Coût unitaire")
plt.ylabel("Coût global")
plt.grid(True)
plt.legend()

# Sous-plot 3 : Coût fixe
plt.subplot(3, 1, 3)
plt.plot(fixed_costs, costs_fixed, label="Coût fixe variable", color='red')
plt.title("Coût global en fonction du coût fixe")
plt.xlabel("Coût fixe")
plt.ylabel("Coût global")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
