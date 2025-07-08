import numpy as np

# Données des VaaS disponibles
vaas = [
    {'uid': 'LY1', 'cost': 40, 'speed': 60, 'emission': 60, 'coverd_regions': [69], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'LY2', 'cost': 50, 'speed': 80, 'emission': 45, 'coverd_regions': [69], 'climatisation': True, 'places_min': 5, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'LY3', 'cost': 45, 'speed': 75, 'emission': 50, 'coverd_regions': [69], 'climatisation': True, 'places_min': 4, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'BS1', 'cost': 38, 'speed': 70, 'emission': 40, 'coverd_regions': [25], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'BS2', 'cost': 42, 'speed': 85, 'emission': 38, 'coverd_regions': [25], 'climatisation': True, 'places_min': 5, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'BS3', 'cost': 39, 'speed': 65, 'emission': 43, 'coverd_regions': [25], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'ST1', 'cost': 48, 'speed': 85, 'emission': 35, 'coverd_regions': [67], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'ST2', 'cost': 52, 'speed': 90, 'emission': 37, 'coverd_regions': [67], 'climatisation': True, 'places_min': 4, 'support_velo': True, 'type_energy': 'électrique'},
    {'uid': 'ST3', 'cost': 50, 'speed': 78, 'emission': 39, 'coverd_regions': [67], 'climatisation': True, 'places_min': 3, 'support_velo': True, 'type_energy': 'électrique'}
]

regions = [69, 25, 67]
budget = 60

# Étape 1 : Filtrage des VaaS par région
candidates_per_region = {region: [] for region in regions}

for idx, vaa in enumerate(vaas):
    if (
        vaa["climatisation"] and
        vaa["places_min"] >= 3 and
        vaa["support_velo"] and
        vaa["type_energy"] == "électrique"
    ):
        for region in vaa["coverd_regions"]:
            if region in candidates_per_region:
                candidates_per_region[region].append(idx)

# Étape 2 : Fonctions de composantes
def compute_total_cost(indices):
    return sum(vaas[candidates_per_region[region][i]]["cost"] for region, i in zip(regions, indices))

def compute_total_time(indices):
    return sum(1 / vaas[candidates_per_region[region][i]]["speed"] for region, i in zip(regions, indices))

def compute_total_emission(indices):
    return sum(vaas[candidates_per_region[region][i]]["emission"] for region, i in zip(regions, indices))

def objective_function(indices):
    resolved_indices = [
        candidates_per_region[region][i]
        for region, i in zip(regions, indices)
    ]

    cost_total = compute_total_cost(indices)
    time_total = compute_total_time(indices)
    emissions_total = compute_total_emission(indices)

    penalty = 0

    for region, global_index in zip(regions, resolved_indices):
        if region not in vaas[global_index]["coverd_regions"]:
            penalty += 1e6

    if budget is not None and cost_total > budget:
        penalty += (cost_total - budget) ** 2

    return cost_total + time_total + emissions_total + penalty

# Étape 3 : Implémentation de l'algorithme de recherche taboue
def tabu_search():
    # Initialisation des paramètres
    num_iterations = 100
    best_solution = None
    best_score = float('inf')

    lb = [0] * len(regions)
    ub = [len(candidates_per_region[region]) - 1 for region in regions]

    # Recherche tabou
    for _ in range(num_iterations):
        current_solution = [np.random.randint(lb[i], ub[i] + 1) for i in range(len(regions))]  
        current_score = objective_function(current_solution)

        if current_score < best_score:
            best_score = current_score
            best_solution = current_solution

    return {
        "best_score": best_score,
        "best_solution": best_solution
    }

# Étape 4 : Fonction optimize()
def optimize():
    result = tabu_search()
    return {
        "best_score": result["best_score"],
        "best_solution": result["best_solution"]
    }

# Étape 5 : Variables exportées
regions_exported = regions
candidates_per_region_exported = {
    region: candidates_per_region[region]
    for region in regions
}

if __name__ == "__main__":
    result = optimize()
    print("=== Résultat de l'optimisation ===")
    print(result)
