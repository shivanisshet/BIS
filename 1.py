
import numpy as np
import matplotlib.pyplot as plt


POPULATION_SIZE = 300
GENERATIONS = 100
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.005

BITS_PER_VARIABLE = 16
CHROMOSOME_LENGTH = BITS_PER_VARIABLE * 2

X_MIN, X_MAX = 0, 1000
Y_MIN, Y_MAX = 0, 1000

BUDGET = 150000
LABOR_COST = 180
CAPITAL_COST = 220

ELITE_COUNT = 2

np.random.seed(42)




def objective_function(x, y):
    return 150 * (x ** 0.7) * (y ** 0.3)




def decode_chromosome(chromosome):

    x_bits = chromosome[:BITS_PER_VARIABLE]
    y_bits = chromosome[BITS_PER_VARIABLE:]

    x_decimal = int("".join(map(str, x_bits)), 2)
    y_decimal = int("".join(map(str, y_bits)), 2)

    max_value = (2 ** BITS_PER_VARIABLE) - 1

    x = X_MIN + (x_decimal / max_value) * (X_MAX - X_MIN)
    y = Y_MIN + (y_decimal / max_value) * (Y_MAX - Y_MIN)

    return x, y




def fitness_function(chromosome):

    x, y = decode_chromosome(chromosome)

    total_cost = LABOR_COST * x + CAPITAL_COST * y

    if total_cost > BUDGET:
        return 0

    return objective_function(x, y)




def create_population():

    return np.random.randint(
        0, 2,
        size=(POPULATION_SIZE, CHROMOSOME_LENGTH)
    )




def selection(population, fitness_values):

    selected = []

    for _ in range(2):

        candidates = np.random.choice(
            len(population), 3, replace=False
        )

        winner = candidates[
            np.argmax(fitness_values[candidates])
        ]

        selected.append(population[winner].copy())

    return selected




def crossover(parent1, parent2):

    child1 = parent1.copy()
    child2 = parent2.copy()

    if np.random.rand() < CROSSOVER_RATE:

        point = np.random.randint(1, CHROMOSOME_LENGTH)

        child1 = np.concatenate((
            parent1[:point],
            parent2[point:]
        ))

        child2 = np.concatenate((
            parent2[:point],
            parent1[point:]
        ))

    return child1, child2




def mutation(chromosome):

    for i in range(CHROMOSOME_LENGTH):

        if np.random.rand() < MUTATION_RATE:

            chromosome[i] = 1 - chromosome[i]

    return chromosome



population = create_population()

best_fitness_history = []
average_fitness_history = []

best_x_history = []
best_y_history = []

global_best_fitness = -1
global_best_chromosome = None

for generation in range(GENERATIONS):

    fitness_values = np.array([
        fitness_function(chromosome)
        for chromosome in population
    ])

    best_index = np.argmax(fitness_values)

    generation_best = fitness_values[best_index]
    average_fitness = np.mean(fitness_values)

    if generation_best > global_best_fitness:

        global_best_fitness = generation_best

        global_best_chromosome = population[
            best_index
        ].copy()

 
    best_x, best_y = decode_chromosome(
        global_best_chromosome
    )

    best_fitness_history.append(global_best_fitness)
    average_fitness_history.append(average_fitness)

    best_x_history.append(best_x)
    best_y_history.append(best_y)

    print(
        f"Generation {generation + 1:3d} | "
        f"Best Fitness: {global_best_fitness:.4f} | "
        f"Labor: {best_x:.2f} | "
        f"Capital: {best_y:.2f}"
    )

    
    elite_indices = np.argsort(fitness_values)[-ELITE_COUNT:]

    new_population = [
        population[i].copy()
        for i in elite_indices
    ]

   
    while len(new_population) < POPULATION_SIZE:

        parent1, parent2 = selection(
            population, fitness_values
        )

        child1, child2 = crossover(parent1, parent2)

        child1 = mutation(child1)
        child2 = mutation(child2)

        new_population.append(child1)

        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = np.array(new_population)




optimal_x, optimal_y = decode_chromosome(
    global_best_chromosome
)

maximum_output = objective_function(
    optimal_x, optimal_y
)

total_cost = (
    LABOR_COST * optimal_x +
    CAPITAL_COST * optimal_y
)

print("\n========== FINAL OPTIMIZATION RESULTS ==========")

print(f"Optimal Labor (X):     {optimal_x:.4f}")
print(f"Optimal Capital (Y):   {optimal_y:.4f}")
print(f"Maximum Output (Z):    {maximum_output:.4f}")
print(f"Total Budget Used:     {total_cost:.4f}")
print(f"Budget Available:      {BUDGET:.4f}")
print(f"Budget Remaining:      {BUDGET-total_cost:.4f}")




plt.figure(figsize=(10, 5))

plt.plot(
    best_fitness_history,
    label="Best Fitness",
    color="blue"
)

plt.plot(
    average_fitness_history,
    label="Average Fitness",
    color="orange"
)

plt.xlabel("Generation")
plt.ylabel("Production Output")
plt.title("GA Convergence - Cobb-Douglas Optimization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 5))

plt.plot(best_x_history, color="green")

plt.axhline(
    y=583.33,
    color="red",
    linestyle="--",
    label="Analytical Labor = 583.33"
)

plt.xlabel("Generation")
plt.ylabel("Labor (X)")
plt.title("Convergence of Labor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 5))

plt.plot(best_y_history, color="purple")

plt.axhline(
    y=204.55,
    color="red",
    linestyle="--",
    label="Analytical Capital = 204.55"
)

plt.xlabel("Generation")
plt.ylabel("Capital (Y)")
plt.title("Convergence of Capital")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



from mpl_toolkits.mplot3d import Axes3D

x_values = np.linspace(0, 1000, 100)
y_values = np.linspace(0, 1000, 100)

X, Y = np.meshgrid(x_values, y_values)

Z = 150 * (X ** 0.7) * (Y ** 0.3)

# Apply budget constraint to visualization
Z[
    (LABOR_COST * X + CAPITAL_COST * Y) > BUDGET
] = np.nan

fig = plt.figure(figsize=(11, 7))

ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Y, Z,
    cmap="viridis",
    alpha=0.8
)

ax.scatter(
    optimal_x,
    optimal_y,
    maximum_output,
    color="red",
    s=100,
    label="GA Optimal Solution"
)

ax.set_xlabel("Labor (X)")
ax.set_ylabel("Capital (Y)")
ax.set_zlabel("Production Output (Z)")

ax.set_title("Cobb-Douglas Function Optimization using GA")

ax.legend()

plt.tight_layout()
plt.show()
