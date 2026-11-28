
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

POP_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.02
ELITE_SIZE = 2

ECA_RULE = 110       # Try 90 or 110
K_MIN = 1
K_MAX = 30

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ==========================================================
# 2. LOAD AND PREPROCESS DATASET
# ==========================================================

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Validation split from training data
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train
)


# ==========================================================
# 3. BINARY CHROMOSOME ENCODING
# ==========================================================

N_BITS = int(np.ceil(np.log2(K_MAX + 1)))


def encode(value):
    return np.array(
        list(np.binary_repr(value, width=N_BITS)),
        dtype=int
    )


def decode(chromosome):
    value = int("".join(map(str, chromosome)), 2)

    return int(np.clip(value, K_MIN, K_MAX))


# ==========================================================
# 4. FITNESS FUNCTION USING KNN
# ==========================================================

def fitness(chromosome):

    k = decode(chromosome)

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform"
    )

    model.fit(X_fit, y_fit)

    predictions = model.predict(X_val)

    return accuracy_score(y_val, predictions)


# ==========================================================
# 5. INITIAL POPULATION
# ==========================================================

def initialize_population():

    return rng.integers(
        0, 2,
        size=(POP_SIZE, N_BITS)
    )


# ==========================================================
# 6. ROULETTE WHEEL SELECTION
# ==========================================================

def roulette_selection(population, fitness_values):

    # Shift fitness to ensure non-negative probabilities
    probabilities = fitness_values - fitness_values.min() + 1e-8

    probabilities = probabilities / probabilities.sum()

    index = rng.choice(
        len(population),
        p=probabilities
    )

    return population[index].copy()


# ==========================================================
# 7. ELEMENTARY CELLULAR AUTOMATON
# ==========================================================

def eca_rule(left, center, right, rule=110):

    pattern = (
        left * 4 +
        center * 2 +
        right
    )

    return (rule >> pattern) & 1


def eca_chromosome(chromosome, rule=110):

    n = len(chromosome)

    child = np.zeros(n, dtype=int)

    for i in range(n):

        left = chromosome[i - 1] if i > 0 else 0

        center = chromosome[i]

        right = chromosome[i + 1] if i < n - 1 else 0

        child[i] = eca_rule(
            left, center, right, rule
        )

    return child


# ==========================================================
# 8. CGACELL ECA MIXED CROSSOVER
#    Three-parent cellular automaton crossover
# ==========================================================

def cgacell_eca_crossover(parent1, parent2, parent3,
                          rule=110):

    n = len(parent1)

    child = np.zeros(n, dtype=int)

    for i in range(n):

        child[i] = eca_rule(
            parent1[i],
            parent2[i],
            parent3[i],
            rule
        )

    return child


# ==========================================================
# 9. TWO-DIMENSIONAL CELLULAR AUTOMATON
# ==========================================================

def ca_2d_crossover(population, rule=6,
                    neighborhood="von_neumann",
                    generations=1):

    grid = population.copy()

    rows, cols = grid.shape

    if neighborhood == "von_neumann":
        offsets = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1)
        ]

    elif neighborhood == "moore":
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    else:
        raise ValueError("Unknown neighborhood")

    for _ in range(generations):

        new_grid = np.zeros_like(grid)

        for i in range(rows):
            for j in range(cols):

                neighborhood_bits = []

                for di, dj in offsets:

                    ni = i + di
                    nj = j + dj

                    if (0 <= ni < rows and
                            0 <= nj < cols):
                        neighborhood_bits.append(
                            grid[ni, nj]
                        )
                    else:
                        neighborhood_bits.append(0)

                pattern = 0

                for bit in neighborhood_bits:
                    pattern = (pattern << 1) | int(bit)

                new_grid[i, j] = (
                    rule >> pattern
                ) & 1

        grid = new_grid

    return grid


# ==========================================================
# 10. CLASSICAL CROSSOVER OPERATORS
# ==========================================================

def single_point_crossover(p1, p2):

    point = rng.integers(1, len(p1))

    return np.concatenate([
        p1[:point],
        p2[point:]
    ])


def two_point_crossover(p1, p2):

    points = sorted(
        rng.choice(
            np.arange(1, len(p1)),
            size=2,
            replace=False
        )
    )

    a, b = points

    child = p1.copy()
    child[a:b] = p2[a:b]

    return child


def uniform_crossover(p1, p2):

    mask = rng.integers(0, 2, size=len(p1))

    return np.where(mask == 1, p1, p2)


# ==========================================================
# 11. MUTATION
# ==========================================================

def mutation(chromosome):

    child = chromosome.copy()

    for i in range(len(child)):

        if rng.random() < MUTATION_RATE:

            child[i] = 1 - child[i]

    return child


# ==========================================================
# 12. GENETIC ALGORITHM
# ==========================================================

def genetic_algorithm(method="cgacell",
                      rule=110):

    population = initialize_population()

    history = []

    best_chromosome = None
    best_fitness = -1

    for generation in range(GENERATIONS):

        fitness_values = np.array([
            fitness(ch) for ch in population
        ])

        best_idx = np.argmax(fitness_values)

        generation_best = fitness_values[best_idx]

        if generation_best > best_fitness:

            best_fitness = generation_best

            best_chromosome = population[
                best_idx
            ].copy()

        history.append(best_fitness)

        # Elitism
        elite_indices = np.argsort(
            fitness_values
        )[-ELITE_SIZE:]

        new_population = [
            population[i].copy()
            for i in elite_indices
        ]

        # Generate new offspring
        while len(new_population) < POP_SIZE:

            p1 = roulette_selection(
                population, fitness_values
            )

            p2 = roulette_selection(
                population, fitness_values
            )

            p3 = roulette_selection(
                population, fitness_values
            )

            if method == "cgacell":

                child = cgacell_eca_crossover(
                    p1, p2, p3, rule
                )

            elif method == "eca_individual":

                child = eca_chromosome(
                    p1, rule
                )

            elif method == "single":

                child = single_point_crossover(
                    p1, p2
                )

            elif method == "two_point":

                child = two_point_crossover(
                    p1, p2
                )

            elif method == "uniform":

                child = uniform_crossover(
                    p1, p2
                )

            else:
                raise ValueError("Unknown method")

            child = mutation(child)

            new_population.append(child)

        population = np.array(new_population)

        if (generation + 1) % 10 == 0:

            print(
                f"{method} | Generation "
                f"{generation + 1} | "
                f"Best Accuracy: "
                f"{best_fitness:.4f} | "
                f"k = {decode(best_chromosome)}"
            )

    return best_chromosome, history


# ==========================================================
# 13. RUN EXPERIMENTS
# ==========================================================

methods = [
    "cgacell",
    "eca_individual",
    "single",
    "two_point",
    "uniform"
]

results = {}
histories = {}

for method in methods:

    print("\nRunning:", method)

    chromosome, history = genetic_algorithm(
        method=method,
        rule=ECA_RULE
    )

    results[method] = chromosome
    histories[method] = history


# ==========================================================
# 14. FINAL TEST PERFORMANCE
# ==========================================================

print("\n========== FINAL RESULTS ==========")

result_table = []

for method in methods:

    chromosome = results[method]

    best_k = decode(chromosome)

    model = KNeighborsClassifier(
        n_neighbors=best_k
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test, predictions
    )

    result_table.append({
        "Method": method,
        "Optimal K": best_k,
        "Validation Accuracy": fitness(chromosome),
        "Test Accuracy": test_accuracy * 100
    })

    print(
        f"{method:20s} "
        f"K={best_k:3d} "
        f"Test Accuracy={test_accuracy * 100:.2f}%"
    )

df = pd.DataFrame(result_table)

print("\n", df.to_string(index=False))


# ==========================================================
# 15. CONVERGENCE GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

for method in methods:

    plt.plot(
        histories[method],
        label=method
    )

plt.xlabel("Generation")
plt.ylabel("Best Validation Accuracy")

plt.title(
    "CGACell Genetic Algorithm vs Traditional GA"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================================
# 16. FINAL ACCURACY COMPARISON
# ==========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Method"],
    df["Test Accuracy"],
    color=[
        "green", "orange", "blue",
        "purple", "red"
    ]
)

plt.xlabel("Crossover Method")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Performance Comparison of Crossover Operators"
)

plt.xticks(rotation=20)

plt.grid(axis="y")

plt.tight_layout()
plt.show()
