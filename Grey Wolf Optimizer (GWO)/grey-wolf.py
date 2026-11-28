
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ============================================================
# 1. BENCHMARK FUNCTIONS
# ============================================================

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock(x):
    return np.sum(
        100*(x[1:] - x[:-1]**2)**2
        + (x[:-1] - 1)**2
    )

def ackley(x):
    n = len(x)
    return (
        -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))
        - np.exp(np.sum(np.cos(2*np.pi*x))/n)
        + 20 + np.e
    )

def michalewicz(x, m=10):
    i = np.arange(1, len(x)+1)
    return -np.sum(
        np.sin(x) *
        np.sin(i*x**2/np.pi)**(2*m)
    )


# ============================================================
# 2. STANDARD GRAY WOLF OPTIMIZATION
# ============================================================

def GWO(objective, lb, ub, dim,
        population=50, iterations=500, seed=42):

    rng = np.random.default_rng(seed)

    wolves = rng.uniform(
        lb, ub, (population, dim)
    )

    history = []

    for t in range(iterations):

        fitness = np.array([
            objective(w) for w in wolves
        ])

        order = np.argsort(fitness)

        alpha = wolves[order[0]].copy()
        beta = wolves[order[1]].copy()
        delta = wolves[order[2]].copy()

        a = 2 - 2*t/iterations

        new_wolves = np.zeros_like(wolves)

        for i in range(population):

            positions = []

            for leader in [alpha, beta, delta]:

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                A = 2*a*r1 - a
                C = 2*r2

                D = np.abs(C*leader - wolves[i])

                X = leader - A*D

                positions.append(X)

            new_wolves[i] = np.mean(
                positions, axis=0
            )

        wolves = np.clip(new_wolves, lb, ub)

        best = min(
            objective(w) for w in wolves
        )

        history.append(best)

    final_fitness = np.array([
        objective(w) for w in wolves
    ])

    best_index = np.argmin(final_fitness)

    return (
        wolves[best_index],
        final_fitness[best_index],
        history
    )


# ============================================================
# 3. PMS-GWO INSPIRED ALGORITHM
# ============================================================

def PMS_GWO(objective, lb, ub, dim,
            population=50,
            iterations=500,
            escape_rate=0.08,
            deception_rate=0.02,
            crowding_weight=0.03,
            refine_weight=0.5,
            seed=42):

    rng = np.random.default_rng(seed)

    wolves = rng.uniform(
        lb, ub, (population, dim)
    )

    prey = wolves[0].copy()

    history = []

    for t in range(iterations):

        # ----------------------------------------------------
        # STEP 1: Evaluate population
        # ----------------------------------------------------

        fitness = np.array([
            objective(w) for w in wolves
        ])

        order = np.argsort(fitness)

        alpha = wolves[order[0]].copy()
        beta = wolves[order[1]].copy()
        delta = wolves[order[2]].copy()

        alpha_fit = fitness[order[0]]

        # ----------------------------------------------------
        # STEP 2: Dynamic leadership
        # ----------------------------------------------------

        # Leaders are reassigned using current fitness ranking.

        # ----------------------------------------------------
        # STEP 3: Adaptive parameters
        # ----------------------------------------------------

        progress = t / iterations

        a = 2*(1-progress)

        new_wolves = wolves.copy()

        # Normalize fitness differences for hybrid coefficient
        fitness_range = (
            np.max(fitness) - np.min(fitness) + 1e-12
        )

        # ----------------------------------------------------
        # STEP 4: Update every wolf
        # ----------------------------------------------------

        for i in range(population):

            current = wolves[i].copy()

            # Standard GWO leader-guided movement

            leader_positions = []

            for leader in [alpha, beta, delta]:

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                A = 2*a*r1 - a
                C = 2*r2

                D = np.abs(C*leader - current)

                candidate = leader - A*D

                leader_positions.append(candidate)

            gwo_position = np.mean(
                leader_positions, axis=0
            )

            # ------------------------------------------------
            # STEP 5: Hybrid exploration-exploitation
            # ------------------------------------------------

            fitness_difference = abs(
                fitness[i] - alpha_fit
            )

            C_hybrid = 1 - (
                fitness_difference / fitness_range
            )

            C_hybrid = np.clip(
                C_hybrid, 0, 1
            )

            random_movement = rng.normal(
                0, 1, dim
            ) * (1-progress)

            hybrid_position = (
                current
                + refine_weight*C_hybrid*(alpha-current)
                + 0.1*random_movement
            )

            # ------------------------------------------------
            # STEP 6: Crowding-distance mechanism
            # ------------------------------------------------

            distances = np.linalg.norm(
                wolves-current, axis=1
            )

            distances[i] = np.inf

            nearest_index = np.argmin(distances)

            nearest = wolves[nearest_index]

            crowding_movement = (
                current-nearest
            )

            # ------------------------------------------------
            # STEP 7: Combine movements
            # ------------------------------------------------

            candidate = (
                0.5*gwo_position
                + 0.5*hybrid_position
                + crowding_weight*crowding_movement
            )

            # ------------------------------------------------
            # STEP 8: Step-up / step-down decision
            # ------------------------------------------------

            candidate = np.clip(
                candidate, lb, ub
            )

            candidate_fitness = objective(candidate)

            # Step-up: accept an improving candidate
            if candidate_fitness < fitness[i]:

                new_wolves[i] = candidate

            else:

                # Step-down: exploratory perturbation
                step_down = (
                    current
                    + rng.normal(0, 1, dim)
                    * (ub-lb)
                    * 0.02
                )

                step_down = np.clip(
                    step_down, lb, ub
                )

                # Greedy acceptance protects good solutions
                if objective(step_down) < fitness[i]:
                    new_wolves[i] = step_down
                else:
                    new_wolves[i] = current

        wolves = np.clip(
            new_wolves, lb, ub
        )

        # ----------------------------------------------------
        # STEP 9: Prey escape mechanism
        # ----------------------------------------------------

        fitness = np.array([
            objective(w) for w in wolves
        ])

        order = np.argsort(fitness)

        alpha = wolves[order[0]].copy()

        distances_to_alpha = np.linalg.norm(
            wolves-alpha, axis=1
        )

        # Trigger escape if the population is crowded
        if np.mean(distances_to_alpha) < 0.05*(ub-lb):

            prey = prey + escape_rate * np.sign(
                prey-alpha
            ) * rng.normal(
                0, 1, dim
            ) * (ub-lb)

        # ----------------------------------------------------
        # STEP 10: Prey deception mechanism
        # ----------------------------------------------------

        deceptive_prey = (
            prey
            + deception_rate
            * rng.normal(0, 1, dim)
            * (ub-lb)
        )

        deceptive_prey = np.clip(
            deceptive_prey, lb, ub
        )

        # Occasionally use deceptive position
        if rng.random() < 0.10:

            index = rng.integers(population)

            if objective(deceptive_prey) < objective(
                wolves[index]
            ):
                wolves[index] = deceptive_prey

        # ----------------------------------------------------
        # STEP 11: Preserve the best solution
        # ----------------------------------------------------

        fitness = np.array([
            objective(w) for w in wolves
        ])

        best_index = np.argmin(fitness)

        best_position = wolves[best_index].copy()

        history.append(fitness[best_index])

        # Maintain prey reference as best-known solution
        prey = best_position.copy()

    # Final result

    fitness = np.array([
        objective(w) for w in wolves
    ])

    best_index = np.argmin(fitness)

    return (
        wolves[best_index],
        fitness[best_index],
        history
    )


# ============================================================
# 4. EXPERIMENTAL COMPARISON
# ============================================================

benchmarks = {

    "Sphere": {
        "function": sphere,
        "lb": -100,
        "ub": 100,
        "dim": 30
    },

    "Rastrigin": {
        "function": rastrigin,
        "lb": -5.12,
        "ub": 5.12,
        "dim": 30
    },

    "Rosenbrock": {
        "function": rosenbrock,
        "lb": -30,
        "ub": 30,
        "dim": 30
    },

    "Ackley": {
        "function": ackley,
        "lb": -32,
        "ub": 32,
        "dim": 30
    },

    "Michalewicz": {
        "function": michalewicz,
        "lb": 0,
        "ub": np.pi,
        "dim": 10
    }
}


# ============================================================
# 5. RUN BOTH ALGORITHMS
# ============================================================

results = []

all_histories = {}

for name, config in benchmarks.items():

    objective = config["function"]
    lb = config["lb"]
    ub = config["ub"]
    dim = config["dim"]

    print("\nBenchmark:", name)

    # Standard GWO
    start = time.perf_counter()

    gwo_position, gwo_value, gwo_history = GWO(
        objective, lb, ub, dim
    )

    gwo_time = time.perf_counter()-start

    # PMS-GWO
    start = time.perf_counter()

    pms_position, pms_value, pms_history = PMS_GWO(
        objective, lb, ub, dim
    )

    pms_time = time.perf_counter()-start

    results.append({
        "Function": name,
        "GWO Best": gwo_value,
        "PMS-GWO Best": pms_value,
        "GWO Runtime": gwo_time,
        "PMS-GWO Runtime": pms_time
    })

    all_histories[name] = {
        "GWO": gwo_history,
        "PMS-GWO": pms_history
    }

    print("GWO best fitness:", gwo_value)
    print("PMS-GWO best fitness:", pms_value)

    print("GWO runtime:", gwo_time)
    print("PMS-GWO runtime:", pms_time)


# ============================================================
# 6. RESULTS TABLE
# ============================================================

df = pd.DataFrame(results)

print("\nFINAL PERFORMANCE COMPARISON")

print(df.to_string(index=False))


# ============================================================
# 7. CONVERGENCE GRAPHS
# ============================================================

fig, axes = plt.subplots(
    2, 3, figsize=(16, 9)
)

axes = axes.flatten()

for i, name in enumerate(benchmarks):

    axes[i].plot(
        all_histories[name]["GWO"],
        label="GWO",
        linewidth=2
    )

    axes[i].plot(
        all_histories[name]["PMS-GWO"],
        label="PMS-GWO",
        linewidth=2
    )

    axes[i].set_title(name)

    axes[i].set_xlabel("Iterations")

    axes[i].set_ylabel("Best Fitness")

    axes[i].set_yscale("symlog")

    axes[i].grid(True, alpha=0.3)

    axes[i].legend()

axes[-1].axis("off")

plt.suptitle(
    "GWO vs PMS-GWO Convergence Comparison",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 8. RUNTIME COMPARISON
# ============================================================

x = np.arange(len(df))

width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x-width/2,
    df["GWO Runtime"],
    width,
    label="GWO"
)

plt.bar(
    x+width/2,
    df["PMS-GWO Runtime"],
    width,
    label="PMS-GWO"
)

plt.xticks(x, df["Function"])

plt.ylabel("Runtime (seconds)")

plt.title("Runtime Comparison")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# 9. ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 5))

for algorithm in ["GWO", "PMS-GWO"]:

    values = [
        df.loc[i, algorithm + " Best"]
        for i in range(len(df))
    ]

    plt.plot(
        df["Function"],
        values,
        marker="o",
        label=algorithm
    )

plt.ylabel("Best Fitness")

plt.title("Optimization Accuracy Comparison")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# 10. SAVE EXPERIMENTAL RESULTS
# ============================================================

df.to_csv(
    "PMS_GWO_Experimental_Results.csv",
    index=False
)

print("\nResults saved successfully.")
