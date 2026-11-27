
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# ============================================================
# 1. WORKFLOW TASKS AND CLOUD VIRTUAL MACHINES
# ============================================================

# Each task has:
# length     = computational workload (MI)
# min_cpu    = minimum required CPU capacity
# min_bw     = minimum required bandwidth
# max_price  = maximum allowed VM price
# predecessors = dependent tasks and data transfer size (MB)

tasks = [
    {"id": 0, "length": 500,  "min_cpu": 100, "min_bw": 50,
     "max_price": 0.20, "pred": {}},

    {"id": 1, "length": 1000, "min_cpu": 200, "min_bw": 50,
     "max_price": 0.30, "pred": {0: 100}},

    {"id": 2, "length": 800, "min_cpu": 150, "min_bw": 50,
     "max_price": 0.30, "pred": {0: 150}},

    {"id": 3, "length": 1200, "min_cpu": 250, "min_bw": 100,
     "max_price": 0.50, "pred": {1: 200}},

    {"id": 4, "length": 700, "min_cpu": 150, "min_bw": 50,
     "max_price": 0.30, "pred": {1: 100, 2: 150}},

    {"id": 5, "length": 1500, "min_cpu": 300, "min_bw": 100,
     "max_price": 0.60, "pred": {3: 200, 4: 250}},

    {"id": 6, "length": 900, "min_cpu": 200, "min_bw": 50,
     "max_price": 0.40, "pred": {4: 100}},

    {"id": 7, "length": 600, "min_cpu": 100, "min_bw": 50,
     "max_price": 0.30, "pred": {5: 150, 6: 100}}
]

# Cloud virtual machines
# mips = Million Instructions Per Second
# bandwidth = MB per second
# price = cost per second

vms = [
    {"id": 0, "mips": 500,  "bandwidth": 100,
     "price": 0.10},

    {"id": 1, "mips": 1000, "bandwidth": 200,
     "price": 0.18},

    {"id": 2, "mips": 1500, "bandwidth": 300,
     "price": 0.25},

    {"id": 3, "mips": 2000, "bandwidth": 400,
     "price": 0.35}
]

N_TASKS = len(tasks)
N_VMS = len(vms)

# ============================================================
# 2. WORKFLOW DAG
# ============================================================

G = nx.DiGraph()

for task in tasks:
    G.add_node(task["id"])

    for parent in task["pred"]:
        G.add_edge(parent, task["id"])

# Check workflow validity
if not nx.is_directed_acyclic_graph(G):
    raise ValueError("Workflow must be a Directed Acyclic Graph!")

# Topological task execution order
task_order = list(nx.topological_sort(G))

print("Workflow execution order:", task_order)

# ============================================================
# 3. CHECK VM RESOURCE REQUIREMENTS
# ============================================================

def is_eligible(task, vm):
    return (
        vm["mips"] >= task["min_cpu"]
        and vm["bandwidth"] >= task["min_bw"]
        and vm["price"] <= task["max_price"]
    )


eligible_vms = []

for task in tasks:
    eligible = [
        vm["id"] for vm in vms
        if is_eligible(task, vm)
    ]

    if not eligible:
        raise ValueError(
            f"No eligible VM for task {task['id']}"
        )

    eligible_vms.append(eligible)

# ============================================================
# 4. FITNESS FUNCTION: MAKESPAN AND EXECUTION COST
# ============================================================

def evaluate(position):

    # Convert continuous PSO position into VM assignments
    assignment = np.rint(position).astype(int)
    assignment = np.clip(assignment, 0, N_VMS - 1)

    # Validate resource eligibility
    for i in range(N_TASKS):
        if assignment[i] not in eligible_vms[i]:
            return 1e9, 0, 0, assignment

    vm_available = np.zeros(N_VMS)

    start_times = np.zeros(N_TASKS)
    finish_times = np.zeros(N_TASKS)

    execution_cost = 0

    # Schedule tasks in topological order
    for task_id in task_order:

        task = tasks[task_id]
        vm_id = assignment[task_id]
        vm = vms[vm_id]

        # Calculate task execution time
        execution_time = task["length"] / vm["mips"]

        # Determine when predecessor tasks finish
        dependency_finish = 0

        for parent_id, data_size in task["pred"].items():

            parent_vm = assignment[parent_id]

            transfer_time = 0

            # Transfer is required if tasks use different VMs
            if parent_vm != vm_id:

                bandwidth = min(
                    vm["bandwidth"],
                    vms[parent_vm]["bandwidth"]
                )

                transfer_time = data_size / bandwidth

            arrival_time = (
                finish_times[parent_id] + transfer_time
            )

            dependency_finish = max(
                dependency_finish,
                arrival_time
            )

        # VM must also be available
        start_time = max(
            dependency_finish,
            vm_available[vm_id]
        )

        finish_time = start_time + execution_time

        start_times[task_id] = start_time
        finish_times[task_id] = finish_time

        vm_available[vm_id] = finish_time

        execution_cost += execution_time * vm["price"]

    makespan = max(finish_times)

    return (
        makespan,
        execution_cost,
        start_times,
        assignment
    )

# ============================================================
# 5. PARTICLE INITIALIZATION
# ============================================================

def initialize_particles(
    population,
    enhanced=False
):

    particles = np.zeros((population, N_TASKS))

    # Sort tasks by descending workload
    sorted_tasks = sorted(
        range(N_TASKS),
        key=lambda i: tasks[i]["length"],
        reverse=True
    )

    for p in range(population):

        # Random initialization
        for i in range(N_TASKS):
            particles[p, i] = np.random.choice(
                eligible_vms[i]
            )

        if enhanced:

            # Heuristic initialization:
            # Prioritize large tasks for high-performance VMs.
            # Initialize approximately 25% of tasks heuristically.

            count = max(1, int(0.25 * N_TASKS))

            for task_id in sorted_tasks[:count]:

                possible_vms = eligible_vms[task_id]

                best_vm = max(
                    possible_vms,
                    key=lambda vm_id: vms[vm_id]["mips"]
                )

                particles[p, task_id] = best_vm

    return particles

# ============================================================
# 6. PARTICLE SWARM OPTIMIZATION
# ============================================================

def run_pso(
    enhanced=False,
    population=30,
    iterations=100
):

    # PSO parameters
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    velocity_max = 1.5

    # Initialize positions and velocities
    positions = initialize_particles(
        population,
        enhanced
    )

    velocities = np.random.uniform(
        -1, 1,
        (population, N_TASKS)
    )

    # Evaluate initial particles
    pbest_positions = positions.copy()

    pbest_scores = np.array([
        evaluate(p)[0] for p in positions
    ])

    best_index = np.argmin(pbest_scores)

    gbest_position = pbest_positions[
        best_index
    ].copy()

    gbest_score = pbest_scores[best_index]

    convergence = []

    # ========================================================
    # PSO ITERATIONS
    # ========================================================

    for iteration in range(iterations):

        for i in range(population):

            # Evaluate current particle
            score = evaluate(positions[i])[0]

            # Update personal best
            if score < pbest_scores[i]:

                pbest_scores[i] = score

                pbest_positions[i] = positions[i].copy()

            # Update global best
            if score < gbest_score:

                gbest_score = score

                gbest_position = positions[i].copy()

            # Random values
            r1 = np.random.random(N_TASKS)
            r2 = np.random.random(N_TASKS)

            # Velocity update equation
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (
                    pbest_positions[i] - positions[i]
                )
                + c2 * r2 * (
                    gbest_position - positions[i]
                )
            )

            # Limit velocity
            velocities[i] = np.clip(
                velocities[i],
                -velocity_max,
                velocity_max
            )

            # Position update
            positions[i] += velocities[i]

            # Keep position within VM range
            positions[i] = np.clip(
                positions[i],
                0,
                N_VMS - 1
            )

        convergence.append(gbest_score)

    return (
        gbest_position,
        gbest_score,
        convergence
    )

# ============================================================
# 7. RUN TRADITIONAL PSO
# ============================================================

np.random.seed(42)

print("\nRunning Traditional PSO...")

pso_position, pso_makespan, pso_curve = run_pso(
    enhanced=False
)

# ============================================================
# 8. RUN ENHANCED PSO (EPSO)
# ============================================================

np.random.seed(42)

print("Running Enhanced PSO...")

epso_position, epso_makespan, epso_curve = run_pso(
    enhanced=True
)

# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

pso_result = evaluate(pso_position)
epso_result = evaluate(epso_position)

print("\n========== OPTIMIZATION RESULTS ==========")

print("\nTraditional PSO")
print("Makespan:", round(pso_result[0], 4), "seconds")
print("Execution cost:", round(pso_result[1], 4))

print("\nEnhanced PSO")
print("Makespan:", round(epso_result[0], 4), "seconds")
print("Execution cost:", round(epso_result[1], 4))

improvement = (
    (pso_result[0] - epso_result[0])
    / pso_result[0]
) * 100

print(
    "\nMakespan improvement:",
    round(improvement, 2), "%"
)

# ============================================================
# 10. TASK ALLOCATION TABLE
# ============================================================

rows = []

for i in range(N_TASKS):

    vm_id = int(epso_result[3][i])

    rows.append({
        "Task": f"T{i}",
        "Assigned VM": f"VM{vm_id}",
        "Start Time": round(epso_result[2][i], 4),
        "Execution Time": round(
            tasks[i]["length"] / vms[vm_id]["mips"],
            4
        ),
        "Finish Time": round(
            epso_result[2][i]
            + tasks[i]["length"] / vms[vm_id]["mips"],
            4
        )
    })

df = pd.DataFrame(rows)

print("\n========== EPSO TASK SCHEDULE ==========")
print(df.to_string(index=False))

# ============================================================
# 11. CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    pso_curve,
    label="Traditional PSO",
    linewidth=2
)

plt.plot(
    epso_curve,
    label="Enhanced PSO",
    linewidth=2
)

plt.xlabel("Iterations")
plt.ylabel("Best Makespan (seconds)")

plt.title(
    "Convergence Comparison: PSO vs EPSO"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 12. MAKESPAN COMPARISON
# ============================================================

plt.figure(figsize=(7, 5))

algorithms = ["Traditional PSO", "Enhanced PSO"]

makespans = [
    pso_result[0],
    epso_result[0]
]

plt.bar(
    algorithms,
    makespans,
    color=["steelblue", "orange"]
)

plt.ylabel("Makespan (seconds)")
plt.title("Workflow Scheduling Performance")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 13. EXECUTION COST COMPARISON
# ============================================================

plt.figure(figsize=(7, 5))

costs = [
    pso_result[1],
    epso_result[1]
]

plt.bar(
    algorithms,
    costs,
    color=["steelblue", "orange"]
)

plt.ylabel("Execution Cost")
plt.title("Execution Cost Comparison")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
