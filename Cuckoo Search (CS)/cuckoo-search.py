
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Parameters
n = 15
iterations = 30
pa = 0.25
alpha = 0.5
dimensions = X_train.shape[1]

# 4. KNN classifier
def classifier():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

# 5. Convert continuous solution into binary
def binary_solution(position):
    sigmoid = 1 / (1 + np.exp(-np.clip(position, -20, 20)))
    return (np.random.rand(dimensions) < sigmoid).astype(int)

# 6. Fitness function
def fitness(position):
    selected = position.astype(bool)

    if np.sum(selected) == 0:
        return 1.0

    model = classifier()

    accuracy = cross_val_score(
        model,
        X_train[:, selected],
        y_train,
        cv=3,
        scoring='accuracy'
    ).mean()

    feature_ratio = np.sum(selected) / dimensions

    return 0.99 * (1 - accuracy) + 0.01 * feature_ratio

# 7. Levy flight
def levy_flight(dim):
    beta = 1.5

    sigma = (
        math.gamma(1 + beta) *
        np.sin(np.pi * beta / 2) /
        (math.gamma((1 + beta) / 2) *
         beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)

    return u / (np.abs(v) ** (1 / beta) + 1e-10)

# 8. Initialize nests
np.random.seed(42)

nests = np.random.uniform(-1, 1, (n, dimensions))
solutions = np.array([binary_solution(x) for x in nests])

scores = np.array([fitness(x) for x in solutions])

best_history = []
feature_history = []

# 9. Cuckoo Search
for iteration in range(iterations):

    for i in range(n):

        # Generate new solution using Levy flight
        new_position = (
            nests[i] +
            alpha * levy_flight(dimensions)
        )

        new_solution = binary_solution(new_position)
        new_score = fitness(new_solution)

        # Random nest selection
        j = np.random.randint(n)

        if new_score < scores[j]:
            nests[j] = new_position
            solutions[j] = new_solution
            scores[j] = new_score

    # Abandon worst nests
    abandon_count = max(1, int(pa * n))
    worst = np.argsort(scores)[-abandon_count:]

    for j in worst:
        nests[j] = np.random.uniform(-1, 1, dimensions)
        solutions[j] = binary_solution(nests[j])
        scores[j] = fitness(solutions[j])

    # Preserve best solution
    best_index = np.argmin(scores)
    current_best = solutions[best_index].copy()
    current_score = scores[best_index]

    best_history.append(current_score)
    feature_history.append(np.sum(current_best))

    print(
        f"Iteration {iteration+1}: "
        f"Fitness = {current_score:.5f}, "
        f"Features = {np.sum(current_best)}"
    )

# 10. Best selected features
best_index = np.argmin(scores)
best_solution = solutions[best_index]

selected_features = feature_names[best_solution == 1]

print("\nSelected Features:")
print(selected_features)

print("\nTotal Selected Features:", len(selected_features))

# 11. Train final model using selected features
model = classifier()

model.fit(
    X_train[:, best_solution == 1],
    y_train
)

predictions = model.predict(
    X_test[:, best_solution == 1]
)

final_accuracy = accuracy_score(y_test, predictions)

print("\nFinal Test Accuracy:",
      round(final_accuracy * 100, 2), "%")

# 12. Baseline accuracy using all features
baseline = classifier()
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

print("Baseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

# 13. Convergence graph
plt.figure(figsize=(8, 5))
plt.plot(best_history, color='blue', linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("Cuckoo Search Convergence")
plt.grid(True)
plt.show()

# 14. Feature selection graph
plt.figure(figsize=(8, 5))
plt.plot(feature_history, color='green', linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.title("Feature Selection Over Iterations")
plt.grid(True)
plt.show()

# 15. Accuracy comparison graph
plt.figure(figsize=(7, 5))
plt.bar(
    ["All Features", "CS Selected Features"],
    [baseline_accuracy * 100, final_accuracy * 100],
    color=['orange', 'green']
)
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy Comparison")
plt.ylim(0, 100)
plt.show()
