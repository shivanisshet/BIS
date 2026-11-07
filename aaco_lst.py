
import numpy as np
import matplotlib.pyplot as plt


class AACO_LST:

    def __init__(
        self,
        coordinates,
        n_ants=50,
        max_iterations=1000,
        alpha_base=2,
        beta_base=3,
        evaporation=0.5,
        local_evaporation=0.1,
        Q=1,
        lam=0.5,
        gamma=0.9,
        omega=0.8,
        stagnation_limit=10,
        seed=42
    ):

        self.coords = np.asarray(coordinates, dtype=float)

        self.n = len(coordinates)
        self.m = n_ants
        self.max_iter = max_iterations

        self.A = alpha_base
        self.B = beta_base

        self.rho0 = evaporation
        self.rho = evaporation

        self.epsilon = local_evaporation
        self.Q = Q

        self.lam = lam
        self.gamma = gamma
        self.omega = omega

        self.s0 = stagnation_limit

        self.rng = np.random.default_rng(seed)

      
        self.distance = self.calculate_distance()

        
        self.Lnn = self.greedy_solution()

        
        self.tau0 = 1.0 / (self.m * self.Lnn)

        self.pheromone = np.full(
            (self.n, self.n),
            self.tau0
        )

        np.fill_diagonal(self.pheromone, 0)

        self.best_route = None
        self.best_length = float("inf")

        self.history = []

        self.previous_best = float("inf")
        self.stagnation = 0

   

    def calculate_distance(self):

        diff = (
            self.coords[:, None, :]
            - self.coords[None, :, :]
        )

        return np.sqrt(np.sum(diff ** 2, axis=2))

    

    def greedy_solution(self):

        unvisited = set(range(self.n))

        current = 0
        unvisited.remove(current)

        length = 0

        while unvisited:

            next_city = min(
                unvisited,
                key=lambda city: self.distance[current, city]
            )

            length += self.distance[current, next_city]

            current = next_city
            unvisited.remove(current)

        length += self.distance[current, 0]

        return length

  
    def adaptive_parameters(self, iteration):

        r1 = self.rng.random()
        r2 = self.rng.random()

        alpha = (
            np.cos(
                r1 * iteration * np.pi /
                (2 * self.max_iter)
            ) + self.A
        )

        beta = (
            np.sin(
                r2 * iteration * np.pi /
                (2 * self.max_iter)
            ) + self.B
        )

        return alpha, beta

   

    def select_next_city(
        self,
        current,
        unvisited,
        alpha,
        beta
    ):

        cities = np.array(list(unvisited))

        pheromone = self.pheromone[current, cities]
        heuristic = 1.0 / (
            self.distance[current, cities] + 1e-12
        )

       
        log_prob = (
            alpha * np.log(pheromone + 1e-30)
            + beta * np.log(heuristic + 1e-30)
        )

        log_prob -= np.max(log_prob)

        probabilities = np.exp(log_prob)

        probabilities /= np.sum(probabilities)

        next_city = self.rng.choice(
            cities,
            p=probabilities
        )

        return int(next_city)

    def construct_route(self, alpha, beta):

        start = self.rng.integers(self.n)

        route = [start]

        unvisited = set(range(self.n))
        unvisited.remove(start)

        current = start

        while unvisited:

            next_city = self.select_next_city(
                current,
                unvisited,
                alpha,
                beta
            )

            route.append(next_city)

           
            i = current
            j = next_city

            new_tau = (
                (1 - self.epsilon)
                * self.pheromone[i, j]
                + self.epsilon * self.tau0
            )

            self.pheromone[i, j] = new_tau
            self.pheromone[j, i] = new_tau

            current = next_city
            unvisited.remove(next_city)

        return np.array(route)

   

    def route_length(self, route):

        return np.sum(
            self.distance[
                route,
                np.roll(route, -1)
            ]
        )

   

    def local_search(self, route):

        route = route.copy()

        n = len(route)

   
        for i in range(n):

            a = route[i]
            b = route[(i + 1) % n]
            c = route[(i + 2) % n]
            d = route[(i + 3) % n]

            old_length = (
                self.distance[a, b]
                + self.distance[c, d]
            )

            new_length = (
                self.distance[a, c]
                + self.distance[b, d]
            )

            if new_length < old_length:

              
                if i + 2 < n:
                    route[i + 1:i + 3] = (
                        route[i + 1:i + 3][::-1]
                    )

                else:
                    
                    route = np.roll(route, -i)
                    route[1:3] = route[1:3][::-1]
                    route = np.roll(route, i)

        return route

   

    def global_update(self, routes, lengths):

       
        order = np.argsort(lengths)

        selected_count = max(
            1,
            int(self.lam * self.m)
        )

        rho = self.rho

        self.pheromone *= (1 - rho)

        for rank_position, ant_index in enumerate(
            order[:selected_count]
        ):

            route = routes[ant_index]
            length = lengths[ant_index]

            weight = selected_count - rank_position

            deposit = (
                weight * self.Q / (length + 1e-12)
            )

            for i in range(self.n):

                a = route[i]
                b = route[(i + 1) % self.n]

                self.pheromone[a, b] += rho * deposit
                self.pheromone[b, a] += rho * deposit

        np.fill_diagonal(self.pheromone, 0)

       
        self.pheromone = np.maximum(
            self.pheromone,
            1e-15
        )

   
    def update_evaporation(self, iteration):

        if iteration < self.omega * self.max_iter:

            self.rho = self.rho0

        elif self.stagnation > self.s0:

            self.rho *= self.gamma

        self.rho = max(self.rho, 0.01)

   
    def optimize(self):

        print("\nStarting AACO-LST Optimization\n")

        for iteration in range(1, self.max_iter + 1):

            alpha, beta = self.adaptive_parameters(
                iteration
            )

            routes = []
            lengths = []

            
            for ant in range(self.m):

                route = self.construct_route(
                    alpha,
                    beta
                )

                length = self.route_length(route)

                routes.append(route)
                lengths.append(length)

            lengths = np.array(lengths)

            order = np.argsort(lengths)

            selected_count = max(
                1,
                int(self.lam * self.m)
            )

            for ant_index in order[:selected_count]:

                improved_route = self.local_search(
                    routes[ant_index]
                )

                improved_length = self.route_length(
                    improved_route
                )

                if improved_length < lengths[ant_index]:

                    routes[ant_index] = improved_route
                    lengths[ant_index] = improved_length

            iteration_best = np.argmin(lengths)

            if lengths[iteration_best] < self.best_length:

                self.best_length = lengths[iteration_best]

                self.best_route = routes[
                    iteration_best
                ].copy()

            
            self.global_update(routes, lengths)

          
            if abs(
                self.previous_best - self.best_length
            ) < 1e-10:

                self.stagnation += 1

            else:

                self.stagnation = 0

            self.previous_best = self.best_length

            
            self.update_evaporation(iteration)

            self.history.append(self.best_length)

            if iteration % 50 == 0 or iteration == 1:

                print(
                    f"Iteration: {iteration:4d} | "
                    f"Best Distance: {self.best_length:.4f} | "
                    f"Alpha: {alpha:.3f} | "
                    f"Beta: {beta:.3f} | "
                    f"Rho: {self.rho:.4f}"
                )

        print("\nOptimization Completed!")

        print(
            "Best Distance:",
            round(self.best_length, 4)
        )

        print("Best Route:", self.best_route)

        return self.best_route, self.best_length

 

    def plot_convergence(self):

        plt.figure(figsize=(9, 5))

        plt.plot(
            self.history,
            color="blue",
            linewidth=2
        )

        plt.xlabel("Iterations")
        plt.ylabel("Best Tour Distance")

        plt.title("AACO-LST Convergence Curve")

        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

  

    def plot_route(self):

        route = self.best_route

        closed_route = np.append(route, route[0])

        plt.figure(figsize=(8, 7))

        plt.plot(
            self.coords[closed_route, 0],
            self.coords[closed_route, 1],
            "b-o",
            markersize=4,
            linewidth=1.5
        )

        plt.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            color="red",
            s=30
        )

        plt.title(
            f"AACO-LST Optimal Route\n"
            f"Distance = {self.best_length:.4f}"
        )

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":



    np.random.seed(42)

    coordinates = np.random.rand(50, 2) * 100

    aaco = AACO_LST(
        coordinates=coordinates,
        n_ants=50,
        max_iterations=1000,
        alpha_base=2,
        beta_base=3,
        evaporation=0.5,
        local_evaporation=0.1,
        Q=1,
        lam=0.5,
        gamma=0.9,
        omega=0.8,
        stagnation_limit=10,
        seed=42
    )

  
    best_route, best_distance = aaco.optimize()

    aaco.plot_convergence()

    aaco.plot_route()
