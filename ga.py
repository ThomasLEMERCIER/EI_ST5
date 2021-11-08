import random

import numpy as np

class GA:
    def __init__(self, Indiv, fitness_function, config):

        self.population = []
        self.pop_size = config.pop_size

        self.elite_size = int(config.elitist_factor * config.pop_size)

        self.parents_selection_method = config.parents_selection_method

        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate

        self.get_fitness_score = lambda indv: indv.fitness_score

        # Generate initial population
        for _ in range(self.pop_size):
            self.population.append( Indiv(
                                shape=config.shape, fitness_function=fitness_function,
                                starting_interval=config.starting_interval, interval=config.interval,
                                crossover_method=config.crossover_method, mutation_method=config.mutation_method) )

        self.population.sort(key=self.get_fitness_score)
        self.best_indv = self.population[-1]


    def parents_selection(self, method='tournament'):
        """
        Roulette Wheel Selection : Each parent is chosen randomly based on their fitness score
        ie one with a great fitness score
        has more chance to get chosen to be parent

        Stochastic Universal Sampling (SUS) : Each parent is chosen at the same time randomly
        based on their fitness score, very similar to Roulette Wheel Selection
        but the two parents can't be the same individual

        Tournament Selection : We choose k-candidat for the first parent and the one with
        the best fitness value is the chosen one.

        """

        if method == 'tournament':
            tournoi = random.sample(self.population, k=int(0.1 * self.pop_size + 2))
            parents = sorted(tournoi, key=self.get_fitness_score)[-2:]
        elif method == 'sus':
            fitness_exponential = np.exp(np.fromiter((indv.fitness_score for indv in self.population), dtype=np.float32))
            s = np.sum(fitness_exponential)
            p = s/2
            rd = random.random() * p
            pointers = [rd + i * p for i in range(2)]
            parents = []
            for point in pointers:
                i = 0
                while (x:=np.sum(fitness_exponential[:i+1])) < point:
                    i += 1
                parents.append(self.population[i])
        elif method == 'roulette':
            fitness_exponential = np.exp(np.array([indv.fitness_score for indv in self.population], dtype=np.float32))
            proba = fitness_exponential / np.sum(fitness_exponential)
            parents = np.random.choice(self.population, size=2, p=proba)
        else:
            NotImplementedError(f"The following parents selection is not implemented : {method}")
        return parents

    def survivor_selection(self, elite):
        """
        Age based selection : Only keep the newest population

        Fitness based selection : Only keep the indivituals with the best fitness score
        """

        self.population = sorted(self.population + elite, key=self.get_fitness_score)[-self.pop_size:]
        self.best_indv = self.population[-1]

    def step(self):

        elite = []

        self.population.sort(key=self.get_fitness_score)

        for idv in self.population[-self.elite_size:]:
            elite.append(idv.copy())

        # Crossover
        for _ in range(self.pop_size//2):
            # Choose parents
            parent_1, parent_2 = self.parents_selection(self.parents_selection_method)

            if random.random() < self.crossover_rate:
                # Do the crossover
                baby_1, baby_2 = parent_1.crossover(parent_2)

                self.population.append(baby_1)
                self.population.append(baby_2)

        # Mutation
        for individual in self.population:
            if random.random() < self.mutation_rate:
                individual.mutate()

        # survivor selection
        self.survivor_selection(elite)
