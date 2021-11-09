import numpy as np

class Individual_vector():
    def __init__(self, shape, fitness_function):

        self.shape = shape

        self.chromosomes = np.random.uniform(size=self.shape)
        
        self.fitness_function = fitness_function
        self.fit()

    def mutate(self):
        self.chromosomes = 1 - self.chromosomes
        self.fit()

    def crossover(self, individual):
        alpha = np.random.uniform()
        beta = np.random.uniform()
        
        baby_1 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function)
        baby_2 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function)

        baby_1.chromosomes = alpha * self.chromosomes + individual.chromosomes * (1-alpha)
        baby_2.chromosomes = beta * self.chromosomes + individual.chromosomes * (1-beta)

        baby_1.fit()
        baby_2.fit()

        return baby_1, baby_2

    def fit(self):
        self.fitness_score = self.fitness_function(self.chromosomes)

    def copy(self):
        duplicate = Individual_vector( shape=self.shape, fitness_function=self.fitness_function)
        duplicate.chromosomes = self.chromosomes.copy()
        duplicate.fitness_score = self.fitness_score
        return duplicate
