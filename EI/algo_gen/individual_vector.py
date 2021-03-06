import numpy as np
from numpy.random.mtrand import beta
import EI._env as _env
import random as rd
import EI.utils as utils

class Individual_vector():
    def __init__(self, shape, fitness_function, normalization_indices, beta, domain):

        self.shape = shape

        self.beta = beta
        self.domain = domain
        self.chromosomes = np.random.uniform(size=self.shape)
        self.normalization_indices = normalization_indices
        self.normalize()

        self.fitness_function = fitness_function
        self.fit()

    def mutate(self):
        boundary_indices = np.logical_not(self.normalization_indices)
        self.chromosomes[boundary_indices] = np.random.permutation(self.chromosomes[boundary_indices])
        self.normalize()
        self.fit()
    

    def crossover(self, individual):
        alpha = np.random.uniform()
        beta = np.random.uniform()
        
        baby_1 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        baby_2 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)

        baby_1.chromosomes = alpha * self.chromosomes + individual.chromosomes * (1-alpha)
        baby_2.chromosomes = beta * self.chromosomes + individual.chromosomes * (1-beta)

        baby_1.normalize()
        baby_2.normalize()

        baby_1.fit()
        baby_2.fit()

        return baby_1, baby_2

    def fit(self):
        self.fitness_score = self.fitness_function(self.chromosomes)

    def copy(self):
        duplicate = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        duplicate.chromosomes = self.chromosomes.copy()
        duplicate.fitness_score = self.fitness_score
        return duplicate

    def normalize(self):
        # -- set to zero outside of boundary
        self.chromosomes[self.normalization_indices] = 0

        # -- constraint on density
        self.chromosomes=utils.project(self.chromosomes, self.beta, self.domain)


class Individual_vector_better():
    def __init__(self, shape, fitness_function, normalization_indices, beta, domain):
        self.shape = shape
        self.beta = int(beta)
        self.domain = domain
        self.fitness_function = fitness_function
        self.normalization_indices = normalization_indices
        #Le chromosome est un vecteur de taille shape nulle hors de la fronti??re :
        self.chromosomes = np.zeros(shape)
        #Sur la fronti??re, beta composantes al??atoires sont ?? 1 :
        boundary_indices = np.logical_not(self.normalization_indices)
        L = [1] * self.beta + [0] * (np.sum(boundary_indices) - self.beta) 
        rd.shuffle(L)
        self.chromosomes[boundary_indices] = L
        self.fit()
    
    def mutate(self):
        boundary_indices = np.logical_not(self.normalization_indices)
        n_boundary = np.sum(boundary_indices)
        n_permutation = int(n_boundary * rd.random() * 0.4)     #0 ?? 20 permutation (al??toire) va s'??changer pour un vecteur de taille 100
        indices = list(zip(*np.where(boundary_indices)))
        for _ in range(n_permutation):
            i, j = rd.sample(indices, k = 2)
            self.chromosomes[i],  self.chromosomes[j] = self.chromosomes[j], self.chromosomes[i]
        self.fit()

    def crossover(self, individual):
        #Les composantes ?? 1 ?? la fois chez self et individual sont transmises ?? 1. Pour les 1 restants ?? transmettre afin que les enfants poss??dent beta 1, on choisit au hasard entre
        #self et individual puis on transmet une nouvelle composante ?? 1.
        baby_1 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        baby_2 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        indices1 = set(zip(*np.where(self.chromosomes == 1)))          #Indices o?? self.chr vaut 1.
        indices2 = set(zip(*np.where(individual.chromosomes == 1)))    #Indices o?? individual.chr vaut 1.    
        indices12_both = indices1.intersection(indices2)               #self.chr et ind.chr valent tous deux 1.
        indices1_only = list(indices1 - indices2)                      #Seul self.chr vaut 1.
        indices2_only = list(indices2 - indices1)                      #Seul indiv.chr vaut 1.
        N = self.beta - len(indices12_both) - 1                            
        for baby in (baby_1, baby_2):
            baby.chromosomes = np.zeros(self.shape)
            for ind in indices12_both:
                baby.chromosomes[ind] = 1
            rd.shuffle(indices1_only)
            rd.shuffle(indices2_only)
            for k in range(N):
                if rd.random() > 0.5:
                    ind = indices1_only[k]
                else:
                    ind = indices2_only[k]
                baby.chromosomes[ind] = 1
            baby.fit()
        return baby_1, baby_2

    def fit(self):
        self.fitness_score = self.fitness_function(self.chromosomes)

    def copy(self):
        duplicate = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        duplicate.chromosomes = self.chromosomes.copy()
        duplicate.fitness_score = self.fitness_score
        return duplicate