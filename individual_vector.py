import numpy as np
import _env

PRECISION_BETA = 1e-1

def projector(domain, l,chi):
    indices = np.where(domain == _env.NODE_ROBIN)
    new_chi = np.copy(chi) 
    new_chi[indices] += l
    new_chi = np.maximum(0, np.minimum(1, new_chi))
    return new_chi

def dicho_l(x, beta, lmin, lmax, domain):
    lmid = (lmax + lmin) / 2
    x_new = projector(domain, lmid, x)
    beta_current = np.sum(x_new)
    #print("Beta current: ", beta_current, "for: ", lmin, lmax, lmid)
    if abs(beta_current - beta) <= PRECISION_BETA:
        return lmid
    if beta_current >= beta:
        return dicho_l(x, beta, lmin, lmid, domain) 
    else:
        return dicho_l(x, beta, lmid, lmax, domain)

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
        self.chromosomes = 1 - self.chromosomes
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
        l = dicho_l(self.chromosomes, self.beta, -1, 1, self.domain)
        self.chromosomes=projector(self.domain, l, self.chromosomes)
