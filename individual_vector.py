import enum
import numpy as np
from numpy.linalg import norm
import _env
import random as rd

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
        #self.chromosomes = 1 - self.chromosomes
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
        l = dicho_l(self.chromosomes, self.beta, -1, 1, self.domain)
        self.chromosomes=projector(self.domain, l, self.chromosomes)



class Individual_vector_better():
    def __init__(self, shape, fitness_function, normalization_indices, beta, domain):
        self.shape = shape
        self.beta = int(beta)
        self.domain = domain
        self.fitness_function = fitness_function
        self.normalization_indices = normalization_indices
        #Le chromosome est un vecteur de taille shape nulle hors de la frontière :
        self.chromosomes = np.zeros(shape)
        #Sur la frontière, beta composantes aléatoires sont à 1 :
        boundary_indices = np.logical_not(self.normalization_indices)
        L = [1] * self.beta + [0] * (np.sum(boundary_indices) - self.beta) 
        rd.shuffle(L)
        self.chromosomes[boundary_indices] = L
        self.fit()
    
    def mutate(self):
        boundary_indices = np.logical_not(self.normalization_indices)
        n_boundary = np.sum(boundary_indices)
        n_permutation = int(n_boundary * rd.random() * 0.4)     #0 à 20 permutation (alétoire) va s'échanger pour un vecteur de taille 100
        indices = list(zip(*np.where(boundary_indices)))
        for _ in range(n_permutation):
            i, j = rd.sample(indices, k = 2)
            self.chromosomes[i],  self.chromosomes[j] = self.chromosomes[j], self.chromosomes[i]
        self.fit()

    # def mutate(self):
    #     boundary_indices = np.logical_not(self.normalization_indices)

    #     indices = list(zip(*np.where(boundary_indices)))
    #     n_boundary = len(indices)
    #     n_1_to_move = rd.randint(0, self.beta - 1) #Numéro du 1 à bouger parmi les beta 1 que possède self.
    #     k = 0
    #     for i in range(len(indices)):
    #         indice = indices[i]
    #         if self.chromosomes[indice] == 1:
    #             k += 1
    #             if k == n_1_to_move:
    #                 indice_next = indices[(i+1) % n_boundary]
    #                 indice_previous = indices[(i-1) % n_boundary]
    #                 if rd.random() > 0.5:
    #                     self.chromosomes[indice_previous],  self.chromosomes[indice] = self.chromosomes[indice], self.chromosomes[indice_previous]
    #                 else:
    #                     self.chromosomes[indice_next],  self.chromosomes[indice] = self.chromosomes[indice], self.chromosomes[indice_next]
    #                 break
    #     self.fit()

    def crossover(self, individual):
        #Les composantes à 1 à la fois chez self et individual sont transmises à 1. Pour les 1 restants à transmettre afin que les enfants possèdent beta 1, on choisit au hasard entre
        #self et individual puis on transmet une nouvelle composante à 1.
        baby_1 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        baby_2 = Individual_vector( shape=self.shape, fitness_function=self.fitness_function, normalization_indices=self.normalization_indices, beta=self.beta, domain=self.domain)
        boundary_indices = np.logical_not(self.normalization_indices)
        indices1 = set(zip(*np.where(self.chromosomes == 1)))          #Indices où self.chr vaut 1.
        indices2 = set(zip(*np.where(individual.chromosomes == 1)))    #Indices où individual.chr vaut 1.    
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