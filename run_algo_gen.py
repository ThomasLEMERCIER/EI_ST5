from ga import GA
import _env
from demo_control_polycopie import *
import postprocessing
import preprocessing
import processing
import individual_vector
import numpy as np
import matplotlib.pyplot as plt

def fitness_function(chi, domain_omega, spacestep, wavenumber, Alpha):
    p = compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    return J(domain_omega, p, spacestep, None, None)

def solve():


    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 10  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    c0 = 340
    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0
    material = "MELAMINE"
    omega = wavenumber * c0 
    precision = 15

    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    beta = np.sum(chi)
    # -- define absorbing material
    Alpha = compute_alpha(material, omega, precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    print("Alpha: ", Alpha)
    print("Beta: ", beta) 
    fitness = lambda x: -fitness_function(x, domain_omega, spacestep, wavenumber, Alpha)
    energy_function = lambda x: fitness_function(x, domain_omega, spacestep, wavenumber, Alpha)
    normalization_indices = np.where(domain_omega != _env.NODE_ROBIN)

    ga = GA(individual_vector.Individual_vector, chi.shape, normalization_indices, beta, domain_omega, fitness, 100, 0, "tournament", 0.2, 0.8)

    iterations = [0]
    energy = [energy_function(ga.best_indv.chromosomes)]

    for e in range(10):

        print("Epoch: ", e)
        ga.step()
        iterations.append(iterations[-1]+1)
        energy.append(ga.best_indv.fitness_score)

    plt.plot(iterations, energy)
    plt.show()




if __name__ == "__main__":

    solve()