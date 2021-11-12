from ga import GA
import _env
import utils
import pde.postprocessing
import pde.preprocessing
import pde.processing
import alpha
import individual_vector
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fitness_function(chi, domain_omega, spacestep, wavenumber, Alpha):
    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy = utils.J(domain_omega, p, spacestep, None, None)
    return energy 

def solve():


    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 20  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    c0 = 340
    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0
    material = "MELAMINE"
    omega = wavenumber * c0 
    precision = 15

    domain_omega, x, y, _, _ = pde.preprocessing._set_geometry_of_domain(M, N, level)

    # -- define material density matrix
    chi = pde.preprocessing._set_chi(M, N, x, y)
    chi = pde.preprocessing.set2zero(chi, domain_omega)
    beta = np.sum(chi)
    # -- define absorbing material
    Alpha = alpha.compute_alpha(material, omega, precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    print("Alpha: ", Alpha)
    print("Beta: ", beta) 
    fitness = lambda x: -fitness_function(x, domain_omega, spacestep, wavenumber, Alpha)
    energy_function = lambda x: fitness_function(x, domain_omega, spacestep, wavenumber, Alpha)
    normalization_indices = (domain_omega != _env.NODE_ROBIN)


    ga = GA(individual_vector.Individual_vector_better, chi.shape, normalization_indices, beta, domain_omega, fitness, 50, 0.15, "tournament", 0.2, 0.8)

    iterations = [0]
    energy = [energy_function(ga.best_indv.chromosomes)]
    energy_mean = energy.copy()
    u0 = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, ga.best_indv.chromosomes)
    chi0 = ga.best_indv.chromosomes
    plt.figure()
    plt.ion()

    for e in tqdm(range(25)):
        iterations.append(iterations[-1]+1)
        energy.append(energy_function(ga.best_indv.chromosomes))
        energy_mean.append(utils.avg([energy_function(indiv.chromosomes) for indiv in ga.population]))
        plt.clf()
        plt.plot(iterations, energy, label = "lowest energy")
        plt.plot(iterations, energy_mean, label = "mean energy")
        plt.legend()
        plt.pause(1e-3)
        ga.step()
        

    plt.plot(iterations, energy)
    plt.show()


    un = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, ga.best_indv.chromosomes)
    chin = ga.best_indv.chromosomes
    pde.postprocessing._plot_uncontroled_solution(u0, chi0)
    pde.postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    pde.postprocessing._plot_error(err)
    pde.postprocessing._plot_energy_history(energy)


if __name__ == "__main__":

    import cProfile
    cProfile.run("solve()", "output.dat")

    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()