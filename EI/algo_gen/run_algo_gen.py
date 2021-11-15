from EI.algo_gen.ga import GA
import EI._env as _env
import EI.utils as utils
import EI.pde.postprocessing
import EI.pde.preprocessing
import EI.pde.processing
pde = EI.pde
import EI.alpha as alpha
import EI.algo_gen.individual_vector as individual_vector

# -- External import 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    fitness = lambda x: -utils.energy(x, domain_omega, spacestep, wavenumber, Alpha)
    energy_function = lambda x: utils.energy(x, domain_omega, spacestep, wavenumber, Alpha)
    normalization_indices = (domain_omega != _env.NODE_ROBIN)


    ga = GA(individual_vector.Individual_vector, chi.shape, normalization_indices, beta, domain_omega, fitness, 50, 0.15, "tournament", 0.2, 0.8)

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

def run_algo_gen(chi0, domain_omega, spacestep, wavenumber, Alpha, K):

    chi = np.copy(chi0)

    beta = np.sum(chi)
    normalization_indices = domain_omega != _env.NODE_ROBIN

    fitness_function = lambda x: -utils.energy(x, domain_omega, spacestep, wavenumber, Alpha)

    ga = GA(individual_vector.Individual_vector, chi.shape, normalization_indices, beta, domain_omega, fitness_function, 50, 0.15, "tournament", 0.2, 0.8)

    energy = []

    for e in range(K):
        print(f"--- Iteration: {e+1} ---")
        energy.append(utils.energy(ga.best_indv.chromosomes, domain_omega, spacestep, wavenumber, Alpha))
        ga.step()

    chi = ga.best_indv.chromosomes
    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p

def test_algo_gen():

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
    fitness = lambda x: -utils.energy(x, domain_omega, spacestep, wavenumber, Alpha)
    energy_function = lambda x: utils.energy(x, domain_omega, spacestep, wavenumber, Alpha)
    normalization_indices = (domain_omega != _env.NODE_ROBIN)


    ga = GA(individual_vector.Individual_vector, chi.shape, normalization_indices, beta, domain_omega, fitness, 50, 0.15, "tournament", 0.2, 0.8)

    u0 = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, ga.best_indv.chromosomes)
    chi0 = ga.best_indv.chromosomes


    for indv in ga.population:

        print(np.sum(indv.chromosomes) - beta)
        #print(indv.chromosomes[normalization_indices] == 0)

    for e in range(1):
        ga.step()
        
    for indv in ga.population:

        print(np.sum(indv.chromosomes) - beta)
        #print(indv.chromosomes[normalization_indices] == 0)

    print("Test done!")


    
if __name__ == "__main__":

    """
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
    """