from ga import GA
from demo_control_polycopie import *
import postprocessing
import preprocessing
import processing
import individual_vector

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
    # -- define absorbing material
    Alpha = compute_alpha(material, omega, precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    print("Alpha: ", Alpha)
    
    fitness = lambda x: fitness_function(x, domain_omega, spacestep, wavenumber, Alpha)

    ga = GA(individual_vector.Individual_vector, fitness, jj)

if __name__ == "__main__":

    solve()