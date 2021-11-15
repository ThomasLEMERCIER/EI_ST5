# -- Solver
from EI import algo_opti
import EI.pde.preprocessing
import EI.pde.processing
import EI.pde.postprocessing
import EI.alpha

# -- Optimization algorithm
import EI.algo_opti.directGradientDescent
algo_opti = EI.algo_opti

import EI.utils as utils
# -- external import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi

def main():

    # ----------------------------------------------------------------------
    # -- Define the structure of the problem
    # ----------------------------------------------------------------------
    N = 20  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    levels = [0, 1, 2] # level of the fractal
    spacestep = 1.0 / N  # mesh size
    c0 = 340
    # -- set parameters of the partial differential equation
    alpha_precision = 15

    frequency = 4000
    omega = 2 * pi / frequency
    wavenumber = omega / c0

    material = "SUTHERLAND" 
    Alpha = EI.alpha.compute_alpha(material, omega, alpha_precision)
    Alpha = Alpha[0] + Alpha[1] * 1j

    K = 5 

    plt.figure()
    ax = plt.subplot(111)

    energy = []

    for level in tqdm(levels):

        # -- set geometry of domain
        domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

        # -- define material density matrix
        chi0 = EI.pde.preprocessing._set_chi(M, N, x, y)
        chi0 = EI.pde.preprocessing.set2zero(chi0, domain_omega)

        # -- compute optimization
        chi, e, _ = EI.algo_opti.directGradientDescent.DirectGradientDescent(chi0, domain_omega, spacestep, wavenumber, Alpha, K)
        # --- en of optimization

        energy.append(e[-1])

    ax.set_xlabel("Level of the fractal")
    ax.set_ylabel("Energy")
    plt.plot(levels, energy)
    plt.title(f"Energy as a function of the fractal level")
    plt.show()

if __name__ == "__main__":
    main()