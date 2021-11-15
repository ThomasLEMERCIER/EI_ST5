# -- Solver
from EI import algo_opti
import EI.pde.preprocessing
import EI.pde.processing
import EI.pde.postprocessing
import EI.alpha

# -- Optimization algorithm
import EI.algo_opti.GradientDescent
import EI.algo_opti.softGD
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
    level = 1 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    c0 = 340
    # -- set parameters of the partial differential equation
    alpha_precision = 15

    # -- set geometry of domain
    domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

    # -- define material density matrix
    chi0 = EI.pde.preprocessing._set_chi(M, N, x, y)
    chi0 = EI.pde.preprocessing.set2zero(chi0, domain_omega)
    # -- define absorbing material
    materials = ["BIRCHLT", "SUTHERLAND", "POLYURETHANE", "MELAMINE", "BIRCHLT"]
    frequencies = np.linspace(4000, 4500, 50)

    K = 5 
    # algo_opti.directGradientDescent.DirectGradientDescent_Adam

    plt.figure()
    ax = plt.subplot(111)
    for material in tqdm(materials):

        energy = []

        for frequency in (frequencies):

            omega = 2 * pi / frequency
            wavenumber = omega / c0

            Alpha = EI.alpha.compute_alpha(material, omega, alpha_precision)
            Alpha = Alpha[0] + Alpha[1] * 1j

            # -- compute optimization
            chi, _, _ = EI.algo_opti.GradientDescent.evolutive_lr_ProjectedGradientDescent(chi0, domain_omega, spacestep, wavenumber, Alpha, K)
            # --- en of optimization

            chi = utils.project_to_admissible_set(chi)
            energy.append(utils.energy(chi, domain_omega, spacestep, wavenumber, Alpha))

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Energy")
        plt.plot(frequencies, energy, label=material)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Energy")
    plt.title(f"Energy as a function of the frequency for different materials")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()