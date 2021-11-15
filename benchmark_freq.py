# -- Solver
from math import pi
from EI import algo_opti
import EI.pde.preprocessing
import EI.pde.processing
import EI.pde.postprocessing
import EI.alpha
import EI._env as _env

# -- Optimization algorithm
import EI.algo_opti.directGradientDescent

import EI.utils as utils
# -- external import
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    frequencies = np.linspace(100, 4000, 200)
    material = "SUTHERLAND"
    alpha_precision = 15

    # -- set geometry of domain
    domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

    # -- define material density matrix
    chi = EI.pde.preprocessing._set_chi(M, N, x, y)
    chi = EI.pde.preprocessing.set2zero(chi, domain_omega)
    # -- define material density matrix
    chi0 = np.random.uniform(size=chi.shape)
    normalization_indices = domain_omega != _env.NODE_ROBIN
    # -- set to zero outside of boundary
    chi0[normalization_indices] = 0
    # -- constraint on density
    chi0=utils.project(chi0, np.sum(chi), domain_omega)
    del chi

    K = 5

    energy = []

    plt.figure()
    ax = plt.subplot(111)
    for frequency in tqdm(frequencies):

        omega = 2 * pi / frequency
        wavenumber = omega / c0

        # -- define absorbing material
        Alpha = EI.alpha.compute_alpha(material, omega, alpha_precision)
        Alpha = Alpha[0] + Alpha[1] * 1j

        # -- compute optimization
        _, e, _ = EI.algo_opti.directGradientDescent.DirectGradientDescent_Adam(chi0, domain_omega, spacestep, wavenumber, Alpha, K)
        # --- en of optimization

        energy.append(e[-1])

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Energy")
    plt.plot(frequencies, energy)
    plt.title(f"Energy as a function of frequency")
    plt.show()

if __name__ == "__main__":
    main()