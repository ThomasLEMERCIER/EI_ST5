# -- Solver
from EI import algo_opti
import EI.pde.preprocessing
import EI.pde.processing
import EI.pde.postprocessing
import EI.alpha

# -- Optimization algorithm
import EI.algo_opti.directGradientDescent
import EI.algo_opti.GradientDescent
import EI.algo_opti.softGD
algo_opti = EI.algo_opti

# -- Outside module
import numpy as np
import matplotlib.pyplot as plt

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
    kx = -1.0
    ky = -1.0
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    material = "MELAMINE"
    omega = wavenumber * c0 
    alpha_precision = 15

    # -- set geometry of domain
    domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

    # -- define material density matrix
    chi0 = EI.pde.preprocessing._set_chi(M, N, x, y)
    chi0 = EI.pde.preprocessing.set2zero(chi0, domain_omega)
    # -- define absorbing material
    Alpha = EI.alpha.compute_alpha(material, omega, alpha_precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    alpha_rob = Alpha * chi0


    K = 10 
    # algo_opti.directGradientDescent.DirectGradientDescent_Adam
    algos = [   algo_opti.GradientDescent.evolutive_lr_ProjectedGradientDescent,
                algo_opti.GradientDescent.ProjectedGradientDescent,
                algo_opti.GradientDescent.evolutive_lr_ProjectedGradientDescent_Adam,
                algo_opti.directGradientDescent.DirectGradientDescent,
                algo_opti.directGradientDescent.DirectGradientDescent_Adam,
                algo_opti.softGD.soft_evolutive_lr_ProjectedGradientDescent,
                algo_opti.softGD.soft_evolutive_lr_ProjectedGradientDescent_Adam,
                algo_opti.softGD.soft_ProjectedGradientDescent,
    ]
    iterations = np.arange(K+1)
    plt.figure()
    for algo in algos:


        # -- compute optimization
        chi, energy, u = algo(chi0, domain_omega, spacestep, wavenumber, Alpha, K)
        # --- en of optimization

        plt.plot(iterations, energy, label=algo.__name__)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()