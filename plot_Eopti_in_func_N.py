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
    level = 1 # level of the fractal
    c0 = 340
    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
    material = "MELAMINE"
    omega = wavenumber * c0 
    alpha_precision = 15

    # -- define absorbing material
    Alpha = EI.alpha.compute_alpha(material, omega, alpha_precision)
    Alpha = Alpha[0] + Alpha[1] * 1j


    K = 3
    # algo_opti.directGradientDescent.DirectGradientDescent_Adam
    algos = [   algo_opti.GradientDescent.evolutive_lr_ProjectedGradientDescent,
                algo_opti.GradientDescent.ProjectedGradientDescent,
                algo_opti.softGD.soft_evolutive_lr_ProjectedGradientDescent,
                algo_opti.softGD.soft_evolutive_lr_ProjectedGradientDescent_Adam,
                algo_opti.softGD.soft_ProjectedGradientDescent,
                algo_opti.GradientDescent.evolutive_lr_ProjectedGradientDescent_Adam,
                algo_opti.directGradientDescent.DirectGradientDescent,
                algo_opti.directGradientDescent.DirectGradientDescent_Adam,
                
    ]

    L_N = np.linspace(10, 100, 5).tolist()
    plt.figure()
    for algo in algos:
        print("Optimizing energy for various N for algo =", algo.__name__)
        energy_opti = list()
        for N in L_N:
            # -- set geometry of domain
            N = int(N)
            print("N = ", N)
            M = 2*N
            spacestep = 1.0 / N  # mesh size

            domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

            # -- define material density matrix
            chi0 = EI.pde.preprocessing._set_chi(M, N, x, y)
            chi0 = EI.pde.preprocessing.set2zero(chi0, domain_omega)

            # -- compute optimization
            chi, energy, u = algo(chi0, domain_omega, spacestep, wavenumber, Alpha, K)
            # --- en of optimization

            energy_opti.append(energy[-1])
        plt.plot(L_N, energy_opti, label=algo.__name__)
        plt.legend()
        plt.savefig("Eopti_in_func_N")

if __name__ == "__main__":
    main()