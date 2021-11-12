# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy
import os
import math

plt = matplotlib.pyplot
np = numpy

import EI
import EI.alpha
import EI.pde.postprocessing
 
# -- Optimization algorithm
import EI.algo_opti.directGradientDescent
import EI.algo_opti.GradientDescent
import EI.algo_opti.softGD
algo_opti = EI.algo_opti

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Define the algo of optimization
    ALGO = EI.algo_opti.softGD.soft_evolutive_lr_ProjectedGradientDescent_Adam
    K = 10
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # -- Define the structure of the problem
    # ----------------------------------------------------------------------
    N = 15  # number of points along x-axis
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

    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = EI.pde.preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = EI.pde.preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = EI.pde.preprocessing._set_geometry_of_domain(M, N, level)

    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- define material density matrix
    chi = EI.pde.preprocessing._set_chi(M, N, x, y)
    chi = EI.pde.preprocessing.set2zero(chi, domain_omega)
    # -- define absorbing material
    Alpha = EI.alpha.compute_alpha(material, omega, precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    alpha_rob = Alpha * chi

    # -- compute finite difference solution
    u = EI.pde.processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()
    

    # -- compute optimization
    chi, energy, u = ALGO(chi, domain_omega, spacestep, wavenumber, Alpha, K)
    # --- en of optimization

    chin = chi.copy()    
    un = u.copy()

    # -- plot chi, u, and energy
    EI.pde.postprocessing._plot_uncontroled_solution(u0, chi0)
    EI.pde.postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    EI.pde.postprocessing._plot_error(err)
    EI.pde.postprocessing._plot_energy_history(energy)
    print("Energie finale obtenue :", energy[-1])
    print('End.')
