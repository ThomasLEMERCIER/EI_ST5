import numpy as np
import _env
import preprocessing
import processing

def projector(domain, l,chi):
    indices = np.where(domain == _env.NODE_ROBIN)
    new_chi = np.copy(chi) 
    new_chi[indices] += l
    new_chi = np.maximum(0, np.minimum(1, new_chi))
    return new_chi

def compute_p(domain_omega, spacestep, wavenumber, Alpha, chi):

    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    alpha_rob = Alpha * chi

    p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return p

def compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi):
                        
    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    alpha_rob = Alpha * chi

    f_adjoint = - 2 * np.conjugate(p)

    q = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adjoint, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return q
