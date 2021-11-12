import numpy
from numpy.lib.index_tricks import index_exp
np = numpy
import EI._env
_env = EI._env
import EI.pde.preprocessing
import EI.pde.processing
pde = EI.pde
import matplotlib.pyplot as plt

def clamp_to_boundary(domain, matrix):
    matrix[domain != _env.NODE_ROBIN] = 0
    return matrix

def projector(domain, l, x):
    indices = np.where(domain == _env.NODE_ROBIN)
    new_x = np.copy(x) 
    new_x[indices] += l
    new_x = np.maximum(0, np.minimum(1, new_x))
    return new_x

def project(x, beta, domain):
    l = dicho_l(x, beta, -np.max(x), 1-np.min(x), domain)
    x = projector(domain, l, x)
    return x

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def softmax_project(x, beta, domain):
    indices = np.where(domain == _env.NODE_ROBIN)
    x[indices] = beta * softmax(x[indices])
    return x

def compute_p(domain_omega, spacestep, wavenumber, Alpha, chi):

    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = pde.preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = pde.preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0

    alpha_rob = Alpha * chi

    p = pde.processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return p

def compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi):
                        
    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = pde.preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = pde.preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0

    alpha_rob = Alpha * chi

    f = - 2 * np.conjugate(p)

    q = pde.processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return q


def energy(chi, domain, spacestep, wavenumber, Alpha):

    p = compute_p(  domain_omega=domain,
                    spacestep=spacestep,
                    wavenumber=wavenumber,
                    Alpha=Alpha,
                    chi=chi)
    
    e = J(domain_omega=domain,
                    p=p,
                    spacestep=spacestep,
                    mu1=None,
                    V_0=None)

    return e

def J(domain_omega, p, spacestep, mu1=None, V_0=None):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)
    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """

    p_conj = np.conjugate(p)
    p_square_norm = np.real(p * p_conj)
    energy = np.sum(p_square_norm)

    return energy * spacestep * spacestep

def diff_J(p, q, alpha):
    return - np.real(alpha * p * q)

def print_on_boundary(matrix, domain_omega, msg=""):
    indices = np.where(domain_omega == _env.NODE_ROBIN)
    print(msg, matrix[indices])

def get_neighbors_values(matrix, index):
    (M, N) = matrix.shape
    x, y = index
    values = np.zeros(shape=(8), dtype=matrix.dtype)
    for i in range(-1 if x > 0 else 0, 2 if x < M else 1):
        for j in range(-1 if y > 0 else 0, 2 if y < N-1 else 1):
           values[i*3+j] = matrix[x + i, y+ j] 
    return values

def shift_on_boundary(matrix, domain_omega):

    indices = np.where(domain_omega == _env.NODE_ROBIN)

    new_matrix = np.zeros_like(matrix)

    for index in zip(*indices):
        neighbors_values = get_neighbors_values(matrix, index)
        new_matrix[index] = np.sum(neighbors_values) / max(1, np.sum(neighbors_values != 0))

    return new_matrix

def avg(L):
    return sum(L)/len(L)

def dicho_l(x, beta, lmin, lmax, domain, precision=1e-3):
    lmid = (lmax + lmin) / 2
    x_new = projector(domain, lmid, x)
    beta_current = np.sum(x_new)
    #print("Beta target: ", beta, "Beta current: ", beta_current, "for: ", lmin, lmax, lmid)
    if abs(beta_current - beta) <= precision:
        return lmid
    if beta_current >= beta:
        return dicho_l(x, beta, lmin, lmid, domain, precision) 
    else:
        return dicho_l(x, beta, lmid, lmax, domain, precision)

def compute_grad_J_euler(chi, beta, domain, spacestep, wavenumber, Alpha, h=1e-3):
    """
    grad_J = J(chi + h) - J(chi) / h
    
    """
    (M, N) = domain.shape

    grad_J = np.zeros((M, N))
    energy0 = energy(chi, domain, spacestep, wavenumber, Alpha)

    for i, j in zip(*np.where(domain == _env.NODE_ROBIN)):

        chih = np.copy(chi)
        chih[i, j] = chih[i, j] + h

        # -- project chih
        chih = project(chih, beta, domain)

        energyh = energy(chih, domain, spacestep, wavenumber, Alpha)

        grad_J[i, j] = (energyh - energy0) / h

    return grad_J

def compute_all(chi, domain, spacestep, wavenumber, Alpha):

    p = compute_p(  domain_omega=domain,
                    spacestep=spacestep,
                    wavenumber=wavenumber,
                    Alpha=Alpha,
                    chi=chi)
    
    q = compute_q(  p=p,
                    domain_omega=domain,
                    spacestep=spacestep,
                    wavenumber=wavenumber,
                    Alpha=Alpha,
                    chi=chi)

    e = J(  domain_omega=domain,
            p=p,
            spacestep=spacestep,
            mu1=None,
            V_0=None)

    grad_J = diff_J(p=p, q=q, alpha=Alpha)
    grad_J = shift_on_boundary(matrix=grad_J, domain_omega=domain)

    return p, q, e, grad_J