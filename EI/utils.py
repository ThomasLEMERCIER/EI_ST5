import numpy
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

def J(domain_omega, p, spacestep, mu1, V_0):
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
    energy = np.sum(p_square_norm) * spacestep * spacestep

    return energy * spacestep * spacestep

def diff_J(p, q, alpha):
    return - np.real(alpha * p * q)

def print_on_boundary(matrix, domain_omega, msg=""):
    indices = np.where(domain_omega == _env.NODE_ROBIN)
    print(msg, matrix[indices])

def plot_energy(Ene):
    plt.clf()
    plt.plot(Ene)
    plt.pause(1e-2)

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

def compute_l(matrix, beta, domain, precision=1e-1, l_step=1e-3):
    l = 0
    matrix = projector(domain, l, matrix)
    beta_current = np.sum(matrix)
    while abs(beta_current - beta) >= precision:
        if beta_current >= beta:
            l -= l_step
        else:
            l += l_step
        matrix = projector(domain, l, matrix)
        beta_current = np.sum(matrix)
    return l

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