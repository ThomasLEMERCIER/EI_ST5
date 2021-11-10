import numpy as np
import _env
import preprocessing
import processing
import matplotlib.pyplot as plt

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

def grad_shifted(grad,domain_omega):
    (M, N) = np.shape(domain_omega)

    indices_x,indices_y = np.where(domain_omega == _env.NODE_ROBIN)
    for i in range(len(indices_x)):
        x,y=indices_x[i],indices_y[i]
        if x>0:
            bas=grad[x-1,y]
        else:
            bas=0
        if x<M-1:
            haut=grad[x+1,y]
        else:
            haut=0
        if y>0:
            gauche=grad[x,y-1]
        else:
            gauche=0
        if y<N-1:
            droite=grad[x,y+1]
        else:
            droite=0
        grad[x,y]=(bas+haut+droite+gauche)/max(1,(4-[bas,haut,gauche,droite].count(0)))
    grad=preprocessing.set2zero(grad,domain_omega)
    return grad

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
    p_norm = np.real(p * p_conj)
    energy = np.sum(p_norm) * spacestep * spacestep

    return energy

def diff_J(p, q, alpha, domain_omega):
    return diff_J_shifted(p, q, alpha, domain_omega)
    return - numpy.real(alpha * p * q)

def diff_J_shifted(p, q, alpha, domain_omega):
    #Cas où la frontière est linéaire : domain_omega[N, 0:N] = _env.NODE_ROBIN
    M, N = domain_omega.shape
    p_shifted = p.copy()
    p_shifted[N, :] = p[N-1, :]
    q_shifted = q.copy()
    q_shifted[N, :] = q[N-1, :]
    # extract_on_boundary(p_shifted, domain_omega)
    # extract_on_boundary(q_shifted, domain_omega)
    # extract_on_boundary(p*q, domain_omega)
    return - np.real(alpha * p_shifted * q_shifted)

def extract_on_boundary(matrix, domain_omega):
    indices = np.where(domain_omega == _env.NODE_ROBIN)
    print(indices)

def plot_energy(Ene):
    plt.clf()
    plt.plot(Ene)
    plt.pause(1e-3)
