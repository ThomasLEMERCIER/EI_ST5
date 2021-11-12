import EI.utils
utils = EI.utils
import matplotlib.pyplot as plt
import numpy as np
import EI._env
_env = EI._env

def DirectGradientDescent(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=1):
    chi = np.copy(chi0)
    
    beta = np.sum(chi)
    energy = list()


    for k in range(K): 

        print(f"--- Iteration: {k+1} ---")

        energy.append(utils.energy(chi, domain_omega, spacestep, wavenumber, Alpha))

        grad_J = utils.compute_grad_J_euler(chi, beta, domain_omega, spacestep, wavenumber, Alpha)

        chi = chi - lr * grad_J
        chi = utils.project(chi, beta, domain_omega)

    
    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p


def DirectGradientDescent_Adam(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=1):
    """
    Descent de gradient directe avec l'optimizer Adam et la notion de momentum.
    Performance : 0.49
    """
    chi = np.copy(chi0)
    
    beta = np.sum(chi)
    energy = list()

    beta1 = 0.9 
    beta2 = 0.999
    epsilon = 1e-8

    m = np.zeros_like(chi)
    v = np.zeros_like(chi)

    for k in range(K): 

        print(f"--- Iteration: {k+1} ---")

        energy.append(utils.energy(chi, domain_omega, spacestep, wavenumber, Alpha))

        grad_J = utils.compute_grad_J_euler(chi, beta, domain_omega, spacestep, wavenumber, Alpha)

        m = beta1 * m + (1 - beta1) * grad_J
        v = beta2 * v + (1 - beta2) * grad_J * grad_J

        m_norm = m / (1 - (beta1 ** (k+1)))
        v_norm = v / (1 - (beta2 ** (k+1)))

        corrected_grad_J = m_norm / (np.sqrt(v_norm) + epsilon)

        chi = chi - lr * corrected_grad_J
        chi = utils.project(chi, beta, domain_omega)

    
    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p
