import numpy as np
import EI.utils
utils = EI.utils

def ProjectedGradientDescent(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=5):

    chi = np.copy(chi0)
    beta = np.sum(chi)

    energy = []

    for k in range(0, K):

        print(f"--- Iteration: {k+1} ---")

        p, q, e, grad_J = utils.compute_all(chi, domain_omega, spacestep, wavenumber, Alpha)

        energy.append(e)

        chi_next = chi-lr*grad_J
        chi_next = utils.project(chi_next, beta, domain_omega)
        

    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p

def evolutive_lr_ProjectedGradientDescent(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=5, lr_min=1e-8):

    chi = np.copy(chi0)
    beta = np.sum(chi)

    energy = []

    for k in range(0, K):
        print(f"--- Iteration: {k+1} ---")

        p, q, e, grad_J = utils.compute_all(chi, domain_omega, spacestep, wavenumber, Alpha)

        energy.append(e)

        e_next = e

        while e_next >= e and lr > lr_min:
            chi_next = chi-lr*grad_J
            chi_next = utils.project(chi_next, beta, domain_omega)
        
            e_next = utils.energy(chi_next, domain_omega, spacestep, wavenumber, Alpha)

            if e_next >= e:
                lr = lr/2
            else:
                lr = lr * 1.1

        chi=chi_next

    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p

def evolutive_lr_ProjectedGradientDescent_Adam(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=5, lr_min=1e-8):

    chi = np.copy(chi0)
    beta = np.sum(chi)

    energy = []


    beta1 = 0.9 
    beta2 = 0.999
    epsilon = 1e-8

    m = np.zeros_like(chi)
    v = np.zeros_like(chi)

    for k in range(0, K):
        print(f"--- Iteration: {k+1} ---")

        p, q, e, grad_J = utils.compute_all(chi, domain_omega, spacestep, wavenumber, Alpha)

        energy.append(e)

        e_next = e

        m = beta1 * m + (1 - beta1) * grad_J
        v = beta2 * v + (1 - beta2) * grad_J * grad_J

        m_norm = m / (1 - (beta1 ** (k+1)))
        v_norm = v / (1 - (beta2 ** (k+1)))

        corrected_grad_J = m_norm / (np.sqrt(v_norm) + epsilon)

        while e_next >= e and lr > lr_min:

            chi_next = chi-lr*corrected_grad_J
            chi_next = utils.project(chi_next, beta, domain_omega)
        
            e_next = utils.energy(chi_next, domain_omega, spacestep, wavenumber, Alpha)

            if e_next >= e:
                lr = lr/2
            else:
                lr = lr * 1.1
        


        chi=chi_next

    p = utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    energy.append(utils.J(domain_omega, p, spacestep))

    return chi, energy, p
