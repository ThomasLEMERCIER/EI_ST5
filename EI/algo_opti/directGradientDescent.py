import EI.utils
utils = EI.utils
import matplotlib.pyplot as plt
import numpy as np
import EI._env
_env = EI._env

def DirectGradientDescent(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Descent de gradient directe : on calcule le gradient avec la formule (f(chi+h) - f(chi)) / h , et non avec -Re(a*p*q)
    où f est la fonction de calcul de l'énergie précédée de la fonction de respect de contrainte pour chi.
    Performance : 0.60
    """
    
    k = 0
    beta = np.sum(chi)
    (M, N) = np.shape(domain_omega)
    energy = list()

    alpha = 1
    h = 0.01

    def Energy(chi):
        l = utils.dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi= utils.projector(domain_omega, l, chi)
        p= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return utils.J(domain_omega, p, spacestep, None, None)

    while k < K:
        print('---- iteration number = ', k)
        k += 1

        currentEnergy = Energy(chi)
        energy.append(currentEnergy)

        grad_Energy = np.zeros((M, N))
        for i, j in zip(*np.where(domain_omega == _env.NODE_ROBIN)):
            H = np.zeros((M, N))
            H[i, j] = h
            grad_Energy[i, j] = (Energy(chi + H) - currentEnergy) / h
        chi = chi - alpha * grad_Energy

    print('end. computing solution of Helmholtz problem')
    l = utils.dicho_l(chi, beta, -1, 1, domain_omega)
    chi= utils.projector(domain_omega, l, chi)
    p= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    return chi, energy, p


def DirectGradientDescent_Adam(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Descent de gradient directe avec l'optimizer Adam et la notion de momentum.
    Performance : 0.49
    """
    plt.figure()
    plt.ion()

    k = 0
    beta = np.sum(chi)
    (M, N) = np.shape(domain_omega)
    energy = list()

    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    h = 0.01

    m = np.zeros((M, N))
    v = np.zeros((M, N))

    def Energy(chi):
        l = utils.dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi= utils.projector(domain_omega, l, chi)
        p= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return utils.J(domain_omega, p, spacestep, None, None)

    for k in range(K):
        print('---- iteration number = ', k)
        
        energy.append(Energy(chi))

        grad_Energy = np.zeros((M, N))
        for i, j in zip(*np.where(domain_omega == _env.NODE_ROBIN)):
            H = np.zeros((M, N))
            H[i, j] = h
            grad_Energy[i, j] = (Energy(chi + H) - Energy(chi)) / h
        
        m = beta1 * m + (1-beta1) * grad_Energy
        v = beta2 * v + (1-beta2) * grad_Energy * grad_Energy
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (np.sqrt(v_bias_corrected) + eps)

    p= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    return chi, energy, p