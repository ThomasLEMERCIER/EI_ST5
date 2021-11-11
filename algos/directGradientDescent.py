from algo_optimisation import *
from utils import *
import matplotlib.pyplot as plt
import numpy
import _env

def DirectGradientDescent(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Descent de gradient directe : on calcule le gradient avec la formule (f(chi+h) - f(chi)) / h , et non avec -Re(a*p*q)
    où f est la fonction de calcul de l'énergie précédée de la fonction de respect de contrainte pour chi.
    Performance : 0.60
    """
    plt.figure()
    plt.ion()
    
    k = 0
    beta = np.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()

    alpha = 1
    h = 0.01

    def Energy(chi):
        l = dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi=projector(domain_omega, l, chi)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return J(domain_omega, p, spacestep, None, None)

    while k < K:
        print('---- iteration number = ', k)
        k += 1

        currentEnergy = Energy(chi)
        energy.append(currentEnergy)
        plot_energy(energy)

        grad_Energy = numpy.zeros((M, N))
        for i, j in zip(*np.where(domain_omega == _env.NODE_ROBIN)):
            H = numpy.zeros((M, N))
            H[i, j] = h
            grad_Energy[i, j] = (Energy(chi + H) - currentEnergy) / h
        chi = chi - alpha * grad_Energy

    print('end. computing solution of Helmholtz problem')
    l = dicho_l(chi, beta, -1, 1, domain_omega)
    chi=projector(domain_omega, l, chi)
    p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
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
    (M, N) = numpy.shape(domain_omega)
    energy = list()

    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    h = 0.01

    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))

    def Energy(chi):
        l = dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi=projector(domain_omega, l, chi)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return J(domain_omega, p, spacestep, None, None)

    while k < K:
        print('---- iteration number = ', k)
        k += 1
        
        energy.append(Energy(chi))
        plot_energy(energy)

        grad_Energy = numpy.zeros((M, N))
        for i, j in zip(*np.where(domain_omega == _env.NODE_ROBIN)):
            H = numpy.zeros((M, N))
            H[i, j] = h
            grad_Energy[i, j] = (Energy(chi + H) - Energy(chi)) / h
        
        m = beta1 * m + (1-beta1) * grad_Energy
        v = beta2 * v + (1-beta2) * grad_Energy * grad_Energy
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (numpy.sqrt(v_bias_corrected) + eps)

    print('end. computing solution of Helmholtz problem')
    p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
    return chi, energy, p