from algo_optimisation import *
from utils import *
import matplotlib.pyplot as plt
import numpy
import _env

def DirectGradientDescent(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    """
    Descent de gradient directe : on calcule le gradient avec la formule (f(chi+h) - f(chi)) / h , et non avec -Re(a*p*q)
    où f est la fonction de calcul de l'énergie précédée de la fonction de respect de contrainte pour chi.
    Performance : 0.6
    """
    plt.figure()
    plt.ion()

    k = 0
    beta = np.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = list()

    alpha = 1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    h = 0.2

    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))

    def Energy(chi):
        l = dicho_l(chi, beta, -1, 1, domain_omega)
        chi=projector(domain_omega, l, chi)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return J(domain_omega, p, spacestep, mu1, V_0)

    while k < numb_iter:
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
    return chi, energy, p, grad_Energy


def DirectGradientDescent_Adam(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    """
    Descent de gradient directe avec l'optimizer Adam et la notion de momentum.
    Performance : 0.64
    """
    plt.figure()
    plt.ion()

    k = 0
    beta = np.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100000
    energy = list()

    alpha = 1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    h = 0.2

    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))

    def Energy(chi):
        l = dicho_l(chi, beta, -1, 1, domain_omega)
        chi=projector(domain_omega, l, chi)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        return J(domain_omega, p, spacestep, mu1, V_0)

    while k < numb_iter:
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
    return chi, energy, p, grad_Energy