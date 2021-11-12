# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy
plt = matplotlib.pyplot
# MRG packages
import EI.utils
utils = EI.utils


def SGD(chi, domain_omega, spacestep, wavenumber, Alpha, K):

    """
    Descente de gradient classique. On ne cherche pas à absolument à baisser E.
    Performance : mauvaise (E augmente)
    """
    
    k = 0
    beta = numpy.sum(chi)
    energy = list()
    
    mu = 0.00001
    while k < K:
        print('---- iteration number = ', k)
        p=utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        q=utils.compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        E=utils.J(domain_omega, p, spacestep, None, None)

        grad_J = utils.diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = utils.shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        l = utils.dicho_l(chi, beta, - numpy.max(chi), 1 - numpy.min(chi), domain_omega, precision=1e-3)
        chi=utils.projector(domain_omega, l, chi-mu*grad_J)             #Descente de gradient sous contraintes "X[k] in [0, 1]"

        energy.append(E) 

    return chi, energy, p


def SGD_Adam(chi, domain_omega, spacestep, wavenumber, Alpha, K):

    """
    Descente de gradient classique. On ne cherche pas à absolument à baisser E.
    Performance : mauvaise (E augmente)
    """
    beta = numpy.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()
    
    alpha = 0.00001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))
    
    for k in range(K):
        p=utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        q=utils.compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        E=utils.J(domain_omega, p, spacestep, None, None)

        grad_J = utils.diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = utils.shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        m = beta1 * m + (1-beta1) * grad_J
        v = beta2 * v + (1-beta2) * grad_J * grad_J
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (numpy.sqrt(v_bias_corrected) + eps)

        l = utils.dicho_l(chi, beta, - numpy.max(chi), 1 - numpy.min(chi), domain_omega, precision=1e-3)
        chi = utils.projector(domain_omega, l, chi)            #Descente de gradient sous contraintes "X[k] in [0, 1]"

        energy.append(E) 

    return chi, energy, p




