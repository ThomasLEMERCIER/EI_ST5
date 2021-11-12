# -*- coding: utf-8 -*-

# Python packages
import matplotlib.pyplot
import numpy
import os
import math

plt = matplotlib.pyplot

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
from utils import *
from algo_optimisation import *
#import solutions
from alpha import compute_alpha


def SGD(chi, domain_omega, spacestep, wavenumber, Alpha, K):

    """
    Descente de gradient classique. On ne cherche pas à absolument à baisser E.
    Performance : mauvaise (E augmente)
    """
    plt.figure()
    plt.ion()
    
    k = 0
    beta = numpy.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()
    
    mu = 0.1
    K = 100
    while k < K:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, None, None)

        print('3.5. computing gradient clipped')
        grad_J = diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        print('4. gradient descent')
        chi = chi - mu * grad_J
        l = dicho_l(chi, beta, - numpy.max(chi), 1 - numpy.min(chi), domain_omega, precision=1e-3)
        chi=projector(domain_omega, l, chi)             #Descente de gradient sous contraintes "X[k] in [0, 1]"

        energy.append(E) 
        plot_energy(energy)

        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p


def SGD_Adam(chi, domain_omega, spacestep, wavenumber, Alpha, K):

    """
    Descente de gradient classique. On ne cherche pas à absolument à baisser E.
    Performance : mauvaise (E augmente)
    """
    plt.figure()
    plt.ion()
    
    k = 0
    beta = numpy.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()
    
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))
    
    while k < K:
        k += 1
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, None, None)

        print('3.5. computing gradient clipped')
        grad_J = diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        m = beta1 * m + (1-beta1) * grad_J
        v = beta2 * v + (1-beta2) * grad_J * grad_J
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (numpy.sqrt(v_bias_corrected) + eps)

        print('4. gradient descent')
        l = dicho_l(chi, beta, - numpy.max(chi), 1 - numpy.min(chi), domain_omega, precision=1e-3)
        chi = projector(domain_omega, l, chi)            #Descente de gradient sous contraintes "X[k] in [0, 1]"

        energy.append(E) 
        plot_energy(energy)

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p




