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


def proposed_algo(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Algorithme proposé par les slides d'É. Savin. On s'assure que l'énergie décroit avant de faire la descente de gradient. Pour cela on réduit le LR.
    Performance : 0.75
    """
    plt.figure()
    plt.ion()
    
    k = 0
    beta = numpy.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()

    mu = 5
    while k < K and mu > 10 ** -5:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, None, None)
        E_next = E
        energy.append(E)

        print('3.5. computing gradient clipped')
        grad_J = diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        while E_next >= E and mu > 10 ** -5:
        #Tant que l'énergie ne s'améliore pas, et que l'on a pas atteint un minimum, on fait une descente de gradient avec un lr plus petit.
        #On passe à l'itération suivante si l'énergie baisse.
            print('4. gradient descent trial')
            chi_candidate = chi-mu*grad_J
            l = dicho_l(chi_candidate, beta, - numpy.max(chi_candidate), 1 - numpy.min(chi_candidate), domain_omega, precision=1e-3)
            chi_next=projector(domain_omega, l, chi_candidate)            #Descente de gradient sous contraintes "X[k] in [0, 1]"
               
            p_next=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)  #Calcul du p possiblement meilleur. 
            E_next=J(domain_omega, p_next, spacestep, None, None)                     #Calcul de l'E possiblement plus faible.
            if E_next < E:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            
            energy.append(E_next)
            plot_energy(energy)
            print(mu)
        chi = chi_next        
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p


def proposed_algo_2(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Identique à l'algo proposé mais en réinitialisant le LR à sa valeur initiale après chaque gradient step.
    Plus lent, mais converge mieux.
    Performance : 0.7
    """
    plt.figure()
    plt.ion()
    
    k = 0
    beta = numpy.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    energy = list()

    while k < K:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, None, None)
        E_next = E
        energy.append(E)

        print('3.5. computing gradient clipped')
        grad_J = diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        mu = 100
        while E_next >= E and mu > 10 ** -5:
        #Tant que l'énergie ne s'améliore pas, et que l'on a pas atteint un minimum, on fait une descente de gradient avec un lr plus petit.
        #On passe à l'itération suivante si l'énergie baisse.
            print('4. gradient descent trial')
            chi_candidate = chi-mu*grad_J
            l = dicho_l(chi_candidate, beta, - numpy.max(chi_candidate), 1 - numpy.min(chi_candidate), domain_omega, precision=1e-3)
            chi_next=projector(domain_omega, l, chi_candidate)             #Descente de gradient sous contraintes "X[k] in [0, 1]"
               
            p_next=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)  #Calcul du p possiblement meilleur. 
            E_next=J(domain_omega, p_next, spacestep, None, None)                     #Calcul de l'E possiblement plus faible.
            if E_next < E:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            
            energy.append(E_next)
            plot_energy(energy)

        chi = chi_next        
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p


