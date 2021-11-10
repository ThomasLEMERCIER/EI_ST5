from algo_optimisation import *
from utils import *
import _env
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))
def diff_sigmoid(x):
    return math.exp(-x) * sigmoid(x) ** 2
def logit(y):
    math.log(y/(1-y))

def softmax(chi):
    exps = np.exp(chi)
    chi_bar = exps / np.sum(exps)
    return chi_bar

def Jacob_softmax(chi):
    exps = np.exp(chi)
    sum_exps = np.sum(np.exp(chi))
    J = np.diagflat(exps) / sum_exps
    A, B = J.shape
    for i in range(A):
        for j in range(B):
            J[i, j] -= math.exp(chi[i] + chi[j]) / (sum_exps * sum_exps)
    return J


def softGD(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Ici, on considère que chi_bar les paramètres donnés en entrée du calcul de p, q et E qui sont contraints d'etre entre 0 et beta et d'etre de somme beta, 
    sont fonctions d'autres paramètres plus libres chi : chi_bar = beta * softmax(chi).
    On fait alors une descente de gradient : chi = chi - alpha * beta * grad(E)(softmax(chi)) * Jacobian(softmax)(chi)    , règle de la chaine. 
    """
    plt.figure()
    plt.ion()

    #Vectorisation
    
    
    beta = np.sum(chi)
    (M, N) = np.shape(domain_omega)
    k = 0
    energy = list()

    #Learning rate
    alpha = 0.1
    K = 200
    
    while k < K:
        print('---- iteration number = ', k)
        k += 1

        #Softmax
        chi_bar = softmax(chi)

        #Calcul de grad_chi(E)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, beta * chi_bar)
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, beta * chi_bar)
        E=J(domain_omega, p, spacestep, None, None)
        energy.append(E)
        plot_energy(energy)
        grad_chiBar_E = diff_J(p,q,Alpha)      
        grad_chiBar_E = grad_shifted(grad_chiBar_E, domain_omega)                
        grad_chiBar_E = np.matrix.flatten(grad_chiBar_E)

        #Calcul de la jacobienne de softmax en chi
        chi_vector = np.matrix.flatten(chi)
        Jacobian_chi_Softmax = Jacob_softmax(chi_vector)

        #Calcul de grad_chi(E)
        grad_chi_E = beta * grad_chiBar_E @ Jacobian_chi_Softmax
        grad_chi_E = grad_chi_E.reshape((M,N))
        chi = chi - alpha * grad_chi_E
   
        energy.append(E)
        plot_energy(energy)

    print('end. computing solution of Helmholtz problem')
    print("Performance :", energy[-1])
    return chi, energy, p