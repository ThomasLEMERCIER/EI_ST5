import EI.utils
import EI._env
_env = EI._env
utils = EI.utils
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))
def diff_sigmoid(x):
    return math.exp(-x) * sigmoid(x) ** 2
def logit(y):
    math.log(y/(1-y))

def softmax(chi, domain_omega):
    indices = domain_omega == _env.NODE_ROBIN
    chi_bar = np.copy(chi)
    chi_bar[np.logical_not(indices)] = 0
    exps = np.exp(chi_bar[indices])
    chi_bar[indices] = exps / np.sum(exps)
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
    Adam est implémenté.
    Performance : nulle (E augmente)
    """
    (M, N) = np.shape(domain_omega)
    k = 0
    energy = list()

    beta = np.sum(chi)

    #Hyper Params
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros((M, N))
    v = np.zeros((M, N))

    for k in range(K):
        print('---- iteration number = ', k)
        #Softmax
        chi_bar = beta * softmax(chi, domain_omega)

        #Calcul de grad_chi(E)
        p= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_bar)
        q= utils.compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi_bar)
        E= utils.J(domain_omega, p, spacestep, None, None)
        grad_chiBar_E = utils.diff_J(p,q,Alpha)      
        grad_chiBar_E = utils.grad_shifted(grad_chiBar_E, domain_omega)                
        grad_chiBar_E = np.matrix.flatten(grad_chiBar_E)

        #Calcul de la jacobienne de softmax en chi
        chi_vector = np.matrix.flatten(chi)
        Jacobian_chi_Softmax = Jacob_softmax(chi_vector)

        #Calcul de grad_chi(E)
        grad_chi_E = beta * np.transpose(Jacobian_chi_Softmax) @ grad_chiBar_E
        grad_chi_E = grad_chi_E.reshape((M,N))
        
        m = beta1 * m + (1-beta1) * grad_chi_E
        v = beta2 * v + (1-beta2) * grad_chi_E * grad_chi_E
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (np.sqrt(v_bias_corrected) + eps)
   
    return chi, energy, p



def softDirectGD(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    On évalue le gradient avec la méthode de calcul à la limite f(x+h) - f(x) / h
    Performance : 0.44
    """
    plt.figure()
    plt.ion()

    #Vectorisation
    beta = np.sum(chi)
    (M, N) = np.shape(domain_omega)
    k = 0
    energy = list()

    def E(chi_bar):
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_bar)
        return J(domain_omega, p, spacestep, None, None)

    #Hyper Params
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    h = 0.1

    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))

    K = 100
    while k < K:
        print('---- iteration number = ', k)
        k += 1

        #Softmax
        chi_bar = beta * softmax(chi, domain_omega)

        #Calcul de E(chi)
        currentEnergy = E(chi_bar)
        energy.append(currentEnergy)
        plot_energy(energy)

        #Calcul de grad_chiBar(E)
        grad_chiBar_E = numpy.zeros((M, N))
        for i, j in zip(*np.where(domain_omega == _env.NODE_ROBIN)):
            H = numpy.zeros((M, N))
            H[i, j] = h
            grad_chiBar_E[i, j] = (E(chi_bar + H) - currentEnergy) / h
        grad_chiBar_E = np.matrix.flatten(grad_chiBar_E)

        #Calcul de la jacobienne de softmax en chi
        chi_vector = np.matrix.flatten(chi)
        Jacobian_chi_Softmax = Jacob_softmax(chi_vector)

        #Calcul de grad_chi(E)
        grad_chi_E = beta * np.transpose(Jacobian_chi_Softmax) @ grad_chiBar_E
        grad_chi_E = grad_chi_E.reshape((M,N))
        
        m = beta1 * m + (1-beta1) * grad_chi_E
        v = beta2 * v + (1-beta2) * grad_chi_E * grad_chi_E
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (numpy.sqrt(v_bias_corrected) + eps)
        

    print('end. computing solution of Helmholtz problem')
    print("Performance :", energy[-1])
    p = compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_bar)
    return chi_bar, energy, p