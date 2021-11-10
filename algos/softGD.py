import matplotlib.pyplot as plt
import numpy
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))
def diff_sigmoid(x):
    return math.exp(-x) * sigmoid(x) ** 2
def logit(y):
    math.log(y/(1-y))
def softmax(X):
    norm = numpy.sum(numpy.exp(X), axis = None)
    print(norm)
    return numpy.exp(X) / norm
def grad_softmax(X):
    
    return

def softGD(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Descente de gradient vis à vis des paramètres theta, où chi = sigmoid(theta).
    """
    plt.figure()
    plt.ion()

    beta = np.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    k = 0
    energy = list()

    alpha = 0.01
    
    while k < K:
        print('---- iteration number = ', k)
        k += 1
        #Calcul de chi_bar le softmax de chi
        chi_bar = numpy.softmax

        #Calcul de grad_chi(E)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_bar)
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi_bar)
        E=J(domain_omega, p, spacestep, None, None)
        energy.append(E)
        plot_energy(energy)
        grad_J = diff_J(p,q,Alpha, domain_omega)                     #Gradient de E vis à vis des points du domaine
        grad_J = grad_shifted(grad_J, domain_omega)                  #Gradient clip à zero en tout les points non frontaliers
        

        grad_J *= 1
        chi = chi - alpha * grad_J
        
        l = dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi=projector(domain_omega, l, chi)
   
        energy.append(E)
        plot_energy(energy)
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J