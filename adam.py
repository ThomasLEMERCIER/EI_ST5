from algo_optimisation import *
from utils import *
import matplotlib.pyplot as plt
import numpy

def GD_Adam(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    plt.figure()
    plt.ion()

    k = 0
    beta = np.sum(chi)
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100000
    energy = list()

    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = 0
    v = 0
    
    while k < numb_iter:
        print('---- iteration number = ', k)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        E=J(domain_omega, p, spacestep, mu1, V_0)
        energy.append(E)

        grad_J = diff_J(p,q,Alpha, domain_omega)                     #Gradient de E vis à vis des points du domaine
        grad_J = grad_shifted(grad_J, domain_omega)                  #Gradient clip à zero en tout les points non frontaliers
        l = dicho_l(chi-mu*grad_J, beta, -1, 1, domain_omega)
        chi=projector(domain_omega, l, chi-mu*grad_J)
   
        energy.append(E)
        plot_energy(energy)
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J