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

    alpha = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = numpy.zeros((M, N))
    v = numpy.zeros((M, N))
    
    while k < numb_iter:
        print('---- iteration number = ', k)
        k += 1

        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        E=J(domain_omega, p, spacestep, mu1, V_0)
        energy.append(E)
        plot_energy(energy)
        grad_J = diff_J(p,q,Alpha, domain_omega)                     #Gradient de E vis à vis des points du domaine
        grad_J = grad_shifted(grad_J, domain_omega)                  #Gradient clip à zero en tout les points non frontaliers
        
        m = beta1 * m + (1-beta1) * grad_J
        v = beta2 * v + (1-beta2) * grad_J * grad_J
        m_bias_corrected = m/(1 - beta1 ** k)
        v_bias_corrected = v/(1 - beta2 ** k)
        chi = chi - alpha * m_bias_corrected / (numpy.sqrt(v_bias_corrected) + eps)
        
        l = dicho_l(chi, beta, -np.max(chi), 1-np.min(chi), domain_omega)
        chi=projector(domain_omega, l, chi)
   
        energy.append(E)
        plot_energy(energy)
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J