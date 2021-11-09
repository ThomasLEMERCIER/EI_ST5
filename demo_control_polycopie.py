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
#import solutions
from alpha import compute_alpha


def your_optimization_procedure(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    """This function return the optimized density.
    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can it up to 1).
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 5
    energy = list()

    mu = 10
    while k < (numb_iter := 20):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, mu1, V_0)
        energy.append(E)
        E_next = E + 1

        mu = 5
        while E < E_next and mu > 10 ** -5:
        #Tant que l'énergie ne s'améliore pas, et que l'on a pas atteint un minimum, on fait une descente de gradient.
        #On passe à l'itération suivante si l'énergie baisse.
            l=0
            print('4. computing parametric gradient')
            grad_J=diff_J(p,q,Alpha, domain_omega)                  #Gradient de E vis à vis des points du domaine
            clipped_grad_J = grad_shifted(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
            chi_next=projector(l,chi-mu*clipped_grad_J)             #Descente de gradient sous contraintes "X[k] in [0, 1]"
            
            while abs(numpy.sum(chi_next)*spacestep-beta)>eps1:     #Respect de la contrainte "sum(X) * spacestep = Beta"
                if numpy.sum(chi_next)*spacestep>=beta:
                    l=l-eps2
                else:
                    l=l+eps2
                chi_next=projector(l,chi-mu*clipped_grad_J)

            p_next=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)  #Calcul du p possiblement meilleur. 
            E_next=J(domain_omega, p_next, spacestep, mu1, V_0)                     #Calcul de l'E possiblement plus faible.

            if E_next<E:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2

            print(mu)
            chi=chi_next
            energy.append(E_next)
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J

def procedure2(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    """This function return the optimized density.
    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can it up to 1).
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 5
    energy = list()

    mu = 0.01
    while k < (numb_iter := 20):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem')
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        print('2. computing solution of adjoint problem')
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        print('3. computing objective function')
        E=J(domain_omega, p, spacestep, mu1, V_0)
        energy.append(E)
        E_next = E + 1
        print('3.5. computing gradient clipped')
        grad_J=diff_J(p,q,Alpha, domain_omega)                  #Gradient de E vis à vis des points du domaine
        clipped_grad_J = grad_shifted(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        mu = 8
        while E < E_next and mu > 10 ** -5:
        #Tant que l'énergie ne s'améliore pas, et que l'on a pas atteint un minimum, on fait une descente de gradient avec un lr plus petit.
        #On passe à l'itération suivante si l'énergie baisse.
            print('4. gradient descent trial')
            l = 0
            chi_next=projector(l,chi-mu*clipped_grad_J)             #Descente de gradient sous contraintes "X[k] in [0, 1]"
            while abs(numpy.sum(chi_next)*spacestep-beta)>eps1:     #Respect de la contrainte "sum(X) * spacestep = Beta"
                if numpy.sum(chi_next)*spacestep>=beta:
                    l=l-eps2
                else:
                    l=l+eps2
                chi_next=projector(l,chi-mu*clipped_grad_J)
                
            p_next=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)  #Calcul du p possiblement meilleur. 
            E_next=J(domain_omega, p_next, spacestep, mu1, V_0)                     #Calcul de l'E possiblement plus faible.

            if E_next<E:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2

            print(mu)
            energy.append(E_next)
            plot_energy(energy)
        
        #Amélioration de chi une fois que la descente de gradient a vraiment amélioré (baissé) E.
        chi = chi_next
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J

def SGD(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, eps1, eps2, beta, V_0):
    """This function return the optimized density.
    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can it up to 1).
    """
    plt.figure()
    plt.ion()

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100000
    energy = list()

    mu = 0.01
    while k < numb_iter:
        print('---- iteration number = ', k)
        p=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi)
        q=compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi)
        E=J(domain_omega, p, spacestep, mu1, V_0)
        energy.append(E)
        grad_J=diff_J(p,q,Alpha, domain_omega)                  #Gradient de E vis à vis des points du domaine
        clipped_grad_J = grad_shifted(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        l = 0
        chi_next=projector(l,chi-mu*clipped_grad_J)             #Descente de gradient sous contraintes "X[k] in [0, 1]"
        while abs(numpy.sum(chi_next)*spacestep-beta)>eps1:     #Respect de la contrainte "sum(X) * spacestep = Beta"
            if numpy.sum(chi_next)*spacestep>=beta:
                l=l-eps2
            else:
                l=l+eps2
            chi_next=projector(l,chi-mu*clipped_grad_J)
        chi = chi_next    

        energy.append(E)
        plt.clf()
        plt.plot(energy)
        plt.pause(1e-3)
        k += 1

    print('end. computing solution of Helmholtz problem')
    return chi, energy, p, grad_J


def grad_shifted(grad,domain_omega):
    (M, N) = numpy.shape(domain_omega)

    indices_x,indices_y = numpy.where(domain_omega == _env.NODE_ROBIN)
    for i in range(len(indices_x)):
        x,y=indices_x[i],indices_y[i]
        if x>0:
            bas=grad[x-1,y]
        else:
            bas=0
        if x<M-1:
            haut=grad[x+1,y]
        else:
            haut=0
        if y>0:
            gauche=grad[x,y-1]
        else:
            gauche=0
        if y<N-1:
            droite=grad[x,y+1]
        else:
            droite=0
        grad[x,y]=(bas+haut+droite+gauche)/max(1,(4-[bas,haut,gauche,droite].count(0)))
    grad=preprocessing.set2zero(grad,domain_omega)
    return grad

def projector(l,chi):
    for i in range(len(chi)):
        for j in range(len(chi[i])):
            chi[i][j]=max(0,min(chi[i,j]+l,1))
    return chi

def J(domain_omega, p, spacestep, mu1, V_0):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)
    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """

    p_conj = numpy.conjugate(p)
    p_norm = numpy.real(p * p_conj)
    energy = numpy.sum(p_norm) * spacestep * spacestep

    return energy

def compute_p(domain_omega, spacestep, wavenumber, Alpha, chi):

    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    alpha_rob = Alpha * chi

    p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return p

def compute_q(p, domain_omega, spacestep, wavenumber, Alpha, chi):
                        
    (M, N) = domain_omega.shape   
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    f_dir[:, :] = 0.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    alpha_rob = Alpha * chi

    f_adjoint = - 2 * numpy.conjugate(p)

    q = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adjoint, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    return q

def diff_J(p, q, alpha, domain_omega):
    return diff_J_shifted(p, q, alpha, domain_omega)
    return - numpy.real(alpha * p * q)

def diff_J_shifted(p, q, alpha, domain_omega):
    #Cas où la frontière est linéaire : domain_omega[N, 0:N] = _env.NODE_ROBIN
    M, N = domain_omega.shape
    p_shifted = p.copy()
    p_shifted[N, :] = p[N-1, :]
    q_shifted = q.copy()
    q_shifted[N, :] = q[N-1, :]
    # extract_on_boundary(p_shifted, domain_omega)
    # extract_on_boundary(q_shifted, domain_omega)
    # extract_on_boundary(p*q, domain_omega)
    return - numpy.real(alpha * p_shifted * q_shifted)

def extract_on_boundary(matrix, domain_omega):
    indices = numpy.where(domain_omega == _env.NODE_ROBIN)
    print(indices)

def plot_energy(Ene):
    matplotlib.pyplot.close()
    matplotlib.pyplot.plot(Ene)
    matplotlib.pyplot.savefig("ENERGIE")

if __name__ == '__main__':
    ALGO = SGD
    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0 # level of the fractal
    spacestep = 1.0 / N  # mesh size
    c0 = 340
    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0
    material = "MELAMINE"
    omega = wavenumber * c0 
    precision = 15

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    # -- define absorbing material
    Alpha = compute_alpha(material, omega, precision)
    Alpha = Alpha[0] + Alpha[1] * 1j
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional



    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    chi, energy, u, grad = ALGO(domain_omega, spacestep, wavenumber, Alpha, chi, mu, mu1, 1e-2, 1e-2, 0.3, V_0)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    print('End.')
