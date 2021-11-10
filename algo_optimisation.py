import numpy as np
import utils

def dicho_l(x, beta, lmin, lmax, domain, precision=1e-3):
    lmid = (lmax + lmin) / 2
    x_new = utils.projector(domain, lmid, x)
    beta_current = np.sum(x_new)
<<<<<<< HEAD
=======
    #print("Beta target: ", beta, "Beta current: ", beta_current, "for: ", lmin, lmax, lmid)
>>>>>>> ffdee7315a5e527b5ba71bffb28ef6728ccd99a0
    if abs(beta_current - beta) <= precision:
        return lmid
    if beta_current >= beta:
        return dicho_l(x, beta, lmin, lmid, domain, precision) 
    else:
        return dicho_l(x, beta, lmid, lmax, domain, precision)

def proposed_algo(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=1e-3, lr_min=1e-5):

    beta = np.sum(chi0)
    chi = np.copy(chi0)

    energy = []

    for k in range(0, K):
        print(f"--- Iteration: {k+1} ---")

        p = utils.compute_p(  domain_omega=domain_omega,
                        spacestep=spacestep,
                        wavenumber=wavenumber,
                        Alpha=Alpha,
                        chi=chi)
        
        q = utils.compute_q(p=p,
                            domain_omega=domain_omega,
                            spacestep=spacestep,
                            wavenumber=wavenumber,
                            Alpha=Alpha,
                            chi=chi)

        e = utils.J(domain_omega=domain_omega,
                    p=p,
                    spacestep=spacestep,
                    mu1=None,
                    V_0=None)

        energy.append(e)
        print(f"energy at the beginning: {e}, lr at the beginning: {lr}")

        diff_J = utils.diff_J(p=p, q=q, alpha=Alpha)
        diff_J = utils.shift_on_boundary(matrix=diff_J, domain_omega=domain_omega)

        utils.print_on_boundary(diff_J, domain_omega, "Diff J: \n")

        while e >= utils.J(domain_omega=domain_omega, p=p, spacestep=spacestep, mu1=None, V_0=None) and lr > lr_min:
            print(f"--- Looping ---")
            l = utils.compute_l(matrix=chi-lr*diff_J, beta=beta, domain=domain_omega)
            chi_next = utils.projector(domain=domain_omega, l=l, x=chi-lr*diff_J)
        
            p_next = utils.compute_p(  domain_omega=domain_omega,
                            spacestep=spacestep,
                            wavenumber=wavenumber,
                            Alpha=Alpha,
                            chi=chi_next)

            e = utils.J(domain_omega=domain_omega,
                        p=p_next,
                        spacestep=spacestep,
                        mu1=None,
                        V_0=None)

            if e > utils.J(domain_omega=domain_omega, p=p, spacestep=spacestep, mu1=None, V_0=None):
                lr = lr/2

        chi=chi_next
        energy.append(e)

    p = p_next

    return chi, energy, p

def dicho_algo(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=1e-3, lr_min=1e-5):

    beta = np.sum(chi0)
    chi = np.copy(chi0)

    energy = []

    for k in range(0, K):
        print(f"--- Iteration: {k+1} ---")

        p = utils.compute_p(  domain_omega=domain_omega,
                        spacestep=spacestep,
                        wavenumber=wavenumber,
                        Alpha=Alpha,
                        chi=chi)
        
        q = utils.compute_q(p=p,
                            domain_omega=domain_omega,
                            spacestep=spacestep,
                            wavenumber=wavenumber,
                            Alpha=Alpha,
                            chi=chi)

        e = utils.J(domain_omega=domain_omega,
                    p=p,
                    spacestep=spacestep,
                    mu1=None,
                    V_0=None)

        energy.append(e)
        print(f"energy at the beginning: {e}, lr at the beginning: {lr}")

        diff_J = utils.diff_J(p=p, q=q, alpha=Alpha)
        diff_J = utils.shift_on_boundary(matrix=diff_J, domain_omega=domain_omega)

        utils.print_on_boundary(diff_J, domain_omega, "Diff J: \n")

        while e >= utils.J(domain_omega=domain_omega, p=p, spacestep=spacestep, mu1=None, V_0=None) and lr > lr_min:
            print(f"--- Looping ---")
            l = dicho_l(x=chi-lr*diff_J, beta=beta, lmin=-np.max(chi), lmax=1-np.min(chi), domain=domain_omega)
            chi_next = utils.projector(domain=domain_omega, l=l, x=chi-lr*diff_J)
        
            p_next = utils.compute_p(  domain_omega=domain_omega,
                            spacestep=spacestep,
                            wavenumber=wavenumber,
                            Alpha=Alpha,
                            chi=chi_next)

            e = utils.J(domain_omega=domain_omega,
                        p=p_next,
                        spacestep=spacestep,
                        mu1=None,
                        V_0=None)

            if e > utils.J(domain_omega=domain_omega, p=p, spacestep=spacestep, mu1=None, V_0=None):
                lr = lr/2

        chi=chi_next
        energy.append(e)

    p = p_next

    return chi, energy, p

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
    plt.figure()
    plt.ion()

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
        E_next=E
        while E_next>=E and mu > 10 ** -5:
            print('4. computing parametric gradient')
            grad_J=diff_J(p,q,Alpha, domain_omega)
            grad_J = preprocessing.set2zero(grad_J, domain_omega)
            l = dicho_l(chi-mu*grad_J, beta, -1, 1, domain_omega)
            chi_next=projector(domain_omega, l, chi-mu*grad_J)
            p_next=compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)
            E_next=J(domain_omega, p_next, spacestep, mu1, V_0)
            print(E,E_next,mu)
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
            plot_energy(energy)
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
    plt.figure()
    plt.ion()

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 5
    energy = list()

    mu = 0.1
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

    mu = 1
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