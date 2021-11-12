import numpy as np
import EI.utils
utils = EI.utils

def proposed_algo(chi0, domain_omega, spacestep, wavenumber, Alpha, K, lr=5, lr_min=1e-8):

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

        #utils.print_on_boundary(diff_J, domain_omega, "Diff J: \n")

        e_next = e

        while e_next >= e and lr > lr_min:
            print(f"--- Looping ---")
            chi_next = chi-lr*diff_J # -- Gradient descent
            l = utils.dicho_l(x=chi_next, beta=beta, lmin=-np.max(chi_next), lmax=1-np.min(chi_next), domain=domain_omega) # -- Compute projector
            chi_next = utils.projector(domain=domain_omega, l=l, x=chi_next) # -- Projection
        
            p_next = utils.compute_p(  domain_omega=domain_omega,
                            spacestep=spacestep,
                            wavenumber=wavenumber,
                            Alpha=Alpha,
                            chi=chi_next)

            e_next = utils.J(domain_omega=domain_omega,
                        p=p_next,
                        spacestep=spacestep,
                        mu1=None,
                        V_0=None)

            if e_next > e:
                lr = lr/2
            else:
                lr = lr * 1.1

        chi=chi_next

    p = p_next
    energy.append(e_next)

    return chi, energy, p

def proposed_algo_reset(chi, domain_omega, spacestep, wavenumber, Alpha, K):
    """
    Identique à l'algo proposé mais en réinitialisant le LR à sa valeur initiale après chaque gradient step.
    Plus lent, mais converge mieux.
    Performance : 0.7
    """
    beta = np.sum(chi)
    energy = []

    for k in range(K):

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

        E = utils.J(domain_omega=domain_omega,
                    p=p,
                    spacestep=spacestep,
                    mu1=None,
                    V_0=None)

        E_next = E
        energy.append(E)

        grad_J = utils.diff_J(p,q,Alpha)                  #Gradient de E vis à vis des points du domaine
        grad_J = utils.shift_on_boundary(grad_J, domain_omega)     #Gradient clip à zero en tout les points non frontaliers
    
        mu = 5 
        while E_next >= E and mu > 10 ** -5:
        #Tant que l'énergie ne s'améliore pas, et que l'on a pas atteint un minimum, on fait une descente de gradient avec un lr plus petit.
        #On passe à l'itération suivante si l'énergie baisse.
            chi_candidate = chi-mu*grad_J
            l = utils.dicho_l(chi_candidate, beta, - np.max(chi_candidate), 1 - np.min(chi_candidate), domain_omega, precision=1e-3)
            chi_next= utils.projector(domain_omega, l, chi_candidate)             #Descente de gradient sous contraintes "X[k] in [0, 1]"
               
            p_next= utils.compute_p(domain_omega, spacestep, wavenumber, Alpha, chi_next)  #Calcul du p possiblement meilleur. 
            E_next= utils.J(domain_omega, p_next, spacestep, None, None)                     #Calcul de l'E possiblement plus faible.
            if E_next < E:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            
            energy.append(E_next)

        chi = chi_next        
    return chi, energy, p