import numpy as np
from demo_control_polycopie import Alpha
import utils

def dicho_l(x, beta, lmin, lmax, domain, precision=1e-3):
    lmid = (lmax + lmin) / 2
    x_new = utils.projector(domain, lmid, x)
    beta_current = np.sum(x_new)
    #print("Beta current: ", beta_current, "for: ", lmin, lmax, lmid)
    if abs(beta_current - beta) <= precision:
        return lmid
    if beta_current >= beta:
        return dicho_l(x, beta, lmin, lmid, domain, precision) 
    else:
        return dicho_l(x, beta, lmid, lmax, domain, precision)

def proposed_algo(chi0, lr):

    for k in range(0, K):

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
