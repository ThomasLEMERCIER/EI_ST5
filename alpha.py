from math import pi, sqrt
from cmath import exp
from scipy.optimize import minimize
from tqdm import tqdm


def error(alpha, precision, initial_parameters):
    alpha = complex(alpha[0], alpha[1])
    return sum(error_k(n * pi / initial_parameters["L"], alpha, initial_parameters) for n in range(-precision, precision+1))

def compute_parameters(k, initial_parameters):
    parameters = initial_parameters.copy()
    temp =  k * k - parameters["xi0"] * parameters["w"] * parameters["w"] / parameters["eta0"]
    if temp >= 0:
        parameters["lambda0"] = sqrt(temp)
    else:
        parameters["lambda0"] = complex(0,1) * sqrt(-temp)
    temp = k*k - initial_parameters["xi1"] * parameters["w"] * parameters["w"] / initial_parameters["eta1"]
    s_temp = initial_parameters["a"] * initial_parameters["w"] / initial_parameters["eta1"]
    t_temp = 1/sqrt(2)
    racine = temp + sqrt(temp * temp + s_temp * s_temp)
    parameters["lambda1"] = t_temp * sqrt(racine)
    racine -= 2 * temp
    parameters["lambda1"] -= t_temp * complex(0,1) * sqrt(racine)
    return parameters


def FFT_g(k):
    return 0 if k != 0 else 1

def f(x, parameters):
    return (parameters["lambda0"] * parameters["eta0"] - x) * exp(-parameters["lambda0"] * parameters["L"]) + (parameters["lambda0"] * parameters["eta0"] + x) * exp(parameters["lambda0"] * parameters["L"])

def X(k, alpha, parameters):
    x = (parameters["lambda0"] * parameters["eta0"] - parameters["lambda1"] * parameters["eta1"]) / f(parameters["lambda1"] * parameters["eta1"], parameters)
    x -= (parameters["lambda0"] * parameters["eta0"] - alpha) / f(alpha, parameters)
    return FFT_g(k) * x

def Y(k, alpha, parameters):
    x = (parameters["lambda0"] * parameters["eta0"] + parameters["lambda1"] * parameters["eta1"]) / f(parameters["lambda1"] * parameters["eta1"], parameters)
    x -= (parameters["lambda0"] * parameters["eta0"] + alpha) / f(alpha, parameters)
    return FFT_g(k) * x

def error_k(k, alpha, initial_parameters):
    parameters = compute_parameters(k, initial_parameters)
    x = X(k, alpha, parameters)
    y = Y(k, alpha, parameters)
    if k * k >= parameters["xi0"] * parameters["w"] * parameters["w"] / parameters["eta0"]:
        e_k = (parameters["A"] + parameters["B"] * k**2 ) * ( ( 1 / ( 2 * parameters["lambda0"] ) ) * ( abs(x)**2 * ( 1 - exp(-2*parameters["lambda0"]*parameters["L"]) ) + abs(y)**2 * ( exp(2*parameters["lambda0"]*parameters["L"]) -1 ) ) + 2 * parameters["L"] * (x * y.conjugate()).real)
        e_k += 0.5 * parameters["B"] * parameters["lambda0"] * ( abs(x)**2 * ( 1-exp(-2*parameters["lambda0"]*parameters["L"]) ) + abs(y)**2 * ( exp(2*parameters["lambda0"]*parameters["L"]) -1 ))
        e_k -= 2 * parameters["L"] * parameters["B"] * parameters["lambda0"] **2 * (x * y.conjugate()).real
    else:
        e_k = ( parameters["A"] + parameters["B"] * k**2 ) * ( parameters["L"] * ( abs(x)**2 + abs(y)**2 ) + complex(0,1) * ( x * y.conjugate() * (1-exp(-2*parameters["lambda0"]*parameters["L"])) ).imag / parameters["lambda0"] )
        e_k += parameters["B"] * parameters["L"] * abs(parameters["lambda0"])**2 * ( abs(x)**2 + abs(y)**2 )
        e_k += complex(0,1) * parameters["B"] * parameters["lambda0"] * ( x * y.conjugate() * ( 1-exp(-2*parameters["lambda0"]*parameters["L"]) ) ).imag
    return e_k

def material_parameters(material, omega):
    if material == 'BIRCHLT':
        # Birch LT
        phi = 0.529  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 151429.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.37  # tortuosity
        c_0 = 340.0
    elif material == 'MELAMINE':
        # Melamine foam
        phi = 0.99  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 14000.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.02  # tortuosity
        c_0 = 340.0
    elif material == 'POLYURETHANE':
        # Polyurethane foam
        phi = 0.98  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 45000.0  # resitivity
        rho_0 = 1.2
        alpha_h = 2.01  # tortuosity
        c_0 = 340.0
    elif material == 'SUTHERLAND':
        # Sutherland's law
        a = 0.555
        b = 120.0
        T0 = 524.07
        TC = 20
        TR = 9 / 5 * (273.15 + TC)
        mu0 = 1.827 * 1e-05
        mu = mu0 * (a * T0 + b) / (a * TR + b) * (TR / T0) ** (3 / 2)
        k0 = 1.6853 * 1e-09
        sigma = mu / k0

        phi = 0.7
        gamma_p = 7.0 / 5.0
        rho_0 = 1.2
        alpha_h = 1.15
        c_0 = 340.0
    else:
        NotImplementedError(f"The following material is not yet implemented: {material}")

    # parameters of the geometry
    L = 0.01

    # parameters of the material (cont.)
    eta_0 = 1.0
    xi_0 = 1.0 / (c_0 ** 2)
    eta_1 = phi / alpha_h
    xi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    return {"L": L, "A":1, "B":1, "xi0": xi_0, "eta0": eta_0, "w": omega, "a":a, "xi1": xi_1,"eta1": eta_1}

def compute_alpha(material, omega, precision):
    initial_parameters = material_parameters(material, omega)
    result = minimize(lambda x: error(x, precision, initial_parameters).real, [-1,1])
    return result.x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    precision = 15
    omegas = np.linspace(1, 30_000, 100)

    materials = ["BIRCHLT", "MELAMINE", "POLYURETHANE", "SUTHERLAND"]


    fig = plt.figure()
    ax_re = fig.add_subplot(111)
    ax_re.set_xlabel(r"$\omega$")
    ax_re.set_ylabel(r"Re($\alpha$)")


    fig = plt.figure()
    ax_im = fig.add_subplot(111)
    ax_im.set_xlabel(r"$\omega$")
    ax_im.set_ylabel(r"Im($\alpha$)")

    for material in materials:
        alphas_real = []
        alphas_img = []
        for omega in tqdm(omegas):
            initial_parameters = material_parameters(material, omega)
            result = minimize(lambda x: error(x, precision, initial_parameters).real, [-1,1])
            alpha = result.x
            alphas_real.append(alpha[0])
            alphas_img.append(alpha[1])

        ax_re.plot(omegas, alphas_real, label=material)
        ax_im.plot(omegas, alphas_img, label=material)


    ax_re.legend()
    ax_im.legend()
    plt.show()



