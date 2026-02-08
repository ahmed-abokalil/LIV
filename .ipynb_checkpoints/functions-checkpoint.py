import numpy as np
import torch as tn
from scipy.integrate import quad
import lhapdf
from constants import *
from scipy.integrate import simps
pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)

factor = 4 * alpha**2*np.pi / (3 * Nc)

def f_s(x, tau, flavor, Q2):
    tau_x = tau / x
    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_flavor_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_anti_flavor_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2


def num_derivative(func, x, h=1e-8, *args):
    return (func(x + h, *args) - func(x - h, *args)) / (2 * h)


def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau/x
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(-flavor, t, Q2), tau_x)

    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    
    return (1/x * pdf_flavor_x * f_fbar_tau_x_prime + \
           1/x * f_f_tau_x_prime * pdf_anti_flavor_x)


def sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2):
    tau_x = tau/x
    f_s_val = f_s(x, tau, flavor, Q2)
    f_p_s_val = f_prime_s(x, tau, flavor, Q2)
    s_eff = 2 * (p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3])
    # Pre-calculate contractions
    c11, c12, c21, c22 = tn.einsum('mn,m,n->', C, p1, p1), tn.einsum('mn,m,n->', C, p1, p2), \
                         tn.einsum('mn,m,n->', C, p2, p1), tn.einsum('mn,m,n->', C, p2, p2)
    
    sum_c = c11 + c12 + c21 + c22
    weighted_c = (x * c11 + tau_x * c12 + tau_x * c21 + x * c22)
    term2_plus_term3 = (2/s_eff) * (1 + x/tau_x) * sum_c * f_s_val + (2/s_eff) * weighted_c * f_p_s_val
    return f_s_val, term2_plus_term3

def integrate_sigma_hat_prime_sm(tau, flavor, Q2):
    def integrand1(x):
        tau_x = tau/x
        return f_s(x, tau, flavor, Q2) * tau_x
    
    result1, _ = quad(integrand1, tau, 1)
    return result1

def integrate_sigma_hat_prime_sme(tau, C, p1, p2, flavor, Q2):
    def integrand2(x):
        tau_x = tau / x
        _, term2_plus_term3 = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term2_plus_term3 * tau_x

    result2, _ = quad(integrand2, tau, 1)

    return result2


def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g):
    return abs((((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g))
            
def term_3(Q2, e_f, g):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2

def summation_terms(Q2, e_f, g):
    return  (term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, e_f, g))
    

def d_sigma_sm(Q2, quark_couplings):
    tau = Q2 / s
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)

        termL = summation_terms(Q2, e_f, g_fL)
        termR = summation_terms(Q2, e_f, g_fR)

        d_sigma +=  (termL+ termR )* integral
    
    d_sigmasm =  factor * 0.389379 * 1e9* d_sigma    # Conversion from GeV^-2 to Pb
    return d_sigmasm


def d_sigma(Q2, CL, CR, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0

    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral1 = integrate_sigma_hat_prime_sme(tau, CL, p1, p2, flavor, Q2)
        integral2 = integrate_sigma_hat_prime_sme(tau, CR, p1, p2, flavor, Q2)
        sum_terms_L = summation_terms(Q2, e_f, g_fL)
        sum_terms_R = summation_terms(Q2, e_f, g_fR)

        d_sigmaL +=  sum_terms_L * integral1
        d_sigmaR += sum_terms_R * integral2

    return factor* 0.389379 * 1e9*(d_sigmaL + d_sigmaR)  # This is d\sigma / dQ^2

def d_sigma_full(Q2, CL, CR, p1, p2, quark_couplings):
    liv = d_sigma(Q2, CL, CR, p1, p2, quark_couplings)
    sm = d_sigma_sm(Q2, quark_couplings)
    return sm + liv  # This is also d\sigma / dQ^2

def sigma_sm(Qmin, Qmax, quark_couplings):
    def inet(Q2):
        return d_sigma_sm(Q2, quark_couplings)
    int, _ = quad(inet, Qmin**2, Qmax**2)
    return int
    
    
def sme(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps=100):
    num_steps = int(num_steps)
    
    # Ensure that num_steps is odd for Simpson's rule to work properly
    if num_steps % 2 == 0:
        num_steps += 1

    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps)
    
    # Loop over Q2 values and calculate d_sigma for each
    integrand_values = np.array([d_sigma(Q2, CL, CR, p1, p2, quark_couplings) for Q2 in Q2_values])
    
    # Use Simpson's rule to integrate over the Q2_values array
    integral_liv = simps(integrand_values, Q2_values)
    
    return integral_liv


def sigma_full(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps=100):
    num_steps = int(num_steps)
    
    if num_steps % 2 == 0:
        num_steps += 1

    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps)
    
    # Calculate the integrand values at each Q2 point
    integrand_values = np.array([d_sigma_full(Q2, CL, CR, p1, p2, quark_couplings) for Q2 in Q2_values])

    # Integrate using Simpson's rule
    integral_full = simps(integrand_values, Q2_values)

    return integral_full

def dsigma_dQ(Q2, quark_couplings):
    Q= np.sqrt(Q2)
    sm = d_sigma_sm(Q2, quark_couplings)
    
    return 2*Q*sm

def dsigma_dQ_1(Q2, quark_couplings):
    Q= np.sqrt(Q2)
    tau = Q2 / s
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)

        sum_terms = 2*term_1(Q2, e_f)

        d_sigma +=  sum_terms * integral
    
    d_sigmasm =  factor * 0.389379 * 1e9* d_sigma    # Conversion from GeV^-2 to Pb
    return 2*Q*d_sigmasm


def dsigma_dQ_2(Q2, quark_couplings):
    Q= np.sqrt(Q2)
    tau = Q2 / s
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)

        sum_terms1 = term_2(Q2, e_f, g_fL)+ term_2(Q2, e_f, g_fR)
        sum_terms2 = term_2(Q2, e_f, g_fL)+ term_2(Q2, e_f, g_fL)
        d_sigma +=  (sum_terms1+sum_terms2) * integral
    
    d_sigmasm =  factor * 0.389379 * 1e9* d_sigma    # Conversion from GeV^-2 to Pb
    return 2*Q*d_sigmasm

def dsigma_dQ_3(Q2, quark_couplings):
    Q= np.sqrt(Q2)
    tau = Q2 / s
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)

        sum_terms =  term_3(Q2, e_f, g_fL)+ term_3(Q2, e_f, g_fR)

        d_sigma +=  sum_terms * integral
    
    d_sigmasm =  factor * 0.389379 * 1e9* d_sigma    # Conversion from GeV^-2 to Pb
    return 2*Q*d_sigmasm
