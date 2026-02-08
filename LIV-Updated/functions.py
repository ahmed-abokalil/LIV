import numpy as np
import torch as tn
import lhapdf
import warnings
from scipy.integrate import quad, simpson, IntegrationWarning
from constants import *

# Initialize PDF set
pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)
factor = 4 * alpha**2 * np.pi / (3 * Nc)


def f_s(x, tau, flavor, Q2):
    tau_x = tau / x
    pdf_f_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_f_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_af_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_af_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    return (1/x * pdf_f_x * 1/tau_x * pdf_af_tau_x) + (1/tau_x * pdf_f_tau_x * 1/x * pdf_af_x)

def num_derivative(func, x, h=1e-8):
    return (func(x + h) - func(x - h)) / (2 * h)

def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau/x
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(-flavor, t, Q2), tau_x)
    
    return (1/x * pdf.xfxQ2(flavor, x, Q2) * f_fbar_tau_x_prime + \
            1/x * f_f_tau_x_prime * pdf.xfxQ2(-flavor, x, Q2))

# Cross-Section Components

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
    #The factors are chosen to align with specific momentum conventions and c to optimize computational efficiency.
    term2_plus_term3 = (2/s_eff) * (1 + x/tau_x) * sum_c * f_s_val + (2/s_eff) * weighted_c * f_p_s_val
    return f_s_val, term2_plus_term3

def term_1(Q2, e_f):
    return e_f**2 / (2 * Q2**2)
    
def term_2(Q2, e_f, g):
    denom = (Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2
    prefix = (1 - m_Z**2/Q2) / denom
    coupling = (1 - 4*sin2th_w) / (4 * sin2th_w * (1 - sin2th_w))
    return abs(prefix * coupling * e_f * g)
            
def term_3(Q2, e_f, g):
    denom = (Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2
    coupling = (1 + (1 - 4*sin2th_w)**2) / (32 * sin2th_w**2 * (1 - sin2th_w)**2)
    return (1 / denom) * coupling * g**2

def summation_terms(Q2, e_f, g):
    return term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, e_f, g)


def d_sigma_sm(Q2, quark_couplings):
    tau = Q2 / s
    d_sig = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral, _ = quad(lambda x: f_s(x, tau, flavor, Q2) * (tau/x), tau, 1, limit=100)
        d_sig += (summation_terms(Q2, e_f, g_fL) + summation_terms(Q2, e_f, g_fR)) * integral
    
    return factor * 0.389379 * 1e9 * d_sig

def d_sigma(Q2, CL, CR, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigL, d_sigR = 0, 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        # SME terms
        def integrand(x, C):
            _, liv_term = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
            return liv_term * (tau/x)
        
        intL, _ = quad(integrand, tau, 1, args=(CL,), limit=100)
        intR, _ = quad(integrand, tau, 1, args=(CR,), limit=100)
        
        d_sigL += summation_terms(Q2, e_f, g_fL) * intL
        d_sigR += summation_terms(Q2, e_f, g_fR) * intR

    return factor * 0.389379 * 1e9 * (d_sigL + d_sigR)

# Final Observables 

def sigma_sm(Qmin, Qmax, quark_couplings):
    # SM Total Cross Section
    val, _ = quad(lambda Q2: d_sigma_sm(Q2, quark_couplings), Qmin**2, Qmax**2, limit=100)
    return val

def sme(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps=101):
    # SME Total Contribution
    Q2_vals = np.linspace(Q_min**2, Q_max**2, int(num_steps))
    integrand = np.array([d_sigma(Q2, CL, CR, p1, p2, quark_couplings) for Q2 in Q2_vals])
    return simpson(integrand, x=Q2_vals)

def sigma_full(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps=101):
    # Combined Result
    return sigma_sm(Q_min, Q_max, quark_couplings) + \
           sme(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps)

# Differential Plots (dsigma/dQ)

def dsigma_dQ(Q2, quark_couplings):
    return 2 * np.sqrt(Q2) * d_sigma_sm(Q2, quark_couplings)

def dsigma_dQ_components(Q2, quark_couplings, mode=1):
    tau = Q2 / s
    d_sig = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral, _ = quad(lambda x: f_s(x, tau, flavor, Q2) * (tau/x), tau, 1, limit=100)
        if mode == 1: # Photon
            terms = 2 * term_1(Q2, e_f)
        elif mode == 2: # Interference
            terms = (term_2(Q2, e_f, g_fL) + term_2(Q2, e_f, g_fR)) * 2
        else: # Z-boson
            terms = term_3(Q2, e_f, g_fL) + term_3(Q2, e_f, g_fR)
        d_sig += terms * integral
    return 2 * np.sqrt(Q2) * factor * 0.389379 * 1e9 * d_sig

# Parallel Processing Workers for M4 Mac

def compute_result_worker(args, sigma_sm_value):

    pm, pn, quark_couplings, CL1, CL2, CL3, CL4, CR = args
    Q_min, Q_max = 70, 80  
    
    r1 = sme(Q_min, Q_max, CL1, CR, pm, pn, quark_couplings)
    r2 = sme(Q_min, Q_max, CL2, CR, pm, pn, quark_couplings)
    r3 = sme(Q_min, Q_max, CL3, CR, pm, pn, quark_couplings)
    r4 = sme(Q_min, Q_max, CL4, CR, pm, pn, quark_couplings)
    
    return {
        'result_sme1': r1 + sigma_sm_value,
        'result_sme2': r2 + sigma_sm_value,
        'result_sme3': r3 + sigma_sm_value,
        'result_sme4': r4 + sigma_sm_value
    }


def compute_sme_bin_worker(Q_range, CL, CR, p1, p2, quark_couplings):
    import warnings
    from scipy.integrate import IntegrationWarning
    warnings.simplefilter("ignore", IntegrationWarning)
    
    Q_start, Q_end = Q_range
    return sme(Q_start, Q_end, CL, CR, p1, p2, quark_couplings)