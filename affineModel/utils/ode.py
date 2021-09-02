"""
ODE function for solving affine model
"""
from typing import List
import numpy as np


def d_alpha_beta(alpha_beta: List[float], t,
                 u, r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j) -> List[float]:

    """
    :param alpha_beta: time zero value. alpha_real[0]=0, alpha_img[0]=0. beta_real[0]=0, beta_imag[0]=0
    :param t: time vector
    :param u: domian of characteristic function u
    :param r: risk free rate
    :param sigma: vol level
    :param rho:
    :param kappa:
    :param xi:
    :param lambda_0: constant part for jump intensity
    :param lambda_1: coefficient on teh volatility for jump intensity
    :param m_v:  vol jumps with exponential distribution, sqrt(m_v) is the average of vol jump in terms of SD
    :param mu_j:
    :param rho_j: correlation between stock price jump and vol jump
    :param sigma_j: once jump arrives. stock price jumps with log normal distribution with the sigma_j int terms of SD
    :return: list of given time changes in alpha and beta real and imag value
    """
    alpha_real = alpha_beta[0]
    alpha_imag = alpha_beta[1]
    beta_real = alpha_beta[2]
    beta_imag = alpha_beta[3]

    # compute the expected jump size
    m_j = np.exp(mu_j + 0.5 * (sigma_j * sigma_j)) / (1 - m_v * rho_j) - 1

    # functions of beta real and beta imag
    temp1 = 1 - m_v * beta_real
    temp2 = m_v * (beta_imag + rho_j * u)

    a_beta_real_beta_imag = temp1 / (temp1 * temp1 + temp2 * temp2)
    b_beta_real_beta_imag = temp2 / (temp1 * temp1 + temp2 * temp2)

    # ODE1: transition of alpha real part
    d_alpha_real = -lambda_0 + beta_real * kappa * (sigma * sigma) \
                   + lambda_0 * np.exp(-0.5 * ((u * u) * (sigma_j * sigma_j))) \
                   * (np.cos(mu_j * u) * a_beta_real_beta_imag - np.sin(mu_j * u) * b_beta_real_beta_imag)

    # ODE2: transition of alpha imag part
    d_alpha_imag = u * (r - m_j * lambda_0) + beta_imag * kappa * (sigma * sigma) \
                   + lambda_0 * np.exp(-0.5 * ((u * u) * (sigma_j * sigma_j))) \
                   * (np.cos(mu_j * u) * b_beta_real_beta_imag + np.sin(mu_j * u) * a_beta_real_beta_imag)

    # ODE 3: transition of beta real part
    d_beta_real = -lambda_1 - beta_real * kappa + 0.5 * (-(u * u)) - u * beta_imag * xi * rho \
                   + 0.5 * (beta_real * beta_real - beta_imag * beta_imag) * (xi * xi) \
                   + lambda_1 * np.exp(-0.5 * ((u * u) * (sigma_j * sigma_j))) \
                   * (np.cos(mu_j * u) * a_beta_real_beta_imag - np.sin(mu_j * u) * b_beta_real_beta_imag)

    # ODE 4: transition of beta imag part
    d_beta_imag = -u * m_j * lambda_1 * kappa - beta_imag * kappa + 0.5 * (-u) + u * beta_real * xi * rho \
                   + beta_real * beta_imag * (xi * xi) + lambda_1 * np.exp(-0.5 * ((u * u) * (sigma_j * sigma_j))) \
                   * (np.cos(mu_j * u) * b_beta_real_beta_imag + np.sin(mu_j * u) * a_beta_real_beta_imag)

    return [d_alpha_real, d_alpha_imag, d_beta_real, d_beta_imag]


def chi_k(u, a, b, c, d):
    tmp1 = np.cos(u * (d-a)) * np.exp(d) - np.cos(u * (c - a)) * np.exp(c)
    tmp2 = u * np.sin(u*(d-a)) * np.exp(d) - u * np.sin(u * (d-a)) * np.exp(c)
    chi_k = 1. / (1 + u * u) * (tmp1 + tmp2)
    return chi_k


def psi_k(u, a, b, c, d):
    psi_k = np.zeros(len(u))
    psi_k[0] = d - c
    psi_k[1:] = (np.sin(u[1:] * (d - a)) - np.sin(u[1:] * (c - a))) * (1 / u[1:])
    return psi_k

