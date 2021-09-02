from numpy import matlib
from scipy.integrate import odeint
from affineModel.utils.ode import d_alpha_beta, chi_k, psi_k
from affineModel.utils.blackscholes import implvol_put, put
import time
import numpy as np


class AffineModelwithJump:
    def __init__(self, s_0, sigma_mean, sigma_0,  r, k_upper, a, b, T, days_in_year, moneyness, time_to_maturity):
        # state variable
        self.s_0 = s_0  # 100  # spot price
        self.sigma = sigma_mean  #
        self.v_0 = sigma_0 * sigma_0   # variance
        self.r = r
        self.k_upper = k_upper  # 100
        self.k = np.arange(0, k_upper + 1)  # k from 0 to 100 np.arange(0, 101)
        self.a = a
        self.b = b
        self.u_vec = np.pi * self.k / (b - a)  # length of 101

        # discrete the time domain
        self.days_in_year = days_in_year
        self.T_large = T
        self.number_of_days = T * days_in_year
        self.time_vec = np.arange(0, self.number_of_days + 1) / self.days_in_year

        # add option moneyness and time_to_maturity
        self.moneyness = moneyness
        self.time_to_maturity = time_to_maturity

    def solve_odes(self, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j):
        # Solving for the coefficients of characteristic function
        solution = np.zeros((len(self.u_vec) * len(self.time_vec), 4))
        k1 = np.kron(self.u_vec, np.ones(len(self.time_vec)))
        k2 = matlib.repmat(self.time_vec, 1, len(self.u_vec)).transpose()
        k2 = k2.reshape((k2.shape[0],))
        ut_identifier = np.column_stack((k1, k2))  # use for locating the relative data

        # define initial condition of ODEs and solve for ODE system
        for i in range(0, len(self.u_vec)):
            y0 = np.zeros(4)
            y = odeint(func=d_alpha_beta, y0=y0, t=self.time_vec,
                       args=(self.u_vec[i], self.r, self.sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j))
            solution[i * len(self.time_vec): (i + 1) * len(self.time_vec), :] = y

        alpha_real = solution[:, 0]
        alpha_imag = solution[:, 1]
        beta_real = solution[:, 2]
        beta_imag = solution[:, 3]
        charac_fun_wo_so = np.exp(alpha_real + 1j * alpha_imag + (beta_real + 1j * beta_imag) * self.v_0)

        return charac_fun_wo_so, ut_identifier

    def put_option_price(self, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j):
        # compute for F_k and V_k, comutation of Density through CF
        # moneyness = np.linspace(-1, 1, 11)
        # time_to_maturity = np.linspace(0.2, 1, 5)
        charac_fun_wo_so, ut_identifier = self.solve_odes(rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j)
        v_k_matrix = np.zeros((len(self.moneyness) * len(self.time_to_maturity), len(self.k)))
        f_k_matrix = np.zeros((len(self.moneyness) * len(self.time_to_maturity), len(self.k)))

        for k in range(0, len(self.moneyness)):
            for t in range(0, len(self.time_to_maturity)):
                T = self.time_to_maturity[t]
                K = self.s_0 * np.exp(self.moneyness[k] * self.sigma * np.sqrt(T))
                v_k_matrix[k * len(self.time_to_maturity) + t, :] = 2. / (self.b - self.a) * K * \
                                                                    (psi_k(self.u_vec, self.a, self.b, self.a, 0)
                                                                     - chi_k(self.u_vec, self.a, self.b, self.a, 0))
                tmp = np.exp(1j * (np.log(self.s_0 / K) - self.a) * self.u_vec) \
                      * charac_fun_wo_so[np.round(ut_identifier[:, 1] * self.days_in_year) == np.round(self.days_in_year * T)]
                f_k_matrix[k * len(self.time_to_maturity) + t, :] = np.real(tmp)
                f_k_matrix[k * len(self.time_to_maturity) + t, 0] = 0.5 * f_k_matrix[k * len(self.time_to_maturity) + t, 0]

        # Find the price of Put option over the Moneyness and Time to Maturity plane, [len(moneyness), len(time)]
        f_k_times_v_k = v_k_matrix * f_k_matrix
        k_sigma_f_k_v_k = f_k_times_v_k.sum(axis=1)
        k_sigma_f_k_v_k_matrix = k_sigma_f_k_v_k.reshape((len(self.moneyness), len(self.time_to_maturity)))
        put_price_matrix = matlib.repmat(np.exp(-self.r * self.time_to_maturity), len(self.moneyness), 1) \
                           * k_sigma_f_k_v_k_matrix

        return put_price_matrix

    def put_option_stochastic_vol_no_jump(self, theta, y):
        put_price_matrix = self.put_option_price(rho=theta[0],
                                                 kappa=theta[1],
                                                 xi=theta[2],
                                                 lambda_0=0,
                                                 lambda_1=0,
                                                 m_v=0,
                                                 mu_j=0,
                                                 rho_j=0,
                                                 sigma_j=0)
        # put_price_vec: [pv(m0, t0), pv(m0, t1), ..., pv(m0, tn), pv(m1, t0), ..., pv(m1, tn),...pv(mn,tn)]
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        return np.sum((y - put_price_vec) ** 2) / len(y)

    def put_option_stocastic_vol_stock_jump(self, theta, y):
        put_price_matrix = self.put_option_price(rho=theta[0],
                                                 kappa=theta[1],
                                                 xi=theta[2],
                                                 lambda_0=0.01,
                                                 lambda_1=0.01,
                                                 m_v=0,
                                                 mu_j=theta[3],
                                                 rho_j=0,
                                                 sigma_j=theta[4])
        # put_price_vec: [pv(m0, t0), pv(m0, t1), ..., pv(m0, tn), pv(m1, t0), ..., pv(m1, tn),...pv(mn,tn)]
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        return np.sum((y - put_price_vec) ** 2).mean()

    def put_option_stocastic_vol_all_jump(self, theta, y):
        put_price_matrix = self.put_option_price(rho=theta[0],
                                                 kappa=theta[1],
                                                 xi=theta[2],
                                                 lambda_0=0.01,
                                                 lambda_1=0.01,
                                                 m_v=theta[3],
                                                 mu_j=theta[4],
                                                 rho_j=theta[5],
                                                 sigma_j=theta[6])
        # put_price_vec: [pv(m0, t0), pv(m0, t1), ..., pv(m0, tn), pv(m1, t0), ..., pv(m1, tn),...pv(mn,tn)]
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        return np.sum((y - put_price_vec) ** 2).mean()










