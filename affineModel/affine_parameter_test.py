from numpy import matlib
from scipy.integrate import odeint
from affineModel.utils.ode import d_alpha_beta, chi_k, psi_k
from affineModel.utils.blackscholes import implvol_put, put
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0, moneyness,
                           time_to_maturity):
    k_upper = 100
    k = np.arange(0, k_upper + 1)  # k from 0 to 100
    a = -5
    b = 5
    u_vec = np.pi * k / (b - a)  # length of 101

    # discrete the time domain
    days_in_year = 250
    T_large = 1
    number_of_days = T_large * days_in_year
    time_vec = np.arange(0, number_of_days + 1) / days_in_year  # length of 253

    # Solving for the coefficients of characteristic function
    solution = np.zeros((len(u_vec) * len(time_vec), 4))
    k1 = np.kron(u_vec, np.ones(len(time_vec)))
    k2 = matlib.repmat(time_vec, 1, len(u_vec)).transpose()
    k2 = k2.reshape((k2.shape[0], ))
    UT_Identifier = np.column_stack((k1, k2))  # use for locating the relative data

    # define initial condition of ODEs and solve for ODE system
    s = time.time()
    for i in range(0, len(u_vec)):
        y0 = np.zeros(4)
        y = odeint(func=d_alpha_beta, y0=y0, t=time_vec,
                   args=(u_vec[i], r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j))
        solution[i * len(time_vec): (i+1) * len(time_vec), :] = y
    e = time.time()
    print(f'solving ODE system takes {e - s} seconds')

    alpha_real = solution[:, 0]
    alpha_imag = solution[:, 1]
    beta_real = solution[:, 2]
    beta_imag = solution[:, 3]

    charac_fun_wo_so = np.exp(alpha_real + 1j * alpha_imag + (beta_real + 1j * beta_imag) * v_0)

    # compute for F_k and V_k, comutation of Density through CF
    v_k_matrix = np.zeros((len(moneyness) * len(time_to_maturity), len(k)))
    f_k_matrix = np.zeros((len(moneyness) * len(time_to_maturity), len(k)))

    s = time.time()

    for k in range(0, len(moneyness)):
        for t in range(0, len(time_to_maturity)):
            T = time_to_maturity[t]
            K = s_0 * np.exp(moneyness[k] * sigma * np.sqrt(T))
            v_k_matrix[k * len(time_to_maturity) + t, :] = 2. / (b - a) * K * (psi_k(u_vec, a, b, a, 0)
                                                           - chi_k(u_vec, a, b, a, 0))
            f_k_matrix[k * len(time_to_maturity) + t, :] = np.real(np.exp(1j * (np.log(s_0 / K) - a) * u_vec)
                                                           * charac_fun_wo_so[np.round(UT_Identifier[:, 1]*days_in_year)
                                                           == np.round(days_in_year * T)])
            f_k_matrix[k * len(time_to_maturity) + t, 0] = 0.5 * f_k_matrix[k * len(time_to_maturity) + t, 0]
    e = time.time()
    print(f'compute f_k v_k takes {e - s} seconds')

    # Find the price of Put option over the Moneyness and Time to Maturity plane
    f_k_times_v_k = v_k_matrix * f_k_matrix
    k_sigma_f_k_v_k = f_k_times_v_k.sum(axis=1)
    k_sigma_f_k_v_k_matrix = k_sigma_f_k_v_k.reshape((len(moneyness), len(time_to_maturity)))
    put_price_matrix = matlib.repmat(np.exp(-r * time_to_maturity), len(moneyness), 1) * k_sigma_f_k_v_k_matrix
    return put_price_matrix


def compare_put_with_bs(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0, moneyness,
                        time_to_maturity):
    put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                              s_0, v_0, moneyness, time_to_maturity)
    impl_vol_matrix = np.zeros((len(moneyness), len(time_to_maturity)))
    BS_Price = np.zeros((len(moneyness), len(time_to_maturity)))

    for k in range(0, len(moneyness)):
        for t in range(0, len(time_to_maturity)):
            T = time_to_maturity[t]
            K = s_0 * np.exp(moneyness[k] * sigma * np.sqrt(T))
            impl_vol = implvol_put(put_price_matrix[k, t], s_0, K, T, r, 0)
            BS_Price[k, t] = put(s_0, K, sigma, T, r, 0)
            impl_vol_matrix[k, t] = impl_vol
    a, b = BS_Price.shape
    array1 = ((BS_Price - put_price_matrix) ** 2).reshape(a * b, )
    mean_square_error = np.average(array1)
    print(mean_square_error)
    df1 = pd.DataFrame(put_price_matrix)
    df2 = pd.DataFrame(BS_Price)
    df1.to_csv('./output/put_price_affine.csv')
    df2.to_csv('./output/put_price_bs.csv')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X, Y = np.meshgrid(moneyness, time_to_maturity)
    Z = impl_vol_matrix.reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(0, 0.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlabel('log-moneyness')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied vol')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(fname='./output/impl_vol_surface')
    plt.show()


def rho_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                                moneyness, time_to_maturity):
    rho_dict = {}
    for rho in rho:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j,
                                                  sigma_j, s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        rho_dict[round(rho, 1)] = put_price_vec
    rho_df = pd.DataFrame.from_dict(rho_dict)
    rho_df['log-moneyness'] = np.array(moneyness)
    rho_df.set_index('log-moneyness', inplace=True, drop=True)
    rho_df.to_csv('./output/rho_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = rho_df.iloc[i, :].plot()
        ax.set_xlabel('rho')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_rho')
    return 0


def kappa_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                                moneyness, time_to_maturity):
    kappa_dict = {}
    for k in kappa:
        put_price_matrix = affine_model_put_price(r, sigma, rho, k, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        kappa_dict[str(k)] = put_price_vec
    kappa_df = pd.DataFrame.from_dict(kappa_dict)
    kappa_df['log-moneyness'] = np.array(moneyness)
    kappa_df.set_index('log-moneyness', inplace=True, drop=True)
    kappa_df.to_csv('./output/kappa_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = kappa_df.iloc[i, :].plot()
        ax.set_xlabel('kappa')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_kappa_positive_rho')
    return 0


def xi_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                             moneyness, time_to_maturity):
    xi_dict = {}
    for xi in xi:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        xi_dict[round(xi, 2)] = put_price_vec
    xi_df = pd.DataFrame.from_dict(xi_dict)
    xi_df['log-moneyness'] = np.array(moneyness)
    xi_df.set_index('log-moneyness', inplace=True, drop=True)
    xi_df.to_csv('./output/xi_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = xi_df.iloc[i, :].plot()
        ax.set_xlabel('xi')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_xi_negative')
    return 0


def lambda1_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                                  moneyness, time_to_maturity):
    lambda1_dict = {}
    for lambda_1 in lambda_1:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        lambda1_dict[round(lambda_1, 2)] = put_price_vec
    lambda1_df = pd.DataFrame.from_dict(lambda1_dict)
    lambda1_df['log-moneyness'] = np.array(moneyness)
    lambda1_df.set_index('log-moneyness', inplace=True, drop=True)
    lambda1_df.to_csv('./output/lambda1_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = lambda1_df.iloc[i, :].plot()
        ax.set_xlabel('lambda1')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_lambda1')
        plt.show()
    return 0


def muj_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                              moneyness, time_to_maturity):
    muj_dict = {}
    for mu_j in mu_j:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        muj_dict[round(mu_j, 2)] = put_price_vec
    muj_df = pd.DataFrame.from_dict(muj_dict)
    muj_df['log-moneyness'] = np.array(moneyness)
    muj_df.set_index('log-moneyness', inplace=True, drop=True)
    muj_df.to_csv('./output/muj_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = muj_df.iloc[i, :].plot()
        ax.set_xlabel('mu_j')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_muj')
        plt.show()
    return 0


def sigmaj_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                                 moneyness, time_to_maturity):
    sigmaj_dict = {}
    for sigma_j in sigma_j:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        sigmaj_dict[round(sigma_j, 2)] = put_price_vec
    sigmaj_df = pd.DataFrame.from_dict(sigmaj_dict)
    sigmaj_df['log-moneyness'] = np.array(moneyness)
    sigmaj_df.set_index('log-moneyness', inplace=True, drop=True)
    sigmaj_df.to_csv('./output/sigmaj_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = sigmaj_df.iloc[i, :].plot()
        ax.set_xlabel('sigma_j')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_sigma_j')
        plt.show()
    return 0


def mv_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                             moneyness, time_to_maturity):
    mv_dict = {}
    for m_v in m_v:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        mv_dict[round(m_v, 2)] = put_price_vec
    mv_df = pd.DataFrame.from_dict(mv_dict)
    mv_df['log-moneyness'] = np.array(moneyness)
    mv_df.set_index('log-moneyness', inplace=True, drop=True)
    mv_df.to_csv('./output/mv_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = mv_df.iloc[i, :].plot()
        ax.set_xlabel('m_v')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_mv')
        plt.show()
    return 0


def rhoj_impact_stochastic_vol(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j, s_0, v_0,
                              moneyness, time_to_maturity):
    rhoj_dict = {}
    for rho_j in rho_j:
        put_price_matrix = affine_model_put_price(r, sigma, rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j, rho_j, sigma_j,
                                                  s_0, v_0, moneyness, time_to_maturity)
        put_price_vec = put_price_matrix.reshape((put_price_matrix.shape[0] * put_price_matrix.shape[1],))
        rhoj_dict[round(rho_j, 2)] = put_price_vec
    rhoj_df = pd.DataFrame.from_dict(rhoj_dict)
    rhoj_df['log-moneyness'] = np.array(moneyness)
    rhoj_df.set_index('log-moneyness', inplace=True, drop=True)
    rhoj_df.to_csv('./output/rhoj_against_put_price.csv')
    moneyness_dict = {0: 'otm', 5: 'atm', -1: 'itm'}

    for i in [0, 5, -1]:
        plt.figure()
        ax = rhoj_df.iloc[i, :].plot()
        ax.set_xlabel('rho_j')
        ax.set_ylabel('price')
        moneyness_name = moneyness_dict[i]
        plt.savefig(fname=f'./output/{moneyness_name}_put_price_rhoj')
        plt.show()
    return 0


def main_test(test_number):
    if test_number == 'bs':  # compare with BS formula
        compare_put_with_bs(r=0.02, sigma=0.3, rho=0, kappa=0, xi=0,
                            lambda_0=0, lambda_1=0, m_v=0, mu_j=0, rho_j=0, sigma_j=0,
                            s_0=100, v_0=0.3*0.3,
                            moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.linspace(0.2, 1, 5))
    if test_number == 'rho':  # stochastic vol and impact analysis of rho
        rho_impact_stochastic_vol(r=0.02, sigma=0.15, rho=np.linspace(-1, 1, 11), kappa=2, xi=0.5,
                                  lambda_0=0, lambda_1=0, m_v=0, mu_j=0, rho_j=0, sigma_j=0,
                                  s_0=100, v_0=0.2*0.2,
                                  moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'kappa':  # stochastic vol and impact analysis of kappa
        kappa_impact_stochastic_vol(r=0.02, sigma=0.15, rho=0.5, kappa=np.linspace(0, 10, 11), xi=0.5,
                                    lambda_0=0, lambda_1=0, m_v=0, mu_j=0, rho_j=0, sigma_j=0,
                                    s_0=100, v_0=0.2*0.2,
                                    moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'xi':  # stochastic vol and impact analysis of xi
        xi_impact_stochastic_vol(r=0.02, sigma=0.15, rho=-0.5, kappa=2, xi=np.linspace(0, 2, 21),
                                 lambda_0=0, lambda_1=0, m_v=0, mu_j=0, rho_j=0, sigma_j=0,
                                 s_0=100, v_0=0.2*0.2,
                                 moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'lambda1':  # stochastic vol and jump impact analysis of lambda1
        lambda1_impact_stochastic_vol(r=0.02, sigma=0.15, rho=0.5, kappa=2, xi=0.5,
                                      lambda_0=0.1, lambda_1=np.linspace(0, 1, 11), m_v=0, mu_j=0.1, rho_j=0, sigma_j=0.1,
                                      s_0=100, v_0=0.2*0.2,
                                      moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'mu_j':  # stochastic vol and jump impact analysis of mu_j
        muj_impact_stochastic_vol(r=0.02, sigma=0.15, rho=0.5, kappa=2, xi=0.5,
                                  lambda_0=0.1, lambda_1=0.1, m_v=0, mu_j=np.linspace(0, 1, 11), rho_j=0, sigma_j=0.1,
                                  s_0=100, v_0=0.2*0.2,
                                  moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'sigma_j':  # stochastic vol and jump impact analysis of mu_j
        sigmaj_impact_stochastic_vol(r=0.02, sigma=0.15, rho=0.5, kappa=2, xi=0.5,
                                     lambda_0=0.1, lambda_1=0.1, m_v=0, mu_j=0.1, rho_j=0, sigma_j=np.linspace(0, 1, 11),
                                     s_0=100, v_0=0.2*0.2,
                                     moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'm_v':  # stochastic vol and jump impact analysis of m_v
        mv_impact_stochastic_vol(r=0.02, sigma=0.15, rho=0.5, kappa=2, xi=0.5,
                                 lambda_0=0.1, lambda_1=0.1, m_v=np.linspace(0, 1, 11), mu_j=0.1, rho_j=0.1, sigma_j=0.1,
                                 s_0=100, v_0=0.2*0.2,
                                 moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))

    if test_number == 'rho_j':  # stochastic vol and jump impact analysis of rhoj
        rhoj_impact_stochastic_vol(r=0.02, sigma=0.15, rho=-0.5, kappa=2, xi=0.5,
                                   lambda_0=0.1, lambda_1=0.1, m_v=0.1, mu_j=0.1, rho_j=np.linspace(-1, 1, 11), sigma_j=0.1,
                                   s_0=100, v_0=0.2*0.2,
                                   moneyness=np.linspace(-1, 1, 11), time_to_maturity=np.array([1.]))


if __name__ == '__main__':
    main_test('rho_j')