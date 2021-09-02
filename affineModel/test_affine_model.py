import numpy as np
import pandas as pd
from typing import AnyStr
from affineModel.utils import dateUtils
from affineModel.utils.blackscholes import implvol_put, implvol_call
from affineModel.affine_model_with_jump import AffineModelwithJump
import scipy.optimize as opt
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator
from affineModel.utils.blackscholes import implvol_put


def get_historical_price(date: AnyStr, option_type: AnyStr):
    """
    :param date: extract given date data
    :param optiontype: extract given option type data
    :return: data frame
    """
    option_df = pd.read_csv('./data/affine_option.csv')
    option_price_df = pd.read_csv('./data/affine_option_price.csv')
    stock_price_df = pd.read_csv('./data/affine_stock_price.csv')

    test_df = option_df.merge(option_price_df[['date', 'contract', 'price']],
                              left_on=['option_code', 'tradedate'],
                              right_on=['contract', 'date'],
                              how='left')

    test_df = test_df[['tradedate', 'option_code', 'last_tradedate',
                       'call_put', 'strike_price', 'price', 'underlying_code']]

    test_df = test_df.merge(stock_price_df[['date', 'close']],
                            left_on='tradedate',
                            right_on='date',
                            how='left')

    test_df = test_df[['tradedate', 'option_code', 'call_put', 'strike_price', 'last_tradedate',
                       'price', 'underlying_code', 'close']]

    return test_df[(test_df['tradedate'] == date) & (test_df['call_put'] == option_type)]


def get_option_data(date: AnyStr, option_type: AnyStr):
    """
    get the stock price, moneyness vector, time to maturity vector, and option price by given date and option_type
    put_price_vec: [pv(m0, t0), pv(m0, t1), ..., pv(m0, tn), pv(m1, t0), ..., pv(m1, tn),...pv(mn,tn)]
    """
    option_df = get_historical_price(date, option_type)
    option_df.to_csv(f'./output/market option data for {date}.csv')
    option_df['moneyness'] = np.log(option_df['strike_price'] * 1. / option_df['close'])
    option_df['number_of_days'] = option_df.apply(lambda x:
                                                  dateUtils.business_day_count(x['tradedate'], x['last_tradedate']),
                                                  axis=1)
    option_df['time_to_maturity'] = option_df['number_of_days']
    opt_market_df = option_df.reset_index(drop=True)
    s1 = opt_market_df.groupby('moneyness').count()['price'] == 4
    opt_market_df = opt_market_df.set_index(keys='moneyness', drop=False, append=False)
    opt_market_df.rename(columns={'moneyness': 'moneyness_col'}, inplace=True)
    opt_market_df = opt_market_df[s1].sort_values(by=['moneyness_col', 'time_to_maturity'])
    moneyness = opt_market_df['moneyness_col'].drop_duplicates(keep='last').values
    time_to_maturity = opt_market_df['time_to_maturity'].drop_duplicates(keep='last').values / 244.
    put_price = opt_market_df['price'].values
    stock_price = opt_market_df['close'].values[0]
    T_largest = time_to_maturity[-1]

    return stock_price, T_largest, moneyness, time_to_maturity, put_price


def get_option_vol_surface(date: AnyStr, option_type: AnyStr):
    """
    get given date and option type market vol surface using blackshcoles implied vol
    """
    stock_price, T_largest, moneyness, time_to_maturity, put_price = get_option_data(date, option_type)
    put_price_matrix = np.array(put_price).reshape((len(moneyness), len(time_to_maturity)))
    r = 0.02
    impl_vol_matrix = np.zeros((len(moneyness), len(time_to_maturity)))
    for k in range(0, len(moneyness)):
        for t in range(0, len(time_to_maturity)):
            T = time_to_maturity[t]
            K = stock_price * np.exp(moneyness[k])
            if option_type == "认沽":
                impl_vol = implvol_put(put_price_matrix[k, t], stock_price, K, T, r, 0)
            else:
                impl_vol = implvol_call(put_price_matrix[k, t], stock_price, K, T, r, 0)
            impl_vol_matrix[k, t] = impl_vol

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    X, Y = np.meshgrid(moneyness, time_to_maturity)
    Z = impl_vol_matrix.reshape(X.shape)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(0.1, 0.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('log-moneyness')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied vol')
    plt.savefig(fname='./output/impl_vol_surface_20210701')
    plt.show()


def calibration_no_jump(val_str):
    s_0, T_largest, moneyness, time_to_maturity, put_price = get_option_data(val_str, '认沽')
    sigma_mean = 0.15
    sigma_0 = 0.2
    r = 0.02
    k_upper = 100
    a = -5
    b = 5
    days_in_year = 245
    case1 = AffineModelwithJump(s_0=s_0, sigma_mean=sigma_mean, sigma_0=sigma_0, r=r,
                                k_upper=k_upper, a=a, b=b, T=T_largest, days_in_year=days_in_year,
                                moneyness=moneyness, time_to_maturity=time_to_maturity)
    rho = np.linspace(-1., 1, 11)
    kappa = np.linspace(0, 1.5, 4)
    xi = np.linspace(0, 1, 11)
    print(case1.put_option_stochastic_vol_no_jump([0.6, 0.01, 0.3 ], put_price))
    param = []
    mean_square_error = []
    for i in rho:
        for j in kappa:
            for x in xi:
                theta = [i, j, x]
                mse = case1.put_option_stochastic_vol_no_jump(theta, put_price)
                param.append(theta)
                mean_square_error.append(mse)
                print(f'{i, j, x} parameters finish calculation')

    no_jump_df = pd.DataFrame({'param': param, 'mean_square_error': mean_square_error})
    no_jump_df.to_csv(f'./output/mse with no stock and vol jump as of date {val_str}.csv')


def calibration_stock_jump(val_str):
    s_0, T_largest, moneyness, time_to_maturity, put_price = get_option_data(val_str, '认沽')
    sigma_mean = 0.15
    sigma_0 = 0.2
    r = 0.02
    k_upper = 100
    a = -5
    b = 5
    days_in_year = 245
    case1 = AffineModelwithJump(s_0=s_0, sigma_mean=sigma_mean, sigma_0=sigma_0, r=r,
                                k_upper=k_upper, a=a, b=b, T=T_largest, days_in_year=days_in_year,
                                moneyness=moneyness, time_to_maturity=time_to_maturity)
    rho = np.linspace(0, 1, 6)
    kappa = 0.1
    xi = np.linspace(0, 1, 6)
    mu_j = np.linspace(-1, 0, 6)
    sigma_j = np.linspace(0, 1, 6)
    param = []
    mean_square_error = []
    for i in rho:
        for x in xi:
            for j in mu_j:
                for sigma in sigma_j:
                    theta = [i, kappa, x, j, sigma]
                    mse = case1.put_option_stocastic_vol_stock_jump(theta, put_price)
                    param.append(theta)
                    mean_square_error.append(mse)
                    print(f'{i, x, j, sigma}  finish ')
    no_jump_df = pd.DataFrame({'param': param, 'mean_square_error': mean_square_error})
    no_jump_df.to_csv(f'./output/mse with stock jump as of date {val_str}.csv')


def calibration_stock_vol_jump(val_str):
    s_0, T_largest, moneyness, time_to_maturity, put_price = get_option_data(val_str, '认沽')
    sigma_mean = 0.15
    sigma_0 = 0.2
    r = 0.02
    k_upper = 100
    a = -5
    b = 5
    days_in_year = 245
    case1 = AffineModelwithJump(s_0=s_0, sigma_mean=sigma_mean, sigma_0=sigma_0, r=r,
                                k_upper=k_upper, a=a, b=b, T=T_largest, days_in_year=days_in_year,
                                moneyness=moneyness, time_to_maturity=time_to_maturity)
    rho = np.linspace(0, 1, 6)
    kappa = 0.01
    xi = np.linspace(0, 1, 6)
    mu_j = np.linspace(-1, 0, 6)
    sigma_j = np.linspace(0, 1, 6)
    m_v = np.linspace(-2.1, 2.1, 6)
    rho_j = np.linspace(-1.1, 1.1, 6)
    # print(case1.put_option_stocastic_vol_stock_jump([0.9, 0.01, 0.43, 1.26, -0.51, 0.5, 1.5], put_price))
    param = []
    mean_square_error = []
    for i in rho:
        for x in xi:
            for muj in mu_j:
                for sigmaj in sigma_j:
                    for mv in m_v:
                        for rhoj in rho_j:
                            theta = [i, kappa, x, mv, muj, rhoj, sigmaj]
                            mse = case1.put_option_stocastic_vol_all_jump(theta, put_price)
                            param.append(theta)
                            mean_square_error.append(mse)
                            print(f'{i, kappa, x, mv, muj, rhoj, sigmaj}  finish ')
    no_jump_df = pd.DataFrame({'param': param, 'mean_square_error': mean_square_error})
    no_jump_df.to_csv(f'./output/mse with stock and vol jump as of date {val_str}.csv')


val_str = '2021-08-02'
get_option_vol_surface(val_str, '认沽')
calibration_no_jump(val_str)
calibration_stock_jump(val_str)
calibration_stock_vol_jump(val_str)

# parameters:[rho, kappa, xi]
# rho = np.linspace(-1., 1, 21)
# kappa = 2
# xi = np.linspace(0, 1, 11)
# s = time.time()
# param = []
# mean_square_error = []
# for i in rho:
#     for j in kappa:
#         for x in xi:
#             theta = [i, j, x]
#             mse = case1.put_option_stochastic_vol_no_jump(theta, put_price)
#             param.append(theta)
#             mean_square_error.append(mse)
#             print(f'{i, j, x} parameters finish calculation')
#
# no_jump_df = pd.DataFrame({'param': param, 'mean_square_error': mean_square_error})
# no_jump_df.to_csv(f'mse with no stock and vol jum as of date {val_str}.csv')


# theta_guess_1 = [-0.5, 2, 0.5]
# print('start to calibration the model without stock or vol jump')
# s = time.time()
# results1 = opt.minimize(fun=case1.put_option_stochastic_vol_no_jump, x0=theta_guess_1, args=put_price)  # tol=1e-5
# e = time.time()
# print(f'calibration takes {e-s} seconds')
# print(results1)
# print("calibration parameters: ", results1.x)

# """
# # parameters:[rho, kappa, xi, lambda_0, lambda_1, m_v, mu_j. rho_j , sigma_j
# theta_guess_2 = [0.2, 2, 0.5, 0.1, 0.1, 0.2, 0.2, -0.1, 0.2]
# print('start to calibration the model without stock and vol jump')
# s = time.time()
# results2 = opt.minimize(fun=case1.put_option_stocastic_vol_stock_jump, x0=theta_guess_2, args=put_price, tol=1e-3)  # tol=1e-5
# e = time.time()
# print(f'calibration takes {e-s} seconds')
# print(results2)
# print("calibration parameters: ", results2.x)
#
# """
# #parameters:[lambda_0, lambda_1, m_v, mu_j. rho_j , sigma_j
# theta_guess_3 = 0.1
# print('start to calibration the model without stock and vol jump')
# s = time.time()
# results2 = opt.minimize(fun=case1.put_option_stocastic_only_jump, x0=theta_guess_3, args=put_price, tol=1e-3)  # tol=1e-5
# e = time.time()
# print(f'calibration takes {e-s} seconds')
# print(results2)
# print("calibration parameters: ", results2.x)
