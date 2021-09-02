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

r = 0.02
sigma = 0.3
rho = 0
kappa = 2  # 2
xi = 0.5

# jump intensity paramete
lambda_0 = 0  # constant part for jump intensity, lambda_s
lambda_1 = 0  # coefficient on volatility for jump intensity lambda_v

# jump size paramter
m_v = 0   # 0.2 * 0.2 vol jumps with exponential distribution, sqrt(m_v) is the average of vol jump in terms of SD
mu_j = 0
rho_j = 0  # -0.2  # correlation between stock price jump and vol jump
sigma_j = 0  # 0.1  # once jump arrives. stock price jumps with log normal distribution with the sigma_j int terms of SD

# state variable
s_0 = 100  # spot price
v_0 = sigma * sigma  # variance
k_upper = 100
k = np.arange(0, 101)  # k from 0 to 100
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
moneyness = np.linspace(-1, 1, 11)
time_to_maturity = np.linspace(0.2, 1, 5)
# time_to_maturity = np.array([0.08163, 0.16327, 0.23673, 0.48163])
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

# Proof of the code with the implied volatility
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
array1 = ((BS_Price - put_price_matrix) ** 2).reshape(a*b, )
mean_square_error = np.average(array1)
print(BS_Price)
print(put_price_matrix)
print(mean_square_error)
print(moneyness)
print(time_to_maturity)

df1 = pd.DataFrame(put_price_matrix)
df2 = pd.DataFrame(BS_Price)
df1.to_csv('put_price_affine.csv')
df2.to_csv('put_price_bs.csv')
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

plt.savefig(fname='impl_vol_surface')
plt.show()













