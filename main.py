import numpy as np
import initialization as ini
import data
import json
import codecs
import multiprocessing
from expectation import *
from optimization import *
import matplotlib.pyplot as plt

st_prob = np.array([0.3, 0.7], dtype=np.float64).reshape(-1, 1)
trans_prob = np.array([[0.7, 0.3], [0.3, 0.7]])  # np.ones([2,2], dtype=np.float64)/2
con_dens = np.ones([2, 30], dtype=np.float64)

''' 
for i in range(10):
    predic_prob, flt_prob, ll = hamilton_filter(st_prob, trans_prob, con_dens)
    st_prob, smoothed_prob, joint_smoothed_prob = kim_smoother(flt_prob, trans_prob)
    print(flt_prob)
    print(predic_prob)
    print(smoothed_prob)
    print(joint_smoothed_prob)
    trans_prob = estimate_transition_prob(smoothed_prob, joint_smoothed_prob)
    print(trans_prob)
    '''
'''
for i in range(10):
    smoothed_joint_probabilities, smoothed_marginal_probabilities, ll = smoothed(st_prob, trans_prob, con_dens)
    regime_transition = em_regime_transition(smoothed_marginal_probabilities, smoothed_joint_probabilities)
    trans_prob_temp = regime_transition_matrix(regime_transition, 2)
    trans_prob = trans_prob_temp[:, :, 0]
    print(f'this is ll:{ll}')
    print(smoothed_marginal_probabilities)
    st_prob = initial_probabilities(trans_prob_temp).reshape(-1, 1)
'''
draws = 1
b0_list = []

for i in range(draws):
    b0 = np.random.normal(0.01, 4, size=(4, 4))
    b0_list.append(b0)




### EM algorithm starts here
beta = np.array([0, 0, 0, 1]).reshape(-1, 1)
initialize = ini.Initializes(data.df, 3, 2, b0_list[0], beta=beta)
initialize.run_initialization()

residuals = initialize.u_hat
sigma = initialize.sigmas

print(sigma)
regimes = 2
k_vars = 4


transition_prob = initialize.p
start_prob = initialize.e_0_0
b_mat = initialize.initial_b
delta_yt = initialize.delta_y_t
zt = initialize.z_t_1


initial_guess = np.ones([k_vars*k_vars+(regimes-1)*k_vars, 1])

initial_guess[:k_vars*k_vars, [0]] = b_mat.T.reshape(-1,1)

initial_guess = np.squeeze(initial_guess)
initial_guess = initial_guess + np.random.normal(1, 1, size=initial_guess.shape)
for i in range(200):
    loglikelihood, smoothed_prob, joint_smoothed_prob = expectation_run(sigma,
                                                                        residuals,
                                                                        start_prob,
                                                                        transition_prob)
    print(f'this is  ll :{loglikelihood}')
    transition_prob, \
        start_prob, \
        initial_guess, \
        b_matrix, \
        lam_m,\
        sigma,\
        wls_params, residuals = optimization_run(smoothed_prob, joint_smoothed_prob, initial_guess, residuals, zt, delta_yt)
    print(transition_prob)
print(residuals)
print(sigma[:, :, 0])
print(sigma[:, :, 1])
print(smoothed_prob)

import pandas as pd

df2 = pd.DataFrame()
df2['smoothed_prob_regime1'] = smoothed_prob[0, :]
df2['smoothed_prob_regime1'].plot()
plt.show()