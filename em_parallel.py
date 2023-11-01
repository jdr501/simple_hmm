import initialization as ini
import numpy as np
from expectation import *
from optimization import *
from functools import partial
import multiprocessing

def parallel_run(data_frame, lags, regimes, random_b_matrix, max_iteration, tolerance, beta=None):
    draws = 10
    k_vars = data_frame.shape[1]
    b0_list = []
    for i in range(draws):
        b0 = np.random.normal(0, 0.1, size=(k_vars, k_vars))
        b0_list.append(b0)\
            result= em_algorithm(data_frame, lags, regimes, random_b_matrix,
                                                      max_iteration, tolerance, beta=beta)
        d = 0
        with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
            for result in pool.map(draw, b0_list):
                results.update({f'{d}': result})
                d += 1
 N = pool.map(partial(func, b=second_arg), a_args)
file_path = "/results.json"

json.dump(results, codecs.open(file_path, 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True,
          indent=4)


def em_algorithm(data_frame, lags, regimes, random_b_matrix, max_iteration, tolerance, beta=None):
    initial_class = ini.Initializes(data_frame, lags, regimes, random_b_matrix, beta=beta)
    initial_class.run_initialization()

    residuals = initial_class.u_hat
    sigma = initial_class.sigmas
    transition_prob = initial_class.p
    start_prob = initial_class.e_0_0
    b_mat = initial_class.initial_b
    delta_yt = initial_class.delta_y_t
    zt = initial_class.z_t_1
    vec_b = b_mat.T.reshape(-1, 1)
    vec_lambda = np.ones([b_mat.shape[0] * regimes, 1])
    initial_guess = np.concatenate((vec_b, vec_lambda), axis=1)

    llf = []
    itr = 1
    while itr <= max_iteration or (itr > 2 and (abs(llf[-1] - llf[-2]) / abs(llf[-2])) < tolerance):
        loglikelihood, \
            start_prob, \
            smoothed_prob, \
            joint_smoothed_prob = expectation_run(sigma, residuals, start_prob, transition_prob)
        llf.append(loglikelihood)

        transition_prob, \
            initial_guess, \
            b_matrix, \
            lam_m, sigma, \
            wls_params, residuals = optimization_run(smoothed_prob, joint_smoothed_prob,
                                                     initial_guess, residuals, zt, delta_yt)
        itr += 1

    em_result = {'likelihood': loglikelihood,
                 'start_prob': start_prob,
                 'smoothed_prob': smoothed_prob,
                 'joint_smoothed_prob': joint_smoothed_prob,
                 'transition_prob': transition_prob,
                 'initial_guess': initial_guess,
                 '': b_matrix,
                 'b_matrix': lam_m,
                 'sigma': sigma,
                 'wls_params': wls_params,
                 'residuals': residuals,
                 'llf': llf}

    return em_result
