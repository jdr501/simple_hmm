import warnings
from statsmodels.tools.sm_exceptions import EstimationWarning
import numpy as np
from scipy.optimize import minimize
from expectation import normal_cond_dens

def optimization_run(smoothed_prob, joint_smoothed_prob,
                     initial_guess, residuals, zt, delta_yt):
    transition_prob, start_prob = estimate_transition_prob(smoothed_prob, joint_smoothed_prob)

    initial_guess, \
        b_matrix, \
        lam_m, \
        sigma = numerical_opt_b_lambda(initial_guess, residuals, smoothed_prob)
    wls_params, residuals = wls_estimate(sigma, zt, delta_yt, smoothed_prob)

    return transition_prob, start_prob, initial_guess, b_matrix, lam_m, sigma, wls_params, residuals


def estimate_transition_prob(smoothed_prob, joint_smoothed_prob):
    regime_transition = em_regime_transition(smoothed_prob, joint_smoothed_prob)
    trans_prob_temp = regime_transition_matrix(regime_transition, 2)
    start_prob = initial_probabilities(trans_prob_temp).reshape(-1, 1)
    transition_prob = trans_prob_temp[:, :, 0]
    return transition_prob, start_prob


def sigma_likelihood(x, residuals, smoothed_prob):
    """
    :param x: must be a column vector of guesses
    :param residuals:
    :param smoothed_prob:
    :return:
    """

    k_vars, obs = residuals.shape
    regimes = smoothed_prob.shape[0]
    b_matrix, lam_m = reconstitute_b_lambda(x, k_vars, regimes)
    sigma = sigma_estimate(b_matrix, lam_m)
    condi_dens = normal_cond_dens(sigma,residuals)
    weighted_dens  = condi_dens*smoothed_prob
    # weighted squared sum of residuals
    weighted_sum_res = np.zeros([k_vars, k_vars, regimes])
    for regime in range(regimes):
        temp_wt_sum = 0
        for t in range(obs):
            temp_wt_sum = temp_wt_sum + \
                          smoothed_prob[regime, t] * \
                          residuals[:, [t]] @ residuals[:, [t]].T
        weighted_sum_res[:, :, regime] = temp_wt_sum

    # likelihood terms
    b_matrix_trans_inv = np.linalg.pinv(b_matrix.T)
    b_matrix_inv = np.linalg.pinv(b_matrix)

    term_1 = obs * np.log(abs(np.linalg.det(b_matrix))) / 2 # TODO change back to original sum(smoothed_prob[0, :])
    term_2 = np.trace(b_matrix_trans_inv @ b_matrix_inv @ weighted_sum_res[:, :, 0]) / 2
    term_3 = 0
    term_4 = 0
    for regime in range(regimes):
        lam_inv = np.linalg.pinv(lam_m[:, :, regime - 1])

        term_3 += np.sum(smoothed_prob[regime, :]) * np.log(np.linalg.det(lam_m[:, :, regime - 1])) / 2
        term_4 += np.trace(b_matrix_trans_inv @ lam_inv @ b_matrix_inv @ weighted_sum_res[:, :, regime]) / 2
    negative_likelihood = term_1 + term_2 + term_3 + term_4

    return - np.sum(weighted_dens)


def reconstitute_b_lambda(x, k_vars, regimes):
    x = x.reshape(-1, 1)
    lam_m = np.zeros([k_vars, k_vars, regimes - 1])
    b_matrix = x[:k_vars * k_vars, [0]].reshape(k_vars, k_vars).T
    identity_mat = np.eye(k_vars)
    for regime in range(regimes - 1):
        if regime == 0:
            start = k_vars * k_vars
            end = start + k_vars
        else:
            start = end.copy()
            end = start + k_vars
        lam_m[:, :, regime] = identity_mat * x[start:end, [0]]

    return b_matrix, lam_m


def sigma_estimate(b_matrix, lam_m):
    regimes = 1 + lam_m.shape[2]
    k_vars = b_matrix.shape[0]
    sigma = np.zeros([k_vars, k_vars, regimes])
    for regime in range(regimes):
        if regime == 0:
            sigma[:, :, 0] = b_matrix @ b_matrix.T
        else:
            sigma[:, :, regime] = b_matrix @ lam_m[:, :, regime - 1] @ b_matrix.T
    return sigma


def numerical_opt_b_lambda(initial_guess, residuals, smoothed_prob):
    regimes = smoothed_prob.shape[0]
    input_args = (residuals, smoothed_prob)
    k_vars = residuals.shape[0]
    bound_list = []
    for i in range(len(initial_guess)):
        if i < k_vars ** 2:
            bound_list.append((None, None))
        else:
            bound_list.append((0.01, None))
    bound_list = tuple(bound_list)
    # input_args = residuals, smoothed_prob
    opt ={'maxiter': 10000}
    b_lambda_result = minimize(sigma_likelihood,
                               initial_guess,
                               args=input_args,
                               tol=1e-5,
                               method='L-BFGS-B',
                               bounds=bound_list,
                               options=opt)

    print(b_lambda_result.message)
    print(f'this is the function value: {b_lambda_result.fun}')
    b_matrix, lam_m = reconstitute_b_lambda(b_lambda_result.x, k_vars, regimes)
    sigma = sigma_estimate(b_matrix, lam_m)
    return b_lambda_result.x, b_matrix, lam_m, sigma


def wls_estimate(sigma, zt, delta_yt, smoothed_prob):
    regimes, obs = smoothed_prob.shape
    sigma_inv = np.zeros(sigma.shape)

    for regime in range(regimes):
        sigma_inv[:, :, regime] = np.linalg.pinv(sigma[:, :, regime])

    tsum_denom = 0
    m_sum_denom = 0
    tsum_numo = 0
    m_sum_numo = 0
    for t in range(obs):
        for regime in range(regimes):
            m_sum_denom += np.kron(smoothed_prob[regime, t] * zt[:, [t]] @ zt[:, [t]].T, sigma_inv[:, :, regime])
            m_sum_numo += np.kron(smoothed_prob[regime, t] * zt[:, [t]], sigma_inv[:, :, regime]) @ delta_yt[:, [t]]
        tsum_denom += m_sum_denom
        tsum_numo += m_sum_numo

    wls_params = np.linalg.pinv(tsum_denom) @ tsum_numo

    residuals = residuals_estimate(delta_yt, zt, wls_params)

    return wls_params, residuals


def residuals_estimate(delta_yt, zt, wls_params):
    k_vars, obs = delta_yt.shape
    residuals = np.zeros(delta_yt.shape)

    for t in range(obs):
        residuals[:, [t]] = delta_yt[:, [t]] - np.kron(zt[:, [t]].T, np.eye(k_vars)) @ wls_params
    return residuals


def em_regime_transition(smoothed_marginal_probabilities, smoothed_joint_probabilities):
    """
    EM step for regime transition probabilities
    """
    k_regimes = 2  ## test at two
    # Marginalize the smoothed joint probabilities to just S_t, S_{t-1} | T
    tmp = smoothed_joint_probabilities
    for i in range(tmp.ndim - 3):
        print(f'this is i {i}')
        tmp = np.sum(tmp, -2)
    smoothed_joint_probabilities = tmp
    # Transition parameters (recall we're not yet supporting TVTP here)

    regime_transition = np.zeros((k_regimes, 1))
    for i in range(k_regimes):  # S_{t_1}
        for j in range(k_regimes-1):  # S_t
            regime_transition[i, j] = (
                    np.sum(smoothed_joint_probabilities[j, i]) /
                    np.sum(smoothed_marginal_probabilities[i]))

        # It may be the case that due to rounding error this estimates
        # transition probabilities that sum to greater than one. If so,
        # re-scale the probabilities and warn the user that something
        # is not quite right
        delta = np.sum(regime_transition[i]) - 1
        if delta > 0:
            warnings.warn('Invalid regime transition probabilities'
                          ' estimated in EM iteration; probabilities have'
                          ' been re-scaled to continue estimation.',
                          EstimationWarning)
            regime_transition[i] /= 1 + delta + 1e-6
    return regime_transition


def regime_transition_matrix(regime_transition, k_regimes):
    """
    Construct the left-stochastic transition matrix

    Notes
    -----
    This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
    are time-varying transition probabilities, it will be shaped
    (k_regimes, k_regimes, nobs).

    The (i,j)th element of this matrix is the probability of transitioning
    from regime j to regime i; thus the previous regime is represented in a
    column and the next regime is represented by a row.

    It is left-stochastic, meaning that each column sums to one (because
    it is certain that from one regime (j) you will transition to *some
    other regime*).
    """
    #transition_matrix = regime_transition.reshape(k_regimes,k_regimes,1)
    if True:
        transition_matrix = np.zeros((k_regimes, k_regimes, 1), dtype=np.float64)
        transition_matrix[:-1, :, 0] = np.reshape(regime_transition,
                                                  (k_regimes - 1, k_regimes))
        transition_matrix[-1, :, 0] = (
                1 - np.sum(transition_matrix[:-1, :, 0], axis=0))

    return transition_matrix


def initial_probabilities(regime_transition):
    """
    Retrieve initial probabilities
    """
    if regime_transition.ndim == 3:
        regime_transition = regime_transition[..., 0]
    m = regime_transition.shape[0]
    A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
    try:
        probabilities = np.linalg.pinv(A)[:, -1]
    except np.linalg.LinAlgError:
        raise RuntimeError('Steady-state probabilities could not be'
                           ' constructed.')

    # Slightly bound probabilities away from zero (for filters in log
    # space)
    probabilities = np.maximum(probabilities, 1e-20)

    return probabilities
