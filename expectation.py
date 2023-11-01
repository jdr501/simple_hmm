import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tsa.regime_switching.markov_switching import cy_kim_smoother_log, cy_hamilton_filter_log


def expectation_run(sigma, residuals, start_prob, transition_prob):
    conditional_density = normal_cond_dens(sigma, residuals)
    joint_smoothed_prob, \
        smoothed_prob, \
        loglikelihood = smoothed(start_prob, transition_prob, conditional_density)

    return loglikelihood, smoothed_prob, joint_smoothed_prob


def normal_cond_dens(sigma, residuals):
    regimes = sigma.shape[2]
    obs = residuals.shape[1]
    conditional_density = np.zeros([regimes, obs])  # y_t|s_t = j conditional density of Y for a given state
    for r in range(regimes):
        conditional_density[r, :] = multivariate_normal(mean=None,
                                                        cov=sigma[:, :, r]).pdf(residuals.T).T

    return conditional_density


def smoothed(initial_prob, transition_prob, conditional_density):
    initial_prob = np.squeeze(initial_prob)
    regimes, obs = conditional_density.shape
    trans_prob = transition_prob.reshape(regimes, regimes, 1)
    cond_dens = np.zeros([regimes, 2, obs])
    for t in range(obs):
        cond_dens[:, 0, [t]] = conditional_density[:, [t]]
        cond_dens[:, 1, [t]] = conditional_density[:, [t]]

    filtered_marginal_probabilities, \
        predicted_joint_probabilities, \
        joint_loglikelihoods, \
        filtered_joint_probabilities, \
        predicted_joint_probabilities_log, \
        filtered_joint_probabilities_log = cy_hamilton_filter_log(initial_prob, trans_prob, cond_dens,
                                                                  0)  # model order doesn't matter in this case
    smoothed_joint_probabilities, \
        smoothed_marginal_probabilities = cy_kim_smoother_log(trans_prob, predicted_joint_probabilities_log,
                                                              filtered_joint_probabilities_log)

    return smoothed_joint_probabilities, smoothed_marginal_probabilities, sum(joint_loglikelihoods)
