# junk from expectation


def hamilton_filter(start_prob, transition_prob, conditional_density):
    regimes, obs = conditional_density.shape
    predicted_prob = np.zeros([regimes, obs], dtype=np.float64)
    filtered_prob = np.zeros([regimes, obs + 1], dtype=np.float64)
    ll_array = np.zeros(obs, dtype=np.float64)

    for t in range(obs + 1):
        if t == 0:
            filtered_prob[:, [t]] = start_prob
        else:
            predicted_prob[:, [t - 1]] = transition_prob @ filtered_prob[:, [t - 1]]
            filtered_prob[:, [t]] = conditional_density[:, [t - 1]] * predicted_prob[:, [t - 1]]
            ll_array[t - 1] = np.sum(filtered_prob[:, [t]])
            filtered_prob[:, [t]] = filtered_prob[:, [t]] / ll_array[t - 1]

    loglikelihood = np.sum(np.log(ll_array))

    return predicted_prob, filtered_prob, loglikelihood


def kim_smoother(filtered_prob, transition_prob):
    regimes, t_length = filtered_prob.shape
    smoothed_prob = np.zeros(filtered_prob.shape, dtype=np.float64)
    joint_smoothed_prob = np.zeros([regimes * regimes, t_length - 1], dtype=np.float64)

    for t in range(t_length - 1, -1, -1):  # iteration going from T...0
        if t == t_length - 1:
            smoothed_prob[:, [t]] = filtered_prob[:, [t]]
        else:

            smoothed_prob[:, [t]] = (transition_prob.T @
                                     (smoothed_prob[:, [t + 1]] /
                                      (transition_prob @ filtered_prob[:, [t]])
                                      )) * filtered_prob[:, [t]]
    for t in range(t_length - 1):
        joint_smoothed_prob[:, [t]] = transition_prob.T.flatten('F').reshape(-1, 1) * \
                                      np.kron(
                                          (smoothed_prob[:, [t + 1]] / (transition_prob @ filtered_prob[:, [t]])),
                                          filtered_prob[:, [t]])
    smoothed_prob = smoothed_prob
    start_prob = smoothed_prob[:, [0]] / np.sum(smoothed_prob[:, [0]])
    smoothed_prob = smoothed_prob[:, 1:]
    return start_prob, smoothed_prob, joint_smoothed_prob




def estimate_transition_prob(smoothed_prob, joint_smoothed_prob):
    regimes = smoothed_prob.shape[0]
    print(f'this is the number of regimes {regimes}')
    print('============')
    print(np.sum(joint_smoothed_prob[0], 0) / np.sum(joint_smoothed_prob[0:1]))
    print(np.sum(joint_smoothed_prob[1], 0) / np.sum(joint_smoothed_prob[0:1]))
    print(np.sum(joint_smoothed_prob[0:1]))
    print('============')
    vec_p_trans = np.sum(joint_smoothed_prob, 1).reshape(-1, 1) / np.kron(np.ones([regimes, 1]),
                                                                          np.sum(smoothed_prob, 1).reshape(-1, 1))

    # print(f'this is vec_p_trans {vec_p_trans}')
    # vec_p_trans[0, 0] = vec_p_trans[0, 0] / (vec_p_trans[0, 0] + vec_p_trans[1, 0])
    # vec_p_trans[1, 0] = vec_p_trans[0, 0] / (vec_p_trans[0, 0] + vec_p_trans[1, 0])
    # vec_p_trans[2, 0] = vec_p_trans[2, 0] / (vec_p_trans[2, 0] + vec_p_trans[3, 0])
    # vec_p_trans[3, 0] = vec_p_trans[3, 0] / (vec_p_trans[2, 0] + vec_p_trans[3, 0])

    transition_prob = vec_p_trans.reshape(regimes, regimes)
    for regime in range(regimes):
        print(regime)
        if np.sum(transition_prob[regime, :]) < 1.000:
            print(f'regime {regime + 1}  sum is greater than one {np.sum(transition_prob[regime, :])}')
    transition_prob = transition_prob / np.sum(transition_prob, 1).reshape(-1, 1)
    transition_prob = np.maximum(transition_prob, 1e-20)
    return transition_prob
