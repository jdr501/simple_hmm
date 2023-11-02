import numpy as np
import scipy as sp
from statsmodels.tsa.vector_ar import vecm


class Initializes:

    def __init__(self, data, lag, regimes, b0, beta=None):
        """
        :param data: Input pandas dataset with time variable as the first column
        :param lag: Number of Delta_yt-1 lags
        :param regimes: Number of regimes
        :param b0: Random number  matrix to add to initial b_hat estimate
        :return: initial parameters
        """
        self.u_hat = None
        self.sigmas = None
        self.p = None
        self.e_0_0 = None
        self.lamda_m = None
        self.initial_b = None
        self.initial_theta_hat = None
        self.k = None
        self.t = None
        self.beta = beta
        self.delta_y_t = None
        self.z_t_1 = None
        self.data = data
        self.lag = lag
        self.regimes = regimes
        self.b0 = b0

    def data_matrix(self):
        model = vecm.VECM(endog=self.data,
                          k_ar_diff=self.lag,
                          coint_rank=1,
                          dates=self.data.index,
                          deterministic="colo")
        if self.beta is None:
            self.beta = model.fit().beta

        data_mat = vecm._endog_matrices(model.y, model.endog, model.exog, self.lag, "colo")
        self.k, self.t = data_mat[0].shape

        self.delta_y_t = data_mat[1]  # left-hand variables

        beta_trn_y_t_1 = np.array(
            [(self.beta.T @ data_mat[2][:, i].T)
             for i in range(self.t)]).T

        delta_y_t_lags = data_mat[3][: -(self.k + 2)]

        v0_v1 = data_mat[3][-(self.k + 2):-self.k]

        self.z_t_1 = np.vstack((v0_v1, beta_trn_y_t_1, delta_y_t_lags))

    def initial_params(self):
        z_t_1 = self.z_t_1
        z_len = z_t_1.shape[0]
        delta_y_t = self.delta_y_t
        k = self.k
        t = self.t
        denominator_col = z_len * k
        denominator = np.zeros([denominator_col, denominator_col])
        numerator = np.zeros([denominator_col, 1])
        identity_k = np.identity(k)

        for i in range(t):
            z = z_t_1[:, [i]]
            dy = delta_y_t[:, [i]]
            denominator += np.kron((z @ z.T), identity_k)
            numerator += np.kron(z, identity_k) @ dy

        denominator = np.linalg.pinv(denominator)

        self.initial_theta_hat = denominator @ numerator
        print(f'this is parameters:{self.initial_theta_hat}')

    def initial_b_matrix(self):
        identity_k = np.identity(self.k)
        self.u_hat = resid(self.delta_y_t, self.z_t_1, self.initial_theta_hat, identity_k)

        u_sum = np.zeros([self.k, self.k])
        for i in range(self.u_hat.shape[1]):
            u_sum += self.u_hat[:, [i]] @ self.u_hat[:, [i]].T
        b = 1 / self.t * u_sum

        b = vecm._mat_sqrt(b)
        b2 = sp.linalg.sqrtm(b) + + self.b0
        self.initial_b = b + self.b0

    def other_initial_params(self):
        self.lamda_m = np.identity(self.k)
        self.e_0_0 = np.array([[0.5], [0.5]])

        self.p = np.ones([self.regimes, self.regimes]) / self.regimes

        sigmas = np.zeros([self.k, self.k, self.regimes])
        for regime in range(self.regimes):
            if regime == 0:
                sigmas[:, :, regime] = self.initial_b @ self.initial_b.T
            else:
                sigmas[:, :, regime] = self.initial_b @ self.lamda_m @ self.initial_b.T
        self.sigmas = sigmas

    def run_initialization(self):
        self.data_matrix()
        self.initial_params()
        self.initial_b_matrix()
        self.other_initial_params()


def resid(dy_t, z_t, theta, ik=None):
    if ik is None:
        ik = np.identity(dy_t.shape[0])
    u_hat = np.zeros(dy_t.shape)
    for i in range(dy_t.shape[1]):
        z = z_t[:, [i]]
        dy = dy_t[:, [i]]
        u_hat[:, [i]] = dy - np.kron(z.T, ik) @ theta

    return u_hat
