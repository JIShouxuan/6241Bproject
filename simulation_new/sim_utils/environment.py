import numpy as np
from collections import namedtuple


def sample_cov_matrix(d, k, random_state, sd=1.):
    W = random_state.randn(d, k)
    S = W.dot(W.T) + np.diag(random_state.rand(d))
    inv_std = np.diag(1. / np.sqrt(np.diag(S)))
    corr = inv_std.dot(S).dot(inv_std)
    diag_std = np.eye(d) * np.repeat(sd, d)
    cov = diag_std.dot(corr).dot(diag_std)
    return cov


class Environment(object):

    def __init__(self, n_env, n_actions, inv_seed, non_inv_seed, train=True):
        self.n_env = n_env
        self.n_actions = n_actions
        inv_state = np.random.RandomState(inv_seed)
        self.bH3 = inv_state.normal(0, 1) #delta, corresponds to U, not effected by the environment
        self.bH0 = inv_state.normal(0, 1) #eta, corrsponds to X3, not effected by the environment
        self.bR = inv_state.normal(0, 1, size=(n_actions, 3)) #3 is the number of related covariates and reward is related to X2, X3 and U
        non_inv_state = np.random.RandomState(non_inv_seed)
        if train:
            # scale = 1
            # cov1 = sample_cov_matrix(self.n_env, 3, non_inv_state, scale)
            # cov2 = sample_cov_matrix(self.n_env, 3, non_inv_state, scale)
            # self.bH1 = non_inv_state.multivariate_normal(np.zeros(self.n_env), cov=cov1)[:, np.newaxis]
            # self.bH2 = non_inv_state.multivariate_normal(np.zeros(self.n_env), cov=cov2)[:, np.newaxis]
            scale = 2
            self.bH1 = non_inv_state.normal(size=(self.n_env, 1), scale=scale) #gamma_e, corresponds to X1
            self.bH2 = non_inv_state.normal(size=(self.n_env, 1), scale=scale) #alpha_e, corresponds to X2
        else:
            scale = 4
            self.bH1 = non_inv_state.normal(size=(self.n_env, 1), scale=scale) #gamma_e, corresponds to X1
            self.bH2 = non_inv_state.normal(size=(self.n_env, 1), scale=scale) #alpha_e, corresponds to X2

        self.e = np.stack([self.bH1, self.bH2])

    def gen_data(self, train_policy, s_size):
        # gendata
        D = []
        for env in range(self.n_env):
            # H3 is U
            # H2 is X2
            # H1 is X1
            # H0 is X3
            H0, H1, H2, H3 = get_state(s_size, self.bH3, self.bH0, bH1=self.bH1[env, :], bH2=self.bH2[env, :])
            # visible state: X1, X2, X3
            # X output: X3, X1, X2
            X = np.stack([H0, H1, H2], axis=1)
            # choose actions
            A, P = train_policy.get_actions(X)
            # Reward is related to X2, X3 and U
            R = reward(H0, H2, H3, self.bR)

            E = np.repeat(env, R.shape[0])
            D += [(X, A, R, P, E)]

        X, A, R, P, E = [np.stack(arr).reshape(s_size * self.n_env, -1) for arr in list(zip(*D))]
        A, P, E = A.squeeze(), P.squeeze(), E.squeeze()

        return X, A, R, P, E

    def get_corr(self):
        s_size = 10000
        H3_vec = []
        H1_vec = []
        for env in range(self.n_env):
            H0, H1, H2, H3 = get_state(s_size, self.bH3, self.bH0, bH1=self.bH1[env, :], bH2=self.bH2[env, :])
            H3_vec += [H3]
            H1_vec += [H1]
        corr = np.array([np.corrcoef(h3, h1)[1, 0] for h3,h1 in zip(H3_vec, H1_vec)])
        # H3_vec = np.concatenate(H3_vec)
        # H1_vec = np.concatenate(H1_vec)

        return corr


def get_state(n, bH3, bH0, bH1, bH2):
    H3 = np.random.normal(bH3, size=(n,))
    H2 = np.random.normal(bH2, size=(n,))
    H1 = np.random.normal(bH1 * H3)
    H0 = np.random.normal(bH0 * H3)

    return H0, H1, H2, H3


def reward(H0, H2, H3, bR):
    # compute reward
    return bR.dot(np.stack([H0, H2, H3])).T
