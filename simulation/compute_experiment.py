# Add invariant test into the path
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant-policy-learning-main/invariant_test_utils')))
sys.path.append(os.path.abspath(os.path.join('..', 'invariant-policy-learning-main/simulation/sim_utils')))

import numpy as np

from sklearn.linear_model import LinearRegression

from environment import Environment
from learner import train_lsq
from policy import RandomPolicy, Policy

from invariant_test import invariance_test_actions_resample, fit_m, rate_fn

n_actions = 3
inv_seed = 111
seed = 0
target_sets = {r'$\emptyset$': [],
               'X1': [0],
               'X2': [1],
               'X1,X2': [0, 1]}
np.random.seed(seed)

n_envs = [2, 6, 10]
s_sizes = [1000, 3000, 9000, 27000, 81000, 243000]

random_policy = RandomPolicy(n_actions)
loggin_env = Environment(1, n_actions, 1, 1)
X_log, A_log, R_log, _, _ = loggin_env.gen_data(random_policy, int(1e4))
loggin_policy = Policy(train_lsq(X_log, R_log, target_sets['X1,X2']), target_sets['X1,X2'], 1.75)

def verify_policies(s_size,
                    n_env,
                    draw_seed,
                    loggin_policy=loggin_policy,
                    inv_seed=inv_seed,
                    non_inv_seed=seed,
                    alpha=0.05):
    model = LinearRegression()
    train_size = int(s_size / n_env)

    train_env = Environment(n_env, n_actions, inv_seed=inv_seed, non_inv_seed=non_inv_seed, train=True)
    np.random.seed(draw_seed)
    X, A, R, P, E = train_env.gen_data(loggin_policy, train_size)
    # get target and weights
    Y = R[np.arange(len(R)), A]
    W = 1 / P

    is_invs = {}
    m = fit_m(A, X, W, n_iter=10)
    rate = rate_fn(1, m / s_size)
    for subset in target_sets.items():
        target_name, p_val = invariance_test_actions_resample(model, subset,
                                                              X, A, Y, E, W, rate)
        is_invs[target_name] = p_val >= alpha

    return is_invs


# function for computing experiment
def compute_experiment(i):
    ret_dict = {'Acceptance Rate': [], 'Policy': [], 'n_env': [], 'Sample Size': []}
    for s_size in s_sizes:
        for n_env in n_envs:
            is_invs = verify_policies(s_size, n_env, draw_seed=(i, s_size, n_env))

            for policy_name, is_inv in is_invs.items():
                ret_dict['Acceptance Rate'] += [is_inv]
                ret_dict['n_env'] += [n_env]
                ret_dict['Policy'] += [policy_name]
                ret_dict['Sample Size'] += [s_size]

    return ret_dict