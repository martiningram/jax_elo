import numpy as onp
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from jax.ops import index_update
from jax.lax import scan
from collections import namedtuple, defaultdict
from scipy.optimize import minimize
from ml_tools.lin_alg import num_triangular_elts
from ml_tools.jax import (pos_def_mat_from_tri_elts, weighted_sum,
                          logistic_normal_integral_approx)
from ml_tools.flattening import flatten_and_summarise, reconstruct_np
from tqdm import tqdm


# TODO: Make a function which gets the final results out

EloFunctions = namedtuple('EloFunctions',
                          'log_post_jac_x,log_post_hess_x,predictive_lik_fun'
                          ',parse_theta_fun,win_prob_fun')
EloParams = namedtuple('EloParams', 'theta,cov_mat')


@partial(jit, static_argnums=4)
def calculate_update(mu, cov_mat, a, y, elo_functions, elo_params):

    lik = elo_functions.predictive_lik_fun(mu, mu, a, cov_mat,
                                           elo_params.theta, y)

    # Evaluate Jacobian and Hessian at the current guess
    mode_jac = elo_functions.log_post_jac_x(mu, mu, cov_mat, a,
                                            elo_params.theta, y)
    mode_hess = elo_functions.log_post_hess_x(mu, mu, cov_mat, a,
                                              elo_params.theta, y)

    # Get the updated guess from linearising
    new_x = -jnp.linalg.solve(mode_hess, mode_jac)

    return new_x + mu, lik


@jit
def calculate_win_prob(mu1, mu2, a, cov_mat, pre_factor=1.):

    full_mu = jnp.concatenate([mu1, mu2])
    full_cov_mat = jnp.kron(jnp.eye(2), cov_mat)

    latent_mean, latent_var = weighted_sum(full_mu, full_cov_mat, a)

    return logistic_normal_integral_approx(
        pre_factor * latent_mean, pre_factor**2 * latent_var)


@partial(jit, static_argnums=4)
def compute_update(mu1, mu2, a, y, elo_functions, elo_params):

    mu = jnp.concatenate([mu1, mu2])
    cov_full = jnp.kron(jnp.eye(2), elo_params.cov_mat)

    # TODO: The name clash isn't good. Update this.
    new_mu, lik = calculate_update(mu, cov_full, a, y, elo_functions,
                                   elo_params)

    new_mu1, new_mu2 = jnp.split(new_mu, 2)

    return new_mu1, new_mu2, lik


def update_ratings(carry, x, elo_functions, elo_params):

    cur_winner, cur_loser, cur_a, cur_y = x

    new_winner_mean, new_loser_mean, lik = compute_update(
        carry[cur_winner], carry[cur_loser], cur_a, cur_y, elo_functions,
        elo_params)

    carry = index_update(carry, cur_winner, new_winner_mean)
    carry = index_update(carry, cur_loser, new_loser_mean)

    return carry, lik


@partial(jit, static_argnums=4)
def calculate_ratings_scan(winners_array, losers_array, a_full, y_full,
                           elo_functions, elo_params, init):

    fun_to_scan = partial(update_ratings, elo_functions=elo_functions,
                          elo_params=elo_params)

    ratings, liks = scan(fun_to_scan, init, [winners_array, losers_array,
                                             a_full, y_full])

    return ratings, jnp.sum(liks)


def calculate_ratings_history(winners, losers, a_full, y_full, elo_functions,
                              elo_params):

    ratings = defaultdict(lambda: jnp.zeros(a_full.shape[1] // 2))
    history = list()

    for cur_winner, cur_loser, cur_a, cur_y in zip(
            tqdm(winners), losers, a_full, y_full):

        mu1, mu2 = ratings[cur_winner], ratings[cur_loser]

        prior_win_prob = elo_functions.win_prob_fun(
            mu1, mu2, cur_a, elo_params.cov_mat)

        new_mu1, new_mu2, lik = compute_update(
            mu1, mu2, cur_a, cur_y, elo_functions, elo_params)

        history.append({'winner': cur_winner,
                        'loser': cur_loser,
                        'prior_mu_winner': mu1,
                        'prior_mu_loser': mu2,
                        'prior_win_prob': prior_win_prob})

        ratings[cur_winner] = new_mu1
        ratings[cur_loser] = new_mu2

    return history

def get_starting_elts(cov_mat):

    L = jnp.linalg.cholesky(cov_mat)
    elts = L[onp.tril_indices_from(L)]

    return elts


def update_params(x, params, functions, summaries, verbose=True):

    n_latent = params.cov_mat.shape[0]

    # TODO: Allow for covariance matrix to be different, e.g. allow
    # independences
    cov_mat = pos_def_mat_from_tri_elts(
        x[:num_triangular_elts(n_latent)], n_latent)

    theta = functions.parse_theta_fun(x[num_triangular_elts(n_latent):],
                                      summaries)

    params = EloParams(theta=theta, cov_mat=cov_mat)

    if verbose:
        print(theta)
        print(cov_mat)

    return params


def ratings_lik(*args):

    return calculate_ratings_scan(*args)[1]


def to_optimise(x, start_params, functions, winners_array, losers_array,
                a_full, y_full, summaries, n_players, verbose=True):

    params = update_params(x, start_params, functions, summaries,
                           verbose=verbose)

    init = jnp.zeros((n_players, a_full.shape[1] // 2))

    cur_lik = ratings_lik(winners_array, losers_array, a_full, y_full,
                          functions, params, init)

    return -cur_lik


def optimise_elo(start_params, functions, winners_array, losers_array, a_full,
                 y_full, n_players, tol=1e-3, verbose=True):

    theta_flat, theta_summary = flatten_and_summarise(**start_params.theta)
    start_cov_mat = get_starting_elts(start_params.cov_mat)

    start_elts = jnp.concatenate([start_cov_mat, theta_flat])

    minimize_fun = partial(to_optimise, start_params=start_params,
                           functions=functions, winners_array=winners_array,
                           losers_array=losers_array, a_full=a_full,
                           y_full=y_full, summaries=theta_summary,
                           n_players=n_players)

    minimize_grad = jit(grad(minimize_fun))

    result = minimize(minimize_fun, start_elts, jac=minimize_grad, tol=tol)

    final_params = update_params(result.x, start_params, functions,
                                 theta_summary, verbose=False)

    return final_params, result
