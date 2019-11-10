import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.ops import index_update
from jax.lax import scan
from collections import namedtuple


# TODO: Make a function which gets the final results out

EloFunctions = namedtuple('EloConfig',
                          'log_post_jac_x,log_post_hess_x,predictive_lik_fun')
EloParams = namedtuple('EloParams', 'theta,cov_mat')


@partial(jit, static_argnums=4)
def calculate_update(mu, cov_mat, a, y, elo_functions, elo_params):

    lik = elo_functions.predictive_lik_fun(mu, a, cov_mat, elo_params.theta, y)

    # Evaluate Jacobian and Hessian at the current guess
    mode_jac = elo_functions.log_post_jac_x(mu, mu, cov_mat, a,
                                            elo_params.theta, y)
    mode_hess = elo_functions.log_post_hess_x(mu, mu, cov_mat, a,
                                              elo_params.theta, y)

    # Get the updated guess from linearising
    new_x = -jnp.linalg.solve(mode_hess, mode_jac)

    return new_x + mu, lik


@partial(jit, static_argnums=4)
def compute_update(mu1, mu2, a, y, elo_functions, elo_params):

    mu = jnp.concatenate([mu1, mu2])
    cov_full = jnp.kron(jnp.eye(2), elo_params.cov_mat)
    new_mu, lik = calculate_update(mu, cov_full, a, y, elo_functions,
                                   elo_params)

    new_mu1, new_mu2 = jnp.split(new_mu, 2)

    return new_mu1, new_mu2, lik


@partial(jit, static_argnums=2)
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
