from functools import partial

import jax.numpy as jnp
from jax import jit, grad, hessian
from jax.scipy.stats import norm, multivariate_normal
from jax.scipy.special import expit

from jax_elo.core import EloFunctions, calculate_win_prob
from jax_elo.utils.normals import weighted_sum, logistic_normal_integral_approx
from jax_elo.utils.flattening import reconstruct
from jax_elo.utils.linalg import num_mat_elts, pos_def_mat_from_tri_elts

# TODO: Maybe add some of the other optimisation-related stuff
b = jnp.log(10) / 400.0


@jit
def calculate_likelihood(x, mu, a, theta, y):

    margin = y[0]

    margin_prob = norm.logpdf(
        margin, theta["a1"] * (a @ x) + theta["a2"], theta["sigma_obs"]
    )

    win_prob = jnp.log(expit(b * a @ x))

    return margin_prob + win_prob


@jit
def calculate_marginal_lik(x, mu, a, cov_mat, theta, y):

    margin = y[0]

    latent_mean, latent_var = weighted_sum(x, cov_mat, a)

    margin_prob = norm.logpdf(
        margin,
        theta["a1"] * (latent_mean) + theta["a2"],
        jnp.sqrt(theta["sigma_obs"] ** 2 + theta["a1"] ** 2 * latent_var),
    )

    win_prob = jnp.log(
        logistic_normal_integral_approx(b * latent_mean, b ** 2 * latent_var)
    )

    return win_prob + margin_prob


@jit
def calculate_prior(x, mu, cov_mat, theta):

    return multivariate_normal.logpdf(x, mu, cov_mat)


@jit
def calculate_log_posterior(x, mu, cov_mat, a, theta, y):

    return calculate_likelihood(x, mu, a, theta, y) + calculate_prior(
        x, mu, cov_mat, theta
    )


def parse_theta(flat_theta, summary):

    theta = reconstruct(flat_theta, summary, jnp.reshape)

    n_elts = num_mat_elts(theta["cov_mat"].shape[0])
    theta["cov_mat"] = pos_def_mat_from_tri_elts(theta["cov_mat"], n_elts)

    theta["a1"] = theta["a1"] ** 2
    theta["sigma_obs"] = theta["sigma_obs"] ** 2

    return theta


margin_functions = EloFunctions(
    log_post_jac_x=jit(grad(calculate_log_posterior)),
    log_post_hess_x=jit(hessian(calculate_log_posterior)),
    marginal_lik_fun=calculate_marginal_lik,
    parse_theta_fun=parse_theta,
    win_prob_fun=jit(partial(calculate_win_prob, pre_factor=b)),
)
