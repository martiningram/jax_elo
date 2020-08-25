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

    margin, was_retirement, bo5 = y

    sigma_obs = (1 - bo5) * theta["sigma_obs"] + bo5 * theta["sigma_obs_bo5"]

    # If it wasn't a retirement:
    margin_prob = norm.logpdf(margin, theta["a1"] * (a @ x) + theta["a2"], sigma_obs)

    win_prob = jnp.log(expit((1 + theta["bo5_factor"] * bo5) * b * a @ x))

    # Otherwise:
    # Only the loser's skill matters:
    n_skills = x.shape[0]

    loser_x = x[n_skills // 2 :]
    loser_mu = mu[n_skills // 2 :]
    loser_a = -a[n_skills // 2 :]

    loser_expected_skill = loser_a @ loser_mu
    loser_actual_skill = loser_a @ loser_x

    full_ret_factor = theta["ret_factor"] * (
        1 - theta["skill_factor"] * expit(b * loser_expected_skill)
    )

    ret_prob = expit(
        full_ret_factor * (loser_expected_skill - loser_actual_skill)
        + theta["ret_intercept"]
    )

    prob_retirement = jnp.log(ret_prob)
    prob_not_retirement = jnp.log(1 - ret_prob)

    return (1 - was_retirement) * (
        margin_prob + win_prob + prob_not_retirement
    ) + was_retirement * prob_retirement


@jit
def calculate_marginal_lik(x, mu, a, cov_mat, theta, y):

    margin, was_retirement, bo5 = y

    sigma_obs = (1 - bo5) * theta["sigma_obs"] + bo5 * theta["sigma_obs_bo5"]

    latent_mean, latent_var = weighted_sum(x, cov_mat, a)

    # If it wasn't a retirement:
    margin_prob = norm.logpdf(
        margin,
        theta["a1"] * (latent_mean) + theta["a2"],
        jnp.sqrt(sigma_obs ** 2 + theta["a1"] ** 2 * latent_var),
    )

    win_prob = jnp.log(
        logistic_normal_integral_approx(
            (1 + theta["bo5_factor"] * bo5) * b * latent_mean,
            (1 + theta["bo5_factor"] * bo5) ** 2 * b ** 2 * latent_var,
        )
    )

    n_skills = x.shape[0]

    # Otherwise:
    # Only the loser's skill matters:
    loser_x = x[n_skills // 2 :]
    loser_mu = mu[n_skills // 2 :]
    loser_a = -a[n_skills // 2 :]
    loser_cov_mat = cov_mat[n_skills // 2 :, n_skills // 2 :]

    loser_actual_mean, loser_actual_var = weighted_sum(loser_x, loser_cov_mat, loser_a)
    loser_expected_skill = loser_a @ loser_mu

    full_ret_factor = theta["ret_factor"] * (
        1 - theta["skill_factor"] * expit(b * loser_expected_skill)
    )

    ret_prob = logistic_normal_integral_approx(
        full_ret_factor * (loser_expected_skill - loser_actual_mean)
        + theta["ret_intercept"],
        full_ret_factor ** 2 * loser_actual_var,
    )

    prob_retirement = jnp.log(ret_prob)
    prob_not_retirement = jnp.log(1 - ret_prob)

    return (1 - was_retirement) * (win_prob + margin_prob + prob_not_retirement) + (
        was_retirement * prob_retirement
    )


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

    n_elts = num_mat_elts(len(theta["cov_mat"]))
    theta["cov_mat"] = pos_def_mat_from_tri_elts(theta["cov_mat"], n_elts)

    theta["a1"] = theta["a1"] ** 2
    theta["sigma_obs"] = theta["sigma_obs"] ** 2
    theta["ret_factor"] = theta["ret_factor"] ** 2
    theta["bo5_factor"] = theta["bo5_factor"] ** 2
    theta["sigma_obs_bo5"] = theta["sigma_obs_bo5"] ** 2

    return theta


margin_functions_retirement = EloFunctions(
    log_post_jac_x=jit(grad(calculate_log_posterior)),
    log_post_hess_x=jit(hessian(calculate_log_posterior)),
    marginal_lik_fun=calculate_marginal_lik,
    parse_theta_fun=parse_theta,
    win_prob_fun=jit(partial(calculate_win_prob, pre_factor=b)),
)
