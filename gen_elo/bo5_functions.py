import jax.numpy as jnp
from jax import jit, grad, hessian
from jax.scipy.stats import norm
from jax.scipy.special import expit
from ml_tools.jax import weighted_sum, logistic_normal_integral_approx
from .general import EloFunctions
from ml_tools.flattening import reconstruct
from .margin_functions import calculate_prior


@jit
def calculate_likelihood_bo5(x, mu, a, theta, y):

    margin, is_bo5, is_retirement = y

    logit = a @ x

    margin_prob = norm.logpdf(margin, theta['factor'] * (logit) +
                              theta['offset'], theta['obs_sd'])

    logit_prob = is_bo5 * logit * theta['bo5_factor'] + (1 - is_bo5) * logit

    win_prob = jnp.log(expit(logit_prob))

    _, a_loser = jnp.split(a, 2)
    _, x_loser = jnp.split(x, 2)
    _, mu_loser = jnp.split(mu, 2)

    loser_skill = -a_loser @ x_loser
    loser_mean_skill = -a_loser @ mu_loser

    retirement_prob = jnp.log(expit(-theta['ret_factor'] * (
        loser_mean_skill - loser_skill)))

    return is_retirement * retirement_prob + \
        (1 - is_retirement) * (margin_prob + win_prob)


@jit
def calculate_predictive_lik_bo5(x, mu, a, cov_mat, theta, y):

    margin, is_bo5, is_retirement = y

    latent_mean, latent_var = weighted_sum(x, cov_mat, a)

    margin_prob = norm.logpdf(
        margin, theta['factor'] * (latent_mean) + theta['offset'],
        jnp.sqrt(theta['obs_sd']**2 + theta['factor']**2 * latent_var))

    bo3_prob = (1 - is_bo5) * jnp.log(logistic_normal_integral_approx(
        latent_mean, latent_var))
    bo5_prob = is_bo5 * jnp.log(logistic_normal_integral_approx(
        theta['bo5_factor'] * latent_mean,
        theta['bo5_factor']**2 * latent_var))

    win_prob = bo3_prob + bo5_prob

    _, a_loser = jnp.split(a, 2)
    _, x_loser = jnp.split(x, 2)
    _, mu_loser = jnp.split(mu, 2)
    a_loser = -a_loser

    loser_skill, loser_var = weighted_sum(x_loser, cov_mat[a.shape[0] // 2:,
                                                           a.shape[0] // 2:],
                                          a_loser)

    loser_skill_mu = a_loser @ mu_loser

    retirement_prob = jnp.log(logistic_normal_integral_approx(
        -theta['ret_factor'] * (loser_skill_mu - loser_skill),
        theta['ret_factor']**2 * loser_var))

    return is_retirement * retirement_prob +\
        (1 - is_retirement) * (win_prob + margin_prob)


def parse_theta(x, summary):

    theta = reconstruct(x, summary, jnp.reshape)

    theta['factor'] = theta['factor']**2
    theta['obs_sd'] = theta['obs_sd']**2
    theta['bo5_factor'] = theta['bo5_factor']**2

    return theta


@jit
def calculate_log_posterior(x, mu, cov_mat, a, theta, y):

    return (calculate_likelihood_bo5(x, mu, a, theta, y) +
            calculate_prior(x, mu, cov_mat, theta))


margin_functions_bo5 = EloFunctions(
    log_post_jac_x=jit(grad(calculate_log_posterior)),
    log_post_hess_x=jit(hessian(calculate_log_posterior)),
    predictive_lik_fun=calculate_predictive_lik_bo5,
    parse_theta_fun=parse_theta)
