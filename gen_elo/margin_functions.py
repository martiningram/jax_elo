import jax.numpy as jnp
from jax import jit, grad, hessian
from jax.scipy.stats import norm, multivariate_normal
from jax.scipy.special import expit
from ml_tools.jax import weighted_sum, logistic_normal_integral_approx
from .general import EloFunctions
from ml_tools.flattening import reconstruct

# TODO: Maybe add some of the other optimisation-related stuff


@jit
def calculate_likelihood(x, mu, a, theta, y):

    margin = y[0]

    margin_prob = norm.logpdf(margin, theta['factor'] * (a @ x) +
                              theta['offset'], theta['obs_sd'])

    win_prob = jnp.log(expit(a @ x))

    return margin_prob + win_prob


@jit
def calculate_predictive_lik(x, mu, a, cov_mat, theta, y):

    margin = y[0]

    latent_mean, latent_var = weighted_sum(x, cov_mat, a)

    margin_prob = norm.logpdf(
        margin, theta['factor'] * (latent_mean) + theta['offset'],
        jnp.sqrt(theta['obs_sd']**2 + theta['factor']**2 * latent_var))

    win_prob = jnp.log(logistic_normal_integral_approx(
        latent_mean, latent_var))

    return win_prob + margin_prob


@jit
def calculate_prior(x, mu, cov_mat, theta):

    return multivariate_normal.logpdf(x, mu, cov_mat)


@jit
def calculate_log_posterior(x, mu, cov_mat, a, theta, y):

    return (calculate_likelihood(x, mu, a, theta, y) +
            calculate_prior(x, mu, cov_mat, theta))


def parse_theta(x, summary):

    theta = reconstruct(x, summary, jnp.reshape)

    theta['factor'] = theta['factor']**2
    theta['obs_sd'] = theta['obs_sd']**2

    return theta


margin_functions = EloFunctions(
    log_post_jac_x=jit(grad(calculate_log_posterior)),
    log_post_hess_x=jit(hessian(calculate_log_posterior)),
    predictive_lik_fun=calculate_predictive_lik,
    parse_theta_fun=parse_theta)
