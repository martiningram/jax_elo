import jax.numpy as jnp
from jax import jit, grad, hessian
from jax.scipy.stats import norm, multivariate_normal
from jax.scipy.special import expit
from ml_tools.jax import (weighted_sum, logistic_normal_integral_approx,
                          pos_def_mat_from_tri_elts)
from .general import EloFunctions, EloParams
from ml_tools.lin_alg import num_triangular_elts

# TODO: Maybe add some of the other optimisation-related stuff


@jit
def calculate_likelihood(x, a, theta, y):

    margin = y[0]

    margin_prob = norm.logpdf(margin, theta['factor'] * (a @ x) +
                              theta['offset'], theta['obs_sd'])

    win_prob = jnp.log(expit(a @ x))

    return margin_prob + win_prob


@jit
def calculate_predictive_lik(x, a, cov_mat, theta, y):

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

    return (calculate_likelihood(x, a, theta, y) +
            calculate_prior(x, mu, cov_mat, theta))


# TODO: Some of this should probably move into general.
# I might only have to provide a parse_theta function.
def update_params(x, params, verbose=True):

    n_latent = params.cov_mat.shape[0]

    cov_mat = pos_def_mat_from_tri_elts(
        x[:num_triangular_elts(n_latent)], n_latent)

    theta = {'factor': x[-3]**2, 'offset': x[-2], 'obs_sd': x[-1]**2}

    params = EloParams(theta=theta, cov_mat=cov_mat)

    if verbose:
        print(theta)
        print(cov_mat)

    return params


margin_functions = EloFunctions(
    log_post_jac_x=jit(grad(calculate_log_posterior)),
    log_post_hess_x=jit(hessian(calculate_log_posterior)),
    predictive_lik_fun=calculate_predictive_lik)
