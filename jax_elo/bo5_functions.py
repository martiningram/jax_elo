import jax.numpy as jnp
from jax import jit, grad, hessian
from jax.scipy.stats import norm
from jax.scipy.special import expit
from ml_tools.jax import weighted_sum, logistic_normal_integral_approx
from .general import EloFunctions, calculate_win_prob
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

    a_winner, a_loser = jnp.split(a, 2)
    x_winner, x_loser = jnp.split(x, 2)
    mu_winner, mu_loser = jnp.split(mu, 2)

    loser_skill = -a_loser @ x_loser
    loser_mean_skill = -a_loser @ mu_loser
    winner_skill = a_winner @ x_winner
    winner_mean_skill = a_winner @ mu_winner

    retirement_prob = expit(-theta['ret_factor'] * (
        loser_skill - loser_mean_skill) -
        theta['skill_ret_multiplier'] * jnp.maximum(0., loser_mean_skill)
        + theta['ret_intercept'])

    retirement_prob_winner = expit(
        -theta['ret_factor'] * (winner_skill - winner_mean_skill)
        - theta['skill_ret_multiplier'] * jnp.maximum(0., winner_mean_skill)
        + theta['ret_intercept'])

    # Should add factor here about winner _not_ retiring.
    retirement_lik = is_retirement * jnp.log(retirement_prob) + \
        (1 - is_retirement) * jnp.log(1 - retirement_prob) + \
        jnp.log(1 - retirement_prob_winner)  # winner never retires

    return retirement_lik + (1 - is_retirement) * (margin_prob + win_prob)


def initialise_theta():
    # Gives a reasonable initialisation for theta.
    theta = {
        'ret_factor': 1.,
        'skill_ret_multiplier': 0.,
        'ret_intercept': 0.,
        'factor': jnp.sqrt(0.1),
        'offset': 0.1,
        'obs_sd': jnp.sqrt(0.1),
        'bo5_factor': 1.
    }

    theta = {x: jnp.array(y) for x, y in theta.items()}

    return theta


@jit
def calculate_marginal_lik_bo5(x, mu, a, cov_mat, theta, y):

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

    a_winner, a_loser = jnp.split(a, 2)
    x_winner, x_loser = jnp.split(x, 2)
    mu_winner, mu_loser = jnp.split(mu, 2)
    a_loser = -a_loser

    loser_skill, loser_var = weighted_sum(x_loser, cov_mat[a.shape[0] // 2:,
                                                           a.shape[0] // 2:],
                                          a_loser)

    loser_skill_mu = a_loser @ mu_loser

    winner_skill, winner_var = weighted_sum(
        x_winner, cov_mat[a.shape[0] // 2:, a.shape[0] // 2:], a_winner)

    winner_skill_mu = a_winner @ mu_winner

    retirement_prob = logistic_normal_integral_approx(
        -theta['ret_factor'] * (loser_skill - loser_skill_mu) -
        theta['skill_ret_multiplier'] * jnp.maximum(loser_skill_mu, 0.)
        + theta['ret_intercept'],
        theta['ret_factor']**2 * loser_var)

    retirement_prob_winner = logistic_normal_integral_approx(
        -theta['ret_factor'] * (winner_skill - winner_skill_mu) -
        theta['skill_ret_multiplier'] * jnp.maximum(0., winner_skill_mu)
        + theta['ret_intercept'], theta['ret_factor']**2 * winner_var)

    retirement_lik = is_retirement * jnp.log(retirement_prob) + (
        1 - is_retirement) * jnp.log(1 - retirement_prob) + \
        jnp.log(1 - retirement_prob_winner)  # winner can't have retired

    return retirement_lik + (1 - is_retirement) * (win_prob + margin_prob)


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
    marginal_lik_fun=calculate_marginal_lik_bo5,
    parse_theta_fun=parse_theta,
    win_prob_fun=calculate_win_prob)
