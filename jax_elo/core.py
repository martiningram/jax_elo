import numpy as onp
import jax.numpy as jnp
from jax import jit, grad
from jax.lax import scan
from collections import defaultdict
from typing import NamedTuple, Callable, Dict
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

from jax_elo.utils.normals import weighted_sum, logistic_normal_integral_approx
from jax_elo.utils.flattening import flatten_and_summarise


class EloParams(NamedTuple):
    """
    This Tuple contains the parameters used by Elo.

    Args:
        theta: The parameters used, such as the prior variance.
        cov_mat: The prior covariance for each competitor.
    """

    theta: Dict[str, jnp.ndarray]


class EloFunctions(NamedTuple):
    """This Tuple contains the functions determining each update. In detail,
    they are:

    Note:
        The arguments of each function and how to use them are probably best
        understood by example. Please see the files
        elo_functions/margin_functions.py or elo_functions/basic.py for how to
        implement them in practice, and the jupyter notebook `examples/Best of
        Five extension example`. These files also show that it is not necessary
        to compute the Jacobian and Hessian by hand, since JAX can compute them
        automatically.

    Args:
        log_post_jac_x: This is the Jacobian of the log posterior density with
            respect to x, the vector of skill ratings. It is a function taking
            five arguments. These are the skill vector x, the prior mean mu,
            the vector a mapping from x to the difference in skills, a
            dictionary of parameters theta, and an array of additional
            information y.
        log_post_hess_x: The Hessian of the log posterior density with respect
            to x. It takes the same arguments as log_post_jac_x.
        marginal_lik_fun: This function calculates the log marginal likelihood
            of an observation. It takes the skill vector x, the prior means mu,
            the vector a mapping from skills to skill difference, the
            covariance matrix, the dictionary of parameters theta, and the
            additional observations y.
        parse_theta_fun: This function produces the dictionary of parameters
            theta from a flat vector flat_theta. It takes as its input two
            parameters, the first being the flat vector, and the second being
            the summary of shapes as produced for example by
            flatten_and_summarise. Note that it also has to ensure that the
            elements of flat_theta are valid, e.g. by squaring parameters that
            are constrained to be positive.
        win_prob_fun: The function computing the win probability. This function
            takes four parameters. The five required arguments are the prior
            mean for player 1, the prior mean for player 2, the vector a
            mapping from skills to skill difference, the vector y with
            additional variables, and the parameters in Elo.
    """

    log_post_jac_x: Callable[..., jnp.ndarray]
    log_post_hess_x: Callable[..., jnp.ndarray]
    marginal_lik_fun: Callable[..., float]
    parse_theta_fun: Callable[
        [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]
    ]
    win_prob_fun: Callable[..., float]


@partial(jit, static_argnums=4)
def calculate_update(mu, cov_mat, a, y, elo_functions, elo_params):
    """Calculates the Elo update.

    Args:
        mu: The prior mean
        cov_mat: The prior covariance
        a: The vector mapping from the skill vector to the difference
        y: The outcome
        elo_functions: The functions required to compute the update
        elo_params: The parameters required for the update

    Returns:
    The new mean, as well as the likelihood of the update, as a tuple.
    """

    lik = elo_functions.marginal_lik_fun(mu, mu, a, cov_mat, elo_params.theta, y)

    # Evaluate Jacobian and Hessian at the current guess
    mode_jac = elo_functions.log_post_jac_x(mu, mu, cov_mat, a, elo_params.theta, y)
    mode_hess = elo_functions.log_post_hess_x(mu, mu, cov_mat, a, elo_params.theta, y)

    # Get the updated guess from linearising
    new_x = -jnp.linalg.solve(mode_hess, mode_jac)

    return new_x + mu, lik


@jit
def calculate_win_prob(mu1, mu2, a, y, elo_params, pre_factor=1.0):
    """Calculates the win probability for a match with two competitors.

    Args:
        mu1: Player 1's mean ratings.
        mu2: Player 2's mean ratings.
        a: The vector mapping from the skill vector to the difference in skills
        pre_factor: An optional pre-factor multiplying the difference in
            skills.

    Returns:
    The win probability of player 1.
    """

    full_mu = jnp.concatenate([mu1, mu2])
    full_cov_mat = jnp.kron(jnp.eye(2), elo_params.theta["cov_mat"])

    latent_mean, latent_var = weighted_sum(full_mu, full_cov_mat, a)

    return logistic_normal_integral_approx(
        pre_factor * latent_mean, pre_factor**2 * latent_var
    )


@partial(jit, static_argnums=4)
def concatenate_and_update(mu1, mu2, a, y, elo_functions, elo_params):
    """Combines mu1 and mu2 into a concatenated vector mu and uses this to
    calculate updated means mu1' and mu2'.

    Args:
        mu1: The winner's mean prior to the match.
        mu2: The loser's mean prior to the match.
        a: The vector such that a^T [mu1, mu2] = mu_delta.
        y: The observed outcomes.
        elo_functions: The functions required to compute the update
        elo_params: The parameters required for the update

    Returns:
    A Tuple with three elements: the first two contain the new means, the last
    the log likelihood of the result.
    """

    mu = jnp.concatenate([mu1, mu2])
    cov_full = jnp.kron(jnp.eye(2), elo_params.theta["cov_mat"])

    new_mu, lik = calculate_update(mu, cov_full, a, y, elo_functions, elo_params)

    new_mu1, new_mu2 = jnp.split(new_mu, 2)

    return new_mu1, new_mu2, lik


def update_ratings(carry, x, elo_functions, elo_params):
    """The function to make an update to use in tandem with lax.scan.

    Args:
        carry: The carry, which contains the current ratings in array form so
            that entry [i, j] contains the mean for competitor i on skill j.
        x: The information required to make the update. This the current
            winner's index, the current loser's index, the vector mapping from
            skills to the skill difference a, and the current additional
            outcome information [e.g. the margin] y.
        elo_functions: The functions required to compute the update
        elo_params: The parameters required for the update

    Returns:
    A tuple whose first element is the updated carry [i.e. the updated ratings]
    and whose second element is the likelihood of the current update.
    """

    cur_winner, cur_loser, cur_a, cur_y = x

    new_winner_mean, new_loser_mean, lik = concatenate_and_update(
        carry[cur_winner], carry[cur_loser], cur_a, cur_y, elo_functions, elo_params
    )

    carry = carry.at[cur_winner].set(new_winner_mean)
    carry = carry.at[cur_loser].set(new_loser_mean)

    return carry, lik


@partial(jit, static_argnums=4)
def calculate_ratings_scan(
    winners_array, losers_array, a_full, y_full, elo_functions, elo_params, init
):
    """Calculates the ratings using lax.scan.

    Args:
        winners_array: Array such that entry i gives the index of the winner of
            match i.
        losers_array: Array such that entry i gives the index of the loser of
            match i.
        a_full: A matrix of shape [N, 2L] where N is the number of matches and
            L is the number of skills for each competitor.
        y_full: The full matrix of observed outcomes in addition to win or loss
            [e.g. the margin]. It must be of shape [N, N_Y], where N_Y is the
            number of additional observations [can be zero].
        elo_functions: The functions required to compute the update
        elo_params: The parameters required for the update

    Returns:
    A Tuple whose first element is the ratings after all the updates, and whose
    second is the likelihood for each update.
    """

    fun_to_scan = partial(
        update_ratings, elo_functions=elo_functions, elo_params=elo_params
    )

    ratings, liks = scan(
        fun_to_scan, init, [winners_array, losers_array, a_full, y_full]
    )

    return ratings, liks


def calculate_ratings_history(
    winners, losers, a_full, y_full, elo_functions, elo_params, show_progress=True
):
    """Calculates the full history of ratings.

    Args:
        winners: The names of the winners, as strings.
        losers: The names of the losers, as strings.
        a_full: A matrix of shape [N, 2L] where N is the number of matches and
            L is the number of skills for each competitor.
        y_full: The full matrix of observed outcomes in addition to win or loss
            [e.g. the margin]. It must be of shape [N, N_Y], where N_Y is the
            number of additional observations [can be zero].
        elo_functions: The functions required to compute the update
        elo_params: The parameters required for the update

    Returns:
    A Tuple. The first element is a list of dictionaries, each entry containing
    the entries "winner", "loser", giving their names, respectively; the prior
    mean rating of the winner ["prior_mu_winner"], the prior mean rating of the
    loser ["prior_mu_loser"], and the prior win probability of the winner
    ["prior_win_prob"]. The second element contains a dictionary of the most
    up-to-date ratings for each player.
    """

    ratings = defaultdict(lambda: jnp.zeros(a_full.shape[1] // 2))
    history = list()

    winners = tqdm(winners) if show_progress else winners

    for cur_winner, cur_loser, cur_a, cur_y in zip(
        tqdm(winners), losers, a_full, y_full
    ):

        mu1, mu2 = ratings[cur_winner], ratings[cur_loser]

        prior_win_prob = elo_functions.win_prob_fun(mu1, mu2, cur_a, cur_y, elo_params)

        new_mu1, new_mu2, lik = concatenate_and_update(
            mu1, mu2, cur_a, cur_y, elo_functions, elo_params
        )

        history.append(
            {
                "winner": cur_winner,
                "loser": cur_loser,
                "prior_mu_winner": mu1,
                "prior_mu_loser": mu2,
                "prior_win_prob": prior_win_prob,
            }
        )

        ratings[cur_winner] = new_mu1
        ratings[cur_loser] = new_mu2

    return history, ratings


def get_starting_elts(cov_mat):
    """A helper function which extracts the lower triangular elements of the
    cholesky decomposition of the covariance matrix."""

    L = jnp.linalg.cholesky(cov_mat)
    elts = L[onp.tril_indices_from(L)]

    return elts


def update_params(x, params, functions, summaries, verbose=True):
    """A helper function which translates the flat parameter vector x into the
    NamedTuple of EloParams.

    Args:
        x: The flat vector used by the optimisation routine.
        params: The old parameter settings.
        functions: The functions governing the updates
        summaries: The summaries of array shapes required to convert the flat
            vector x back into its individual components.
        verbose: If verbose, prints the new parameter settings.

    Returns:
    The parameter vector x as the NamedTuple EloParams.
    """

    theta = functions.parse_theta_fun(x, summaries)

    params = EloParams(theta=theta)

    if verbose:
        print("theta:", theta)
        print("cov_mat:", theta["cov_mat"])

    return params


def optimise_elo(
    start_params,
    functions,
    winners_array,
    losers_array,
    a_full,
    y_full,
    n_players,
    tol=1e-3,
    objective_mask=None,
    verbose=True,
):
    """Optimises the parameters for Elo.

    Args:
        start_params: The initial parameters for the optimisation.
        functions: The EloFunctions to use.
        winners_array: Array such that entry i gives the index of the winner of
            match i.
        losers_array: Array such that entry i gives the index of the loser of
            match i.
        a_full: A matrix of shape [N, 2L] where N is the number of matches and
            L is the number of skills for each competitor.
        y_full: The full matrix of observed outcomes in addition to win or loss
            [e.g. the margin]. It must be of shape [N, N_Y], where N_Y is the
            number of additional observations [can be zero].
        n_players: The number of players.
        tol: The tolerance required for the optimisation algorithm to
            terminate.
        objective_mask: If provided, must be a vector of shape [N,], where N is
            the number of matches. It should be one if the log likelihood of the
            match update should be used to compute the objective, and zero
            otherwise. This allows e.g. ignoring an initial set of matches when
            optimising.
        verbose: If True, prints the current settings after each optimisation
            step.

    Note:
        When specifying start_params, please note that they will be passed
        through ``parse_theta'' on the first iteration. Thus, if parse_theta
        constrains any parameters e.g. by squaring, please pass in the square
        root of the desired setting for the parameter. This will hopefully be
        improved in future.

    Returns:
    A Tuple, the first element of which are the optimal parameters found, and
    the second the result of the optimisation routine
    [scipy.optimize.minimize].
    """

    theta_flat, theta_summary = flatten_and_summarise(**start_params.theta)
    start_elts = theta_flat

    minimize_fun = partial(
        _to_optimise,
        start_params=start_params,
        functions=functions,
        winners_array=winners_array,
        losers_array=losers_array,
        a_full=a_full,
        y_full=y_full,
        summaries=theta_summary,
        n_players=n_players,
        objective_mask=objective_mask,
    )

    minimize_grad = jit(grad(minimize_fun))

    result = minimize(minimize_fun, start_elts, jac=minimize_grad, tol=tol)

    final_params = update_params(
        result.x, start_params, functions, theta_summary, verbose=False
    )

    return final_params, result


def _ratings_lik(*args):

    return calculate_ratings_scan(*args)[1]


def _to_optimise(
    x,
    start_params,
    functions,
    winners_array,
    losers_array,
    a_full,
    y_full,
    summaries,
    n_players,
    objective_mask=None,
    verbose=True,
):

    objective_mask = (
        onp.ones(winners_array.shape[0]) if objective_mask is None else objective_mask
    )

    params = update_params(x, start_params, functions, summaries, verbose=verbose)

    init = jnp.zeros((n_players, a_full.shape[1] // 2))

    cur_liks = _ratings_lik(
        winners_array, losers_array, a_full, y_full, functions, params, init
    )

    return jnp.sum(-objective_mask * cur_liks)
