import pandas as pd
import jax.numpy as jnp

from jax_elo.core import (
    EloParams,
    EloFunctions,
    optimise_elo,
    calculate_ratings_history,
    get_starting_elts,
    get_empty_init_covariates,
)
from jax_elo.elo_functions.margin_functions_retirement import (
    margin_functions_retirement,
)
from jax_elo.utils.encoding import encode_players, encode_marks
from jax_elo.utils.flattening import reconstruct
from jax_elo.utils.linalg import num_mat_elts, pos_def_mat_from_tri_elts
from jax.scipy.linalg import block_diag
from functools import partial
from jax.scipy.stats import norm


def parse_theta(flat_theta, summary, use_skill_factor):

    theta = reconstruct(flat_theta, summary, jnp.reshape)

    # Reconstruct surface covariance matrix
    n_elts = num_mat_elts(len(theta["surf_cov_mat_elts"]))
    surf_cov_mat = pos_def_mat_from_tri_elts(theta["surf_cov_mat_elts"], n_elts)
    diag_rank_elts = theta["rank_diag_elts"] ** 2

    full_cov_mat = block_diag(surf_cov_mat, jnp.diag(diag_rank_elts))

    n_surf = surf_cov_mat.shape[1]
    n_tournament = len(diag_rank_elts)

    offsets = jnp.ones((n_tournament - 1, n_surf))
    offsets = offsets * theta["tournament_rank_offsets"]
    offsets = jnp.concatenate(
        [offsets, jnp.zeros((n_tournament - 1, n_tournament))], axis=1
    )
    theta["tournament_rank_offsets"] = offsets

    wildcard_offsets = jnp.zeros((n_tournament, n_surf))
    wildcard_offsets = wildcard_offsets + theta["wildcard_offset"].reshape(-1, 1)
    wildcard_offsets = jnp.concatenate(
        [wildcard_offsets, jnp.zeros((n_tournament, n_tournament))], axis=1
    )

    theta["wildcard_offset"] = wildcard_offsets

    theta["cov_mat"] = full_cov_mat

    theta["a1"] = theta["a1"] ** 2
    theta["a1_challenger"] = theta["a1_challenger"] ** 2

    theta["a2_challenger"] = theta["a2_challenger"] ** 2
    theta["a2"] = theta["a2"] ** 2

    theta["sigma_obs"] = theta["sigma_obs"] ** 2
    theta["sigma_obs_bo5"] = theta["sigma_obs_bo5"] ** 2
    theta["sigma_obs_challenger"] = theta["sigma_obs_challenger"] ** 2

    # TODO: Check!
    theta["ret_factor"] = theta["ret_factor"] ** 2
    # theta["bo5_factor"] = theta["bo5_factor"] ** 2
    theta["skill_factor"] = use_skill_factor * theta["skill_factor"]

    # Fix due to optimisation issues
    theta["bo5_factor"] = 0.43

    # theta["long_break_addition"] = jnp.concatenate(
    #     [jnp.repeat(theta["long_break_addition"], n_surf), jnp.zeros(n_tournament)]
    # )

    # theta["very_long_break_addition"] = jnp.concatenate(
    #     [jnp.repeat(theta["very_long_break_addition"], n_surf), jnp.zeros(n_tournament)]
    # )

    return theta


def create_y(
    margins,
    was_retirement,
    tournament_ranks,
    is_best_of_five,
    is_challenger,
    winner_info,
    loser_info,
):

    rank_ids, _ = encode_marks(tournament_ranks)
    rank_ids = jnp.argmax(rank_ids, axis=1)

    y = {
        "margin": margins,
        "was_retirement": was_retirement,
        "bo5": is_best_of_five,
        "is_challenger": is_challenger,
    }

    # Allow initialisation by tournament rank
    y["match_covariates"] = {"tournament_rank": jnp.array(rank_ids)}
    y["winner_covariates"] = {"is_wildcard": winner_info["is_wildcard"]}
    y["loser_covariates"] = {"is_wildcard": loser_info["is_wildcard"]}

    # y["winner_control"] = {
    #     "is_long_break": winner_info["is_long_break"],
    # }

    # y["loser_control"] = {
    #     "is_long_break": loser_info["is_long_break"],
    # }

    return y


def create_a(surfaces, tournament_ranks):

    dummy_surfaces, surface_names = encode_marks(surfaces)
    dummy_ranks, rank_names = encode_marks(tournament_ranks)

    a = jnp.concatenate(
        [dummy_surfaces, dummy_ranks, -dummy_surfaces, -dummy_ranks], axis=1
    )

    return a, surface_names, rank_names


def fit(
    winners,
    losers,
    surfaces,
    tournament_ranks,
    is_best_of_five,
    margins,
    was_retirement,
    is_challenger,
    winner_info,
    loser_info,
    retirement_skill_factor=True,
    verbose=False,
):
    """Fits the parameters of the correlated skills model.

    Args:
        winners: The names of the winners, as a numpy array.
        losers: The names of the losers, as a numpy array.
        margins: The margins of victory, as a numpy array.
        verbose: If True, prints the progress of the optimisation.

    Returns:
    Tuple: The first element will contain the optimal parameters; the second
    the result from the optimisation routine.
    """

    n_matches = len(winners)

    a, surface_names, rank_names = create_a(surfaces, tournament_ranks)

    n_surfaces = len(surface_names)

    surf_start_cov = jnp.eye(n_surfaces)
    surf_start_cov_elts = get_starting_elts(surf_start_cov)

    # Rank part is diagonal
    n_t_ranks = len(rank_names)
    diag_rank_elts_sqrt = jnp.ones(a.shape[1] // 2 - n_surfaces) * 0.1

    start_theta = {
        "surf_cov_mat_elts": surf_start_cov_elts,
        "rank_diag_elts": diag_rank_elts_sqrt,
        "a1": jnp.array(0.1),
        "a2": jnp.array(0.1),
        "sigma_obs": jnp.array(1.0),
        "ret_factor": jnp.array(1e-4),
        "ret_intercept": jnp.array(-2.0),
        "skill_factor": jnp.array(0.0),
        "bo5_factor": jnp.array(0.1),
        "sigma_obs_bo5": jnp.array(1.0),
        "tournament_rank_offsets": jnp.zeros((len(rank_names) - 1, 1)),
        "a1_challenger": jnp.array(0.1),
        "a2_challenger": jnp.array(0.1),
        "sigma_obs_challenger": jnp.array(1.0),
        "wildcard_offset": jnp.zeros(len(rank_names)),
        # "long_break_addition": jnp.array(0.0),
    }

    dict_base_funs = margin_functions_retirement._asdict()
    dict_base_funs["parse_theta_fun"] = partial(
        parse_theta, use_skill_factor=retirement_skill_factor
    )
    elo_funs = EloFunctions(**dict_base_funs)

    init_params = EloParams(theta=start_theta,)

    winner_ids, loser_ids, names = encode_players(winners, losers)
    n_players = len(names)

    y = create_y(
        margins,
        was_retirement,
        tournament_ranks,
        is_best_of_five,
        is_challenger,
        winner_info,
        loser_info,
    )

    prior_fun = lambda params: (
        norm.logpdf(params.theta["bo5_factor"])
        + jnp.sum(norm.logpdf(params.theta["tournament_rank_offsets"]))
    )

    opt_result = optimise_elo(
        init_params,
        elo_funs,
        winner_ids,
        loser_ids,
        a,
        y,
        n_players,
        prior_fun=prior_fun,
        verbose=verbose,
    )

    return opt_result


def calculate_ratings(
    parameters,
    winners,
    losers,
    surfaces,
    tournament_ranks,
    is_best_of_five,
    margins,
    was_retirement,
    is_challenger,
    winner_info,
    loser_info,
):

    a, surface_names, rank_names = create_a(surfaces, tournament_ranks)

    n_matches = winners.shape[0]
    functions = margin_functions_retirement

    y = create_y(
        margins,
        was_retirement,
        tournament_ranks,
        is_best_of_five,
        is_challenger,
        winner_info,
        loser_info,
    )

    final_ratings, history = calculate_ratings_history(
        winners, losers, a, y, functions, parameters
    )

    return history, final_ratings, surface_names, rank_names


def get_player_skill_history(ratings_history, final_ratings_dict, dates, player_name):

    player_history = list()

    for cur_match, cur_date in zip(ratings_history, dates):

        if player_name not in (cur_match["winner"], cur_match["loser"]):
            continue

        # Otherwise, this is a match we want to use
        is_winner = cur_match["winner"] == player_name

        cur_dict = {"date": cur_date}

        if is_winner:
            cur_dict.update(cur_match["winner_prior_mean"])
        else:
            cur_dict.update(cur_match["loser_prior_mean"])

        player_history.append(cur_dict)

    final_date = max(dates)
    final_player_rating = final_ratings_dict[player_name]
    final_player_rating["date"] = final_date

    player_history.append(final_player_rating)

    return pd.DataFrame(player_history).set_index("date")
