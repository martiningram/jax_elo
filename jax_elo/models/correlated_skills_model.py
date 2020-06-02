import pandas as pd
import jax.numpy as jnp

from ..basic import basic_functions
from ..margin_functions import margin_functions
from ..general import EloParams, optimise_elo, calculate_ratings_history
from ..utils import encode_players, encode_marks


def fit(winners, losers, marks, margins=None, verbose=False):

    n_matches = len(winners)

    if margins is None:
        # TODO: This is inelegant -- the reshaping functions require at least
        # one element in theta. Fix so we don't need to provide this dummy.
        start_theta = {
            'dummy': jnp.array(0.)
        }
    else:
        start_theta = {
            'a1': jnp.sqrt(0.01),
            'a2': jnp.array(0.),
            'sigma_obs': jnp.sqrt(0.1)
        }

    dummy_marks, mark_names = encode_marks(marks)
    a = jnp.concatenate([dummy_marks, -dummy_marks], axis=1)

    n_marks = len(mark_names)
    cov_mat = jnp.eye(n_marks) * 100**2

    init_params = EloParams(
        theta=start_theta,
        cov_mat=cov_mat
    )

    winner_ids, loser_ids, names = encode_players(winners, losers)
    n_players = len(names)

    if margins is None:
        y = jnp.zeros((n_matches, 0))
        functions = basic_functions
    else:
        y = jnp.reshape(margins, (-1, 1))
        functions = margin_functions

    opt_result = optimise_elo(
        init_params, functions, winner_ids, loser_ids, a, y,
        n_players, verbose=verbose)

    return opt_result


def calculate_ratings(parameters, winners, losers, marks, margins=None):

    dummy_marks, mark_names = encode_marks(marks)
    a_full = jnp.concatenate([dummy_marks, -dummy_marks], axis=1)

    n_matches = winners.shape[0]

    if margins is None:
        y = jnp.zeros((n_matches, 0))
        functions = basic_functions
    else:
        y = jnp.reshape(margins, (-1, 1))
        functions = margin_functions

    history, final_ratings = calculate_ratings_history(
        winners, losers, a_full, y, functions, parameters)

    results = list()

    for cur_entry, cur_mark in zip(history, marks):

        cur_dict = {
            'winner': cur_entry['winner'],
            'loser': cur_entry['loser'],
            'winner_prior_mean': {
                x: float(y + 1500) for x, y in zip(
                    mark_names, cur_entry['prior_mu_winner'])},
            'loser_prior_mean': {
                x: float(y + 1500) for x, y in zip(
                    mark_names, cur_entry['prior_mu_loser'])},
            'winner_prior_prob': float(cur_entry['prior_win_prob']),
            'cur_mark': cur_mark
        }

        results.append(cur_dict)

    final_ratings = {
        player_name: {
            mark_name: cur_rating + 1500 for mark_name, cur_rating in
                      zip(mark_names, rating_array)}
        for player_name, rating_array in final_ratings.items()
    }

    return results, final_ratings, mark_names


def predict(ratings, parameters, player, opponent, mark, mark_names):

    player_rating = jnp.array([ratings[player][x] for x in mark_names])
    opponent_rating = jnp.array([ratings[opponent][x] for x in mark_names])

    cur_mark_oh = (mark_names == mark).astype(int)

    cur_a = jnp.concatenate([cur_mark_oh, -cur_mark_oh])

    win_prob = margin_functions.win_prob_fun(
        player_rating, opponent_rating, cur_a, parameters.cov_mat)

    return float(win_prob)


def get_player_skill_history(ratings_history, final_ratings_dict, dates,
                             player_name):

    player_history = list()

    for cur_match, cur_date in zip(ratings_history, dates):

        if player_name not in (cur_match['winner'], cur_match['loser']):
            continue

        # Otherwise, this is a match we want to use
        is_winner = cur_match['winner'] == player_name

        cur_dict = {
            'date': cur_date
        }

        if is_winner:
            cur_dict.update(cur_match['winner_prior_mean'])
        else:
            cur_dict.update(cur_match['loser_prior_mean'])

        player_history.append(cur_dict)

    final_date = max(dates)
    final_player_rating = final_ratings_dict[player_name]
    final_player_rating['date'] = final_date

    player_history.append(final_player_rating)

    return pd.DataFrame(player_history).set_index('date')
