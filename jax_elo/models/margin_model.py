import pandas as pd
import numpy as np
import jax.numpy as jnp
from ..margin_functions import margin_functions
from ..general import EloParams, optimise_elo, calculate_ratings_history
from ..utils import encode_players


def _make_1d_a(n_matches):

    a = jnp.stack([jnp.ones(n_matches), -jnp.ones(n_matches)], axis=1)

    return a


def fit(winners, losers, margins, verbose=False):

    # a1, a2, sigma_obs as defined in the paper, _except_ that we take the sqrt
    # of a1 and sigma_obs since they will be squared later to make sure they
    # remain positive.
    # TODO: Is there a nicer way?
    start_theta = {
        'a1': jnp.sqrt(0.1),
        'a2': jnp.array(0.),
        'sigma_obs': jnp.sqrt(0.1)
    }

    cov_mat = jnp.eye(1)

    init_params = EloParams(
        theta=start_theta,
        cov_mat=cov_mat
    )

    # Get winner and loser ids
    winner_ids, loser_ids, names = encode_players(winners, losers)

    n_matches = winner_ids.shape[0]
    n_players = len(names)

    a = _make_1d_a(n_matches)

    # y will just be the margins, but with an extra dimension
    y = jnp.reshape(margins, (-1, 1))

    opt_result = optimise_elo(
        init_params, margin_functions, winner_ids, loser_ids, a, y,
        n_players, verbose=verbose)

    return opt_result

def calculate_ratings(parameters, winners, losers, margins):

    a_full = _make_1d_a(winners.shape[0])
    y = margins.reshape(-1, 1)

    history, final_ratings = calculate_ratings_history(
        winners, losers, a_full, y, margin_functions, parameters)

    result_df = list()

    for cur_entry in history:

        cur_dict = {
            'winner': cur_entry['winner'],
            'loser': cur_entry['loser'],
            'winner_prior_mean': cur_entry['prior_mu_winner'][0] + 1500,
            'loser_prior_mean': cur_entry['prior_mu_loser'][0] + 1500,
            'winner_prior_prob': cur_entry['prior_win_prob'],
        }

        cur_dict = {x: y if x in ['winner', 'loser'] else float(y) for x, y in
                    cur_dict.items()}

        result_df.append(cur_dict)

    final_ratings = {x: float(y[0]) + 1500 for x, y in final_ratings.items()}

    return pd.DataFrame(result_df), final_ratings


def predict(ratings, parameters, player, opponent):

    player_rating = jnp.array([ratings[player]])
    opponent_rating = jnp.array([ratings[opponent]])

    win_prob = margin_functions.win_prob_fun(
        player_rating, opponent_rating, jnp.array([1, -1]), parameters.cov_mat)

    return float(win_prob)


def get_player_skill_history(ratings_df, final_ratings_dict, dates,
                             player_name):

    relevant = ((ratings_df['winner'] == player_name) |
                (ratings_df['loser'] == player_name))

    relevant_df = ratings_df[relevant]

    relevant_dates = np.array(dates)[relevant]

    ratings = [x.winner_prior_mean if x.winner == player_name
               else x.loser_prior_mean for x in relevant_df.itertuples()]

    final_date = max(dates)

    history = [{'date': x, 'rating': y}
               for x, y in zip(relevant_dates, ratings)]

    history.append({'date': final_date, 'rating':
                    final_ratings_dict[player_name]})

    return pd.DataFrame(history).set_index('date')
