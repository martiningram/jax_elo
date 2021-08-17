import pandas as pd
import jax.numpy as jnp

from jax_elo.core import (
    EloParams,
    optimise_elo,
    calculate_ratings_history,
    get_starting_elts,
)
from jax_elo.elo_functions.basic import basic_functions
from jax_elo.elo_functions.margin_functions import margin_functions
from jax_elo.utils.encoding import encode_players, encode_marks


def fit(winners, losers, marks, objective_mask=None, margins=None, verbose=False):
    """Fits the parameters of the correlated skills model.

    Args:
        winners: The names of the winners, as a numpy array.
        losers: The names of the losers, as a numpy array.
        marks: The names of the marks played on. This could e.g. be surfaces in
            tennis.
        objective_mask: If provided, must be a vector of shape [N,], where N is
            the number of matches. It should be one if the log likelihood of the
            match update should be used to compute the objective, and zero
            otherwise. This allows e.g. ignoring an initial set of matches when
            optimising.
        margins: The margins of victory, as a numpy array. They are optional.
        verbose: If True, prints the progress of the optimisation.

    Returns:
    Tuple: The first element will contain the optimal parameters; the second
    the result from the optimisation routine.
    """

    n_matches = len(winners)

    if margins is None:
        start_theta = {}
    else:
        start_theta = {
            "a1": jnp.sqrt(0.01),
            "a2": jnp.array(0.0),
            "sigma_obs": jnp.sqrt(0.1),
        }

    dummy_marks, mark_names = encode_marks(marks)
    a = jnp.concatenate([dummy_marks, -dummy_marks], axis=1)

    n_marks = len(mark_names)
    cov_mat = jnp.eye(n_marks) * 100 ** 2
    start_elts = get_starting_elts(cov_mat)
    start_theta["cov_mat"] = start_elts

    init_params = EloParams(
        theta=start_theta,
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
        init_params,
        functions,
        winner_ids,
        loser_ids,
        a,
        y,
        n_players,
        verbose=verbose,
        objective_mask=objective_mask,
    )

    return opt_result


def calculate_ratings(parameters, winners, losers, marks, margins=None):
    """Calculates ratings given the parameters.

    Args:
        parameters: The EloParameters to use. Can be found using the fit
            function.
        winners: The names of the winners, as a numpy array.
        losers: The names of the losers, as a numpy array.
        marks: The names of the marks played on. This could e.g. be surfaces in
            tennis.
        margins: The margins of victory, as a numpy array.

    Returns:
    A Tuple whose first element is a list containing the ratings before
    each match, and whose second element is a dictionary of the final ratings
    for each competitor.
    """

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
        winners, losers, a_full, y, functions, parameters
    )

    results = list()

    for cur_entry, cur_mark in zip(history, marks):

        cur_dict = {
            "winner": cur_entry["winner"],
            "loser": cur_entry["loser"],
            "winner_prior_mean": {
                x: float(y + 1500)
                for x, y in zip(mark_names, cur_entry["prior_mu_winner"])
            },
            "loser_prior_mean": {
                x: float(y + 1500)
                for x, y in zip(mark_names, cur_entry["prior_mu_loser"])
            },
            "winner_prior_prob": float(cur_entry["prior_win_prob"]),
            "cur_mark": cur_mark,
        }

        results.append(cur_dict)

    final_ratings = {
        player_name: {
            mark_name: cur_rating + 1500
            for mark_name, cur_rating in zip(mark_names, rating_array)
        }
        for player_name, rating_array in final_ratings.items()
    }

    return results, final_ratings, mark_names


def predict(ratings, parameters, player, opponent, mark, mark_names):
    """Predicts the win probability of a contest between a player and an
    opponent.

    Args:
        ratings: A dictionary mapping names to ratings, obtained e.g. through
            calculate_ratings.
        parameters: The EloParameters to use. Can be found using the fit
            function.
        player: The player to predict the win probability for.
        opponent: The opponent to predict the win probability for.
        mark: The mark played on (e.g. surface in tennis).
        mark_names: The array of different marks used, e.g. as produced by
            calculate_ratings.

    Returns:
    The win probability for the given player.
    """

    player_rating = jnp.array([ratings[player][x] for x in mark_names])
    opponent_rating = jnp.array([ratings[opponent][x] for x in mark_names])

    cur_mark_oh = (mark_names == mark).astype(int)

    cur_a = jnp.concatenate([cur_mark_oh, -cur_mark_oh])

    win_prob = margin_functions.win_prob_fun(
        player_rating, opponent_rating, cur_a, [], parameters
    )

    return float(win_prob)


def get_player_skill_history(ratings_history, final_ratings_dict, dates, player_name):
    """A helper function to extract a player's rating trajectory over time.

    Args:
        ratings_df: The DataFrame of ratings obtained through the predict
            function.
        final_ratings_dict: The dictionary of final ratings obtained through
            the predict function.
        dates: The dates for each match in the ratings_df.
        player_name: The player whose history to find.

    Returns:
    A DataFrame mapping dates to the player ratings on those dates, with one
    column for each of the marks the model was fit to.
    """

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
