import numpy as np
import pandas as pd
import jax.numpy as jnp

from jax_elo.elo_functions.margin_functions import margin_functions
from jax_elo.core import (
    EloParams,
    optimise_elo,
    calculate_ratings_history,
    get_starting_elts,
)
from jax_elo.utils.encoding import encode_players


def _make_1d_a(n_matches):

    a = jnp.stack([jnp.ones(n_matches), -jnp.ones(n_matches)], axis=1)

    return a


def fit(winners, losers, margins, objective_mask=None, verbose=False):
    """Fits a model incorporating the margin of victory.

    Args:
        winners: The names of the winners, as a numpy array.
        losers: The names of the losers, as a numpy array.
        margins: The margins of victory, as a numpy array.
        objective_mask: If provided, must be a vector of shape [N,], where N is
            the number of matches. It should be one if the log likelihood of the
            match update should be used to compute the objective, and zero
            otherwise. This allows e.g. ignoring an initial set of matches when
            optimising.
        verbose: If True, prints the progress of the optimisation.

    Returns:
    Tuple: The first element will contain the optimal parameters; the second
    the result from the optimisation routine.
    """

    cov_mat = jnp.eye(1)
    cov_mat_elts = get_starting_elts(cov_mat)

    # a1, a2, sigma_obs as defined in the paper, _except_ that we take the sqrt
    # of a1 and sigma_obs since they will be squared later to make sure they
    # remain positive.
    # TODO: Is there a nicer way?
    start_theta = {
        "a1": jnp.sqrt(0.1),
        "a2": jnp.array(0.0),
        "sigma_obs": jnp.sqrt(0.1),
        "cov_mat": cov_mat_elts,
    }

    init_params = EloParams(
        theta=start_theta,
    )

    # Get winner and loser ids
    winner_ids, loser_ids, names = encode_players(winners, losers)

    n_matches = winner_ids.shape[0]
    n_players = len(names)

    a = _make_1d_a(n_matches)

    # y will just be the margins, but with an extra dimension
    y = jnp.reshape(margins, (-1, 1))

    opt_result = optimise_elo(
        init_params,
        margin_functions,
        winner_ids,
        loser_ids,
        a,
        y,
        n_players,
        verbose=verbose,
        objective_mask=objective_mask,
    )

    return opt_result


def calculate_ratings(parameters, winners, losers, margins):
    """Calculates ratings given the parameters.

    Args:
        parameters: The EloParameters to use. Can be found using the fit
            function.
        winners: The names of the winners, as a numpy array.
        losers: The names of the losers, as a numpy array.
        margins: The margins of victory, as a numpy array.

    Returns:
    A Tuple whose first element is a DataFrame containing the ratings before
    each match, and whose second element is a dictionary of the final ratings
    for each competitor.
    """

    a_full = _make_1d_a(winners.shape[0])
    y = margins.reshape(-1, 1)

    history, final_ratings = calculate_ratings_history(
        winners, losers, a_full, y, margin_functions, parameters
    )

    result_df = list()

    for cur_entry in history:

        cur_dict = {
            "winner": cur_entry["winner"],
            "loser": cur_entry["loser"],
            "winner_prior_mean": cur_entry["prior_mu_winner"][0] + 1500,
            "loser_prior_mean": cur_entry["prior_mu_loser"][0] + 1500,
            "winner_prior_prob": cur_entry["prior_win_prob"],
        }

        cur_dict = {
            x: y if x in ["winner", "loser"] else float(y) for x, y in cur_dict.items()
        }

        result_df.append(cur_dict)

    final_ratings = {x: float(y[0]) + 1500 for x, y in final_ratings.items()}

    return pd.DataFrame(result_df), final_ratings


def predict(ratings, parameters, player, opponent):
    """Predicts the win probability of a contest between a player and an
    opponent.

    Args:
        ratings: A dictionary mapping names to ratings, obtained e.g. through
            calculate_ratings.
        parameters: The EloParameters to use. Can be found using the fit
            function.
        player: The player to predict the win probability for.
        opponent: The opponent to predict the win probability for.

    Returns:
    The win probability for the given player.
    """

    player_rating = jnp.array([ratings[player]])
    opponent_rating = jnp.array([ratings[opponent]])

    win_prob = margin_functions.win_prob_fun(
        player_rating, opponent_rating, jnp.array([1, -1]), [], parameters
    )

    return float(win_prob)


def get_player_skill_history(ratings_df, final_ratings_dict, dates, player_name):
    """A helper function to extract a player's rating trajectory over time.

    Args:
        ratings_df: The DataFrame of ratings obtained through the predict
            function.
        final_ratings_dict: The dictionary of final ratings obtained through
            the predict function.
        dates: The dates for each match in the ratings_df.
        player_name: The player whose history to find.

    Returns:
    A DataFrame mapping dates to the player ratings on those dates.
    """

    relevant = (ratings_df["winner"] == player_name) | (
        ratings_df["loser"] == player_name
    )

    relevant_df = ratings_df[relevant]

    relevant_dates = np.array(dates)[relevant]

    ratings = [
        x.winner_prior_mean if x.winner == player_name else x.loser_prior_mean
        for x in relevant_df.itertuples()
    ]

    final_date = max(dates)

    history = [{"date": x, "rating": y} for x, y in zip(relevant_dates, ratings)]

    history.append({"date": final_date, "rating": final_ratings_dict[player_name]})

    return pd.DataFrame(history).set_index("date")
