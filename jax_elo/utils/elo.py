import numpy as np
from collections import defaultdict
from scipy.optimize import minimize


def win_probability(elo, opponent_elo):

    return 1 / (1 + 10**(-(elo - opponent_elo) / 400.0))


def compute_elo_ratings(winners, losers, k_func=lambda x: 32.):

    last_ratings = defaultdict(lambda: 1500)
    times_seen = defaultdict(lambda: 0)
    results = list()

    for cur_winner, cur_loser in zip(winners, losers):

        k_winner = k_func(times_seen[cur_winner])
        k_loser = k_func(times_seen[cur_loser])

        cur_elo_winner = last_ratings[cur_winner]
        cur_elo_loser = last_ratings[cur_loser]

        winner_prob = win_probability(cur_elo_winner, cur_elo_loser)

        results.append({'elo_winner': cur_elo_winner,
                        'elo_loser': cur_elo_loser,
                        'winner_prob': winner_prob})

        winner_update = k_winner * (1 - winner_prob)
        loser_update = k_loser * (0 - (1 - winner_prob))

        last_ratings[cur_winner] += winner_update
        last_ratings[cur_loser] += loser_update

        times_seen[cur_winner] += 1
        times_seen[cur_loser] += 1

    return results


def optimise_static_k(winners, losers, tol=1e-2):

    def to_minimise(k):

        k = k[0]

        ratings = compute_elo_ratings(winners, losers, k_func=lambda _: k)
        winner_probs = np.array([x['winner_prob'] for x in ratings])
        log_lik = np.sum(np.log(winner_probs))

        return -log_lik

    result = minimize(to_minimise, [32.], tol=tol)

    return result.x[0], result.success
