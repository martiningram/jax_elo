import numpy as np
import pandas as pd
from glob import glob
from toolz import pipe, partial
from os.path import join, splitext


def get_data(sackmann_dir, tour='atp', keep_davis_cup=False):

    all_csvs = glob(join(sackmann_dir, f'*{tour}_matches_????.csv'))
    all_csvs = sorted(all_csvs, key=lambda x: int(splitext(x)[0][-4:]))

    levels_to_drop = ['C', 'S']

    if not keep_davis_cup:
        levels_to_drop.append('D')

    data = pipe(all_csvs,
                # Read CSV
                lambda y: map(partial(pd.read_csv, encoding="ISO=8859-1"), y),
                # Drop NAs in important fields
                lambda y: map(lambda x: x.dropna(
                    subset=['winner_name', 'loser_name', 'score']),
                    y),
                # Drop retirements and walkovers
                # TODO: Make this optional
                lambda y: map(lambda x:
                              x[~x['score'].astype(str).str.contains(
                                'RET|W/O|DEF|nbsp|Def.')],
                              y),
                # Drop scores that appear truncated
                lambda y: map(lambda x: x[
                    x['score'].astype(str).str.len() > 4],
                    y),
                # Drop challengers and futures
                # TODO: Make this optional too
                lambda y: map(lambda x: x[
                    ~x['tourney_level'].isin(levels_to_drop)],
                    y),
                pd.concat,
                )

    round_numbers = {
        'R128': 1,
        'RR': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'SF': 6,
        'F': 7
    }

    # Drop rounds outside this list
    to_keep = data['round'].isin(round_numbers)
    data = data[to_keep]

    # Add a numerical round number
    data['round_number'] = data['round'].replace(round_numbers)

    # Add date information
    data['tourney_date'] = pd.to_datetime(
        data['tourney_date'].astype(int).astype(str), format='%Y%m%d')
    data['year'] = data['tourney_date'].dt.year

    # Sort by date and round and reset index
    data = data.sort_values(['tourney_date', 'round_number'])
    data = data.reset_index(drop=True)

    data['pts_won_serve_winner'] = data['w_1stWon'] + data['w_2ndWon']
    data['pts_won_serve_loser'] = data['l_1stWon'] + data['l_2ndWon']

    data['pts_played_serve_winner'] = data['w_svpt']
    data['pts_played_serve_loser'] = data['l_svpt']

    # Add serve % won
    data['spw_winner'] = (data['w_1stWon'] + data['w_2ndWon']) / data['w_svpt']
    data['spw_loser'] = (data['l_1stWon'] + data['l_2ndWon']) / data['l_svpt']

    return data
