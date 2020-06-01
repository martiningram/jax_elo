import numpy as np
from sklearn.preprocessing import LabelEncoder
from .general import EloParams
import pandas as pd


def encode_players(winners, losers):

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([winners, losers]))

    return (encoder.transform(winners), encoder.transform(losers),
            encoder.classes_)


def get_surface_a(surfaces):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(surfaces)
    oh = np.zeros((len(surfaces), len(encoder.classes_)))

    oh[np.arange(len(surfaces)), encoded] = 1

    return oh, encoder.classes_
