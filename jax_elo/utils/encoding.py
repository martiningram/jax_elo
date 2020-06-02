import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_players(winners, losers):

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([winners, losers]))

    return (encoder.transform(winners), encoder.transform(losers),
            encoder.classes_)


def encode_marks(marks):

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(marks)
    oh = np.zeros((len(marks), len(encoder.classes_)))

    oh[np.arange(len(marks)), encoded] = 1

    return oh, encoder.classes_
