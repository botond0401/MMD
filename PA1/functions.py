import pandas as pd
import numpy as np

from math import sqrt
from typing import List, Tuple, Union


def create_random_matrices(
        n: int,
        d: int,
        l: int,
        values: Tuple[float, ...] = (-sqrt(3), 0., sqrt(3)),
        probs: Tuple[float, ...] = (1/6, 2/3, 1/6),
        seed: int = 42
) -> List[np.ndarray]:
    """
    Function which creates n random matrices with dimensions dxl using the
    randomized matrix method and returns them in a list.
    """
    np.random.seed(seed)

    random_matrices = [np.random.choice(values, size=(d, l), p=probs) for _ in range(n)]

    return random_matrices


def add_bucket(
        df_features: pd.DataFrame,
        list_random_matrices: List[np.ndarray]
) -> pd.DataFrame:
    """
    Function which returns appends the buckets of tracks to the original df
    e.g.: values like 100010
    """
    for index, R in enumerate(list_random_matrices):
        hashes = (df_features.values @ R > 0).astype(int).astype(str)
        buckets = [''.join(hash) for hash in hashes]
        df_features[f'bucket_{index}'] = buckets

    return df_features


def find_similar_tracks(
        df_features_new: pd.DataFrame,
        df_features_training: pd.DataFrame,
) -> List[pd.DataFrame]:
    """
    Function which finds for all tracks in df_features_new finds the similar tracks in df_features_training and returns their indices.
    e.g.: return a subset of df_features_training for each track in df_features_new
    """
    # the input functions can already contain the columns 'bucket_i' or the function add_bucket can be called inside
    pass


def predict_genre(
        df_features,
        list_df_features_similar,
        df_tracks,
        m,
        k
) -> pd.Series:
    """
    Predict the genre of a song/songs as the majority genre of its
    k-nearest-neighbours defined as the k most similar music tracks as ranked by m
    for each track in df_features return a genre, these can be stored in a pd.Series


    list_df_features_similar is the output of find_similar_tracks
    df_tracks has the genre values

    """
    pass

