import pandas as pd
import numpy as np

from math import sqrt
from typing import List, Tuple


def create_hash_tables(
        n: int,
        d: int,
        l: int,
        values: Tuple[float, ...] = (-sqrt(3), 0., sqrt(3)),
        probs: Tuple[float, ...] = (1/6, 2/3, 1/6),
        seed: int = 42
) -> List[pd.DataFrame]:
    """
    Function which creates n hash tables with dimensions dxl using the randomized matrix method and returns them in a list.
    """
    np.random.seed(seed)

    hash_tables = []

    for _ in range(n):
        random_values = np.random.choice(values, size=(d, l), p=probs)

        df_hash = pd.DataFrame(random_values)
        hash_tables.append(df_hash)

    return hash_tables


def get_bucket(
        df_tracks: pd.DataFrame,
        df_hash_table: pd.DataFrame
) -> pd.Series:
    """
    Function which returns the hash value(s) of tracks stored in ´df_hash_table´.
    e.g.: returns 100010 in a Dataframe
    """
    pass


def find_similar_tracks(
        df_tracks: pd.DataFrame,
        df_tracks_training: pd.DataFrame,
        list_hash_tables: List[pd.DataFrame]
) -> List[int]:
    """
    Function which finds for all tracks in df_tracks finds the similar tracks in df_tracks_training and returns their indices.
    e.g.: returns [3, 5, 57, 108] if df_tracks is only one tracks
    """
    pass


def predict_genre(
        df_features,
        list_df_features_similar,
        df_tracks
        m,
        k
) -> Union[str, List[str]]:
    """
    Predict the genre of a song/songs as the majority genre of its
    k-nearest-neighbours defined as the k most similar music tracks as ranked by m
    """
    pass

