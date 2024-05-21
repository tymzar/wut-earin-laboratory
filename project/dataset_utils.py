from pandas import DataFrame
import os
import pandas as pd
import kaggle


def load_track_details() -> DataFrame:
    if not os.path.isdir("datasets/10-m-tracks"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mcfurland/10-m-beatport-tracks-spotify-audio-features",
            path="datasets/10-m-tracks",
            unzip=True,
        )

    df = pd.read_csv("datasets/10-m-tracks/bp_track.csv")
    return df


def load_artist_release() -> DataFrame:
    if not os.path.isdir("datasets/10-m-tracks"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mcfurland/10-m-beatport-tracks-spotify-audio-features",
            path="datasets/10-m-tracks",
            unzip=True,
        )

    df = pd.read_csv("datasets/10-m-tracks/bp_artist_release.csv")
    return df


def load_artist_details() -> DataFrame:
    if not os.path.isdir("datasets/10-m-tracks"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mcfurland/10-m-beatport-tracks-spotify-audio-features",
            path="datasets/10-m-tracks",
            unzip=True,
        )

    df = pd.read_csv("datasets/10-m-tracks/bp_artist.csv")
    return df
