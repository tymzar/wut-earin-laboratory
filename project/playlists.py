from pandas import DataFrame
import os
import pandas as pd
import kaggle


def load_dataset() -> DataFrame:
    if not os.path.isdir("datasets/spotify-playlists"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "andrewmvd/spotify-playlists",
            path="datasets/spotify-playlists",
            unzip=True,
        )

    df = pd.read_csv(
        "datasets/spotify-playlists/spotify_dataset.csv",
        header=0,
        on_bad_lines="skip",
    )
    return df
