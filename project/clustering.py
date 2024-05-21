import kaggle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib
from pandas import DataFrame
import numpy as np


def preprocess_dataset(df: DataFrame) -> DataFrame:
    df = df.drop(
        columns=[
            "isrc",
            "updated_on",
            "loudness",
            "tempo",
            "uri",
            "track_href",
            "analysis_url",
            "id",
            "type",
        ],
        errors="ignore",
    )

    df = df.sort_index(axis=1)

    try:
        standard_scaler: StandardScaler = joblib.load("trained_scaler")
    except (OSError, IOError) as e:
        standard_scaler = StandardScaler()
        standard_scaler.fit(df)
        joblib.dump(standard_scaler, "trained_scaler")

    data_scaled = standard_scaler.transform(df)
    data = DataFrame(data_scaled, columns=df.columns)
    return data


def load_dataset() -> DataFrame:
    if not os.path.isdir("datasets/10-m-tracks"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mcfurland/10-m-beatport-tracks-spotify-audio-features",
            path="datasets/10-m-tracks",
            unzip=True,
        )

    df = pd.read_csv("datasets/10-m-tracks/audio_features.csv", header=0)
    return df


def train_clustering() -> MiniBatchKMeans:
    data = load_dataset()
    data = preprocess_dataset(data)
    km = MiniBatchKMeans(
        init="k-means++", batch_size=15, n_clusters=6, max_iter=500, verbose=False
    )
    km.fit(data)
    print(km.labels_)
    print(km.cluster_centers_)
    joblib.dump(km, "trained_clastering")

    return km


def find_cluster_members(wanted_cluster):
    data = load_dataset()
    samples_to_get = 1000
    sample = data.sample(samples_to_get, ignore_index=True)
    sample_without_titles = sample.copy()
    sample_without_titles = preprocess_dataset(sample_without_titles)
    try:
        km: MiniBatchKMeans = joblib.load("trained_clastering")
    except (OSError, IOError) as e:
        km = train_clustering()

    predictions = km.predict(sample_without_titles)
    recommendations = []
    for id in range(0, samples_to_get):
        if len(recommendations) >= 5:
            break
        if predictions[id] == wanted_cluster:
            recommendations.append(sample.iloc[id]["isrc"])

    return recommendations

    # a = np.array(km.predict(sample))
    # print(np.where(a = sample, ))
