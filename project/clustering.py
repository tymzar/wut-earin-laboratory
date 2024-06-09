import kaggle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.base import ClusterMixin
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pandas import DataFrame


def prepare_columns(df: DataFrame) -> DataFrame:
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
            "duration_ms",
        ],
        errors="ignore",
    )

    df = df.sort_index(axis=1)
    return df


def preprocess_dataset(df: DataFrame) -> DataFrame:
    df = prepare_columns(df)

    try:
        standard_scaler: StandardScaler = joblib.load("trained_scaler")
    except (OSError, IOError):
        standard_scaler = StandardScaler()
        standard_scaler.fit(prepare_columns(load_dataset()))
        joblib.dump(standard_scaler, "trained_scaler")

    df = DataFrame(df, columns=standard_scaler.feature_names_in_)
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


def train_clustering(model_path, model_type="mk") -> MiniBatchKMeans:
    data = load_dataset()
    data = preprocess_dataset(data)

    if model_type == "mk":
        model = MiniBatchKMeans(
            init="k-means++",
            n_clusters=5,
            max_no_improvement=None,
            batch_size=10 * 256,
            verbose=False,
        )
        model.fit(data)

    else:
        model = GaussianMixture(
            n_components=50,
        )
        model.fit(data)

    joblib.dump(model, model_path)

    return model


def find_cluster_members(km: ClusterMixin, wanted_cluster, temperature: float):
    data = load_dataset()
    samples_to_get = int(data.shape[0] * temperature)
    samples = data.sample(samples_to_get, ignore_index=True)
    sample_without_titles = preprocess_dataset(samples)

    predictions = km.predict(sample_without_titles)
    recommendations = pd.concat(
        [samples, DataFrame(predictions, columns=["prediction"])], axis=1
    )
    recommendations = recommendations.query(f"prediction == {wanted_cluster}")

    return recommendations


def find_most_similar(wanted: DataFrame, candidates: DataFrame):
    candidates = candidates.reset_index()
    preprocessed_candidates: DataFrame = preprocess_dataset(candidates)
    most_similar = cosine_similarity(wanted, preprocessed_candidates)[0]
    most_similar = pd.concat(
        [candidates["isrc"], DataFrame(most_similar, columns=["similarity"])], axis=1
    )
    most_similar = most_similar.sort_values(by="similarity", ascending=False)
    return most_similar.head(5)


def find_popular(samples: DataFrame, popularity: float):
    sp_release, sp_track = get_popularity()
    samples = samples.merge(sp_track, on="isrc").merge(sp_release, on="release_id")

    highest_popularity = samples["popularity"].max()
    query_popularity = popularity * highest_popularity

    return samples.query(f"popularity / total_tracks > {query_popularity}")


def get_popularity() -> list[DataFrame]:
    sp_release = pd.read_csv("datasets/10-m-tracks/sp_release.csv", header=0)
    sp_track = pd.read_csv("datasets/10-m-tracks/sp_track.csv", header=0)
    return [sp_release, sp_track]
