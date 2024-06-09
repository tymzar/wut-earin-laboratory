import spotipy
import os
import pandas as pd
import dotenv
from clustering import (
    preprocess_dataset,
    train_clustering,
    find_cluster_members,
    find_popular,
    find_most_similar,
)
from sklearn.base import ClusterMixin
import joblib


class Pipeline:

    def __initialize_spotify_client(self):

        # load env variables from .env file
        dotenv.load_dotenv()

        self.__spotify_client = spotipy.Spotify(
            client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials()
        )

    def __load_cluster_model(self, cluster_model_path: str):

        if os.path.isfile(cluster_model_path):
            try:
                self.__cluster_model = joblib.load(cluster_model_path)
            except (OSError, IOError, FileNotFoundError):
                self.__cluster_model = train_clustering()
        else:
            self.__cluster_model = train_clustering(cluster_model_path)

    def __init__(
        self,
        popularity: int,
        temperature: int,
        cluster_model_path="trained_clustering_spectral",
    ) -> None:

        self.popularity = popularity
        self.temperature = temperature

        self.__initialize_spotify_client()

        self.__spotify_client: spotipy.Spotify = None

        self.__initialize_spotify_client()

        if self.__spotify_client is None:
            raise Exception("Could not create Spotify API client")

        self.__cluster_model: ClusterMixin = None

        self.__load_cluster_model(cluster_model_path)

    def __retrieve_song_features(self, song_name: str):

        results = self.__spotify_client.search(q=song_name, type="track", limit=1)[
            "tracks"
        ]
        tracks = results["items"]
        features = pd.DataFrame(
            self.__spotify_client.audio_features(
                tracks=[track["id"] for track in tracks]
            )
        )

        return features

    def __print_recommendations(self, recommendations: pd.DataFrame):

        for _, row in recommendations.iterrows():
            print(f"ISRC: {row['isrc']} - Similarity: {row['similarity']}")

            song_query = f"isrc:{row['isrc']}"
            results = self.__spotify_client.search(q=song_query, type="track", limit=1)[
                "tracks"
            ]
            tracks = results["items"]
            for track in tracks:
                print(f"Song: {track['name']}")
                print(
                    f"Artists: {', '.join([artist['name'] for artist in track['artists']])}"
                )
                print(f"URL: {track['external_urls']}\n")

    def recommend(self, reference_song_sample: str):

        print("Retrieving song features...")
        song_features = self.__retrieve_song_features(reference_song_sample)

        self.preprocessed_features = preprocess_dataset(song_features)

        sample_song_cluster = self.__cluster_model.predict(self.preprocessed_features)[
            0
        ]

        cluster_members = find_cluster_members(sample_song_cluster, self.temperature)

        print(f"Selecting songs according to popularity > {self.popularity}")
        popular = find_popular(cluster_members, self.popularity)

        print("Predicting most similar songs...")
        recommendations = find_most_similar(self.preprocessed_features, popular)

        self.__print_recommendations(recommendations)
