import spotipy
import argparse
import pandas as pd

from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from sklearn.base import ClusterMixin
from clustering import *
import playlists


def main(song_name):

    playlists.load_dataset()
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    search_query = song_name
    results = spotify.search(q=search_query, type="track", limit=1)["tracks"]

    tracks = results["items"]
    for track in tracks:
        print(track["name"])
        print([artist["name"] for artist in track["artists"]])
        print(track["external_urls"])

    features = pd.DataFrame(spotify.audio_features(tracks=[track["id"] for track in tracks]))
    preprocessed_features = preprocess_dataset(features)

    try:
        km: ClusterMixin = joblib.load("trained_clastering")
    except (OSError, IOError) as e:
        km = train_clustering()

    prediction = km.predict(preprocessed_features)[0]

    print(prediction)

    cluster_members = find_cluster_members(prediction)
    popular = find_popular(cluster_members)
    recommendations = find_most_similar(preprocessed_features, popular)
    print(features)
    print(recommendations)

    for _, row in recommendations.iterrows():
        isrc = row["isrc"]
        query = f"isrc:{isrc}"
        results = spotify.search(q=query, type="track", limit=1)["tracks"]
        print(f"Similarity: {row['similarity']}")
        tracks = results["items"]
        for track in tracks:
            print(track["name"])
            print([artist["name"] for artist in track["artists"]])
            print(track["external_urls"])
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Song recommendation system")

    parser.add_argument("song_name", type=str, help="Name of the song")

    arguments = parser.parse_args()

    main(arguments.song_name)
