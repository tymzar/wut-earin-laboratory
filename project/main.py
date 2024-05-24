import spotipy
import argparse
import pandas as pd

from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from sklearn.base import ClusterMixin
from clustering import *


def main(song_name, popularity):

    # print("Loading playlists dataset...")
    # playlists.load_dataset()
    print("Logging into Spotify...")
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    search_query = song_name
    print("Searching for song in Spotify...")
    results = spotify.search(q=search_query, type="track", limit=1)["tracks"]

    tracks = results["items"]
    for track in tracks:
        print(track["name"])
        print([artist["name"] for artist in track["artists"]])
        print(track["external_urls"])

    print("Fetching audio features of the song")
    features = pd.DataFrame(
        spotify.audio_features(tracks=[track["id"] for track in tracks])
    )
    print("Preprocessing dataset...")
    preprocessed_features = preprocess_dataset(features)

    try:
        km: ClusterMixin = joblib.load("trained_clastering")
    except (OSError, IOError) as e:
        km = train_clustering()

    print("Predicting song cluster...")
    prediction = km.predict(preprocessed_features)[0]

    print(f"Predicted cluster: {prediction}")

    print("Finding other members of the cluster...")
    cluster_members = find_cluster_members(prediction)
    print("Finding most popular members...")
    popular = find_popular(cluster_members, popularity)
    print("Finding the most similar songs in cluster...")
    recommendations = find_most_similar(preprocessed_features, popular)
    print("Done!")
    print()

    for _, row in recommendations.iterrows():
        isrc = row["isrc"]
        query = f"isrc:{isrc}"
        results = spotify.search(q=query, type="track", limit=1)["tracks"]
        tracks = results["items"]
        for track in tracks:
            print(track["name"])
            print([artist["name"] for artist in track["artists"]])
            print(track["external_urls"])
            print(f"Similarity: {row['similarity']}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Song recommendation system")
    # string with spaces
    parser.add_argument(
        "song_name", type=str, help="Name of the song to be recommended"
    )
    parser.add_argument(
        "--popularity",
        type=int,
        default=1,
        help="Minimum popularity of the song to be considered",
    )

    # python main.py "Shape of You" --popularity 20

    arguments = parser.parse_args()

    main(arguments.song_name, arguments.popularity)
