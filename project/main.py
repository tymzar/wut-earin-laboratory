import spotipy
import argparse
import pandas as pd

from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from sklearn.base import ClusterMixin
from clustering import preprocess_dataset, train_clustering, find_cluster_members

parser = argparse.ArgumentParser(description="Recommend a song")

parser.add_argument("value", help="Name of the input song")

args = parser.parse_args()


spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

search_query = args.value
results = spotify.search(q=search_query, type="track", limit=1)['tracks']

tracks = results['items']
for track in tracks:
    print(track["name"])
    print([artist["name"] for artist in track["artists"]])
    print(track["external_urls"])

features = spotify.audio_features(tracks=[track["id"] for track in tracks])
features = pd.DataFrame(features)
features = preprocess_dataset(features)

try:
    km: ClusterMixin = joblib.load("trained_clastering")
except (OSError, IOError) as e:
    km = train_clustering()

prediction = km.predict(features)[0]

print(prediction)

recommendations = find_cluster_members(prediction)

for recommendation in recommendations:
    query = f"isrc:{recommendation}"
    results = spotify.search(q=query, type="track", limit=1)['tracks']
    tracks = results['items']
    for track in tracks:
        print(track["name"])
        print([artist["name"] for artist in track["artists"]])
        print(track["external_urls"])