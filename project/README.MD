# Song Recommendation System

This Python script uses the Spotipy library to search for a song on Spotify, extract its audio features, and recommend similar songs based on those features.

## Dependencies

- spotipy
- argparse
- pandas
- joblib
- sklearn
- A custom module named 'clustering'

## Environment Variables

- SPOTIPY_CLIENT_ID: The client ID of your Spotify Developer account.
- SPOTIPY_CLIENT_SECRET: The client secret of your Spotify Developer account.

## Kaggle Dataset

We used datesets deployed on Kaggle to train our model. To access the datasets first, go to https://www.kaggle.com/settings/account, log in and click "Create New Token". Next, copy the file kaggle.json to ~/.kaggle/kaggle.json.

## How it works

1. The script logs into Spotify using the `SpotifyClientCredentials` class from the Spotipy library.

2. It then searches for a song on Spotify using the `search` method of the Spotipy client. The search query is the song name provided by the user.

3. The script prints the name, artists, and external URLs of the found tracks.

4. It then fetches the audio features of the song using the `audio_features` method of the Spotipy client. The features are stored in a pandas DataFrame.

5. The script then preprocesses the features and uses a trained clustering model to predict the cluster of the song.

6. It then finds other songs in the same cluster, selects the most popular ones, and finds the most similar songs to the input song.

7. Finally, it prints the names, artists, external URLs, and similarity scores of the recommended songs.

## Usage

To use this script, you need to provide the song name and its popularity as command-line arguments.

```bash
python main.py "song name" --popularity 20
```

> [!NOTE]
> The --popularity argument is optional and defaults to 1. It specifies the minimum popularity of the song to be considered.

## Output

The script outputs a list of recommended songs. Each song is represented by a dictionary containing the following keys:

- 'name': The name of the song.
- 'artists': A list of the song's artists.
- 'external_urls': A dictionary containing the URL of the song on Spotify.
- 'similarity_score': A score representing how similar the song is to the input song.

## Example

Here's an example of how to use the script:

```bash
python main.py "Shape of You" --popularity 20
```

## Note

This script requires a Spotify Developer account and the corresponding client ID and client secret. These should be set as environment variables before running the script.
