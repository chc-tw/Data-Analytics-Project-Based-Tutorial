# Spotify Music Recommendation System

A machine learning-powered music recommendation system that uses Gaussian Mixture Models (GMM) to suggest similar songs based on Spotify's audio features. The system analyzes musical characteristics like danceability, energy, tempo, and valence to find songs with similar acoustic properties.

## Features

- Input up to 5 songs with their artists
- Analyzes audio features using Spotify's API
- Uses GMM clustering to find similar songs
- Displays recommendations with album artwork and Spotify links
- Shows song details including track name, artist, and album

## Prerequisites

- Python 3.10+
- Spotify Developer Account
- Spotify API Access Token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies using Pipenv:
```bash
pipenv install
```


1. Set up your Spotify Developer credentials:
- Create an application in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- Get your Client ID and Client Secret
- Generate an access token

## Running the Application

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

4. Enter your Spotify access token when prompted

5. Input up to 5 songs with their respective artists

6. Click "Get Recommendations" to receive personalized song suggestions

## How It Works

The system uses a pre-trained Gaussian Mixture Model to cluster songs based on their audio features. When you input a song:

1. The app fetches the song's audio features from Spotify's API
2. The GMM model predicts which cluster the song belongs to
3. The system randomly selects another song from the same cluster
4. Recommendations are displayed with album artwork and links to Spotify

The recommendations are based on the audio features of the input songs and the clusters learned by the GMM model. Here are the features used:

- danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
- energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
- key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
- loudness: The overall loudness of a track in decibels (dB)
- mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
- speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
- instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
- valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
- tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration





## Note

- The access token expires periodically, so you'll need to generate a new one if you receive authentication errors
- The quality of recommendations depends on the accuracy of the song and artist names provided

## License

This project is licensed under the MIT License - see the LICENSE file for details.