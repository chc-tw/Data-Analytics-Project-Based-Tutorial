from src.util import get_track_id, get_audio_features, get_track
import streamlit as st
import numpy as np
import sqlite3
import random
import joblib

def get_random_track_from_cluster(cluster_id):
    """Get a random track ID from the same cluster using SQLite"""
    conn = sqlite3.connect('music.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id 
        FROM music_clusters
        WHERE cluster_id = ?
    """, (cluster_id,))
    
    track_ids = cursor.fetchall()
    conn.close()
    
    if not track_ids:
        return None
    
    return random.choice(track_ids)[0]

def get_songs_recommendations(songs_input, access_token):
    try:
        # Get track IDs and audio features for all input songs
        valid_songs = []
        audio_features_list = []
        
        for song_name, artist_name in songs_input:
            if not song_name or not artist_name:
                continue
                
            track_id = get_track_id(song_name, artist_name, access_token)
            if not track_id:
                continue
            
            features = get_audio_features(track_id, access_token)
            if features is not None:
                audio_features_list.append(features)
                valid_songs.append({
                    'song_name': song_name,
                    'artist_name': artist_name,
                    'track_id': track_id
                })

        if not audio_features_list:
            return []

        # Combine all features into a 2D array
        features_array = np.vstack(audio_features_list)
        
        # Predict clusters for all songs at once
        gmm = joblib.load('./model/gmm_model.pkl')
        cluster_ids = gmm.predict(features_array)
        
        # Get recommendations for each song
        recommendations = []
        for i, song_info in enumerate(valid_songs):
            cluster_id = int(cluster_ids[i])
            suggested_track_id = get_random_track_from_cluster(cluster_id)
            
            if suggested_track_id:
                track = get_track(suggested_track_id, access_token)
                recommendations.append({
                    'input_song': song_info['song_name'],
                    'input_artist': song_info['artist_name'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'image_url': track['album']['images'][0]['url'],
                    'song_url': track['external_urls']['spotify']
                })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error processing songs: {str(e)}")
        return []

def main():
    st.title("Music Recommendation System")
    
    # Add access token input field
    access_token = st.text_input(
        "Enter your Spotify Access Token", 
        type="password",
        help="You can get your access token from the Spotify Developer Dashboard"
    )

    if not access_token:
        st.warning("Please enter your Spotify Access Token to continue")
        return

    st.write("Enter 5 songs you like, and we'll recommend similar songs for each!")

    # Create input fields for 5 songs
    songs_input = []
    for i in range(5):
        col1, col2 = st.columns(2)
        with col1:
            song_name = st.text_input(f"Song {i+1} Name", key=f"song_{i}")
        with col2:
            artist_name = st.text_input(f"Artist {i+1} Name", key=f"artist_{i}")
        songs_input.append((song_name, artist_name))

    if st.button("Get Recommendations"):
        if not any(song[0] and song[1] for song in songs_input):
            st.warning("Please enter at least one song with its artist")
            return
            
        with st.spinner('Finding recommendations for your songs...'):
            recommendations = get_songs_recommendations(songs_input, access_token)
            
            if not recommendations:
                st.error("No recommendations found. Please check your inputs and try again.")
                return
                
            st.write("### Your Recommendations:")
            
            for recommendation in recommendations:
                st.write(f"\n#### Based on: {recommendation['input_song']} by {recommendation['input_artist']}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(recommendation['image_url'], width=200)
                
                with col2:
                    st.write(f"**Track:** {recommendation['track_name']}")
                    st.write(f"**Artist:** {recommendation['artist_name']}")
                    st.write(f"**Album:** {recommendation['album_name']}")
                    st.markdown(f"[Listen on Spotify]({recommendation['song_url']})")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
