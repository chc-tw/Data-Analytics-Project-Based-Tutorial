import math
from src.model import GMM
import requests
import numpy as np
circle_of_fifths = {
    0: 0,   # C
    1: 7,   # bD / C#
    2: 2,   # D
    3: 9,   # bE / D#
    4: 4,   # E
    5: 11,  # F
    6: 6,   # bG / F#
    7: 1,   # G
    8: 8,   # bA / G#
    9: 3,   # A
    10: 10, # bB / A#
    11: 5,   # B
    -1: -1, #Unknown
}

def circle_distance(note1, note2):
    """
    計算兩個音階在五度圈中的距離。
    note1, note2: 音階的數字編碼 (0~11)
    """
    pos1 = circle_of_fifths[note1]
    pos2 = circle_of_fifths[note2]
    if pos1 == -1 or pos2 == -1:
        return 6
    distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
    return distance

def get_track_id(track_name, artist_name, access_token):
    # Search API endpoint
    search_url = "https://api.spotify.com/v1/search"
    
    # 組合搜尋查詢
    # query = f"track:{track_name} artist:{artist_name}"
    query = f"remaster%20track:晴天%20artist:周杰倫%20Davis"
    
    # 設置請求頭
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # 設置搜尋參數
    params = {
        "q": query,
        "type": "track",
        "limit": 1  # 只取第一個結果
    }
    
    # 發送請求
    response = requests.get(search_url, headers=headers, params=params)
    print(response.status_code)
    if response.status_code == 200:
        results = response.json()
        if results["tracks"]["items"]:
            return results["tracks"]["items"][0]["id"]
    return None

def get_track(track_id, access_token):
    track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(track_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def get_audio_features(track_id, access_token):
    # Audio Features API endpoint
    features_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(features_url, headers=headers)
    
    if response.status_code == 200:
        features = response.json()
        # Extract only the numeric features in a specific order
        numeric_features = np.array([
            features['danceability'],
            features['energy'],
            features['key'],
            features['loudness'],
            features['mode'],
            features['speechiness'],
            features['acousticness'],
            features['instrumentalness'],
            features['liveness'],
            features['valence'],
            features['tempo']
        ], dtype=float).reshape(1, -1)
        return numeric_features
    return None
