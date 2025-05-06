# Spotify-Recommendation-Algorithms-
This project builds a simple and effective music recommendation system using Spotify song data and the K-Nearest Neighbors (KNN) algorithm.

## 1. Import Libraries
- Essential libraries are imported:
- pandas, numpy for data manipulation.
- sklearn for machine learning tools (e.g., MinMaxScaler, NearestNeighbors).
- warnings to suppress unnecessary warnings.

We'll use content-based filtering using audio features (like tempo, energy, danceability, etc.) to recommend similar songs.

| Singer      | Acousticness | Danceability | Energy | Track ID               | Instrumentalness | Key | Liveness | Loudness | Mode | Song Name            | Speechiness | Tempo   | Spotify Link                                                     | Valence |
| ----------- | ------------ | ------------ | ------ | ---------------------- | ---------------- | --- | -------- | -------- | ---- | -------------------- | ----------- | ------- | ---------------------------------------------------------------- | ------- |
| Aastha Gill | 0.485        | 0.770        | 0.824  | 39ujbBjTwwqUFySaCYDMMT | 0.000000         | 1   | 0.318    | -6.491   | 0    | Proper Patola        | 0.0851      | 172.006 | [Link](https://api.spotify.com/v1/tracks/39ujbBjTwwqUFySaCYDMMT) | 0.490   |
| Aastha Gill | 0.143        | 0.825        | 0.666  | 5cjVsWqIkBQC7acTRhL0RO | 0.000003         | 4   | 0.237    | -4.847   | 0    | Kamariya             | 0.0554      | 96.987  | [Link](https://api.spotify.com/v1/tracks/5cjVsWqIkBQC7acTRhL0RO) | 0.763   |
| Aastha Gill | 0.236        | 0.663        | 0.551  | 3XYvdqcZrTmRntFDDbJkJd | 0.000036         | 3   | 0.0923   | -8.272   | 0    | Buzz (feat. Badshah) | 0.1090      | 113.314 | [Link](https://api.spotify.com/v1/tracks/3XYvdqcZrTmRntFDDbJkJd) | 0.601   |
| Aastha Gill | 0.00323      | 0.919        | 0.571  | 46GBoFCdFZZSjuGaZjZmGv | 0.001680         | 5   | 0.103    | -7.175   | 0    | Saara India          | 0.0687      | 105.007 | [Link](https://api.spotify.com/v1/tracks/46GBoFCdFZZSjuGaZjZmGv) | 0.231   |
| Aastha Gill | 0.129        | 0.867        | 0.720  | 6VwVEIiCro1EMyh9B6Om3v | 0.000000         | 9   | 0.228    | -5.188   | 0    | Drunk n High         | 0.0619      | 104.974 | [Link](https://api.spotify.com/v1/tracks/6VwVEIiCro1EMyh9B6Om3v) | 0.755   |


## 2. Load the Dataset
- Dataset: SingerAndSongs.csv
- Contains ~songs with metadata and audio features (e.g., Energy, Tempo, Danceability).
- data = pd.read_csv("SingerAndSongs.csv", encoding='latin1')

Explore the Dataset Structure
Run this code to understand what columns you have:

| Feature              | Count | Mean    | Std Dev | Min      | 25%     | Median   | 75%      | Max      |
| -------------------- | ----- | ------- | ------- | -------- | ------- | -------- | -------- | -------- |
| **Acousticness**     | 2231  | 0.3962  | 0.2963  | 0.0003   | 0.1180  | 0.3440   | 0.6475   | 0.9940   |
| **Danceability**     | 2231  | 0.6127  | 0.1560  | 0.1560   | 0.5050  | 0.6220   | 0.7340   | 0.9710   |
| **Energy**           | 2231  | 0.6582  | 0.1900  | 0.1030   | 0.5200  | 0.6690   | 0.8155   | 0.9880   |
| **Instrumentalness** | 2231  | 0.0120  | 0.0784  | 0.0000   | 0.0000  | 0.0000   | 0.0002   | 0.9670   |
| **Key**              | 2231  | 5.3967  | 3.4597  | 0.0000   | 2.0000  | 6.0000   | 8.0000   | 11.0000  |
| **Liveness**         | 2231  | 0.1841  | 0.1418  | 0.0222   | 0.0948  | 0.1260   | 0.2455   | 0.9720   |
| **Loudness (dB)**    | 2231  | -7.2509 | 2.9299  | -20.0900 | -8.8415 | -6.8870  | -5.3325  | 0.0030   |
| **Mode**             | 2231  | 0.5737  | 0.4946  | 0.0000   | 0.0000  | 1.0000   | 1.0000   | 1.0000   |
| **Speechiness**      | 2231  | 0.0749  | 0.0696  | 0.0232   | 0.0342  | 0.0483   | 0.0814   | 0.6840   |
| **Tempo (BPM)**      | 2231  | 115.489 | 26.2875 | 55.8320  | 95.9870 | 109.9820 | 130.0980 | 214.0160 |
| **Valence**          | 2231  | 0.5584  | 0.2163  | 0.0394   | 0.3850  | 0.5630   | 0.7310   | 0.9700   |


## 3. Data Cleaning
- Drops irrelevant columns: 'Index', 'Title', 'Artist', 'Top Genre', 'Year'
- Keeps only numeric/audio feature columns.

Available columns: ['Singer', 'acousticness', 'danceability', 'energy', 'id', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'Song name', 'speechiness', 'tempo', 'track_href', 'valence']
Required columns 'track_name' and 'artist_name' not found.
First few column names: Index(['singer', 'acousticness', 'danceability', 'energy', 'id'], dtype='object')

| Danceability | Energy | Loudness | Speechiness | Acousticness | Instrumentalness | Liveness | Valence | Tempo |
| ------------ | ------ | -------- | ----------- | ------------ | ---------------- | -------- | ------- | ----- |
| 0.753        | 0.815  | 0.677    | 0.0937      | 0.488        | 0.000000         | 0.311    | 0.484   | 0.734 |
| 0.821        | 0.636  | 0.759    | 0.0487      | 0.144        | 0.000003         | 0.226    | 0.778   | 0.260 |
| 0.622        | 0.506  | 0.588    | 0.1298      | 0.237        | 0.000037         | 0.074    | 0.603   | 0.363 |
| 0.936        | 0.529  | 0.643    | 0.0689      | 0.003        | 0.001737         | 0.085    | 0.206   | 0.311 |
| 0.872        | 0.697  | 0.742    | 0.0586      | 0.130        | 0.000000         | 0.217    | 0.769   | 0.311 |


## Build Recommendation Model
- Uses NearestNeighbors from sklearn to find songs with similar audio features.
- Fits the model on scaled feature data.

Define the Recommendation Function & Test the Recommender

- ✅ Found: Paani Paani
- ✅ Found: Naagin
- ✅ Found: 52 Non Stop Dilbar Dilbar Remix(Remix By Kedrock,Sd Style)
- ✅ Found: Dj Waley Babu (feat. Aastha Gill)
- ✅ Found: Crazy Lady
- ✅ Found: Badtameez
- ✅ Found: Call Waiting - Reprise
- ✅ Found: Light Kardo Band (feat. Aastha Gill)
- ✅ Found: Kareja Kareja (feat. Aastha Gill)


## Visualize Recommendations

![V3](https://github.com/user-attachments/assets/59bcafde-57c4-4bbb-9c2e-b61ee270a8eb)

![v1](https://github.com/user-attachments/assets/85e1097c-802e-461e-b0f8-8f5b4b196d53)

![v4](https://github.com/user-attachments/assets/ed91db41-9cd4-49d1-93b3-b33c551029de)


Average similarity (accuracy-like score): 99.65%

The plot_similar_songs() function then plots these recommendations in a line graph.

![v2](https://github.com/user-attachments/assets/45a0aa13-dbc9-4ad1-8d84-5f0c9db8a1e9)



## 📝 Conclusion
- This dataset provides audio features for 2,231 tracks, highlighting various attributes like danceability, energy, acousticness, and valence. Key takeaways include:
- Tracks vary widely in energy and danceability, with a mix of acoustic and electronic styles.
- Valence indicates a generally positive mood, while tempo spans from slow to fast tracks.
- Missing columns (track_name and artist_name) can be mapped from available data.

---

### 📄 License

This project is protected under copyright © Md Altamash Alam, 2025.

All rights reserved. Unauthorized copying, distribution, modification, or use of any part of this project without explicit permission is strictly prohibited.

If you wish to use or reference any part of this project for academic, personal, or commercial purposes, please contact the author for permission.

---

© Md Altamash Alam, 2025 – All Rights Reserved.
