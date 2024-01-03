import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from tkinter import ttk
import tkinter as tk


#Bộ data tổng quát
data = pd.read_csv(r"D:/HHTQD_BTL/data.csv")
#Bộ data về các yếu tố bài hát
data1 = pd.read_csv(r"D:/HHTQD_BTL/data_by_year.csv")
#Bộ data về các đặc trưng riêng về âm thanh
data2 = pd.read_csv(r"D:/HHTQD_BTL/data_by_genres.csv")
print(data)
print(data1)
print(data2)

from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness','danceability','energy','instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X,y = data[feature_names],data ['popularity'] 

features = np.array(feature_names)


visualizer = FeatureCorrelation(labels= feature_names)
visualizer.fit(X,y)
visualizer.show()
#Thêm data decade vào dataframe và biểu đồ thể hiện số lượng bản ghi vào mỗi thập kỉ
def get_decade(year):
    period_start = int(year/10)*10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)
sns.set(rc={'figure.figsize':(11 ,6)})
sns.countplot(data['decade'])
plt.show()
#Tạo biểu đồ về các đặc trưng âm thanh qua từng giai đoạn
sound_features = ['acousticness','danceability', 'energy','instrumentalness','liveness','valence']
fig = px.line(data1, x ='year', y = sound_features)
fig.show()
#Biểu đồ thể hiện 10 thể loại âm nhạc phổ biến nhất ('popularity')
top10_genres = data2.nlargest(10,'popularity')
fig = px.bar(top10_genres, x = 'genres', y = ['valence','energy','danceability','acousticness'],
             barmode= 'group')
fig.show()

#Elbow xác định số lượng cụm
#elbow =[]
#for k in range (1,20):
#    kmeans = KMeans(n_clusters= k)
#    kmeans.fit(X)
#    elbow.append(kmeans.inertia_)
    
#plt.figure(figsize=(11, 6))
#plt.plot(range(1, 20), elbow, marker='o')
#plt.title('Phương pháp elbow')
#plt.xlabel('Số lượng cụm  (k)')
#plt.ylabel('Thông số')
#plt.show()
#Phân cụm K-Means

#Tiến hành phân cụm bài hát
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = data2.select_dtypes(np.number)
cluster_pipeline.fit(X)
data2['cluster'] = cluster_pipeline.predict(X)
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = data2['genres']
projection['cluster'] = data2['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

os.environ["SPOTIFY_CLIENT_ID"] = "edb7a7da678343ecb02b1664d2a86bec"
os.environ["SPOTIFY_CLIENT_SECRET"] = "4b5381f65bea450da8971f1a43add621"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

result = recommend_songs([{'name': 'Come As You Are', 'year':1991},
                ],  data)
print(result)  

import tkinter as tk
from tkinter import ttk

def recommend_songs_gui():
    def on_button_click():
        name = entry.get()
        result = recommend_songs([{'name': name, 'year': year_var.get()}], data)
        update_recommendations(result)

    def update_recommendations(recommended_songs):
        recommendation_list.delete(1.0, tk.END)  # Clear the existing text
        for song in recommended_songs:
            recommendation_list.insert(tk.END, f"{song['name']} - {song['artists']} ({song['year']})\n")

    # Create the main window
    window = tk.Tk()
    window.title("Music Recommendation System")
    window.geometry("600x400")
    window.configure(bg="white")

    # Create and place widgets
    entry_label = ttk.Label(window, text="Enter Song Name:", background="white")
    entry_label.grid(row=0, column=0, pady=10)

    entry = ttk.Entry(window)
    entry.grid(row=0, column=1, pady=10)

    year_label = ttk.Label(window, text="Enter Year:", background="white")
    year_label.grid(row=1, column=0, pady=10)

    year_var = tk.StringVar()
    year_entry = ttk.Entry(window, textvariable=year_var)
    year_entry.grid(row=1, column=1, pady=10)

    search_button = ttk.Button(window, text="Search", command=on_button_click)
    search_button.grid(row=2, column=0, columnspan=2, pady=10)

    recommendation_label = ttk.Label(window, text="Recommended Songs:", background="white")
    recommendation_label.grid(row=3, column=0, pady=10)

    recommendation_list = tk.Text(window, height=10, width=40)
    recommendation_list.grid(row=3, column=1, pady=10)

    # Start the Tkinter event loop
    window.mainloop()

# Call the function to display the GUI
recommend_songs_gui()              