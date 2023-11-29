import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pp
import sklearn.decomposition as dc
import sklearn.cluster as cl
import sklearn.metrics as me
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("spotify_songs.csv")
df.shape
df.head()

df.info()
df.describe()

def CorrMat(df):
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def ScatterMatrix(df):
    pd.plotting.scatter_matrix(df);
    # plt.show()

def normalise(df):
    x = df.select_dtypes(include='number').values
    scaler = pp.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    dfnorm = pd.DataFrame(x_scaled,columns = df.select_dtypes(include='number').columns)
    return dfnorm

def boxplot(df):
    dfnorm = normalise(df)
    dfnorm.boxplot(rot=90)
    plt.show()

def large_correlations(df, correlation_matrix):
    corrmat_big = correlation_matrix
    corrmat_big[np.absolute(corrmat_big.values) < 0.1] = 0
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

def highest_correlators(df):
    fig, ([ax1, ax2, ax3],[ax4,ax5,ax6]) = plt.subplots(2, 3)
    ax1.scatter(df['valence'],df['danceability'],s=1)
    ax1.set_xlabel('valence')
    ax1.set_ylabel('danceability')
    ax2.scatter(df['speechiness'],df['danceability'],s=1)
    ax2.set_xlabel('speechiness')
    ax2.set_ylabel('danceability')
    ax3.scatter(df['loudness'],df['energy'],s=1)
    ax3.set_xlabel('loudness')
    ax3.set_ylabel('energy')
    ax4.scatter(df['liveness'],df['energy'],s=1)
    ax4.set_xlabel('liveness')
    ax4.set_ylabel('energy')
    ax5.scatter(df['valence'],df['energy'],s=1)
    ax5.set_xlabel('valence')
    ax5.set_ylabel('energy')
    ax6.scatter(df['tempo'],df['energy'],s=1)
    ax6.set_xlabel('tempo')
    ax6.set_ylabel('energy')
    plt.show()

def seperate_genre(df):
    genres = df['playlist_genre'].unique()
    EDM = normalise(df[df["playlist_genre"]=="edm"])
    Latin = normalise(df[df["playlist_genre"]=="latin"])
    Pop = normalise(df[df["playlist_genre"]=="pop"])
    RnB = normalise(df[df["playlist_genre"]=="r&b"])
    Rap = normalise(df[df["playlist_genre"]=="rap"])
    Rock = normalise(df[df["playlist_genre"]=="rock"])
    return EDM, Latin, Pop, RnB, Rap, Rock

def hist_by_genre(df):
    EDM, Latin, Pop, RnB, Rap, Rock = seperate_genre(df)
    alpha = 0.7
    for column in df.select_dtypes(include="number").columns.values:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        EDM[[column]].hist(ax=ax,alpha=alpha,label='EDM')
        Latin[[column]].hist(ax=ax,alpha=alpha,label='Latin')
        Pop[[column]].hist(ax=ax,alpha=alpha,label='Pop')
        RnB[[column]].hist(ax=ax,alpha=alpha,label='RnB')
        Rap[[column]].hist(ax=ax,alpha=alpha,label='Rap')
        Rock[[column]].hist(ax=ax,alpha=alpha,label='Rock')
        ax.legend()

def var_by_genre(df):
    EDM, Latin, Pop, RnB, Rap, Rock = seperate_genre(df)
    genre_data = [EDM, Latin, Pop, RnB, Rap, Rock]
    variances = np.zeros(13)
    means = np.zeros(6)
    v = 0
    columns = df.select_dtypes(include='number').columns
    for column in columns:
        m = 0
        for genre in genre_data:
            means[m] = genre[column].mean()
            m+=1
        variances[v] = means.std()
        v+=1
    plt.bar(columns,variances)

# call functions here, for example:
highest_correlators(df)

dfnorm = normalise(df)

# Compute cosine similarity between tracks based on selected features
track_features = dfnorm
track_similarity_matrix = cosine_similarity(dfnorm)

track_similarity_df = pd.DataFrame(track_similarity_matrix, index=df['track_id'], columns=df['track_id'])

def find_top_similar_songs(song_id, top_n=5):
    chosen_song_index = df[df['track_id'] == song_id].index[0]
    similarity_scores = track_similarity_df.iloc[chosen_song_index]
    similar_songs = similarity_scores.sort_values(ascending=False)
    top_similar_songs = similar_songs.iloc[1:top_n+1]
    top_similar_song_ids = top_similar_songs.index.tolist()
    top_similar_songs_details = df[df['track_id'].isin(top_similar_song_ids)]

    return top_similar_songs_details

# Try the function with a given track_id
chosen_track_id = '1e8PAfcKUYoKkxPhrHqw4x'
top_similar_songs = find_top_similar_songs(chosen_track_id, top_n=5)
print(top_similar_songs)

def plot_similarity_comparison(chosen_track_id, top_similar_songs):

    columns = df.select_dtypes(include='number').columns
    chosen_song = df[df['track_id'] == chosen_track_id]
    chosen_song_features = chosen_song[columns].values.flatten()
    top_similar_songs_features = top_similar_songs[columns].values

    plt.figure(figsize=(16, 8))
    bar_width = 0.2
    index = range(len(dfnorm))
    plt.bar(index, chosen_song_features, bar_width, label=f'Chosen Song: {chosen_song.iloc[0]["track_name"]}', color='#245eb8')
    
    colors = ['#D4D4D4', '#B4B4B4', '#909090', '#636363', '#494848']

    for i, similar_song_features in enumerate(top_similar_songs_features):
        plt.bar([x + (i + 1) * bar_width for x in index], similar_song_features, bar_width, label=top_similar_songs.iloc[i]['track_name'], color=colors[i])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Features')
    plt.ylabel('Normalized Values')
    plt.title(f'Similarity Comparison of Features for Chosen Song to Top 5 Similar Ones')
    plt.xticks([x + (len(top_similar_songs) + 1) * bar_width / 2 for x in index], col, rotation=45, ha='right') 
    plt.legend(fontsize=10)
    plt.show()

plot_similarity_comparison(chosen_track_id, top_similar_songs)


