import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pp
import sklearn.decomposition as dc
import sklearn.cluster as cl
import sklearn.metrics as me
from yellowbrick.cluster import KElbowVisualizer

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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    sp.cluster.hierarchy.dendrogram(linkage_matrix, **kwargs)
    plt.show()

def clustering(df):
    dfnorm = normalise(df)
    model = cl.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
    model = model.fit(dfnorm)
    plot_dendrogram(model, color_threshold=1.5, no_labels=True)

# call functions here, for example:
highest_correlators(df)


