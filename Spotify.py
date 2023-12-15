import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn.preprocessing as pp
import sklearn.decomposition as dc
import sklearn.cluster as cl
import sklearn.metrics as me
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans #HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm



df = pd.read_csv("spotify_songs.csv")
df.shape
df.head()

df.info()
df.describe()
df.hist(figsize=(14, 9), bins=20)

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

def extract_year(date_str):
    return pd.to_datetime(date_str, format = 'mixed').year

def do_pca(df):
    pca = dc.PCA()
    X = pca.fit_transform(df)
    return X, pca

def analyze_cluster(data, labels, cluster_number):
    data['cluster_label'] = labels
    cluster_points = data[data['cluster_label'] == cluster_number]
    cluster_mean = cluster_points.mean()
    print(f"Cluster {cluster_number} features' means:\n", cluster_mean)



# call functions here, for example:
highest_correlators(df)









# ============================================
#              TEMPORAL ANALYSIS
# ============================================
#
#
# df['year'] = df['track_album_release_date'].apply(extract_year)
#
# #  Histo of songs released by year
# plt.hist(df['year'], bins=20, color='yellow', edgecolor = 'black')
# plt.xlabel('Year')
# plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
# plt.ylabel('Count')
# plt.title('Histogram of Song Release Years')
# plt.grid()
#
# #Average track popularity by year
# mean_popularity = df.groupby('year')['track_popularity'].mean()
# plt.plot(mean_popularity, marker = 'o', color = 'red')
# plt.xlabel('Year')
# plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
# plt.ylabel('Average Popularity')
# plt.title('Average Track Popularity by Year')
# plt.grid(True)
#
# #Set charachteristics
# columns = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
#
# # We group the songs by year and compute the mean fo all features
# average_by_year = df.groupby('year')[columns].mean()
#
# for column in columns:
#     plt.figure(figsize=(10, 6))
#     plt.plot(average_by_year.index, average_by_year[column], marker='o', color = 'aqua')
#     plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
#     plt.xlabel('Year')
#     plt.ylabel('Average ' + column)
#     plt.title('Average ' + column + ' by Year')
#     plt.grid(True)
#     plt.show()
#
# # We build graphs for songs with a popularity value greater than 70
# popular_df = df[df['track_popularity'] > 70]
#
# plt.hist(popular_df['year'], bins=20, color = 'red', alpha=0.8, edgecolor='black')
# plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
# plt.xlabel('Year')
# plt.ylabel('Count')
# plt.title('Histogram of Song Release Years (Popularity > 70)')
# plt.grid(True)
#
# # We group songs by year and compute the mean of each feature 
# # We do the same with songs which have a popularity > 70%
# average_by_year_popular = popular_df.groupby('year')[columns].mean()
#
# for column in columns:
#     plt.figure(figsize=(10, 6))
#     plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
#     plt.plot(average_by_year.index, average_by_year[column], marker='o', color = 'yellow', label='All values')
#     plt.plot(average_by_year_popular.index, average_by_year_popular[column], marker='o', color = 'dodgerblue', label='Popularity > 70')
#     plt.xlabel('Year')
#     plt.ylabel('Average ' + column)
#     plt.grid(True)
#     plt.title('Average ' + column + ' by Year')
#     plt.legend()
#
# # Now we do the same but with songs much less popular (< 2%)
# unpopular_df = df[df['track_popularity'] < 2]
#
# average_by_year_unpopular = unpopular_df.groupby('year')[columns].mean()
#
# for column in columns:
#     plt.figure(figsize=(10, 6))
#     plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
#     plt.plot(average_by_year.index, average_by_year[column], marker='o', color = 'lime', label='All values')
#     plt.plot(average_by_year_unpopular.index, average_by_year_unpopular[column], marker='o', color = 'crimson', label='Popularity < 2')
#     plt.xlabel('Year')
#     plt.ylabel('Average ' + column)
#     plt.title('Average ' + column + ' by Year')
#     plt.legend()
#     plt.grid(True)
#
# grouped = df.groupby('year')
#
# correlations_by_year = {column: [] for column in columns}
# years = []
#
# # Calculate the correlation for each year
# for year, group in grouped:
#     years.append(year)
#     for column in columns:
#         correlation = group[column].corr(group['track_popularity'])
#         correlations_by_year[column].append(correlation)
#
# for column, correlations in correlations_by_year.items():
#     plt.figure(figsize=(10, 6))
#     plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.0f}"))
#     plt.plot(years, correlations, marker='o', color = 'green')
#     plt.xlabel('Year')
#     plt.ylabel('Correlation with track_popularity')
#     plt.title('Correlation of ' + column + ' with track_popularity by Year')
#     plt.grid(True)
#
# # We group songs by artist and count the number of songs for each artist
# song_count = df['track_artist'].value_counts()
#
# # We take the mean popularity of song by each artist
# artist_df = df.groupby('track_artist')['track_popularity'].mean()
#
# # We get the top 5 artists by track popularity
# most_5popular_artists = artist_df.nlargest(5)
#
# # We get the popular songs
# popular80_df = df[df['track_popularity'] > 80]
#
# song_count_popular = popular80_df['track_artist'].value_counts()
#
# top_artists_by_song = song_count_popular.nlargest(5)
#
# plt.figure(figsize=(10, 6))
# plt.barh(most_5popular_artists.index, most_5popular_artists.values, color='skyblue')
# plt.xlabel('Average Popularity')
# plt.title('Top 5 Artists by Average Popularity')
# plt.gca().invert_yaxis()
# plt.grid(True)
#
# plt.figure(figsize=(10, 6))
# plt.barh(top_artists_by_song.index, top_artists_by_song.values, color='skyblue')
# plt.xlabel('Number of Songs with Popularity > 80')
# plt.title('Top 5 Artists by Number of Songs with Popularity > 80')
# plt.gca().invert_yaxis()
# plt.grid(True)
#
# # We create a new column for decades
# df['decade'] = (df['year'] // 10) * 10
#
# # We group songs by decade and genre
# grouped = df.groupby(['decade', 'playlist_genre'])
#
# # We calculate the average popularity of each genre in each decade
# average_popularity = grouped['track_popularity'].mean()
#
# popularity_by_decade = {}
#
# for (decade, genre), popularity in average_popularity.items():
#     if decade not in popularity_by_decade:
#         popularity_by_decade[decade] = []
#     popularity_by_decade[decade].append((genre, popularity))
#
# # Sort genres in each decade by popularity
# for decade in popularity_by_decade:
#     popularity_by_decade[decade].sort(key=lambda x: x[1], reverse=True)
#
#
#
# for decade in sorted(popularity_by_decade):
#     genres = [genre for genre, _ in popularity_by_decade[decade][:5]]
#     popularities = [popularity for _, popularity in popularity_by_decade[decade][:5]]
#     plt.figure(figsize=(8, 6))
#     plt.barh(genres, popularities, color='yellow')
#     plt.ylabel('Genre')
#     plt.xlabel('Average Popularity')
#     plt.title('Top 5 Genres by Average Popularity in the ' + str(decade) + 's')
#     plt.gca().invert_yaxis()
#     plt.show()
#
# # We group songs by artist and decade and count the number of songs per artist
# song_counts = df.groupby(['track_artist', 'decade']).size()
#
# filtered_df = df[df['track_popularity'] > 80]
#
# filtered_song_counts = filtered_df.groupby(['track_artist', 'decade']).size()
#
# # We get the top 5 artists based on the number of songs with a 'track_popularity' value greater than 80 for each decade
# top_artists_by_song_count = filtered_song_counts.groupby('decade').nlargest(5)
#
# for decade, artists in top_artists_by_song_count.groupby('decade'):
#     plt.figure(figsize=(10, 6))
#     plt.barh(artists.index.get_level_values('track_artist'), artists.values, color='orange')
#     plt.xlabel('Number of Songs with Popularity > 80')
#     plt.title('Top 5 Artists by Number of Songs with Popularity > 80 in the ' + str(decade) + 's')
#     plt.gca().invert_yaxis()
#     plt.show()
#
#
# ===================================================
#                   K-MEANS
# ===================================================
    

small_df4 = df[['track_popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
normalise(small_df4)
pca = dc.PCA()
X = pca.fit_transform(small_df4)

# Explained variance - how much each principal component means
exp_var_pca4 = pca.explained_variance_ratio_
print(exp_var_pca4)

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues4 = np.cumsum(exp_var_pca4)

# Create the visualization plot
plt.bar(range(0,len(exp_var_pca4)), exp_var_pca4, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues4)), cum_sum_eigenvalues4, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Apply the clustering algorithm
km = KMeans(n_clusters=4, random_state=0, n_init='auto').fit(X)

# Plot the 2D scatter plot of the pca with clusters
plt.scatter(X[:,0], X[:,1], c = km.labels_, s=2)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='x', s=169, linewidths=3, color='red', zorder=10)

# Plot it in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter_centroids = ax.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], marker='*', markersize=10, linestyle='', color='red', label='Centroids', zorder = 10)
scatter_data = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=km.labels_, s=2, label='Data Points', zorder = 1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.legend()
plt.show()

# Analyze clusters
for i in range(4):
    analyze_cluster(small_df4, km.labels_, i)
    print('\n')


# ==============================================================
#                     Silhouette Method
# ==============================================================
    
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()


# =============================================================
#                         Elbow Method
# =============================================================

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
