import pandas as pd
import numpy as np
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

correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

pd.plotting.scatter_matrix(df);

x = df.select_dtypes(include='number').values
scaler = pp.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
dfnorm = pd.DataFrame(x_scaled,columns = df.select_dtypes(include='number').columns)
dfnorm.boxplot(rot=90)

corrmat_big = correlation_matrix
corrmat_big[np.absolute(corrmat_big.values) < 0.1] = 0
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

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


