import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness','instrumentalness',
            'liveliness','valence', 'tempo']


def parseDF(df: pd.DataFrame):

    fdf = df.filter(items=features)
    
    return fdf.to_numpy()


def train(X: np.ndarray):
    inr = []
    for n in range(1,11):
        km = KMeans(n_clusters=n).fit(X)
        inr.append(km.inertia_)
    
    plt.plot(range(1,11), inr, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    return


def cluster(df: pd.DataFrame, n: int):
    km = KMeans(n_clusters=n).fit(X)

    plt.scatter(df['danceability'], df['energy'], c=km.labels_)
    plt.show()

    return


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")


    X = parseDF(df)
    cluster(df, 3)
