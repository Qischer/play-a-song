import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


features = ['danceability', 'energy', 'loudness',
            'speechiness', 'acousticness','instrumentalness',
            'liveliness','valence']


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


def cluster(df: pd.DataFrame, n: int, labels):
    km = KMeans(n_clusters=n).fit(X)

    fig = plt.figure()

    for i, label in enumerate(labels):
        ax = fig.add_subplot(projection='3d')

        ax.scatter(df[label[0]], df[label[1]], df[label[2]], c=km.labels_)

        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_zlabel(label[2])
    
    plt.tight_layout()
    # plt.show()

    return km


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")

    labels = np.array([
        ['danceability', 'energy', 'loudness'],
        # ['speechiness', 'acousticness','instrumentalness'],
        ])

    ar = df.to_numpy()
    X = parseDF(df)

    cl = cluster(df, 5, labels)

    for i in range(5): 
        print("Label for ",i)
        sample = ar[np.where(cl.labels_ == i)[0]]  
        print(sample)
