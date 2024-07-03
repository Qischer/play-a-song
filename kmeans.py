import numpy as np
import pandas as pd
import main
from sklearn.cluster import KMeans

def train(X: np.ndarray):
    inertia = []
    for n in range(1,11):
        km = KMeans(n_clusters=n).fit(X)
        inertia.append(km.inertia_)
    
    return inertia

def cluster(X: np.ndarray, df: pd.DataFrame, n: int):
    km = KMeans(n_clusters=n).fit(X)

    return km

if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")

    labels = np.array([
        ['danceability', 'energy', 'loudness'],
        # ['speechiness', 'acousticness','instrumentalness'],
        ])

    ar = df.to_numpy()
    X = main.parseDF(df)

    cl = cluster(df, 5, labels)

    for i in range(5): 
        print("Label for ",i)
        sample = ar[np.where(cl.labels_ == i)[0]]  
        print(sample)
