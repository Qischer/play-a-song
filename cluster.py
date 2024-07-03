import dbscan
import kmeans
import matplotlib.pyplot as plt
import pandas as pd
import sys


features = ['danceability', 'energy', 'loudness',
            'speechiness', 'acousticness','instrumentalness',
            'liveliness','valence']

def parseDF(df: pd.DataFrame):
    fdf = df.filter(items=features)
    return fdf.to_numpy()

if __name__ == "__main__":   
    if len(sys.argv) <= 1: 
        print("Print Usage")
    
    match sys.argv[1]:
        case "kmeans":
            print("Run kmeans")
            pass
        case "dbscan":
            print("Run dbscan")
            pass
