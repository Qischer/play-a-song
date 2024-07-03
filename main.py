import pandas as pd
import sys
import display as dsp
import kmeans as km
import numpy as np

features = ['danceability', 'energy', 'loudness',
            'speechiness', 'acousticness','instrumentalness',
            'liveness','valence', 'tempo']

def parseDF(df: pd.DataFrame): 
    df['tempo'] /= 243.372
    df['loudness'] = (df['loudness'] + 49.531) / 53.063
    for f in features: 
        print(f + "--max: " + str(np.max(df[f])) + "--min: " + str(np.min(df[f])))
    fdf = df.filter(items=features)
    return fdf.to_numpy()

def train(X):
    match sys.argv[2]:
        case "kmeans":
            dsp.km_train(km.train(X)) 
            pass
        case  "dbscan":
            print("train dbscan")
            pass
    

if __name__ == "__main__":   
    if len(sys.argv) <= 1: 
        print("Print Usage")
    
    df = pd.read_csv("dataset.csv")
    X = parseDF(df)

    if sys.argv[1] == "-t" or sys.argv[1] == "-T":
        train(X)
        exit(0)
    
    match sys.argv[1]:
        case "kmeans": 
            n = int(sys.argv[2])
            cluster = km.cluster(X=X, df=df, n=n)
           
            if len(sys.argv) >= 3:
                features = sys.argv[3:]
                if len(features) < 3: 
                    print("Please input at least 3 features for plot")
                    exit(1)
                dsp.km_plot(df=df, labels=cluster.labels_, features=features)
            else:
                dsp.km_plot(df=df, labels=cluster.labels_)
            pass
        case "dbscan":
            print("Run dbscan")
            pass
