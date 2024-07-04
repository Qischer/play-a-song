import pandas as pd
import sys
import display as dsp
import kmeans as km
import dbscan

features = ['danceability', 'energy', 'loudness',
            'speechiness', 'acousticness','instrumentalness',
            'liveness','valence', 'tempo']

def parseDF(df: pd.DataFrame): 
    df['tempo'] /= 200
    df['loudness'] = (df['loudness'] + 49.531) / 50
    fdf = df.filter(items=features)
    return fdf.to_numpy()

def train(X):
    match sys.argv[2]:
        case "kmeans":
            dsp.km_train(km.train(X)) 
            pass
        case  "dbscan":
            dbscan.compute(X)
            pass
    

if __name__ == "__main__":   
    if len(sys.argv) <= 1: 
        print("Print Usage")
    
    df = pd.read_csv("dataset.csv")
    X = parseDF(df)

    if sys.argv[1] == "-t" or sys.argv[1] == "-T":
        train(X)
        exit(0)
    
    labels = []
    match sys.argv[1]:
        case "kmeans": 
            n = int(sys.argv[2])
            cluster = km.cluster(X=X, df=df, n=n)
            labels = cluster.labels_
            if len(sys.argv) > 3:
                features = sys.argv[3:]
                if len(features) < 3: 
                    print("Please input at least 3 features for plot")
                    exit(1)
                dsp.scatter_plot(df=df, labels=labels, features=features)
            else:
                dsp.scatter_plot(df=df, labels=labels)
            pass

        case "dbscan":
            labels = dbscan.compute(X)
            pass


