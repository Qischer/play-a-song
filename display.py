import matplotlib.pyplot as plt
import pandas as pd 

def km_train(inertia):
    plt.plot(range(1,11), inertia, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def scatter_plot(df : pd.DataFrame, labels, features=['danceability', 'energy', 'loudness']):
    
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=labels)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    plt.tight_layout()
    plt.show()
