import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":

    df = pd.read_csv("dataset.csv")
    
    dance = np.array(df["danceability"])
    energy = np.array(df["energy"])
    tempo = np.array(df["tempo"])

    gr = np.array(df["track_genre"])
    fig, ax = plt.subplots(1,3)

    # dance - energy
    ax[0,0].scatter(dance, energy)
    ax[0,0].set_title("energy - dance")

    # energy - tempo
    ax[0,1].scatter(energy, tempo)
    ax[0,1].set_title("tempo - energy")

    # dance - tempo
    ax[0,2].scatter(dance, tempo)
    ax[0,2].set_title("temp - dance")

    plt.show()
