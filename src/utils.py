# utils.py

import matplotlib.pyplot as plt

def plot_scores(df, title):
    plt.plot(df["score"].values)
    plt.title(title)
    plt.ylabel("Anomaly Score")
    plt.xlabel("Window Index")
    plt.show()
