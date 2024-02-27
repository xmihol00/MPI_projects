import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys

directory = "performance"
if len(sys.argv) > 1:
    directory = sys.argv[1]

directions = ["Ascending", "Descending"]
communication = ["Batch", "Single"]

for log in [False, True]:
    plt.figure(figsize=(15, 10))

    times_avg = []
    for direction in directions:
        for comm in communication:
            df = pd.read_csv(f"{directory}/{direction}_{comm}.csv".lower(), header=None)
            processes = df.iloc[:, 0]
            times = df.iloc[:, 1]
            times_avg.append(times)
            plt.plot(processes, times, label=f"{direction} - {comm}", marker='o')

    times_avg = np.mean(times_avg, axis=0)
    N = np.arange(2, len(processes) + 2)
    regressor = LinearRegression()
    regressor.fit(N.reshape(-1, 1), times_avg.reshape(-1, 1))
    y_pred = regressor.predict(N.reshape(-1, 1))
    plt.plot(N, y_pred, label="Linear regression on averaged run times", linestyle="--", color="black")

    if log:
        plt.yscale("log")
        plt.ylabel("Time (s) in logarithmic scale")
    else:
        plt.ylabel("Time (s)")
    plt.legend()
    plt.xlabel("Number of processes")
    plt.title("Performance of pipeline merge sort")
    plt.tight_layout()
    plt.savefig(f"{directory}/performance_plot_{'log' if log else 'lin'}.png", dpi=400)
    plt.show()
