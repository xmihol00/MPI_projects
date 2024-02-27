import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directions = ["Ascending", "Descending"]
communication = ["Batch", "Single"]

plt.figure(figsize=(15, 10))

for direction in directions:
    for comm in communication:
        df = pd.read_csv(f"performance/{direction}_{comm}.csv".lower(), header=None)
        processes = df.iloc[:, 0]
        times = df.iloc[:, 1]
        plt.plot(processes, times, label=f"{direction} - {comm}", marker='o')

N = np.arange(2, len(processes) + 2)
M = np.power(2, N)
print(M)

plt.yscale("log")
plt.legend()
plt.xlabel("Number of processes")
plt.ylabel("Time (s)")
plt.title("Performance of pipeline merge sort")
plt.tight_layout()
plt.show()
plt.savefig("performance_plot.png", dpi=400)
