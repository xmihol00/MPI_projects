import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.colors as mcolors

SMALL_SIZE = 12.5
LARGE_SIZE = 15

matplotlib.rc('font', size=SMALL_SIZE)        
matplotlib.rc('axes', titlesize=SMALL_SIZE)   
matplotlib.rc('axes', labelsize=SMALL_SIZE)  
matplotlib.rc('xtick', labelsize=SMALL_SIZE)  
matplotlib.rc('ytick', labelsize=SMALL_SIZE)  
matplotlib.rc('legend', fontsize=SMALL_SIZE)  
matplotlib.rc('figure', titlesize=LARGE_SIZE) 

df = pd.read_csv("fs_settings.csv", sep=";")
df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]
min_iter_time = df["iteration_time"].max()

plt.figure(figsize=(6, 6))
plt.title("FS settings strong scaling", {"fontsize": LARGE_SIZE})

processors = df["mpi_procs"].unique()
ideal_times = np.ones_like(processors)
ideal_times = ideal_times / processors
plt.plot(processors, ideal_times, label="Perfect scaling", linestyle="--", marker="o", markersize=10)

for fs, fs_name, color, marker in zip([1, 0], ["FS -S 1M -c 16", "FS default"], list(mcolors.TABLEAU_COLORS.keys())[1:], ["s", "D"]):
    df_fs = df[df["fs"] == fs]
    for run, marker_fill in zip([0, 1], ["none", "full"]):
        df_run = df_fs[df_fs["run"] == run]
        df_run.reset_index(drop=True, inplace=True)
        df_run.loc[:, "iteration_time"] /= min_iter_time
        plt.plot(df_run["mpi_procs"], df_run["iteration_time"], label=f"MPI 2D 4096x4096 RMA {fs_name} run {run + 1}", marker=marker, markersize=10, color=color, fillstyle=marker_fill)

plt.xlabel("Number of processors")
plt.xscale("log", base=2)
plt.ylabel("Normalized iteration time")
plt.yscale("log", base=2)
plt.legend()
plt.grid()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("fs_settings.png", dpi=500)