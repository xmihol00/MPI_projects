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

plt.figure(figsize=(6, 6))
plt.title("Communication delay comparison", {"fontsize": LARGE_SIZE})

dfs = []
min = 2**32
for filename in ["hybrid_1d_comm_delay_out.csv", "mpi_2d_comm_delay_out.csv"]:
    df = pd.read_csv(filename, sep=";")
    df_min = df[df["mode"] == 1]["comm_delay"].min()
    if df_min < min:
        min = df_min
    dfs.append(df)

for df, launch_type, color in zip(dfs, ["1D hybrid", "2D MPI"], list(mcolors.TABLEAU_COLORS.keys())):
    df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]
    df = df[df["mode"] == 1]
    for domain_size, marker in zip([512, 1024, 2048], ["s", "D", "o"]):
        df_mode = df[df["domain_size"] == domain_size].copy()
        df_mode.reset_index(drop=True, inplace=True)
        df_mode.loc[:, "comm_delay"] /= min
        plt.plot(df_mode["mpi_procs"], df_mode["comm_delay"], label=f"{launch_type} P2P {domain_size}x{domain_size}", marker=marker, markersize=7, color=color, linestyle="--")

plt.xlabel("Number of processors")
plt.xscale("log", base=2)
plt.ylabel("Normalized communication delay")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("communication_delay.png", dpi=500)