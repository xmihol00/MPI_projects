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

fig, ax = plt.subplots(figsize=(12, 6))

dfs = []
min = 2**32
for filename in ["hybrid_1d_halo_delay_out.csv", "mpi_2d_halo_delay_out.csv"]:
    df = pd.read_csv(filename, sep=";")
    df_min = df[df["mode"] == 1]["halo_delay"].min()
    if df_min < min:
        min = df_min
    dfs.append(df)

for df, launch_type, color in zip(dfs, ["1D hybrid", "2D MPI"], list(mcolors.TABLEAU_COLORS.keys())):
    df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]
    df = df[df["mode"] == 1]
    for domain_size, marker in zip([256, 512, 1024, 2048, 4096], ["s", "D", "o", "h", "*"]):
        df_mode = df[df["domain_size"] == domain_size].copy()
        df_mode.reset_index(drop=True, inplace=True)
        df_mode.loc[:, "halo_delay"] /= min
        ax.plot(df_mode["mpi_procs"], df_mode["halo_delay"], label=f"{launch_type} P2P {domain_size}x{domain_size}", marker=marker, markersize=7, color=color, linestyle="--")

ax.set_xlabel("Number of processes")
ax.set_xscale("log", base=2)
ax.set_ylabel("Normalized halo computation time")
ax.set_yscale("log", base=2)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, prop={'size': 10})
plt.grid()
plt.tight_layout()
plt.savefig("halo_computation_time.png", dpi=500)