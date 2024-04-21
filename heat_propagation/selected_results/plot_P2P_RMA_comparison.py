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


dfs = []
min = 2**32
for filename in ["hybrid_1d_comm_delay_out.csv", "mpi_2d_comm_delay_out.csv"]:
    df = pd.read_csv(filename, sep=";")
    df_min = df[df["mode"] == 1]["comm_delay"].min()
    if df_min < min:
        min = df_min
    dfs.append(df)

colors = list(mcolors.TABLEAU_COLORS.keys())
for df, launch_type in zip(dfs, ["1D hybrid", "2D MPI"]):
    fig, ax = plt.subplots(figsize=(12, 6))

    df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]

    color = colors[0]
    df_p2p = df[df["mode"] == 1]
    for domain_size, marker in zip([256, 512, 1024, 2048, 4096], ["s", "D", "o", "h", "*"]):
        df_mode = df_p2p[df_p2p["domain_size"] == domain_size].copy()
        df_mode.reset_index(drop=True, inplace=True)
        df_mode.loc[:, "comm_delay"] /= min
        ax.plot(df_mode["mpi_procs"], df_mode["comm_delay"], label=f"{launch_type} P2P {domain_size}x{domain_size}", marker=marker, markersize=7, color=color, linestyle="--")
    
    color = colors[1]
    df_p2p = df[df["mode"] == 2]
    for domain_size, marker in zip([256, 512, 1024, 2048, 4096], ["s", "D", "o", "h", "*"]):
        df_mode = df_p2p[df_p2p["domain_size"] == domain_size].copy()
        df_mode.reset_index(drop=True, inplace=True)
        df_mode.loc[:, "comm_delay"] /= min
        ax.plot(df_mode["mpi_procs"], df_mode["comm_delay"], label=f"{launch_type} RMA {domain_size}x{domain_size}", marker=marker, markersize=7, color=color, linestyle="--")

    ax.set_xlabel("Number of processes")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("Normalized communication delay")
    ax.set_yscale("log", base=2)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, prop={'size': 10})
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"communication_delay_{launch_type.replace(' ', '_')}.png", dpi=500)