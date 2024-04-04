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

df = pd.read_csv("strong_scaling.csv", sep=";")
df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]

plt.figure(figsize=(6, 6))
plt.title("Strong scaling", {"fontsize": LARGE_SIZE})

processors = df["mpi_procs"].unique()
ideal_times = np.ones_like(processors)
ideal_times = ideal_times / processors
plt.plot(processors, ideal_times, label="Perfect scaling", linestyle="--", marker="o", markersize=8)

for decomposition, launch_type, color, marker in zip(["1D", "2D"], ["Hybrid", "MPI"], list(mcolors.TABLEAU_COLORS.keys())[1:], ["s", "D"]):
    df_decomposition = df[df["decomposition"] == decomposition]
    for domain_size, marker_fill in zip([256, 4096], ["none", "full"]):
        df_domain = df_decomposition[df_decomposition["domain_size"] == domain_size].copy()
        df_domain.reset_index(drop=True, inplace=True)
        df_domain.loc[:, "iteration_time"] /= df_domain.loc[0, "iteration_time"]
        plt.plot(df_domain["mpi_procs"], df_domain["iteration_time"], label=f"{launch_type} {decomposition} {domain_size}x{domain_size} RMA", marker=marker, markersize=8, color=color, fillstyle=marker_fill)

plt.xlabel("Number of processors")
plt.xscale("log", base=2)
plt.ylabel("Normalized iteration time")
plt.yscale("log", base=2)
plt.legend()
plt.grid()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig("strong_scaling.png", dpi=500)