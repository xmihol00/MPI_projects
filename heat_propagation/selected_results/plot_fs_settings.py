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

output_types = [["seq", "par"], ["par"]]

fig, ax = plt.subplots(figsize=(12, 6))
for i, filename, alignment in zip([0, 1], ["fs_comparison", "fs_comparison_aligned"], ["unaligned", "aligned 4 KB"]):
    df = pd.read_csv(f"{filename}.csv", sep=";")
    df.loc[:, "mpi_procs"] *= df.loc[:, "omp_threads"]
    fs_settings = df["stripe"].unique()

    for output_type, color, marker in zip(output_types[i], list(mcolors.TABLEAU_COLORS.keys())[i::2], ["s", "D"]):
        df_output = df[df["output_type"] == output_type]
        for domain_size, marker_fill in zip([4096, 2048, 1024], ["full", "bottom", "top"]):
            df_domain = df_output[df_output["domain_size"] == domain_size].copy()
            df_domain.reset_index(drop=True, inplace=True)
            df_domain.loc[:, "iteration_time"] /= df_domain["iteration_time"].min()
            plt.plot(df_domain["iteration_time"], label=f"{domain_size}x{domain_size} {df_domain['mpi_procs'][0]}P {output_type} I/O {alignment if output_type == 'par' else ''}", marker=marker, markersize=8, color=color, fillstyle=marker_fill, alpha=0.75, linestyle="--")

ax.set_xlabel("Stripe size")
ax.set_ylabel("Normalized iteration time")
ax.set_yscale("log", base=2)
ax.set_xticks(np.arange(0, len(fs_settings), 1), fs_settings)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, prop={'size': 10})
plt.grid()
plt.tight_layout()
plt.savefig(f"fs_comparison.png", dpi=500)