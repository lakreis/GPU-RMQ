import polars as pl
import seaborn as sns

import io
import re
import sys

from pathlib import Path
from matplotlib import pyplot as plt

from static_data import *
from get_and_filter_data import get_prepared_data, filter_optimal_blocksize, filter_approx_optimal_blocksize, \
    split_at_array_size

LOG_LABEL = r"$\mathregular{2^n}$"

sns.set_context('paper', font_scale=0.8)

cell_width = 0.9
cell_height = 0.28

def plot_heatmap(data, lr, log_q, save_name, alg_label, smallest_t):
    print(f"Generating heatmap {save_name}!")
    lr_complete = lr
    lr = int(lr[:3])
    best_tp_in_group = pl.max("time_throughput").over(["n-exp"]).alias("best_tp")
    
    data = data.with_columns(
        ((pl.col("time_throughput") / best_tp_in_group).alias("tp_factor") * 2000).ceil() / 2000,
        pl.col("time_throughput").rank('min').over(["n-exp"]).alias("tp_rank") - 1
        )

    data = data.filter(pl.col("n-exp") >= 20)

    if alg_label == "Hierarchical_vector_load":
        data = data.with_columns(
            pl.concat_str(
                [
                    pl.lit(r"$c\ =\ 2^{"),
                    pl.col("XXX_chunk_size").cast(pl.Utf8),
                    pl.lit(r"}$"),
                ]
            ).alias("param_combination")
        )
    else:
        data = data.with_columns(
            pl.concat_str(
                [
                    pl.lit(r"$g=2^{"),
                    pl.col("XXX_CG_SIZE_LOG").cast(pl.Utf8),
                    pl.lit(r"} \,/\, c=2^{"),
                    pl.col("XXX_chunk_size").cast(pl.Utf8),
                    pl.lit(r"}$"),
                ]
            ).alias("param_combination")
        )

    subset = data.select("param_combination", "n-exp", "tp_factor")
    data_pd = (subset.to_pandas().pivot(index="param_combination", columns="n-exp", values="tp_factor").sort_index())

    n_rows, n_cols = data_pd.shape
    fig, ax = plt.subplots(figsize=(n_cols * cell_width, n_rows * cell_height), dpi=600)

    sns.heatmap(
        data_pd,
        ax=ax,
        cmap="coolwarm_r",      
        vmin=0.8,               
        vmax=1.0,               
        annot=True,
        fmt=".3f",
        cbar=True
    )

    exponents = data_pd.columns
    ax.set_xticklabels([f"$2^{{{int(e)}}}$" for e in exponents])
    ax.tick_params(axis='y', labelrotation=0)

    ax.set_xlabel("n")
    if alg_label == "Hierarchical_vector_load":
        ax.set_ylabel("c")
    else:
        ax.set_ylabel("g / c")

    fig.tight_layout()
    plt.savefig(f"{plot_dir}heatmap-{save_name}-{alg_label}-lr{lr_complete}-log_q{log_q}-smallest_t{smallest_t}.{save_as}", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    plt.close()

def generate_heatmap(device_name, alg_label, log_q, lr, start_times, end_times, save_name, smallest_t):
    if smallest_t:
        pd_data = get_prepared_data(files[alg_label](device_name), alg_label, lr=lr, q=2**log_q, timestamp_start=start_times[alg_label], timestamp_end=end_times[alg_label], groupby_elements=['dev','alg','n','lr', 'q', 'bs', 'nb', 'XXX_chunk_size', 'XXX_CG_SIZE_LOG', 'scan_threshold'], XXX_CG_size_log=None, XXX_CG_amount_log=None, XXX_chunk_size=True, XXX_scan_threshold=None)        
        pd_data = (pd_data.loc[pd_data.groupby(['dev','alg','n','lr', 'q', 'bs', 'nb', 'XXX_chunk_size', 'XXX_CG_SIZE_LOG'])['scan_threshold'].idxmin()].reset_index(drop=True))
    else:
        pd_data = get_prepared_data(files[alg_label](device_name), alg_label, lr=lr, q=2**log_q, timestamp_start=start_times[alg_label], timestamp_end=end_times[alg_label], groupby_elements=['dev','alg','n','lr', 'q', 'bs', 'nb', 'XXX_chunk_size', 'XXX_CG_SIZE_LOG'], XXX_CG_size_log=None, XXX_CG_amount_log=None, XXX_chunk_size=True)        
    pd_data.columns = ["_".join(filter(None, col)) for col in pd_data.columns] 
    data = pl.from_pandas(pd_data)   
    plot_heatmap(data, lr, log_q, save_name, alg_label, smallest_t)


if __name__ == '__main__':
    device_name = sys.argv[1]
    log_q = int(sys.argv[2])
    lr = sys.argv[3]
    save_name=sys.argv[4]
    smallest_t=int(sys.argv[5])
    label=sys.argv[6]

    sys.stdout.flush()
    
    generate_heatmap(device_name, label, log_q, lr, start_times, end_times, save_name, smallest_t)
