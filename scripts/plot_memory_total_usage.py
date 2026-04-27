import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from get_and_filter_data import get_prepared_data, filter_optimal_blocksize, filter_approx_optimal_blocksize, \
    split_at_array_size
from static_data import *

def plot_memory(data_frame, lr, log_q, save_name, labels, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem, plot_max, plot_range_info, plot_width, plot_height):
    print(f"Generating memory plot {save_name}!")
    lr_complete = lr
    lr = int(lr[:3])
    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(plot_width*k*SIZE_MULT,plot_height*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    if plot_range_info:
        plt.title(f"{lrLabels[lr]}", fontsize=range_font_size, pad=range_pad)
    if x_label:
        plt.xlabel("Array size (n)",fontsize=x_label_font_size, labelpad=x_label_pad)
    else:
        plt.xlabel(" ", fontsize=x_label_font_size, labelpad=x_label_pad)
    plt.xticks(range(xlim1,xlim2+1,2), fontsize=x_ticks_font_size)
    plt.tick_params(axis="x", pad=x_ticks_pad)
    plt.xlim(xlim1,xlim2)
    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.35, axis='both', which='major', zorder=0)
    # Create a second y-axis on the right
    ax2 = ax.twinx()
    ax.yaxis.set_visible(False)
    ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.35, axis='both', which='major', zorder=0)
    if y_label:
        plt.ylabel("Used GPU \nmemory $\\left[ GB \\right] $", fontsize=y_label_font_size, labelpad=y_label_pad)
    plt.yticks(fontsize=y_ticks_font_size)
    for i, df in enumerate(data_frame):
        plt.plot(df['n-exp'], max_mem - df['free_memory'], label=final_label_names[labels[i]], linestyle=linestyles[labels[i]],color=colors[labels[i]], zorder=orders[labels[i]], alpha=alphas[labels[i]])

    if legend == 1:
        plt.legend(fontsize=outer_legend_font_size, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    elif legend == 2:
        plt.legend(fontsize=outer_legend_font_size, loc="center right", bbox_to_anchor=(0, 0.5))
    elif legend == 3:
        plt.legend(fontsize=inner_legend_font_size)
    #plt.xscale('log')
    if ylim1 >= 0 and ylim2 >= 0:
        plt.ylim(ylim1, ylim2)

    if plot_max:
        plt.axhline(y=max_mem, color='black', linestyle=':', linewidth=1)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        offset = 0.03 * (y_max - y_min) 
        ax.annotate("GPU memory capacity", xy=(0.02, max_mem - offset), xycoords=("axes fraction", "data"),
            ha="left", va="top", fontsize=10, color="black")


    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$2^{%.0f}$"))
    if optimal_bs:
        plt.savefig(f"{plot_dir}memory_total-optimal-{save_name}-lr{lr_complete}-log_q{log_q}.{save_as}", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.savefig(f"{plot_dir}memory_total-approx-optimal-{save_name}-lr{lr_complete}-log_q{log_q}.{save_as}", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    plt.close()

def save_memory_speedup(df, max_mem):
    combined = pd.concat(df, ignore_index=True)
    combined["total_mem"] = max_mem - combined['free_memory']
    combined.columns = [col[0] if col[1]=='' else f"{col[0]}_{col[1]}" for col in combined.columns]

    ref_algos = ["[GPU] XXX", "[GPU] BASIC VECTOR LOAD"]
    ref = combined[combined['alg'].isin(ref_algos)][['n-exp','lr','total_mem']].rename(columns={'total_mem':'ref_memory'})

    merged = pd.merge(combined, ref, on=['n-exp','lr'], how='left')
    merged['speedup_memory'] = merged['total_mem'] / merged['ref_memory']
    merged['speedup_memory'].fillna(1.0, inplace=True)

    merged[['dev','alg','n-exp','lr','speedup_memory']].to_csv(
        f"{plot_dir}data/memory-optimal-{save_name}-lr{lr}-log_q{log_q}-speedup.csv",
        index=False
    )

def generate_plot(device_name, alg_labels, log_q, lr, start_times, end_times, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem, plot_max, plot_range_info, plot_width, plot_height):
    df = []
    for alg_label in alg_labels:
        data = get_prepared_data(files[alg_label](device_name), alg_label, lr=lr, q=2**log_q, timestamp_start=start_times[alg_label], timestamp_end=end_times[alg_label], groupby_elements=['dev','alg','n','lr', 'q', 'bs', 'nb'])
        if alg_label in ["RTXRMQ", "SER"]:
            if optimal_bs:
                data = filter_optimal_blocksize(data)
            else:
                data = filter_approx_optimal_blocksize(data, alg_label)
        if alg_label == "GPU_RMQ" and split_GPU_RMQ_value > 0:
            print(f"Be aware that the alg Hierarchical_vector_load used here for small array sizes has other parameters")
            small_XXX_data = get_prepared_data(files["Hierarchical_vector_load"](device_name), "Hierarchical_vector_load", lr=lr, q=2 ** log_q,
                                     timestamp_start=start_times["Hierarchical_vector_load"], timestamp_end=end_times["Hierarchical_vector_load"],
                                     groupby_elements=['dev', 'alg', 'n', 'lr', 'q', 'bs', 'nb'], XXX_scan_threshold=Hierarchical_vector_load_optimal_scan_threshold, XXX_CG_size_log=0, XXX_CG_amount_log=Hierarchical_vector_load_optimal_amount_log)
            data = split_at_array_size(small_XXX_data, data, split_GPU_RMQ_value)
        df.append(data)

    # save_memory_speedup(df, max_mem)
    plot_memory(df, lr, log_q, save_name, alg_labels, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem, plot_max, plot_range_info, plot_width, plot_height)

if __name__ == "__main__":
    if len(sys.argv) < 11:
        print("Run with arguments <log_q> <lr> <save_name> <ylim1> <ylim2> <optimal_bs?> <label1> <label2> ..")
        exit()

    device_name = sys.argv[1]
    log_q = int(sys.argv[2])
    lr = sys.argv[3]
    save_name=sys.argv[4]
    xlim1=int(sys.argv[5])
    xlim2=int(sys.argv[6])
    ylim1=int(sys.argv[7])
    ylim2=int(sys.argv[8])
    legend=int(sys.argv[9])
    x_label=int(sys.argv[10])
    y_label=int(sys.argv[11])
    optimal_bs=int(sys.argv[12])
    max_mem_in_GB=float(int(sys.argv[13])*(1e-3))
    plot_max=int(sys.argv[14])
    plot_range_info=int(sys.argv[15])
    plot_width=float(sys.argv[16])
    plot_height=float(sys.argv[17])

    labels=[]
    for i in range(18,len(sys.argv)):
        labels.append(sys.argv[i])

    sys.stdout.flush()
    
    generate_plot(device_name, labels, log_q, lr, start_times, end_times, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem_in_GB, plot_max, plot_range_info, plot_width, plot_height)