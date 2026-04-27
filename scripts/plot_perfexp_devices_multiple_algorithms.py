import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib.lines import Line2D

from get_and_filter_data import get_prepared_data, filter_optimal_blocksize, \
    filter_approx_optimal_blocksize, split_at_array_size
from static_data import *

def plot_time_devices(data_frame, lr, log_q, save_name, labels, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height):
    print("Generating time plot!")
    lr_complete = lr
    lr = int(lr[:3])
    # common plot settings
    k=0.5
    fig = plt.figure(figsize=(plot_width*k*SIZE_MULT,plot_height*k*SIZE_MULT))
    ax = fig.add_subplot(111)
    if x_label:
        plt.xlabel("Array size (n)", fontsize=x_label_font_size, labelpad=x_label_pad)
    else:
        plt.xlabel(" ", fontsize=x_label_font_size, labelpad=x_label_pad)
    if plot_range_info == 1:
        plt.title(f"{final_label_names[alg_label]}, {lrLabels[lr]}", fontsize=range_font_size, pad=range_pad)
    elif plot_range_info == 2:
        plt.title(f"{lrLabels[lr]}", fontsize=range_font_size, pad=range_pad)

    plt.xticks(range(xlim1,xlim2+1,2), fontsize=x_ticks_font_size)
    plt.tick_params(axis="x", pad=x_ticks_pad)
    plt.xlim(xlim1,xlim2)
    plt.grid(color='#e7e7e7', linestyle='--', linewidth=1.35, axis='both', which='major', zorder=0)
    # Create a second y-axis on the right
    ax2 = ax.twinx()
    ax.yaxis.set_visible(False)
    ax2.grid(True, color='#e7e7e7', linestyle='--', linewidth=1.35, axis='both', which='major', zorder=0)
    if y_label:
        plt.ylabel("Time per RMQ $ \\left[ \\frac{ns}{RMQ} \\right] $", fontsize=y_label_font_size, labelpad=y_label_pad)
    plt.yticks(fontsize=y_ticks_font_size)

    algos = []
    handled_algos = []
    devices = []
    handled_devices = []
    for i, df in enumerate(data_frame):
        algo = df["final_algo"].unique()[0]
        device = df["final_device"].unique()[0]
        alg_label = df["alg_label"].unique()[0]
        
        if device in handled_devices:
            line_style = all_linestyles[handled_devices.index(device)]
        else: 
            line_style = all_linestyles[len(handled_devices)]
            handled_devices.append(device)
            devices.append(Line2D([0], [0], linestyle=line_style, color="black", label=device))
        
        color = colors[alg_label]
        if algo not in handled_algos:
            handled_algos.append(algo)
            algos.append(Line2D([0], [0], linestyle="-", color=color, label=algo))
        
        plt.plot(df['n-exp'], df['mean_ns/q'], linestyle=line_style, color=color)

    temp = list(zip(devices, handled_devices))
    temp = sorted(temp, key=lambda x: x[1])
    devices, _ = zip(*temp)
    devices = list(devices)
    diff = max(0, len(devices)-len(algos))
    handles = algos + [Line2D([], [], linestyle="none", label="") for _ in range(diff)] + devices
    legend = plt.legend(handles=handles, ncol=2, frameon=True, columnspacing=1, handletextpad=0.5, handlelength=1.5, loc="lower center", bbox_to_anchor=(0.5, 1))
    legend.set_zorder(100)

    if legend == 1:
        plt.legend(fontsize=outer_legend_font_size, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    elif legend == 2:
        plt.legend(fontsize=outer_legend_font_size, loc="center right", bbox_to_anchor=(0, 0.5))
    elif legend == 3:
        plt.legend(fontsize=inner_legend_font_size)

    if log_scale:
        plt.yscale('log')
    #plt.xscale('log')
    if ylim1 >= 0 and ylim2 >= 0:
        plt.ylim(ylim1, ylim2)

    ax.xaxis.set_major_formatter(FormatStrFormatter(r"$2^{%.0f}$"))
    if optimal_bs:
        plt.savefig(f"{plot_dir}time-devices_multiple_algos-optimal-{save_name}-lr{lr_complete}-log_q{log_q}.{save_as}", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    else:
        plt.savefig(f"{plot_dir}time-devices_multiple_algos-approx-optimal-{save_name}-lr{lr_complete}-log_q{log_q}.{save_as}", dpi=500, facecolor="#ffffff", bbox_inches='tight')
    plt.close()

def generate_plot_time_devices(labels, log_q, lr, start_times, end_times, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height): 
    df = []
    for alg_label in labels.keys():
        for device_name in labels[alg_label]:
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
            data["final_device"] = final_device_names[device_name]
            data["final_algo"] = final_label_names[alg_label]
            data["alg_label"] = alg_label
            df.append(data)
    
    plot_time_devices(df, lr, log_q, save_name, labels, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height)


if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Run with arguments <log_q> <lr> <save_name> <ylim1> <ylim2> <legend?> <x_label?> <y_label?> <y_log_scale?> <label1> <label2> ..")
        exit()

    log_q = int(sys.argv[1])
    lr = sys.argv[2]
    save_name=sys.argv[3]
    xlim1=int(sys.argv[4])
    xlim2=int(sys.argv[5])
    ylim1=float(sys.argv[6])
    ylim2=float(sys.argv[7])
    legend=int(sys.argv[8])
    x_label=int(sys.argv[9])
    y_label=int(sys.argv[10])
    log_scale=int(sys.argv[11])
    optimal_bs=int(sys.argv[12])
    plot_range_info=int(sys.argv[13])
    plot_width=float(sys.argv[14])
    plot_height=float(sys.argv[15])

    i = 16
    labels = {}
    while True:
        if i >= len(sys.argv):
            break
        algo_label=sys.argv[i]
        labels[algo_label] = []
        num_devices=int(sys.argv[i+1])
        for j in range(i+2, i+2+num_devices):
            labels[algo_label].append(sys.argv[j])
        
        i = i+2+num_devices

    sys.stdout.flush()
    
    generate_plot_time_devices(labels, log_q, lr, start_times, end_times, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height)