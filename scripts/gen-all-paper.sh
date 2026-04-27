#!/bin/bash
q=$((26))

x_label=$((0))
y_label=$((0))
optimal_bs=$((1))

echo "----------- Plots for first page ------------"
SCRIPT=plot_perfexp.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -6 4090_first_plot 22 31 1e-2 1e3 3 0 1 1 $optimal_bs 0 4.5 4 RTXRMQ LCA GPU_RMQ

SCRIPT=plot_memory_total_usage.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem_in_GB, plot_max, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -6 4090_first_plot 22 31 0 25 0 0 1 $optimal_bs 24000 1 0 4.5 4 RTXRMQ LCA GPU_RMQ

SCRIPT=plot_perfexp_prep.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -6 4090_first_plot 22 31 1e0 1e3 0 1 1 1 $optimal_bs 0 4.5 4 RTXRMQ LCA GPU_RMQ

echo "--------------------- Total memory usage for chapter 4 -------------------"
SCRIPT=plot_memory_total_usage.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, optimal_bs, max_mem_in_GB, plot_max, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -3 4090 20 31 0 20 3 1 1 $optimal_bs 24000 0 0 6 4 Exhaustive RTXRMQ LCA GPU_RMQ

echo "---------------------- Query time for chapter 4 --------------------------"
SCRIPT=plot_perfexp.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -1 4090 20 31 1e-2 1e4 3 0 0 1 $optimal_bs 0 6 4 HRMQ@128c Exhaustive RTXRMQ LCA GPU_RMQ
python ${SCRIPT} RTXAda4090 $q -2 4090 20 31 1e-2 1e4 0 1 0 1 $optimal_bs 0 6 4 HRMQ@128c Exhaustive RTXRMQ LCA GPU_RMQ
python ${SCRIPT} RTXAda4090 $q -3 4090 20 31 1e-2 1e4 0 0 1 1 $optimal_bs 0 6 4 HRMQ@128c Exhaustive RTXRMQ LCA GPU_RMQ

python ${SCRIPT} RTXAda4090 $q -1 RT_comp 20 31 0 1.5 4 0 0 0 $optimal_bs 0 6 4 XXX Interleaved_in_CUDA Interleaved_in_OptiX Interleaved2 RTXRMQ
python ${SCRIPT} RTXAda4090 $q -2 RT_comp 20 31 0 1.5 0 1 0 0 $optimal_bs 0 6 4 XXX Interleaved_in_CUDA Interleaved_in_OptiX Interleaved2 RTXRMQ
python ${SCRIPT} RTXAda4090 $q -3 RT_comp 20 31 0 1.5 0 0 1 0 $optimal_bs 0 6 4 XXX Interleaved_in_CUDA Interleaved_in_OptiX Interleaved2 RTXRMQ

python ${SCRIPT} RTXAda4090 $q -6 multi_load_comp 20 31 0 0.8 3 1 1 0 $optimal_bs 1 6 4 XXX XXX_multiload

echo "---------------------- Construction time for chapter 4 --------------------------"
SCRIPT=plot_perfexp_prep.py
# device_name, log_q, lr, save_name, xlim1, xlim2, ylim1, ylim2, legend, x_label, y_label, log_scale, optimal_bs, plot_range_info, plot_width, plot_height
python ${SCRIPT} RTXAda4090 $q -6 4090 20 31 1e0 1e5 3 1 1 1 $optimal_bs 0 6 4 HRMQ@128c RTXRMQ LCA GPU_RMQ

echo "---------------------- Comparision of devices for chapter 4 --------------------------"
SCRIPT=plot_perfexp_devices_multiple_algorithms.py
python ${SCRIPT} $q -6 all 20 33 1e-2 15 0 1 1 1 $optimal_bs 3 6 4 RTXRMQ 3 RTX3090 RTXAda4090 RTX6000 LCA 3 RTX3090 RTXAda4090 RTX6000 GPU_RMQ 3 RTX3090 RTXAda4090 RTX6000 

echo "----------------------- Hyperparameter comparision for chapter 4 -----------------------------------"
SCRIPT=plot_combined_heatmap.py
python ${SCRIPT} RTXAda4090 $q -6 - 1 time XXX Hierarchical_vector_load

# python ${SCRIPT} RTX6000 $q -6 RTX6000 1 time XXX Hierarchical_vector_load
