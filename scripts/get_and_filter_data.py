import pandas as pd
import numpy as np
from math import log2, pow
import os
import csv
from static_data import nb_log_precision_upper_bound, bs_log_precision_upper_bound

def get_prepared_data(file, alg_label, alg=None, nt=None, reps=None, n=None, bs=None, nb=None, q=None, lr=None,
                      GPU_BSIZE=1024, timestamp_start= None, timestamp_end= None, 
                      check=True, groupby_elements=['dev','alg','n','lr', 'q', 'bs', 'nb'], raise_error=True,
                      filter_invalid_precision_runs=True, XXX_scan_threshold=64, XXX_CG_size_log=4, XXX_CG_amount_log=1, XXX_chunk_size=False, hyperparameter_columns=[]):
    df = pd.read_csv(file, skip_blank_lines=True, delimiter=",", dtype={"lr": str}).dropna()
    df['timestamp_date'] = pd.to_datetime(df['timestamp'])
    if alg is not None:
        df = df[df.alg == alg]
    if nt is not None and "nt" not in hyperparameter_columns:
        df = df[df.nt == nt]
    if reps is not None:
        df = df[df.reps == reps]
    if n is not None:
        df = df[df.n == n]
    if bs is not None:
        df = df[df.bs == bs]
    if nb is not None:
        df = df[df.nb == nb]
    if q is not None:
        df = df[df.q == q]
    if lr is not None:
        df = df[df.lr == lr]
    if GPU_BSIZE is not None and alg_label != "HRMQ@128c" and "GPU_BSIZE" not in hyperparameter_columns:
        df = df[df.GPU_BSIZE == GPU_BSIZE]
    if timestamp_start is not None:
        start_time = pd.to_datetime(timestamp_start)
        df = df[df.timestamp_date >= start_time]
    if timestamp_end is not None:
        end_time = pd.to_datetime(timestamp_end)
        df = df[df.timestamp_date <= end_time]
    if alg_label in ["XXX", "Interleaved", "Interleaved2", "Interleaved_in_CUDA", "Hierarchical_vector_load", "Interleaved_in_OptiX", "XXX_without_shuffles", "GPU_RMQ"]:
        if XXX_CG_size_log is not None and "XXX_CG_SIZE_LOG" not in hyperparameter_columns:
            df = df[df.XXX_CG_SIZE_LOG == XXX_CG_size_log]
        if XXX_CG_amount_log is not None and "XXX_CG_AMOUNT_LOG" not in hyperparameter_columns:
            df = df[df.XXX_CG_AMOUNT_LOG == XXX_CG_amount_log]
        if XXX_scan_threshold is not None and "scan_threshold" not in hyperparameter_columns:
            df = df[df.scan_threshold == XXX_scan_threshold]
        if XXX_chunk_size and "XXX_chunk_size" not in hyperparameter_columns:
            if alg_label == "Hierarchical_vector_load":
                df['XXX_chunk_size'] = df['XXX_CG_AMOUNT_LOG'] + 2
            else:
                df['XXX_chunk_size'] = df['XXX_CG_AMOUNT_LOG'] + df['XXX_CG_SIZE_LOG']
    if filter_invalid_precision_runs:
        nb_log_upper_bound = nb_log_precision_upper_bound[alg_label]
        bs_log_upper_bound = bs_log_precision_upper_bound[alg_label]
        if nb_log_upper_bound is not None:
            df = df[df.nb <= 2**nb_log_upper_bound]
        if bs_log_upper_bound is not None:
            df = df[df.bs <= 2**bs_log_upper_bound]
    if check:
        if 2 in df.checkResult.unique():
            print_parameters_check_result(df, 2)
            print_checkResult_precision_condition_false(df)
            if raise_error:
                raise ValueError("A trivial check result test failed!")
            else:
                print("A trivial check result test failed!")
        if 0 in df.checkResult.unique():
            print_parameters_check_result(df, 0)
            if raise_error:
                raise ValueError("A check result test failed!")
            else:
                print("A check result test failed!")
        if 4 in df.checkResult.unique():
            print_parameters_check_result(df, 4)
            if raise_error:
                raise ValueError("A check result test failed!")
            else:
                print("A check result test failed!")
    # remove test cases and wrong results
    df = df[df.checkResult.isin([-1, 1, 5])]

    df = df.drop(columns=['timestamp', 'timestamp_date'])
    if len(hyperparameter_columns) > 0:
        groupby_elements += hyperparameter_columns
    df = df.groupby(groupby_elements).agg(["mean", "std"]).reset_index()
    df['n-exp'] = np.log2(df['n'])
    df['q-exp'] = np.log2(df['q'])
    df['memory'] = df['outbuffer', 'mean'] / 1e3
    df['free_memory'] = df['freeGPUMem', 'mean'] / 1e3
    df['construction_time'] = df['construction', 'mean']

    df['mean_ns/q'] = df['ns/q','mean']
    df['mean_t'] = df['t', 'mean']
    df['throughput'] = 1 / (df['mean_ns/q'] * df['outbuffer', 'mean'])

    df['time_throughput'] = df['n'] / df['mean_ns/q']
    return df

def split_at_array_size(first_dataset, second_dataset, last_small_x_value_log):
    df1 = first_dataset[first_dataset['n-exp'] <= last_small_x_value_log]
    df2 = second_dataset[second_dataset['n-exp'] > last_small_x_value_log]
    return pd.concat([df1, df2], ignore_index=True)

def prepare_hyperparameter_data(algo, df, hyperparameter_column):
    if hyperparameter_column == "scan_threshold" and algo != "Interleaved2": 
        print("Be aware that scan thresholds >= 2^10 are filtered per default because they are not efficient (except for Interleaved2!)")
        return [df[df[hyperparameter_column] == para_value] for para_value in np.sort(df[hyperparameter_column].unique()) if int(log2(para_value)) < 10]
    return [df[df[hyperparameter_column] == para_value] for para_value in np.sort(df[hyperparameter_column].unique())]

def print_parameters_check_result(df, checkResult):
    df = df[df['checkResult'] == checkResult]
    print(f"Checking the parameters of all the rows with checkResult = {checkResult}")
    for index, row in df.iterrows():
        print(f"checkResult = {row['checkResult']}, algo = {row['alg']}, n_log = {log2(row['n'])}, lr = {row['lr']}, bs_log = {log2(row['bs'])}")

def print_checkResult_precision_condition_false(df):
    df = df[df['nb'] > 2**22]
    print("Checking the checkResult value of all the rows with nb > 2**22")
    print(df.value_counts("checkResult"))

def print_optimal_blocksize(df, column="ns/q", n=None, lr=None, save_file=None):
    if n is not None:
        df = df[df.n == n]
    if lr is not None:
        df = df[df.lr == lr]
    for lr_elem in df.lr.unique():
        for n_elem in df.n.unique():
            df2 = df[df.n == n_elem]
            df2 = df2[df2.lr == lr_elem]
            optimal_row = df2[df2[column, 'mean'] == df2[column, 'mean'].min()]
            if save_file is not None:
                if not os.path.isfile(save_file):
                    with open(save_file, 'w') as f:
                        writer = csv.writer(f, delimiter=",")
                        writer.writerow(["n", "log n", "lr", "optimal bs", "log optimal bs", "optimal nb", "log optimal nb", "column", "column value"])
                with open(save_file, 'a') as f:
                    writer = csv.writer(f, delimiter=",")
                    writer.writerow([n_elem, int(np.log2(n_elem)), lr_elem, optimal_row.bs.values[0], int(np.log2(optimal_row.bs.values[0])), optimal_row.nb.values[0], int(np.log2(optimal_row.nb.values[0])), column, optimal_row[column, 'mean'].values[0]])
            else:
                print(f"lr: {lr_elem}, n: {n_elem}, optimal blocksize: {optimal_row.bs.values[0]} (log2: {int(np.log2(optimal_row.bs.values[0]))}), num blocks: {optimal_row.nb.values[0]} (log2: {int(np.log2(optimal_row.nb.values[0]))}), {column}: {optimal_row[column, 'mean'].values[0]}")

def filter_optimal_blocksize(df, column="ns/q", n=None, lr=None):
    if n is not None:
        df = df[df.n == n]
    if lr is not None:
        df = df[df.lr == lr]
    result = pd.DataFrame(columns=df.columns)
    for lr_elem in df.lr.unique():
        for n_elem in df.n.unique():
            df2 = df[df.n == n_elem]
            df2 = df2[df2.lr == lr_elem]
            data = df2[df2[column, 'mean'] == df2[column, 'mean'].min()]
            if result.empty:
                result = data.copy()
            else:
                result = pd.concat([result, data])
    return result

def filter_approx_optimal_blocksize(df, algo_label, n=None, lr=None):
    if n is not None:
        df = df[df.n == n]
    if lr is not None:
        df = df[df.lr == lr]
    result = pd.DataFrame(columns=df.columns)
    for lr_elem in df.lr.unique():
        for n_elem in df.n.unique():
            df2 = df[df.n == n_elem]
            df2 = df2[df2.lr == lr_elem]
            data = df2[df2["nb"] == pow(2, get_approx_optimal_nb_log(algo_label, lr_elem, n_elem))]
            if result.empty:
                result = data.copy()
            else:
                result = pd.concat([result, data])
    return result

def get_approx_optimal_nb_log(algo, lr, n):
    n = log2(n)
    if lr == -1:
        if algo in ["RTXRMQ", "SER"]:
            if n in range(10, 15):
                return 0
            elif n in range(15, 18):
                return 9
            elif n in range(18, 20):
                return 11
            elif n in range(20, 23):
                return 14
            elif n in range(23, 28):
                return 9
            elif n in range(28, 33):
                return 17
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
    else:
            if n in range(10, 13):
                return 5
            elif n in range(13, 16):
                return 7
            elif n in range(16, 20):
                return 11
            elif n in range(20, 24):
                return 14
            elif n in range(24, 28):
                return 18
            elif n in range(28, 33):
                return 19
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
    elif lr == -2:
        if algo in ["RTXRMQ", "SER"]:
            if n in range(10, 15):
                return 0
            elif n in range(15, 18):
                return 0
            elif n in range(18, 20):
                return 1
            elif n in range(20, 23):
                return 5
            elif n in range(23, 28):
                return 9
            elif n in range(28, 33):
                return 17
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
        else:
            if n in range(10, 13):
                return 5
            elif n in range(13, 17):
                return 6
            elif n in range(17, 22):
                return 13
            elif n in range(22, 25):
                return 17
            elif n in range(25, 28):
                return 21
            elif n in range(28, 33):
                return 22
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
    else:
        if algo in ["RTXRMQ", "SER"]:
            if n in range(10, 13):
                return 1
            elif n in range(13, 17):
                return 5
            elif n in range(17, 20):
                return 10
            elif n in range(20, 23):
                return 12
            elif n in range(23, 28):
                return 12
            elif n in range(28, 33):
                return 12
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
        else:
            if n in range(10, 13):
                return 1
            elif n in range(13, 16):
                return 1
            elif n in range(16, 20):
                return 11
            elif n in range(20, 24):
                return 12
            elif n in range(24, 28):
                return 2
            elif n in range(28, 33):
                return 4
            else:
                raise ValueError(f"No approx. optimal nb defined for algo {algo} with n = {n} and lr = {lr}")
    
