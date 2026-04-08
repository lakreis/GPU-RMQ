#pragma once
#include <omp.h>
#include "../hrmq/includes/RMQRMM64.h"
#include "tools.h"

// CPU min val
template <typename T, typename nType>
T cpu_min(T *A, nType l, nType r) {
    T min = A[l];
    for (nType i = l; i <= r; ++i) {
        if (A[i] < min){
            min = A[i];
        }
    }
    return min;
}

template <typename T, typename nType, typename n2Type>
T* cpu_rmq(nType n, int nq, T *A, n2Type *Q, int nt, CmdArgs args) {
    int reps = args.reps;
    Timer timer;
    T* out = new T[nq];

    omp_set_num_threads(nt);
    printf(AC_BOLDCYAN "Computing RMQs (%-11s, nt=%2i,r=%-3i).." AC_RESET, algStr[ALG_CPU_BASE], nt, reps); fflush(stdout);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        #pragma omp parallel for shared(out, A, Q)
        for (int i = 0; i < nq; ++i) {
            out[i] = cpu_min<T, nType>(A, Q[i].x, Q[i].y);
        }
    }
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float time_it = timems/reps;
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, timems/(1000.0*reps), (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq, 0, reps, args);
    return out;
}

// CPU min idx
template <typename T, typename nType>
nType cpu_min_idx(T *A, nType l, nType r) {
    T min = A[l];
    nType min_idx = l;
    for (nType i = l; i <= r; ++i) {
        if (A[i] < min){
            min = A[i];
            min_idx = i;
        }
    }
    return min_idx;
}

template <typename T, typename nType, typename n2Type>
nType* cpu_rmq_idx(nType n, int nq, T *A, n2Type *Q, int nt, CmdArgs args) {
    int reps = args.reps;
    Timer timer;
    nType* out = new nType[nq];

    omp_set_num_threads(nt);
    printf(AC_BOLDCYAN "Computing RMQs index (%-11s, nt=%2i,r=%-3i).." AC_RESET, algStrIdx[ALG_CPU_BASE_IDX%10], nt, reps); fflush(stdout);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        #pragma omp parallel for shared(out, A, Q)
        for (int i = 0; i < nq; ++i) {
            out[i] = cpu_min_idx<T, nType>(A, Q[i].x, Q[i].y);
        }
    }
    //printf("dbg5\n"); fflush(stdout);
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float time_it = timems/reps;
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, timems/(1000.0*reps), (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq, 0, reps, args);
    return out;
}

// HRMQ val
template <typename nType, typename n2Type>
int *rmq_rmm_par(nType n, int nq, int *A, n2Type *Q, int nt, CmdArgs args) {
    using namespace rmqrmm;
    omp_set_num_threads(nt);

    int reps = args.reps;
    Timer timer;
    RMQRMM64 *rmq = NULL;
    int* out = new int[nq];

    // create rmq struct
    printf("Creating MinMaxTree......................."); fflush(stdout);
    timer.restart();
    rmq = new RMQRMM64(A, (unsigned long)n);
    uint size = rmq->getSize();
    timer.stop();
    float struct_time = timer.get_elapsed_ms();
    printf("done: %f ms (%f MB)\n", struct_time, (double)size/1e6);

    //printf("%sAnswering Querys [%2i threads]......", AC_BOLDCYAN, nt); fflush(stdout);
    printf(AC_BOLDCYAN "Computing RMQs (%-11s, nt=%2i,r=%-3i).." AC_RESET, algStr[ALG_CPU_HRMQ], nt, reps); fflush(stdout);
    if (args.save_power)
        CPUPowerBegin("HRMQ", 500, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        #pragma omp parallel for shared(rmq, out, A, Q)
        for (int j = 0; j < nq; ++j) {
            nType idx = rmq->queryRMQ(Q[j].x, Q[j].y);
            out[j] = A[idx];
        }
    }
    timer.stop();
    if (args.save_power)
        CPUPowerEnd();
    double timems = timer.get_elapsed_ms();
    float time_it = timems/reps;
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, timems/(1000.0*reps), (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq, struct_time, reps, args, {size, 0});
    return out;
}

template <typename nType, typename n2Type>
nType *rmq_rmm_par_idx(nType n, int nq, int *A, n2Type *Q, int nt, CmdArgs args) {
    using namespace rmqrmm;
    omp_set_num_threads(nt);

    int reps = args.reps;
    Timer timer;
    RMQRMM64 *rmq = NULL;
    nType* out = new nType[nq];

    // create rmq struct
    printf("Creating MinMaxTree......................."); fflush(stdout);
    timer.restart();
    rmq = new RMQRMM64(A, (unsigned long)n);
    uint size = rmq->getSize();
    timer.stop();
    float struct_time = timer.get_elapsed_ms();
    printf("done: %f ms (%f MB)\n", struct_time, (double)size/1e6);

    //printf("%sAnswering Querys [%2i threads]......", AC_BOLDCYAN, nt); fflush(stdout);
    printf(AC_BOLDCYAN "Computing RMQs index (%-11s, nt=%2i,r=%-3i).." AC_RESET, algStrIdx[ALG_CPU_HRMQ_IDX%10], nt, reps); fflush(stdout);
    if (args.save_power)
        CPUPowerBegin("HRMQ_IDX", 500, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        #pragma omp parallel for shared(rmq, out, A, Q)
        for (int j = 0; j < nq; ++j) {
            out[j] = rmq->queryRMQ(Q[j].x, Q[j].y);
        }
    }
    timer.stop();
    if (args.save_power)
        CPUPowerEnd();
    double timems = timer.get_elapsed_ms();
    float time_it = timems/reps;
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, timems/(1000.0*reps), (double)nq/(time_it/1000.0), (double)time_it*1e6/nq);
    write_results(timems, nq, struct_time, reps, args, {size, 0});
    return out;
}
