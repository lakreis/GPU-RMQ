#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cooperative_groups.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#include "constants.cuh"

#if IS_LONG
using nType = uint64_t;
using n2Type = ulong2;
#else
using nType = uint32_t;
using n2Type = uint2;
#endif

#define RAND_TRIVCHECK_LOG_NUM_DIFFQ 15
#define TRIVCHECK_LOG_NUM_DIFFQ 10

// to save arrays, queries and RMQ results as Bit-Stream after computing it once to save evaluation run time (not the actual evaluation results)
// TODO@newUser: write your local path here
std::string directory_save_aux_data = "write/your/path/here";

#define ALG_CPU_BASE              0
#define ALG_CPU_HRMQ              1
#define ALG_GPU_BASE              2
#define ALG_GPU_RTX_BLOCKS        5
#define ALG_GPU_RTX_SER           10
#define ALG_XXX                   16
#define ALG_INTERLEAVED           17
#define ALG_INTERLEAVED2          18
#define ALG_INTERLEAVED_IN_CUDA   19
#define ALG_HIERARCHICAL_VECTOR_LOAD     20
#define ALG_INTERLEAVED_IN_OPTIX  21
#define ALG_XXX_WITHOUT_SHUFFLES  23
#define ALG_XXX_MULTI_LOAD        24

#define ALG_CPU_BASE_IDX        100
#define ALG_CPU_HRMQ_IDX        101
#define ALG_GPU_BASE_IDX        102
#define ALG_GPU_RTX_BLOCKS_IDX  105

const char *algStr[25] = { "[CPU] BASE", "[CPU] HRMQ", "[GPU] BASE", "", "", "[GPU] RTX_blocks", "", "", "", "", "[GPU] RTX_ser", "", "", "", "", "", "[GPU] XXX", "[GPU] Interleaved", "[GPU] Interleaved2", "[GPU] Interleaved in CUDA", "[GPU] BASIC VECTOR LOAD",  "[GPU] Interleaved_in_OptiX", "", "[GPU] XXX without shuffles", "[GPU] XXX multi load"}; 
const char *algStrIdx[25] = { "[CPU] BASE IDX", "[CPU] HRMQ IDX", "[GPU] BASE IDX", "",  "", "[GPU] RTX_blocks IDX",  "",  "", "", "",  "",  "",  "",  "", "", "", "", "", "", "", "", "", "", "", ""};

#include "common/common.h"
#include "common/Timer.h"
#include "common/nvmlPower.hpp"
#include "src/rand.cuh"
#include "src/tools.h"
#include "src/device_tools.cuh"
#include "src/cpu_methods.h"
#include "src/cuda_methods.cuh"
#include "src/rtx_functions.h"
#include "src/rtx.h"
#include "src/xxx.h"
#include "src/interleaved.cuh"

int main(int argc, char *argv[]) {
    printf("----------------------------------\n");
    printf("        RTX-RMQ by Temporal       \n");
    printf("----------------------------------\n");

    CmdArgs args = get_args(argc, argv);

    int reps = args.reps;
    int seed = args.seed;
    int dev = args.dev;
    nType n = args.n;
    nType nb = args.nb;
    nType log_bs = args.log_bs;
    uint32_t q = args.q;
    int lr = args.lr;
    int nt = args.nt;
    int alg = args.alg;

    switch(alg){
        case ALG_INTERLEAVED:
        case ALG_INTERLEAVED2:
        case ALG_XXX:
        case ALG_INTERLEAVED_IN_CUDA:
        case ALG_HIERARCHICAL_VECTOR_LOAD:
        case ALG_INTERLEAVED_IN_OPTIX:
        case ALG_XXX_WITHOUT_SHUFFLES:
        case ALG_XXX_MULTI_LOAD:
            create_XXX_header(args.time_file);
            break;
        default:
            create_header(args.time_file);
            break;
    }

    cudaSetDevice(dev);
    omp_set_num_threads(nt);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    float *hA, *dA;
    int *hAi;

    n2Type *hQ, *dQ;

    Timer timer;
    printf(AC_YELLOW "Generating n=%-10ld values............", n); fflush(stdout);
    hA = get_or_generate_array<nType>(q, lr, n, nt, seed, args);
    CUDA_CHECK( cudaMalloc(&dA, sizeof(float)*n) );
    CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(float)*n, cudaMemcpyHostToDevice) );
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    timer.restart();
    printf(AC_YELLOW "Generating q=%-10i queries...........", q); fflush(stdout);
    hQ = get_or_generate_queries<nType, n2Type>(q, lr, n, nt, seed, args);
    CUDA_CHECK( cudaMalloc(&dQ, sizeof(n2Type)*q) );
    CUDA_CHECK( cudaMemcpy(dQ, hQ, sizeof(n2Type)*q, cudaMemcpyHostToDevice) );

    printf("done: %f secs\n" AC_RESET, timer.get_elapsed_ms()/1000.0f);

    // print_array_dev<nType, float>(n, dA);
    // print_array_dev<nType, n2Type>(q, dQ);

    CUDA_CHECK( cudaDeviceSynchronize() );

    write_results<nType>(dev, alg, n, log_bs, nb, q, lr, reps, nt, args);

    // 2) computation

    float *out;
    nType *outi;

    switch(alg){
        case ALG_CPU_BASE:
            out = cpu_rmq<float, nType, n2Type>(n, q, hA, hQ, nt, args);
            break;
        case ALG_CPU_BASE_IDX:
            outi = cpu_rmq_idx<float, nType, n2Type>(n, q, hA, hQ, nt, args);
            break;
        case ALG_CPU_HRMQ:
            hAi = reinterpret_cast<int*>(hA);
            {
                int* outInt = rmq_rmm_par<nType, n2Type>(n, q, hAi, hQ, nt, args);
                out = reinterpret_cast<float*>(outInt);
            }
            break;
        case ALG_CPU_HRMQ_IDX:
            hAi = reinterpret_cast<int*>(hA);
            outi = rmq_rmm_par_idx<nType, n2Type>(n, q, hAi, hQ, nt, args);
            break;
        case ALG_GPU_BASE:
            out = gpu_rmq_basic<nType, n2Type>(n, q, dA, dQ, args);
            break;
        case ALG_GPU_BASE_IDX:
            outi = gpu_rmq_basic_idx<nType, n2Type>(n, q, dA, dQ, args);
            break;
        case ALG_XXX:
            out = xxx_rmq<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_XXX_WITHOUT_SHUFFLES:
            out = xxx_rmq_without_shuffles<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_INTERLEAVED_IN_CUDA:
            out = xxx_rmq_Interleaved_in_CUDA<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_INTERLEAVED:
        case ALG_INTERLEAVED2:
            out = interleaved_rmq<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_INTERLEAVED_IN_OPTIX:
            out = xxx_rmq_Interleaved_in_OptiX<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_HIERARCHICAL_VECTOR_LOAD:
            out = hierarchical_vector_load<nType, n2Type, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        case ALG_XXX_MULTI_LOAD:
            out = xxx_rmq_multi_load<nType, n2Type, XXX_CG_SIZE_LOG, XXX_CG_AMOUNT_LOG>(n, (1 << log_bs), q, dA, dQ, args);
            break;
        default:
            if (alg < 100)
                out = rtx_rmq<float, nType, n2Type>(alg, n, log_bs, q, dA, dQ, args);
            else
                outi = rtx_rmq<nType, nType, n2Type>(alg, n, log_bs, q, dA, dQ, args);
            break;
    }

    if (args.check){
        printf("\nCHECKING RESULT:\n");
        args.reps = 1;
        int save_time = args.save_time;
        args.save_time = 0;
        args.save_power = 0;

        nType *indices;
        int pass;
        if (alg < 100) {
            float *expected = get_or_generate_expected_val<nType, n2Type>(n, q, dA, dQ, lr, nt, seed, indices, args);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result<nType, n2Type>(hA, hQ, q, expected, out, indices);
        } else {
            nType *expected = get_or_generate_expected_index<nType, n2Type>(n, q, dA, dQ, lr, nt, seed, args);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result_idx<nType, n2Type>(hA, hQ, q, expected, outi);
        }     
        args.save_time = save_time;
        write_check_result(pass, args);
        printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");
    } else if (args.trivialCheck) {
        printf("\nTRIVIALLY CHECKING RESULT:\n");
        args.reps = 1;
        int save_time = args.save_time;
        args.save_time = 0;
        args.save_power = 0;
        int pass;

        uint32_t ori_q = q;
        if (q > (1 << TRIVCHECK_LOG_NUM_DIFFQ)) {
            q = (1 << TRIVCHECK_LOG_NUM_DIFFQ);
        }
                
        if (alg < 100) {
            nType *res_indices;
            float *res = get_or_generate_expected_val<nType, n2Type>(n, q, dA, dQ, lr, nt, seed, res_indices, args);
            float *expected = new float[ori_q];
            fill_repeatedly<float, nType>(res, expected, ori_q);
            nType *indices = new nType[ori_q];
            fill_repeatedly<nType, nType>(res_indices, indices, ori_q);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result<nType, n2Type>(hA, hQ, ori_q, expected, out, indices);
        } else {
            nType *res = get_or_generate_expected_index<nType, n2Type>(n, q, dA, dQ, lr, nt, seed, args);
            nType *expected = new nType[ori_q];
            fill_repeatedly<nType, nType>(res, expected, ori_q);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result_idx<nType, n2Type>(hA, hQ, ori_q, expected, outi);
        }
        args.save_time = save_time;
        write_check_result(pass, args);
        printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");
    } else if (args.randTrivialCheck) {
        printf("\nRANDOMLY TRIVIALLY CHECKING RESULT:\n");
        args.reps = 1;
        int save_time = args.save_time;
        args.save_time = 0;
        args.save_power = 0;
        int pass;

        uint32_t ori_q = q;
        if (q > (1 << RAND_TRIVCHECK_LOG_NUM_DIFFQ)) {
            q = (1 << RAND_TRIVCHECK_LOG_NUM_DIFFQ);
        }
        uint32_t* h_sampled_indices = randomly_sample_indices<uint32_t>(ori_q, q, seed);
        n2Type* h_sampled_queries = (n2Type*) malloc(q * sizeof(n2Type));
        for (uint32_t i = 0; i < q; ++i) {
            h_sampled_queries[i] = hQ[h_sampled_indices[i]];
        }
        n2Type* d_sampled_queries = nullptr;
        CUDA_CHECK( cudaMalloc(&d_sampled_queries, sizeof(n2Type)*q) );
        CUDA_CHECK( cudaMemcpy(d_sampled_queries, h_sampled_queries, sizeof(n2Type)*q, cudaMemcpyHostToDevice) );
        
        if (alg < 100) {
            nType *indices;
            float *expected = get_or_generate_expected_val<nType, n2Type>(n, q, dA, d_sampled_queries, lr, nt, seed, indices, args);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result_rand_sample<nType, n2Type>(hA, hQ, ori_q, q, h_sampled_indices, expected, out, indices);
        } else {
            nType *expected = get_or_generate_expected_index<nType, n2Type>(n, q, dA, dQ, lr, nt, seed, args);
            printf(AC_YELLOW "Checking result..........................." AC_YELLOW); fflush(stdout);
            pass = check_result_idx_rand_sample<nType, n2Type>(hA, hQ, ori_q, q, h_sampled_indices, expected, outi);
        }
        args.save_time = save_time;
        write_check_result(pass, args);
        printf(AC_YELLOW "%s\n" AC_RESET, pass ? "pass" : "failed");
    } else {
        write_check_result(-1, args);
    }   

    printf("Benchmark Finished\n");
    return 0;
}
