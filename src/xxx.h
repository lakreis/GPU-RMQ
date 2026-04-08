#ifndef XXX_H
#define XXX_H
#include <vector>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rtx_functions.h"
#include "device_tools.cuh"

namespace cg = cooperative_groups;

// --- general functions ----------------------------------------------------------------------------------------------------------

template <typename nType, nType cg_size, nType cg_amount>
__device__ __forceinline__
float xxx_static_scan_min(cg::thread_block_tile<cg_size> tile, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    constexpr nType alignment = cg_amount * cg_size;
    const nType start = i_left & ~(alignment - 1);
    const nType offset = start + tile.thread_rank();

    float item = local_min;
    #pragma unroll
    for (nType k = 0; k < cg_amount; ++k) {
        nType i = offset + k * cg_size;
        if (i >= i_left && i < e_right) {
            item = min(item, d_array[i]);
        }
    }
    return item;
}


template <typename nType, uint32_t cg_size>
__device__ __forceinline__
float xxx_scan_min(cg::thread_block_tile<cg_size> tile, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    nType start = i_left & ~(cg_size - 1);

    float item = local_min;
    for (; start < e_right; start += cg_size) {
        nType i = start + tile.thread_rank();
        if (i >= i_left && i < e_right) {
            item = min(item, d_array[i]);
        }
    }
    return item;
}

extern "C" __device__ __forceinline__ 
float atomicMin(float* addr, float value) {
    float old;
    old = __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    return old;
}

// same without cooperative groups
template <typename nType, nType cg_size, nType cg_amount>
//extern "C" 
__device__ __forceinline__
float xxx_static_scan_min(const nType lane_idx, const nType tile_idx, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    constexpr nType alignment = cg_amount * cg_size;
    const nType start = i_left & ~(alignment - 1);
    const nType offset = start + lane_idx;

    float item = local_min;
    #pragma unroll
    for (nType k = 0; k < cg_amount; ++k) {
        nType i = offset + k * cg_size;
        if (i >= i_left && i < e_right) {
            item = min(item, d_array[i]);
        }
    }
    return item;
}

template <typename nType, nType cg_size>
//extern "C"
__device__ __forceinline__
float xxx_scan_min(const nType tile_idx, const nType lane_idx, float local_min, const float* d_array, const nType i_left, const nType e_right) {
    nType start = i_left & ~(cg_size - 1);

    float item = local_min;
    for (; start < e_right; start += cg_size) {
        nType i = start + lane_idx;
        if (i >= i_left && i < e_right) {
            item = min(item, d_array[i]);
        }
    }
    return item;
}

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log>
__global__
void xxx_build_step(const float* d_source, float* d_target, nType source_size, nType target_size) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

    nType tid = blockIdx.x * blockDim.x + threadIdx.x;
    nType num_threads = gridDim.x * blockDim.x;

    for (nType i = tid; i < source_size; i += num_threads) {
        nType gid = i >> cg_size_log;

        if (gid >= target_size) return;

        nType source_start = gid << reduction_factor_log;
        nType source_end = min(source_start + reduction_factor, source_size);

        float local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, INFINITY, d_source, source_start, source_end);
        
        local_min = cg::reduce(tile, local_min, cg::less<float>());
        if (tile.thread_rank() == 0) {
            d_target[gid] = local_min;
        }
    }
}

// --- classic XXX -----------------------------------------------------------------------------------------------------------------------------------------------

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log>
__global__
void xxx_query(uint32_t q, const n2Type* d_query, float* d_output, const float* d_array, const float* d_level_buffer, xxx_metadata_type<nType, n2Type> metadata, nType scan_threshold) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

    float result_min;

    nType l = 0;
    nType r = 0;
    bool to_find = false;
    if (tid < q) {
        l = d_query[tid].x;
        // transform right bound to exclusive
        r = d_query[tid].y + 1;
        to_find = true;
    }
    auto work_queue = tile.ballot(to_find);
    if (!work_queue) return;

    while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        nType local_l = tile.shfl(l, cur_rank);
        nType local_r = tile.shfl(r, cur_rank);
        float local_min = INFINITY;

        // hierarchical lookup starts here
        {
            nType current_left = local_l;
            nType current_right = local_r;
            nType current_level = 0;

            for (; current_level < metadata.num_levels - 1; ++current_level) {
                if (current_right - current_left <= 2*reduction_factor) {
                    break;
                }

                // next multiple of reduction_factor
                nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
                // previous multiple of reduction_factor
                nType prev_right = current_right & ~(reduction_factor - 1);

                // search original array for the first level, otherwise search in the level buffer
                auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
                local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, current_left, next_left);
                local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, prev_right, current_right);

                //printf("thread_idx: %i, current level: %u, cur_rank: %i\n", tid, current_level, cur_rank);
            
                current_left = next_left >> reduction_factor_log;
                current_right = prev_right >> reduction_factor_log;
            }

            // search original array for the first level, otherwise search in the level buffer
            auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
            local_min = xxx_scan_min<nType, cg_size>(tile, local_min, base_ptr, current_left, current_right);
            local_min = cg::reduce(tile, local_min, cg::less<float>());
        }

        if (cur_rank == tile.thread_rank()) {
            result_min = local_min;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (tid < q) {
        d_output[tid] = result_min;
    }
}

// each level reduces the element count by reduction_factor = (cg_size * cg_amount)
template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* xxx_rmq(nType n, nType scan_threshold, uint32_t q, const float *d_array, const n2Type *d_query, CmdArgs args) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    if (scan_threshold < 2 * reduction_factor) {
        fprintf(stderr, "Error: scan_threshold (%u) must be at least 2x reduction_factor (%u) to avoid edge cases\n", scan_threshold, reduction_factor);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    Timer timer;

    float *output, *d_output;
    output = (float*) malloc(q * sizeof(float));

    nType size = n;
    xxx_metadata_type<nType, n2Type> metadata;

    metadata.num_levels = 1;
    metadata.level_offset[0] = ~uint32_t(0);
    metadata.level_size[0] = size;

    nType level_offset = 0;
    while (true) {
        if (metadata.num_levels >= XXX_MAX_LEVELS || size <= scan_threshold) {
            break;
        }
        size = (size + reduction_factor - 1) >> reduction_factor_log;
        level_offset = (level_offset + XXX_LEVEL_ALIGNMENT - 1) & ~(XXX_LEVEL_ALIGNMENT - 1);
        metadata.level_offset[metadata.num_levels] = level_offset;
        metadata.level_size[metadata.num_levels] = size;
        metadata.num_levels++;
        level_offset += size;
    }

    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }

    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build XXX (cg=%2u,amnt=%2u).................", cg_size, cg_amount); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        for (nType level = 0; level < metadata.num_levels - 1; ++level) {
            auto d_source = level == 0 ? d_array : d_level_buffer + metadata.level_offset[level];
            auto d_target = d_level_buffer + metadata.level_offset[level + 1];

            nType thread_count = std::min(metadata.level_size[level + 1] << cg_size_log, (static_cast<nType>(INT32_MAX) >> 1) + 1);            
            dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
            xxx_build_step<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(d_source, d_target, metadata.level_size[level], metadata.level_size[level + 1]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        build_time = timer.get_elapsed_ms();
        mem.out_buffer += level_offset * sizeof(float);
        printf("done: %f ms (%u levels, %f MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1e6);
    }

    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            xxx_query<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(q, d_query, d_output, d_array, d_level_buffer, metadata, scan_threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);
    
        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time, reps, args, mem);
    }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

// --- XXX - adaption 1 (no shuffles) -------------------------------------------------------------------------------------------------------------------------

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log>
__global__
void xxx_query_without_shuffles(uint32_t q, const n2Type* d_query, float* d_output, const float* d_array, const float* d_level_buffer, xxx_metadata_type<nType, n2Type> metadata, nType scan_threshold) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

    float result_min;

    n2Type range;
    const nType tile_idx = tid >> cg_size_log;
    const nType lane_idx = tid % cg_size;
    const uint32_t grid_dim = gridDim.x * blockDim.x;

    // outer for-loop for the case that grid_dim < q
    for (nType query_idx = tile_idx*cg_size; query_idx < q; query_idx += grid_dim) {
        // hierarchical lookup starts here
        for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
            range = d_query[query_idx + active_idx];
            float local_min = INFINITY;
            
            {
                nType current_left = range.x;
                nType current_right = range.y + 1;
                nType current_level = 0;

                for (; current_level < metadata.num_levels - 1; ++current_level) {
                    if (current_right - current_left <= 2*reduction_factor) {
                        break;
                    }

                    // next multiple of reduction_factor
                    nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
                    // previous multiple of reduction_factor
                    nType prev_right = current_right & ~(reduction_factor - 1);

                    // search original array for the first level, otherwise search in the level buffer
                    auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
                    local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, current_left, next_left);
                    local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, prev_right, current_right);

                    //printf("thread_idx: %i, current level: %u, cur_rank: %i\n", tid, current_level, cur_rank);
                    current_left = next_left >> reduction_factor_log;
                    current_right = prev_right >> reduction_factor_log;
                }

                // search original array for the first level, otherwise search in the level buffer
                auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
                local_min = xxx_scan_min<nType, cg_size>(tile, local_min, base_ptr, current_left, current_right);
                local_min = cg::reduce(tile, local_min, cg::less<float>());
            }

            if (active_idx == tile.thread_rank()) {
                result_min = local_min;
            }
        }

        if (query_idx + tile.thread_rank() < q) {
            d_output[query_idx + tile.thread_rank()] = result_min;
        }
    }
}

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* xxx_rmq_without_shuffles(nType n, nType scan_threshold, uint32_t q, const float *d_array, const n2Type *d_query, CmdArgs args) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    if (scan_threshold < 2 * reduction_factor) {
        fprintf(stderr, "Error: scan_threshold (%u) must be at least 2x reduction_factor (%u) to avoid edge cases\n", scan_threshold, reduction_factor);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    Timer timer;

    float *output, *d_output;
    output = (float*) malloc(q * sizeof(float));

    nType size = n;
    xxx_metadata_type<nType, n2Type> metadata;

    metadata.num_levels = 1;
    metadata.level_offset[0] = ~uint32_t(0);
    metadata.level_size[0] = size;

    nType level_offset = 0;
    while (true) {
        if (metadata.num_levels >= XXX_MAX_LEVELS || size <= scan_threshold) {
            break;
        }
        size = (size + reduction_factor - 1) >> reduction_factor_log;
        level_offset = (level_offset + XXX_LEVEL_ALIGNMENT - 1) & ~(XXX_LEVEL_ALIGNMENT - 1);
        metadata.level_offset[metadata.num_levels] = level_offset;
        metadata.level_size[metadata.num_levels] = size;
        metadata.num_levels++;
        level_offset += size;
    }

    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }  

    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build XXX (cg=%2u,amnt=%2u).................", cg_size, cg_amount); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        for (nType level = 0; level < metadata.num_levels - 1; ++level) {
            auto d_source = level == 0 ? d_array : d_level_buffer + metadata.level_offset[level];
            auto d_target = d_level_buffer + metadata.level_offset[level + 1];

            nType thread_count = std::min(metadata.level_size[level + 1] << cg_size_log, (static_cast<nType>(INT32_MAX) >> 1) + 1);            
            dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
            xxx_build_step<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(d_source, d_target, metadata.level_size[level], metadata.level_size[level + 1]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        build_time = timer.get_elapsed_ms();
        mem.out_buffer += level_offset * sizeof(float);
        printf("done: %f ms (%u levels, %f MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1e6);
    }

    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            xxx_query_without_shuffles<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(q, d_query, d_output, d_array, d_level_buffer, metadata, scan_threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);
    
        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time, reps, args, mem);
    }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

// multi load adaption
template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log>
__global__
void xxx_query_multi_load(uint32_t q, const n2Type* d_query, float* d_output, const float* d_array, const float* d_level_buffer, xxx_metadata_type<nType, n2Type> metadata, nType scan_threshold) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto tile = cg::tiled_partition<cg_size>(cg::this_thread_block());

    n2Type range;
    const nType tile_idx = tid >> cg_size_log;
    const nType lane_idx = tid % cg_size;
    const uint32_t grid_dim = gridDim.x * blockDim.x;

    // outer for-loop for the case that grid_dim < q
    for (nType query_idx = tile_idx;  query_idx < q; query_idx += grid_dim) {
        range = d_query[query_idx];
        float local_min = INFINITY;
        
        {
            nType current_left = range.x;
            nType current_right = range.y + 1;
            nType current_level = 0;

            for (; current_level < metadata.num_levels - 1; ++current_level) {
                if (current_right - current_left <= 2*reduction_factor) {
                    break;
                }

                // next multiple of reduction_factor
                nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
                // previous multiple of reduction_factor
                nType prev_right = current_right & ~(reduction_factor - 1);

                // search original array for the first level, otherwise search in the level buffer
                auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
                local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, current_left, next_left);
                local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(tile, local_min, base_ptr, prev_right, current_right);

                //printf("thread_idx: %i, current level: %u, cur_rank: %i\n", tid, current_level, cur_rank);
                current_left = next_left >> reduction_factor_log;
                current_right = prev_right >> reduction_factor_log;
            }

            // search original array for the first level, otherwise search in the level buffer
            auto base_ptr = current_level == 0 ? d_array : d_level_buffer + metadata.level_offset[current_level];
            local_min = xxx_scan_min<nType, cg_size>(tile, local_min, base_ptr, current_left, current_right);
            local_min = cg::reduce(tile, local_min, cg::less<float>());
        }

        if (lane_idx == 0) {
            d_output[tile_idx] = local_min;
        }
    }
}

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* xxx_rmq_multi_load(nType n, nType scan_threshold, uint32_t q, const float *d_array, const n2Type *d_query, CmdArgs args) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    if (scan_threshold < 2 * reduction_factor) {
        fprintf(stderr, "Error: scan_threshold (%u) must be at least 2x reduction_factor (%u) to avoid edge cases\n", scan_threshold, reduction_factor);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    Timer timer;

    float *output, *d_output;
    output = (float*) malloc(q * sizeof(float));

    nType size = n;
    xxx_metadata_type<nType, n2Type> metadata;

    metadata.num_levels = 1;
    metadata.level_offset[0] = ~uint32_t(0);
    metadata.level_size[0] = size;

    nType level_offset = 0;
    while (true) {
        if (metadata.num_levels >= XXX_MAX_LEVELS || size <= scan_threshold) {
            break;
        }
        size = (size + reduction_factor - 1) >> reduction_factor_log;
        level_offset = (level_offset + XXX_LEVEL_ALIGNMENT - 1) & ~(XXX_LEVEL_ALIGNMENT - 1);
        metadata.level_offset[metadata.num_levels] = level_offset;
        metadata.level_size[metadata.num_levels] = size;
        metadata.num_levels++;
        level_offset += size;
    }

    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }  

    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build XXX (cg=%2u,amnt=%2u).................", cg_size, cg_amount); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        for (nType level = 0; level < metadata.num_levels - 1; ++level) {
            auto d_source = level == 0 ? d_array : d_level_buffer + metadata.level_offset[level];
            auto d_target = d_level_buffer + metadata.level_offset[level + 1];

            nType thread_count = std::min(metadata.level_size[level + 1] << cg_size_log, (static_cast<nType>(INT32_MAX) >> 1) + 1);            
            dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
            xxx_build_step<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(d_source, d_target, metadata.level_size[level], metadata.level_size[level + 1]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        build_time = timer.get_elapsed_ms();
        mem.out_buffer += level_offset * sizeof(float);
        printf("done: %f ms (%u levels, %f MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1e6);
    }

    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        nType thread_count = std::min(q << cg_size_log, (static_cast<uint32_t>(INT32_MAX) >> 1) + 1);            
        dim3 block(BSIZE, 1, 1);
        dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            xxx_query_multi_load<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(q, d_query, d_output, d_array, d_level_buffer, metadata, scan_threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);
    
        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time, reps, args, mem);
    }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

// --- XXX - adaption 2 (no shuffles + no reduce) -------------------------------------------------------------------------------------------------------------------------

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log>
__global__
void xxx_query_Interleaved_in_CUDA(uint32_t q, const n2Type* query, float* output, const float* array, const float* level_buffer, xxx_metadata_type<nType, n2Type>metadata, nType scan_threshold) {    
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    n2Type range;
    const nType tile_idx = thread_idx >> cg_size_log;
    const nType lane_idx = thread_idx % cg_size;
    const uint32_t grid_dim = blockDim.x*gridDim.x;
  
    // outer for-loop for the case that grid_dim < q
    for (nType query_idx = tile_idx*cg_size;  query_idx < q; query_idx += grid_dim) {
        // hierarchical lookup starts here
        for (nType active_idx = 0; active_idx < cg_size; active_idx++) {
            range = query[query_idx + active_idx];
            float local_min = INFINITY;

            {
                nType current_left = range.x;
                nType current_right = range.y + 1;
                nType current_level = 0;

                for (; current_level < metadata.num_levels - 1; ++current_level) {
                    if (current_right - current_left <= 2*reduction_factor) {
                        break;
                    }

                    // next multiple of reduction_factor
                    nType next_left = (current_left + reduction_factor - 1) & ~(reduction_factor - 1);
                    // previous multiple of reduction_factor
                    nType prev_right = current_right & ~(reduction_factor - 1);

                    // search original array for the first level, otherwise search in the level buffer
                    const float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
                    local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, current_left, next_left);
                    local_min = xxx_static_scan_min<nType, cg_size, cg_amount>(lane_idx, tile_idx, local_min, base_ptr, prev_right, current_right);

                    //printf("thread_idx: %i, current level: %u, active_idx: %i\n", thread_idx, current_level, active_idx);
                    current_left = next_left >> reduction_factor_log;
                    current_right = prev_right >> reduction_factor_log;
                }

                // search original array for the first level, otherwise search in the level buffer
                const float* base_ptr = current_level == 0 ? array : level_buffer + metadata.level_offset[current_level];
                local_min = xxx_scan_min<nType, cg_size>(tile_idx, lane_idx, local_min, base_ptr, current_left, current_right);
                atomicMin(output + query_idx + active_idx, local_min);
            }
        }
    }
}

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* xxx_rmq_Interleaved_in_CUDA(nType n, nType scan_threshold, uint32_t q, const float *d_array, const n2Type *d_query, CmdArgs args) {
    constexpr nType cg_size = 1u << cg_size_log;
    constexpr nType cg_amount = 1u << cg_amount_log;

    constexpr nType reduction_factor_log = cg_size_log + cg_amount_log;
    constexpr nType reduction_factor = 1u << reduction_factor_log;

    if (scan_threshold < 2 * reduction_factor) {
        fprintf(stderr, "Error: scan_threshold (%u) must be at least 2x reduction_factor (%u) to avoid edge cases\n", scan_threshold, reduction_factor);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    Timer timer;

    float *output, *d_output;
    output = (float*) malloc(q * sizeof(float));

    nType size = n;
    xxx_metadata_type<nType, n2Type>metadata;

    metadata.num_levels = 1;
    metadata.level_offset[0] = ~uint32_t(0);
    metadata.level_size[0] = size;

    nType level_offset = 0;
    while (true) {
        if (metadata.num_levels >= XXX_MAX_LEVELS || size <= scan_threshold) {
            break;
        }
        size = (size + reduction_factor - 1) >> reduction_factor_log;
        level_offset = (level_offset + XXX_LEVEL_ALIGNMENT - 1) & ~(XXX_LEVEL_ALIGNMENT - 1);
        metadata.level_offset[metadata.num_levels] = level_offset;
        metadata.level_size[metadata.num_levels] = size;
        metadata.num_levels++;
        level_offset += size;
    }

    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    } 

    {   
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        kernel_initialise_array<<<grid, block>>>(q, d_output, INFINITY);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build XXX (cg=%2u,amnt=%2u).................", cg_size, cg_amount); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        for (uint32_t level = 0; level < metadata.num_levels - 1; ++level) {
            auto d_source = level == 0 ? d_array : d_level_buffer + metadata.level_offset[level];
            auto d_target = d_level_buffer + metadata.level_offset[level + 1];

            nType thread_count = std::min(metadata.level_size[level + 1] << cg_size_log, (static_cast<nType>(INT32_MAX) >> 1) + 1);
            dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
            xxx_build_step<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(d_source, d_target, metadata.level_size[level], metadata.level_size[level + 1]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        build_time = timer.get_elapsed_ms();
        mem.out_buffer += level_offset * sizeof(float);
        printf("done: %f ms (%u levels, %f MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1e6);
    }

    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            xxx_query_Interleaved_in_CUDA<nType, n2Type, cg_size_log, cg_amount_log><<<grid, block>>>(q, d_query, d_output, d_array, d_level_buffer, metadata, scan_threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);
    
        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time, reps, args, mem);
    }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

// --- XXX for small array sizes with vec4 loads ----------------------------------------------------------------------------------------------

template <typename nType, nType vec_amount_log>
//extern "C" 
__device__ __forceinline__
float vec_scan_min_left(nType vec_n, float local_min, const float4* base_ptr, const nType l) {
    constexpr nType vec_amount = 1u << vec_amount_log;
    constexpr nType alignment = 1u << (vec_amount_log + 2);
    nType start = l & ~(alignment - 1);
    float item = local_min;
    
    #pragma unroll
    for (nType k = 0; k < vec_amount; ++k) {
        nType local_offset = start + 4*k;
        if (local_offset + 3 < l || (local_offset >> 2) >= vec_n) {
            continue;
        }
        float4 entries = base_ptr[local_offset >> 2];
        if (local_offset >= l && entries.x < item) {
            item = entries.x;
        } 
        if (local_offset + 1 >= l && entries.y < item) {
            item = entries.y;
        } 
        if (local_offset + 2 >= l && entries.z < item) {
            item = entries.z;
        } 
        if (local_offset + 3 >= l && entries.w < item) {
            item = entries.w;
        } 
    }
    return item;
}

template <typename nType, nType vec_amount_log>
//extern "C" 
__device__ __forceinline__
float vec_scan_min_right(nType vec_n, float local_min, const float4* base_ptr, const nType r) {
    constexpr nType vec_amount = 1u << vec_amount_log;
    constexpr nType alignment = 1u << (vec_amount_log + 2);
    nType start = r & ~(alignment - 1);
    float item = local_min;

    #pragma unroll
    for (nType k = 0; k < vec_amount; ++k) {
        nType local_offset = start + 4*k;
        if (local_offset >= r) {
            continue;
        }
        float4 entries = base_ptr[local_offset >> 2];
        if (local_offset < r && entries.x < item) {
            item = entries.x;
        } 
        if (local_offset + 1 < r && entries.y < item) {
            item = entries.y;
        } 
        if (local_offset + 2 < r && entries.z < item) {
            item = entries.z;
        } 
        if (local_offset + 3 < r && entries.w < item) {
            item = entries.w;
        } 
    }
    return item;
}

template <typename nType, nType vec_amount_log>
//extern "C" 
__device__ __forceinline__
float vec_scan_min(nType vec_n, float local_min, const float4* base_ptr, const nType l, const nType r) {
    float item = local_min;

    for (nType local_offset = (l & (~3)); local_offset < r; local_offset +=4) {
        float4 entries = base_ptr[local_offset >> 2];
        if (local_offset >= l && local_offset < r && entries.x < item) {
            item = entries.x;
        } 
        if (local_offset + 1 >= l && local_offset + 1 < r && entries.y < item) {
            item = entries.y;
        } 
        if (local_offset + 2 >= l && local_offset + 2 < r && entries.z < item) {
            item = entries.z;
        } 
        if (local_offset + 3 >= l && local_offset + 3 < r && entries.w < item) {
            item = entries.w;
        } 
    }
    return item;
}


template <typename nType, typename n2Type, nType vec_amount_log>
__global__ void kernel_hierarchical_vector_load(const nType vec_n, const uint32_t q, const n2Type* d_query, float* output, float4* vec_d_array, float4* vec_d_level_buffer, xxx_metadata_type<nType, n2Type> metadata, nType scan_threshold){
    constexpr nType reduction_factor_log = vec_amount_log + 2;
    constexpr nType reduction_factor = 1u << reduction_factor_log;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= q) {
        return;
    }

    nType l = d_query[tid].x;
    // right bound exclusive
    nType r = d_query[tid].y + 1;
    float local_min = INFINITY;

    // lookup starts here
    {
        nType current_level = 0;

        for (; current_level < metadata.num_levels - 1; ++current_level) {
            if (r - l <= reduction_factor) {
                break;
            }

            // next multiple of reduction_factor
            nType next_left = (l + reduction_factor - 1) & ~(reduction_factor - 1);
            // previous multiple of reduction_factor
            nType prev_right = r & ~(reduction_factor - 1);
            if (prev_right < next_left) {
                break;
            }

            // search original array for the first level, otherwise search in the level buffer
            auto base_ptr = current_level == 0 ? vec_d_array : vec_d_level_buffer + (metadata.level_offset[current_level] >> 2);
            local_min = vec_scan_min_left<nType, vec_amount_log>(vec_n, local_min, base_ptr, l);
            local_min = vec_scan_min_right<nType, vec_amount_log>(vec_n, local_min, base_ptr, r);

            l = next_left >> reduction_factor_log;
            r = prev_right >> reduction_factor_log;
        }

        // search original array for the first level, otherwise search in the level buffer
        auto base_ptr = current_level == 0 ? vec_d_array : vec_d_level_buffer + (metadata.level_offset[current_level] >> 2);
        local_min = vec_scan_min<nType, vec_amount_log>(vec_n, local_min, base_ptr, l, r);
    }

    output[tid] = local_min;
}

template <typename nType, typename n2Type, nType vec_amount_log = 0>
float* hierarchical_vector_load(const nType n, const nType scan_threshold, const uint32_t q, float *d_array, const n2Type *d_query, CmdArgs args){
    constexpr nType reduction_factor_log = vec_amount_log + 2;
    constexpr nType reduction_factor = 1u << (vec_amount_log + 2);
    constexpr nType vec_amount = 1u << vec_amount_log;

    if (scan_threshold < reduction_factor) {
        fprintf(stderr, "Error: scan_threshold (%u) must be >= reduction_factor (%u) to avoid edge cases\n", scan_threshold, reduction_factor);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    int dev = args.dev;
    int alg = args.alg;
    Timer timer;

    float4* vec4_array = reinterpret_cast<float4*>(d_array); 
    float4* vec4_level_buffer = nullptr;
    // calculate the number of float4's
    nType vec_n = (n >> 2);

    float *output, *d_output;
    output = (float*) malloc(q * sizeof(float));

    nType size = n;
    xxx_metadata_type<nType, n2Type> metadata;

    metadata.num_levels = 1;
    metadata.level_offset[0] = ~uint32_t(0);
    metadata.level_size[0] = size;

    nType level_offset = 0;
    while (true) {
        if (metadata.num_levels >= XXX_MAX_LEVELS || size <= scan_threshold) {
            break;
        }
        size = (size + reduction_factor - 1) >> reduction_factor_log;
        level_offset = (level_offset + XXX_LEVEL_ALIGNMENT - 1) & ~(XXX_LEVEL_ALIGNMENT - 1);
        metadata.level_offset[metadata.num_levels] = level_offset;
        metadata.level_size[metadata.num_levels] = size;
        metadata.num_levels++;
        level_offset += size;
    }

    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }

    if (args.save_power) {
        GPUPowerBegin(algStr[alg], 100, dev, args.power_file);
    }    

    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build XXX (cg=%2u, amnt=%2u).................", 4, 1u << vec_amount_log); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        for (nType level = 0; level < metadata.num_levels - 1; ++level) {
            auto d_source = level == 0 ? d_array : d_level_buffer + metadata.level_offset[level];
            auto d_target = d_level_buffer + metadata.level_offset[level + 1];

            int32_t thread_count = std::min(metadata.level_size[level + 1] << 2, (static_cast<nType>(INT32_MAX) >> 1) + 1);
            dim3 grid((thread_count+BSIZE-1)/BSIZE, 1, 1);
            xxx_build_step<nType, n2Type, 2, vec_amount_log><<<grid, block>>>(d_source, d_target, metadata.level_size[level], metadata.level_size[level + 1]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        vec4_level_buffer = reinterpret_cast<float4*>(d_level_buffer);
        timer.stop();
        build_time = timer.get_elapsed_ms();
        mem.out_buffer += level_offset * sizeof(float);
        printf("done: %f ms (%u levels, %f MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1e6);        
    }
    
    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            kernel_hierarchical_vector_load<nType, n2Type, vec_amount_log><<<grid, block>>>(vec_n, q, d_query, d_output, vec4_array, vec4_level_buffer, metadata, scan_threshold);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);
    
        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time, reps, args, mem);
    }
    if (args.save_power) {
        GPUPowerEnd();
    }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

#endif
