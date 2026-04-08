#pragma once
#include "xxx.h"
#include "rtx_functions.h"
#include "device_tools.cuh"

// --- Interleaved (XXX in OptiX with RT, hence no cooperative group features) -------------------------------------------------------------------------
// has two possible kernels

template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* interleaved_rmq(const nType n, const nType scan_threshold, const uint32_t q, float *d_array, n2Type *d_query, CmdArgs args) {
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
    } else if (scan_threshold >= n) {
        fprintf(stderr, "Error: scan_threshold (%u) must be smaller than n (%u) to avoid edge cases\n", scan_threshold, n);
        if (args.save_time) {
            complete_line(args.time_file);
        }
        exit(EXIT_FAILURE);
    }

    int reps = args.reps;
    int alg = args.alg;
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

    float max = 1e20f;
    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }

    {   
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        kernel_initialise_array<<<grid, block>>>(q, d_output, max);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Build structure
    // XXX Base structure
    VBHMem mem = {0, 0};
    float build_time = 0;
    {
        printf(AC_MAGENTA "Build interleaved XXX (cg=%2u,amnt=%2u).................", cg_size, cg_amount); fflush(stdout);
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
        printf("done: %f ms (%u levels, %u MB)\n" AC_RESET, build_time, metadata.num_levels, mem.out_buffer/1000000);
    }

    // for (nType level = 1; level < metadata.num_levels; ++level) {
    //     printf("level: %i\n", level);
    //     print_array_dev<nType, float>(metadata.level_size[level], d_level_buffer + metadata.level_offset[level]);
    // }

    float geom_time = 0;
    float3 *devVertices;
    uint3 *devTriangles;
    nType N;
    {    
        printf("Generating geometry......................."); fflush(stdout);
        timer.restart();
        nType level = metadata.num_levels - 1;
        float *darray = d_level_buffer + metadata.level_offset[level];
        N = metadata.level_size[level];
        devVertices = gen_vertices_interleaved_dev<nType>(N, darray);
        //print_vertices_dev<nType>(N, devVertices);
        devTriangles = gen_triangles_dev<nType>(N, darray);
        timer.stop();
        geom_time = timer.get_elapsed_ms();
        printf("done: %f ms\n",geom_time); fflush(stdout);
    }

    // RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state, args, true);
    createProgramGroups(state, alg);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // Build Acceleration Structure 
    printf("%sBuild AS on GPU...........................", AC_MAGENTA); fflush(stdout);
    timer.restart();
    buildASFromDeviceData<nType>(mem, state, 3*N, N, devVertices, devTriangles);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    float AS_time = timer.get_elapsed_ms();
    printf("done: %f ms [output: %f MB, temp %f MB]\n" AC_RESET, AS_time, mem.out_buffer/1e6, mem.temp_buffer/1e6);

    // Build optix structure for top X levels combined
    timer.restart();
    ParamsInterleaved<nType, n2Type> params;
    ParamsInterleaved<nType, n2Type> *d_params;

    params.q = q;
    params.handle = state.gas_handle;
    params.query = d_query;
    params.output = d_output;
    params.array = d_array;
    params.level_buffer = d_level_buffer;
    params.scan_threshold = scan_threshold;
    params.metadata = metadata;
    params.min = -1.0f;
    params.max = max;

    unsigned int param_size = sizeof(ParamsInterleaved<nType, n2Type>);
    printf("Parameter size of %7.3f KB............\n", (double)param_size/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&d_params, param_size));
    CUDA_CHECK(cudaMemcpy(d_params, &params, param_size, cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }

    {
        printf(AC_BOLDCYAN "Query interleaved XXX................................."); fflush(stdout);
        timer.restart();
        for (int i = 0; i < reps; ++i) {
            OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params), param_size, &state.sbt, q, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        timer.stop();
        float timems = timer.get_elapsed_ms();
        float avg_time = timems / (1000.0 * reps);
        printf("done: %f secs (avg %f secs) [%.2f RMQs/sec]\n" AC_RESET, timems / 1000, avg_time, (double)q / avg_time);

        if (args.save_time) {
            COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
        }

        write_results(timems, q, build_time + geom_time + AS_time, reps, args, mem);
    }
    // if (args.save_power) {
    //     GPUPowerEnd();
    // }

    CUDA_CHECK(cudaMemcpy(output, d_output, q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_level_buffer));
    return output;
}

// --- XXX in OptiX but without RT! ----------------------------------------------------------------------------------

// same as xxx_rmq_Interleaved_in_CUDA, but with OptiX kernel launched (nevertheless no RT)
// each level reduces the element count by reduction_factor = (cg_size * cg_amount)
template <typename nType, typename n2Type, nType cg_size_log, nType cg_amount_log = 0>
float* xxx_rmq_Interleaved_in_OptiX(nType n, nType scan_threshold, int q, float *d_array, n2Type *d_query, CmdArgs args) {
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
    int alg = args.alg;
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

    float max = 1e20f;
    float* d_level_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, q * sizeof(float)));
    // alloc level buffer only if required
    if (metadata.num_levels > 1) {
        CUDA_CHECK(cudaMalloc(&d_level_buffer, level_offset * sizeof(float)));
    }

    {   
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        kernel_initialise_array<<<grid, block>>>(q, d_output, max);
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

    // RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state, args, true);
    createProgramGroups(state, alg);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // Build optix structure for top X levels combined
    timer.restart();
    ParamsInterleaved<nType, n2Type> params;
    ParamsInterleaved<nType, n2Type> *d_params;

    params.q = q;
    params.handle = state.gas_handle;
    params.query = d_query;
    params.output = d_output;
    params.array = d_array;
    params.level_buffer = d_level_buffer;
    params.scan_threshold = scan_threshold;
    params.metadata = metadata;
    params.min = -1.0f;
    params.max = max;

    unsigned int param_size = sizeof(ParamsInterleaved<nType, n2Type>);
    printf("Parameter size of %7.3f KB............\n", (double)param_size/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&d_params, param_size));
    CUDA_CHECK(cudaMemcpy(d_params, &params, param_size, cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }

    {
        printf(AC_BOLDCYAN "Query XXX................................."); fflush(stdout);
        timer.restart();
        dim3 block(BSIZE, 1, 1);
        dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
        for (int i = 0; i < reps; ++i) {
            OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(d_params), param_size, &state.sbt, q, 1, 1));
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
