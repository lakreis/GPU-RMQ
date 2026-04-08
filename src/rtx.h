#pragma once

long2 make_type2(long x, long y) {
    return make_long2(x, y);
}

int2 make_type2(int x, int y) {
    return make_int2(x, y);
}

ulong2 make_type2(unsigned long x, unsigned long y) {
    return make_ulong2(x, y);
}

uint2 make_type2(unsigned int x, unsigned int y) {
    return make_uint2(x, y);
}

template <typename nType, typename n2Type>
nType* compute_min_blocks(nType n, float* d_array, nType num_blocks, nType block_size) {
    n2Type *queries, *d_queries;
    queries = (n2Type*)malloc(num_blocks * sizeof(n2Type));

    for (nType i = 0; i < num_blocks; ++i) {
        queries[i] = make_type2(i*block_size, (i+1)*block_size-1);
    }

    CUDA_CHECK( cudaMalloc(&d_queries, num_blocks * sizeof(n2Type)) );
    CUDA_CHECK( cudaMemcpy(d_queries, queries, num_blocks * sizeof(n2Type), cudaMemcpyHostToDevice) );

    nType* min_blocks;
    CUDA_CHECK( cudaMalloc(&min_blocks, num_blocks * sizeof(nType)) );

    dim3 block(BSIZE, 1, 1);
    dim3 grid((num_blocks+BSIZE-1)/BSIZE, 1, 1);

    kernel_rmq_basic_idx<nType, n2Type><<<grid, block>>>(n, num_blocks, d_array, d_queries, min_blocks);
    CUDA_CHECK( cudaDeviceSynchronize() );

    return min_blocks;
}


template <typename T, typename nType, typename n2Type>
T* rtx_rmq(int alg, nType n, nType log_bs, uint32_t q, float *darray, n2Type *dquery, CmdArgs args) {
    int dev = args.dev;
    int reps = args.reps;
    nType bs = (1 << log_bs);
    Timer timer;
    T *output, *d_output;
    output = (T*)malloc(q*sizeof(T));

    // 1) Generate geometry from device data
    printf("Generating geometry......................."); fflush(stdout);
    timer.restart();
    float3 *devVertices;
    // nType orig_n = n;
    nType num_blocks;
    devVertices = gen_vertices_blocks_dev<nType>(n, bs, darray);
    num_blocks = (n+bs-1) / bs;
    n += num_blocks;
    uint3 *devTriangles = gen_triangles_dev<nType>(n, darray); 
    // print_array_dev<nType, float>(orig_n, darray);
    // print_array_dev<nType, n2Type>(q, dquery);
    // print_vertices_dev<nType>(n, devVertices);
    timer.stop();
    float geom_time = timer.get_elapsed_ms();
    printf("done: %f ms\n",geom_time); fflush(stdout);

    nType *min_blocks;
    if (alg == ALG_GPU_RTX_BLOCKS_IDX)
        min_blocks = compute_min_blocks<nType, n2Type>(n, darray, num_blocks, bs);

    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config................................"); fflush(stdout);
    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state, args);
    createProgramGroups(state, alg);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());

    // 3) Build Acceleration Structure 
    printf("%sBuild AS on GPU...........................", AC_MAGENTA); fflush(stdout);
    VBHMem mem = {0, 0};
    timer.restart();
    buildASFromDeviceData<nType>(mem, state, 3*n, n, devVertices, devTriangles);
    cudaDeviceSynchronize();
    timer.stop();
    float AS_time = timer.get_elapsed_ms();
    printf("done: %f ms [output: %f MB, temp %f MB]\n" AC_RESET, AS_time, mem.out_buffer/1e6, mem.temp_buffer/1e6);

    // 4) Populate and move parameters to device (ONCE)
    CUDA_CHECK( cudaMalloc(&d_output, q*sizeof(T)) );
    timer.restart();

    Params<nType, n2Type> params;
    Params<nType, n2Type> *device_params;

    nType param_size = sizeof(Params<nType, n2Type>);

    params.handle = state.gas_handle;
    params.min = -1.0f;
    params.max = 10.0f;
    params.output = alg < 100 ? (float*)d_output : nullptr;
    params.idx_output = alg < 100 ? nullptr : (nType*)d_output;

    params.query = nullptr;
    params.iquery = dquery;
    params.num_blocks = (nType)ceil(sqrt(num_blocks + 1));
    params.block_size = bs;
    params.nb = num_blocks;
    if (alg == ALG_GPU_RTX_BLOCKS_IDX) {
        params.min_block = min_blocks;
    }

    printf("(%7.3f MB).........", (double)param_size/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, param_size));
    CUDA_CHECK(cudaMemcpy(device_params, &params, param_size, cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }

    // 5) Computation
    if (alg < 100) {
        printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[alg], reps); fflush(stdout);
    } else {
        printf(AC_BOLDCYAN "Computing RMQs index (%-16s,r=%-3i)..." AC_RESET, algStrIdx[alg%10], reps); fflush(stdout);
    }

    if (args.save_power)
        if (args.alg < 100) {
            GPUPowerBegin(algStr[alg], 100, dev, args.power_file);
        } else {
            GPUPowerBegin(algStrIdx[alg%10], 100, dev, args.power_file);
        }
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), param_size, &state.sbt, q, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    if (args.save_power)
        GPUPowerEnd();
    CUDA_CHECK( cudaMemcpy(output, d_output, q*sizeof(T), cudaMemcpyDeviceToHost) );

    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/avg_time, (double)avg_time*1e9/q);
    
    if (args.save_time) {
        COMPLETE_TIME_FILE_CUDA_ERROR(args.time_file);
    }

    write_results(timems, q, geom_time + AS_time, reps, args, mem);
        
    // 6) clean up
    printf("cleaning up RTX environment..............."); fflush(stdout);
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    for (int i = 0; i < 3; ++i) {
        OPTIX_CHECK(optixProgramGroupDestroy(state.program_groups[i]));
    }
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(device_params));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    printf("done: %f ms\n", timer.get_elapsed_ms());
    return output;
}


