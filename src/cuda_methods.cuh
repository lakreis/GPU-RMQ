#pragma once
// Kernel RMQs basic

template <typename nType, typename n2Type>
__global__ void kernel_rmq_basic(nType n, int q, float *x, n2Type *rmq, float *out){
    nType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    nType l = rmq[tid].x;
    nType r = rmq[tid].y;
    float min = x[l];
    float val;
    for(nType i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
        }
    }
    out[tid] = min;
}

template <typename nType, typename n2Type>
__global__ void kernel_rmq_basic_idx(nType n, int q, float *x, n2Type *rmq, nType *out){
    nType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    nType l = rmq[tid].x;
    nType r = rmq[tid].y;
    float min = x[l];
    float val;
    nType min_idx = l;
    for(nType i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
            min_idx = i;
        }
    }
    out[tid] = min_idx;
}

template <typename nType, typename n2Type>
__global__ void kernel_rmq_basic(nType n, int q, float *x, n2Type *rmq, float *out, nType *indices){
    nType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= q){
        return;
    }
    // solve the tid-th RMQ query in the x array of size n
    nType l = rmq[tid].x;
    nType r = rmq[tid].y;
    float min = x[l];
    float val;
    nType idx_min = l;
    for(nType i=l; i<=r; ++i){
        val = x[i]; 
        if(val < min){
            min = val;
            idx_min = i;
        }
    }
    //printf("thread %i accessing out[%i] putting min %f\n", tid, tid, min);
    out[tid] = min;
    indices[tid] = idx_min;
}

// GPU RMQ basic approach
template <typename nType, typename n2Type>
float* gpu_rmq_basic(nType n, int q, float *devx, n2Type *devrmq, CmdArgs args){
    int reps = args.reps;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[ALG_GPU_BASE], reps); fflush(stdout);
    if (args.save_power)
        GPUPowerBegin(algStr[args.alg], 100, args.dev, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        kernel_rmq_basic<nType, n2Type><<<grid, block>>>(n, q, devx, devrmq, dout);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    if (args.save_power)
        GPUPowerEnd();
    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0, reps, args, {0, 0});
    return hout;
}

template <typename nType, typename n2Type>
nType* gpu_rmq_basic_idx(nType n, int q, float *devx, n2Type *devrmq, CmdArgs args){
    int reps = args.reps;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    nType *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (nType*)malloc(sizeof(nType)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(nType)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs index (%-16s,r=%-3i)..." AC_RESET, algStrIdx[ALG_GPU_BASE_IDX%10], reps); fflush(stdout);
    if (args.save_power)
        GPUPowerBegin(algStrIdx[args.alg%10], 100, args.dev, args.power_file);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        kernel_rmq_basic_idx<nType, n2Type><<<grid, block>>>(n, q, devx, devrmq, dout);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    if (args.save_power)
        GPUPowerEnd();
    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(nType)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0, reps, args, {0, 0});
    return hout;
}

template <typename nType, typename n2Type>
float* gpu_rmq_basic(nType n, int q, float *devx, n2Type *devrmq, CmdArgs args, nType* &indices){
    int reps = args.reps;
    dim3 block(BSIZE, 1, 1);
    dim3 grid((q+BSIZE-1)/BSIZE, 1, 1);
    float *hout, *dout;
    printf("Creating out array........................"); fflush(stdout);
    Timer timer;
    hout = (float*)malloc(sizeof(float)*q);
    indices = (nType*)malloc(sizeof(nType)*q);
    nType *d_indices;
    cudaMalloc(&d_indices, sizeof(nType)*q);
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)*q));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    printf(AC_BOLDCYAN "Computing RMQs (%-16s,r=%-3i)..." AC_RESET, algStr[ALG_GPU_BASE], reps); fflush(stdout);
    timer.restart();
    for (int i = 0; i < reps; ++i) {
        kernel_rmq_basic<nType, n2Type><<<grid, block>>>(n, q, devx, devrmq, dout, d_indices);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timer.stop();
    float timems = timer.get_elapsed_ms();
    float avg_time = timems/(1000.0*reps);
    printf(AC_BOLDCYAN "done: %f secs (avg %f secs): [%.2f RMQs/sec, %f nsec/RMQ]\n" AC_RESET, timems/1000.0, avg_time, (double)q/(timems/1000.0), (double)timems*1e6/q);
    printf("Copying result to host...................."); fflush(stdout);
    timer.restart();
    CUDA_CHECK(cudaMemcpy(hout, dout, sizeof(float)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices, d_indices, sizeof(nType)*q, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(d_indices));
    printf("done: %f secs\n", timer.get_elapsed_ms()/1000.0f);
    write_results(timems, q, 0, reps, args);
    return hout;
}

// functions to verify the correctness of the approaches
template<typename nType, typename n2Type>
float* get_or_generate_expected_val(nType n, uint32_t q, float* dA, n2Type* dQ, int lr, int nt, int seed, nType* &indices, CmdArgs args)
{   
    std::string filename = make_filename<nType>(directory_save_aux_data + "results_value", n, q, lr, seed, args.trivialCheck, args.randTrivialCheck);
    std::string filename_indices = make_filename<nType>(directory_save_aux_data + "results_index", n, q, lr, seed, args.trivialCheck, args.randTrivialCheck);
 
    if (file_exists(filename) && file_exists(filename_indices)) {
        indices = load_array<uint32_t, nType>(filename_indices, q);
        return load_array<uint32_t, float>(filename, q);
    }

    float* data = gpu_rmq_basic<nType, n2Type>(n, q, dA, dQ, args, indices);

    if (args.save_input_data) {
        save_array<uint32_t, float>(filename, data, q);
        save_array<uint32_t, nType>(filename_indices, indices, q);
    }
    return data;
}

template<typename nType, typename n2Type>
nType* get_or_generate_expected_index(nType n, uint32_t q, float* dA, n2Type* dQ, int lr, int nt, int seed, CmdArgs args)
{   
    std::string filename = make_filename<nType>(directory_save_aux_data + "results_index", n, q, lr, seed, args.trivialCheck, args.randTrivialCheck);

    if (file_exists(filename)) {
        return load_array<uint32_t, nType>(filename, q);
    }

    nType* data = gpu_rmq_basic_idx<nType, n2Type>(n, q, dA, dQ, args);

    if (args.save_input_data) {
        save_array<uint32_t, nType>(filename, data, q);
    }
    return data;
}