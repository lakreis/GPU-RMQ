#pragma once
#define PRINT_LIMIT 32

// functions to guarantuee an intact csv output file even in case of errors
#define COMPLETE_TIME_FILE_CUDA_ERROR(filename) complete_time_file(__FILE__, __LINE__, filename)
void complete_time_file(const char* const file, const int line, std::string filename)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        FILE *fp;
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        fp = fopen(filename.c_str(), "a");
        fprintf(fp, "\n");
        fclose(fp);
        exit(1);
    }
}

void complete_line(std::string filename)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, "\n");
    fclose(fp);
}

// print functions to debug the values
__global__ void kernel_print_array_dev(uint32_t n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %f\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint64_t n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%ld] = %.12f\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, int *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %i\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, long *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %ld\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, float4 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%u] = (%f, %f, %f, %f)\n", tid, i, darray[i].x, darray[i].y, darray[i].z, darray[i].w);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint64_t n, long *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%ld] = %ld\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, int2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %i %i\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, uint2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %u %u\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, long2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %ld %ld\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint64_t n, long2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%ld] = %ld %ld\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint64_t n, ulong2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%ld] = %ld %ld\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, bool *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %s\n", tid, i, darray[i] ? "true" : "false");
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, uint32_t *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %i\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint32_t n, float2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %f %f\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_dev(uint64_t n, float2 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%ld] = %f %f\n", tid, i, darray[i].x, darray[i].y);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_array_last_dev(uint32_t n, float4 *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=n-PRINT_LIMIT; i<n; ++i){
        printf("tid %i --> array[%i] = %f %f %f %f\n", tid, i, darray[i].x, darray[i].y, darray[i].z, darray[i].w);
    }
}

__global__ void kernel_print_array_last_dev(uint32_t n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=n-PRINT_LIMIT; i<n; ++i){
        printf("tid %i --> array[%i] = %f\n", tid, i, darray[i]);
    }
}

template <typename nType>
__global__ void kernel_print_vertices_dev(nType ntris, float3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> vertex[%ld] = (%f, %f, %f)\n", tid, 3*i+0, v[3*i+0].x, v[3*i+0].y, v[3*i+0].z);
        printf("tid %i --> vertex[%ld] = (%f, %f, %f)\n", tid, 3*i+1, v[3*i+1].x, v[3*i+1].y, v[3*i+1].z);
        printf("tid %i --> vertex[%ld] = (%f, %f, %f)\n", tid, 3*i+2, v[3*i+2].x, v[3*i+2].y, v[3*i+2].z);
        printf("\n");
    }
    if(i < ntris){
        printf("...\n");
    }
}

__global__ void kernel_print_triangles_dev(int ntris, uint3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> triangle[%i] = (%i, %i, %i)\n", tid, i, v[i].x, v[i].y, v[i].z);
    }
    if(i < ntris){
        printf("...\n");
    }
}

__global__ void kernel_print_triangles_dev(long ntris, uint3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> triangle[%ld] = (%i, %i, %i)\n", tid, i, v[i].x, v[i].y, v[i].z);
    }
    if(i < ntris){
        printf("...\n");
    }
}

template <typename nType, typename arrayType>
void print_array_dev(nType n, arrayType *darray){
    printf("Printing random array:\n");
    kernel_print_array_dev<<<1,1>>>(n, darray);
    cudaDeviceSynchronize();
}

template <typename nType, typename arrayType>
void print_array_last_dev(nType n, arrayType *darray){
    printf("Printing random array:\n");
    kernel_print_array_last_dev<<<1,1>>>(n, darray);
    cudaDeviceSynchronize();
}

template <typename nType>
void print_vertices_dev(nType ntris, float3 *devVertices){
    printf("Printing vertices:\n");
    kernel_print_vertices_dev<nType><<<1,1>>>(ntris, devVertices);
    cudaDeviceSynchronize();
}

template <typename nType>
void print_triangles_dev(nType ntris, uint3 *devTriangles){
    printf("Printing vertices:\n");
    kernel_print_triangles_dev<<<1,1>>>(ntris, devTriangles);
    cudaDeviceSynchronize();
}

template <typename nType>
__global__ void kernel_gen_vertices_blocks(nType num_blocks, nType N, nType bs, float *min_blocks, float *array, float3 *vertices){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    nType n_blocks = ceil(sqrt((double)num_blocks+1));
    nType k = 3*idx;
    if(idx < num_blocks){
        float val = min_blocks[idx];
        // ray hits min on coord (val, l, r)
        float l = (float)(idx+1)/(1<<23);
        float r = (float)(idx-1)/(1<<23);
        float n = 1;

        vertices[k+0] = make_float3(val, l, r);
        vertices[k+1] = make_float3(val, l, 2*n);
        vertices[k+2] = make_float3(val, -1*n, r);
    } else if (idx < N) {
        int64_t sub_idx = idx - num_blocks;
        int64_t bid = sub_idx / bs;
        int64_t lid = sub_idx % bs;
        float val = array[sub_idx];

        int64_t x = (bid + 1) % n_blocks;
        int64_t y = (bid + 1) / n_blocks;
        float l = (float)(lid+1)/bs + 2*x;
        float r = (float)(lid-1)/bs + 2*y;

        vertices[k+0] = make_float3(val, l, r);
        vertices[k+1] = make_float3(val, l, 2*y+2);
        vertices[k+2] = make_float3(val, 2*x-1, r);
        //printf("%i-th element %f  at  %f,  %f\n", sub_idx, val, l, r);
    }
}

template <typename nType>
__global__ void kernel_gen_triangles(nType ntris, float *array, uint3 *triangles){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        triangles[tid] = make_uint3(k, k+1, k+2);
    }
}

template <typename nType>
__global__ void kernel_min_blocks(float *min_blocks, float *darray, nType num_blocks, nType N, nType bs) {
    nType tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;
    nType first = tid * bs;
    float min = darray[first];
    for (nType i = 1; i < bs && i < N - first; ++i) {
        if (darray[i+first] < min)
            min = darray[i+first];
    }
    min_blocks[tid] = min;
}

template <typename nType> 
__global__ void print_darray(float* A, nType n) {
    if (threadIdx.x != 0) return;
    for (nType i = 0; i < n; ++i) {
        printf("%f ", A[i]);
        if (i % 10 == 9)
            printf("\n[%2u] ", i+1);
    }
    printf("\n");
}

template <typename nType>
__global__ void kernel_initialise_array(nType n, float *darray, float val){
    // Assume that 1D grid
    nType tid = threadIdx.x + blockIdx.x * blockDim.x;
    nType num_threads = blockDim.x;

    if (tid >= n) {
        return;
    }

    for(nType i=tid; i<n; i += num_threads){
        darray[i] = val;
    }
}

template <typename nType>
float3* gen_vertices_blocks_dev(nType N, nType bs, float *darray){
    // create array with mins of each block
    nType num_blocks = (N+bs-1) / bs;
    nType ntris = N + num_blocks;

    float *min_blocks;
    cudaMalloc(&min_blocks, sizeof(float)*num_blocks);
    dim3 block(BSIZE, 1, 1);
    dim3 grid_mins((num_blocks+BSIZE-1)/BSIZE,1,1);
    kernel_min_blocks<nType><<<grid_mins, block>>>(min_blocks, darray, num_blocks, N, bs);
    CUDA_CHECK( cudaDeviceSynchronize() );
    // print_darray<nType><<<1,1>>>(min_blocks, num_blocks);

    // vertices data
    float3 *devVertices;
    cudaMalloc(&devVertices, sizeof(float3)*3*ntris);

    // setup states
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices_blocks<nType><<<grid, block>>>(num_blocks, ntris, bs, min_blocks, darray, devVertices);
    CUDA_CHECK( cudaDeviceSynchronize() );
    return devVertices;
}

template <typename nType>
float3* gen_vertices_interleaved_dev(nType N, float *darray){
    // vertices data
    float3 *devVertices;
    cudaMalloc(&devVertices, sizeof(float3)*3*N);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((N+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices_blocks<nType><<<grid, block>>>(N, 0, 0, darray, nullptr, devVertices);
    CUDA_CHECK( cudaDeviceSynchronize() );
    return devVertices;
}

template <typename nType>
uint3* gen_triangles_dev(nType ntris, float *darray){
    // data array
    uint3 *devTriangles;
    cudaMalloc(&devTriangles, sizeof(uint3)*ntris);

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_triangles<nType><<<grid, block>>>(ntris, darray, devTriangles);
    cudaDeviceSynchronize();
    return devTriangles;
}

template <typename nType>
void cpuprint_array(nType np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<np; ++i){
        printf("array [%i] = %f\n", i, hp[i]);
    }
}

template <typename nType, typename n2Type>
void cpuprint_array(nType np, n2Type *dp){
    int2 *hp = new int2[np];
    cudaMemcpy(hp, dp, sizeof(int2)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<np; ++i){
        printf("array[%i] = %i %i\n", i, hp[i].x, hp[i].y);
    }
}

