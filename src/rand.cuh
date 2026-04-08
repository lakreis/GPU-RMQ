#pragma once
#include <curand_kernel.h>
#include <random>
#include <cmath>
#include <omp.h>

// create array par GPU (slow for large array sizes) --------------------------------------
// note: de-parallelization enables to generate array sizes > 2**28, too many CurandStates at once can be too memory intensive
#define GEN_NUM_BLOCKS 4

template <typename nType>
__global__ void kernel_random_array(nType n, int seed, float *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    for (nType i = id; i < n; i+=GEN_NUM_BLOCKS*BSIZE) {
        curandState_t state;
        curand_init(seed, id, 0, &state);
        float x = curand_uniform(&state);
        array[id] = x;
    }
}

template <typename nType>
float* create_random_array_dev(nType n, int seed){
    // data array
    float* darray;
    cudaMalloc(&darray, sizeof(float)*n);

    // gen random numbers
    dim3 block(BSIZE, 1, 1);
    dim3 grid(GEN_NUM_BLOCKS, 1, 1); 
    kernel_random_array<<<grid,block>>>(n, seed, darray);
    cudaDeviceSynchronize();
    return darray;
}

template <typename nType>
__global__ void kernel_random_array(nType n, int max, int lr, curandState *state, int2 *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    int y = lr > 0 ? lr : curand_uniform(&state[id]) * (max-1);
    int x = curand_uniform_double(&state[id]) * (max-y-1);
    array[id].x = x;
    array[id].y = x+y;
}

// create array par CPU ---------------------------------------------------------------------

template <typename nType>
float* random_array_par_cpu(nType n, int seed) {
    float *query = new float[n];

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0, 1);

    for (nType i = 0; i < n; ++i) {
        query[i] = dist(gen); 
    }
    
    return query;
}

// create queries par CPU depending on lr range size --------------------------------------

template <typename nType, typename n2Type>
void fill_queries_constant(n2Type *query, uint32_t q, int lr, nType n, int nt, int seed){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        uint32_t chunk = (q+nt-1)/nt;
        uint32_t begin = chunk*tid;
        uint32_t end   = begin + chunk;
        nType qsize = lr;
        for(uint32_t i=begin; i<q && i<end; ++i){

            std::uniform_int_distribution<nType> lrand(0, n-1 - (qsize-1));
            nType l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
            //printf("thread %i (l,r) -> (%i, %i)\n\n", tid, query[i].x, query[i].y);
        }
    }
}

template <typename nType, typename n2Type>
void fill_queries_uniform(n2Type *query, uint32_t q, int lr, nType n, int nt, int seed){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        std::uniform_int_distribution<nType> dist(1, n);
        uint32_t chunk = (q+nt-1)/nt;
        uint32_t begin = chunk*tid;
        uint32_t end   = begin + chunk;
        for(uint32_t i = begin; i<q && i<end; ++i){
            nType qsize = dist(gen);
            std::uniform_int_distribution<nType> lrand(0, n-1 - (qsize-1));
            nType l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
            //printf("(l,r) -> (%i, %i)\n\n", query[i].x, query[i].y);
        }
    }
}

template <typename nType, typename n2Type>
void fill_queries_lognormal(n2Type *query, uint32_t q, int lr, nType n, int nt, int seed, int scale){
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(seed*tid);
        std::lognormal_distribution<double> dist(log(scale), 0.3);
        uint32_t chunk = (q+nt-1)/nt;
        uint32_t begin = chunk*tid;
        uint32_t end   = begin + chunk;
        for(uint32_t i = begin; i<q && i<end; ++i){
            nType qsize;
            do{ qsize = (nType)dist(gen); }
            while (qsize <= 0 || qsize > n);
            std::uniform_int_distribution<nType> lrand(0, n-1 - (qsize-1));
            nType l = lrand(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
        }
    }
}

template <typename nType, typename n2Type>
void fill_queries_mixed(n2Type *query, uint32_t q, int lr, nType n, int nt, int seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 2);

    std::uniform_int_distribution<nType> dist1(1, n);
    std::lognormal_distribution<double> dist2(log((nType)pow((double)n,0.6)), 0.3);
    std::lognormal_distribution<double> dist3(log((nType)pow((double)n,0.3)), 0.3);

    nType qsize;
    
    for (uint32_t i = 0; i < q; ++i)
    {
        int r = dist(gen);
        // returns -1, -2 or -3
        int curr_lr = static_cast<int>(-(r + 1)); 

        if(curr_lr == -1){
            qsize = dist1(gen);
            std::uniform_int_distribution<nType> lrand1(0, n-1 - (qsize-1));
            nType l = lrand1(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
        }
        else if(curr_lr == -2){
            do{ qsize = (nType)dist2(gen); }
            while (qsize <= 0 || qsize > n);
            std::uniform_int_distribution<nType> lrand2(0, n-1 - (qsize-1));
            nType l = lrand2(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
        }
        else if(curr_lr == -3){
            do{ qsize = (nType)dist3(gen); }
            while (qsize <= 0 || qsize > n);
            std::uniform_int_distribution<nType> lrand3(0, n-1 - (qsize-1));
            nType l = lrand3(gen);
            query[i].x = l;
            query[i].y = l + (qsize - 1);
        }
    }
}

template <typename arrayType, typename nType>
void fill_repeatedly(arrayType *array_filled, arrayType *res_array, nType n) {
    nType index = 0;
    for (nType i = 0; i < n; i++) {
        index = i & ((1 << TRIVCHECK_LOG_NUM_DIFFQ) - 1);
        res_array[i] = array_filled[index]; 
    }
}

template <typename nType, typename n2Type>
n2Type* random_queries_par_cpu(uint32_t q, int lr, nType n, int nt, int seed, bool trivialCheck) {
    // not in parallel for simplicity reasons (error for nt > 1 at least for a small q)
    nt = 1;
    omp_set_num_threads(nt);

    uint32_t ori_q = q;
    // fill the query array with 2**(TRIVCHECK_LOG_NUM_DIFFQ) different queries and repeat them later
    if (trivialCheck && (ori_q > (1 << TRIVCHECK_LOG_NUM_DIFFQ))) {
        q = (1 << TRIVCHECK_LOG_NUM_DIFFQ);
    }

    n2Type *query = new n2Type[q];
    if(lr>0){
        fill_queries_constant<nType, n2Type>(query, q, lr, n, nt, seed);
    }
    else if(lr == -1){
        fill_queries_uniform<nType, n2Type>(query, q, lr, n, nt, seed);
    } else if(lr == -2){
        fill_queries_lognormal<nType, n2Type>(query, q, lr, n, nt, seed, (nType)pow((double)n,0.6));
    } else if(lr == -3){
        fill_queries_lognormal<nType, n2Type>(query, q, lr, n, nt, seed, (nType)pow((double)n,0.3));
    } else if(lr == -4){
        fill_queries_lognormal<nType, n2Type>(query, q, lr, n, nt, seed, (nType)max((nType)1,n/(1<<8)));
    } else if(lr == -5){
        fill_queries_lognormal<nType, n2Type>(query, q, lr, n, nt, seed, (nType)max((nType)1,n/(1<<15)));
    } else if (lr == -6) {
        fill_queries_mixed<nType, n2Type>(query, q, lr, n, nt, seed);
    } 

    if (trivialCheck && (ori_q > (1 << TRIVCHECK_LOG_NUM_DIFFQ))) {
        n2Type *res_query = new n2Type[ori_q];
        fill_repeatedly<n2Type, nType>(query, res_query, ori_q);
        return res_query;
    }
    return query;
}

template <typename nType>
nType* randomly_sample_indices(nType original_size, nType sample_size, int seed) {
    if (original_size <= sample_size) {
        sample_size = original_size;
    }

    nType* indices = (nType*) malloc(sample_size * sizeof(nType));
    if (sample_size == original_size) {
        for (nType i = 0; i < sample_size; ++i) {
            indices[i] = i;
        }
        return indices;
    }

    std::mt19937 gen(seed);
    
    for (nType i = 0; i < sample_size; ++i) {
        std::uniform_int_distribution<nType> dis(0, original_size - 1);
        indices[i] = dis(gen);
    }
    return indices;
}
