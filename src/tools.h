#pragma once
#include <unistd.h>
#include <string>
#include <time.h>
#include <getopt.h>

#define ARG_BS 1
#define ARG_NB 2
#define ARG_REPS 3
#define ARG_DEV 4
#define ARG_NT 5
#define ARG_SEED 6
#define ARG_CHECK 7
#define ARG_TIME 8
#define ARG_POWER 9
#define ARG_TRIVIALCHECK 10
#define ARG_THRESHOLD 11
#define ARG_SAVE_INPUTDATA 12
#define ARG_RAND_TRIVIALCHECK 13

struct VBHMem {
    size_t out_buffer;
    size_t temp_buffer;
};

struct CmdArgs {
#if IS_LONG == 1
    int64_t n, nb;
#else
    int32_t n, nb;
#endif
    uint32_t q, log_bs, log_build_threshold;
    int lr, alg, reps, dev, nt, seed, check, trivialCheck, save_input_data, randTrivialCheck, save_time, save_power;
    std::string time_file, power_file;
};

#define NUM_REQUIRED_ARGS 10
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxrmq <n> <q> <lr> <alg>\n\n" AC_RESET
                    "n   = num elements\n"
                    "q   = num RMQ querys\n"
                    "lr  = length of range; min 1, max n\n"
                    "  >0 -> value\n"
                    "  -1 -> uniform distribution (large values)\n"
                    "  -2 -> lognormal distribution (medium values)\n"
                    "  -3 -> lognormal distribution (small values)\n"
                    "alg = algorithm\n"
                        "0 -> [CPU] BASE\n"
                        "1 -> [CPU] HRMQ\n"
                        "2 -> [GPU] BASE\n"
                        "5 -> [GPU] RTX_blocks (RTXRMQ)\n"
                        "10 -> [GPU] RTX_ser\n"
                        "16 -> [GPU] XXX\n"
                        "17 -> [GPU] Interleaved\n"
                        "18 -> [GPU] Interleaved2\n"
                        "19 -> [GPU] Interleaved in CUDA\n"
                        "20 -> [GPU] BASIC VECTOR LOAD\n"
                        "21 -> [GPU] Interleaved_in_OptiX\n"
                        "23 -> [GPU] XXX without shuffles\n"
                        "24 -> [GPU] XXX multi load\n"
                        "100, 101, 102, 105 -> algs 0 1 2 5 returning indices\n"
                    "\n"
                    "Options:\n"
                    "   --log_bs <block size>         block size for RTX_blocks and scan threshold t for the XXX algorithms (16, 17, 18, 19, 20, 21, 23, 24), both given as logarithm with base 2 (default: 2^15)\n"
                    "   --nb <#blocks>            number of blocks for RTX_blocks (overrides --log_bs)\n"
                    "   --reps <repetitions>      RMQ repeats for the avg time (default: 10)\n"
                    "   --dev <device ID>         device ID (default: 0)\n"
                    "   --nt  <thread num>        number of CPU threads\n"
                    "   --seed <seed>             seed for PRNG\n"
                    "   --check                   check correctness\n"
                    "   --save-time=<file>        \n"
                    "   --save-power=<file>       \n",
                    "   --randTrivialCheck        alternative for --check where only 2^15 random queries are checked (to reduce overall run time)\n"
                    "   --trivialCheck            other alternative for --check where an extra test run with the same 2^10 queries repeated (to reduce overall run time)\n"
                    "   --save-input-data          if set arrays, queries and results for arrays > 2^24 are saved in a specified folder to reduce generation and result checking times\n",
                    algStr[0],
                    algStr[1],
                    algStr[2],
                    algStr[5],
                    algStr[10],
                    algStr[16],
                    algStr[17],
                    algStr[18],
                    algStr[19],
                    algStr[20],
                    algStr[21],
                    algStr[23],
                    algStr[24]
                );
}

CmdArgs get_args(int argc, char *argv[]) {
    if (argc < 5) {
        print_help();
        exit(EXIT_FAILURE);
    }

    CmdArgs args;
#if IS_LONG == 1
    args.n = atol(argv[1]);
#else 
    args.n = atoi(argv[1]);
#endif
    args.q = atoi(argv[2]);
    args.lr = atoi(argv[3]);
    args.alg = atoi(argv[4]);
    if (!args.n || !args.q || !args.lr) {
        print_help();
        exit(EXIT_FAILURE);
    }
    if (args.n < 1) {
        fprintf(stderr, "Error: n=%i is a non positive and therefore invalid array size!\n hint: check if a long was given to a version which can only handle int\n", args.n);
        exit(EXIT_FAILURE);
    }
    if (args.lr > args.n) {
        fprintf(stderr, "Error: lr=%i > n=%i  (lr must be between '1' and 'n')\n", args.lr, args.n);
        exit(EXIT_FAILURE);
    }

    //the default
    args.log_bs = 15;
    args.nb = args.n >> args.log_bs;
    args.reps = 10;
    args.seed = time(0);
    args.dev = 0;
    args.check = 0;
    args.trivialCheck = 0;
    args.save_input_data = 0;
    args.randTrivialCheck = 0;
    args.save_time = 0;
    args.save_power = 0;
    args.nt = 1;
    args.time_file = "";
    args.power_file = "";
    args.log_build_threshold = 0;
    
    static struct option long_option[] = {
        // {name , has_arg, flag, val}
        {"log_bs", required_argument, 0, ARG_BS},
        {"nb", required_argument, 0, ARG_NB},
        {"reps", required_argument, 0, ARG_REPS},
        {"dev", required_argument, 0, ARG_DEV},
        {"nt", required_argument, 0, ARG_NT},
        {"seed", required_argument, 0, ARG_SEED},
        {"check", no_argument, 0, ARG_CHECK},
        {"trivialCheck", no_argument, 0, ARG_TRIVIALCHECK},
        {"save-input-data", no_argument, 0, ARG_SAVE_INPUTDATA},
        {"randTrivialCheck", no_argument, 0, ARG_RAND_TRIVIALCHECK},
        {"save-time", optional_argument, 0, ARG_TIME},
        {"save-power", optional_argument, 0, ARG_POWER},
        {"log_build_threshold", optional_argument, 0, ARG_THRESHOLD},
        {0, 0, 0, 0}
    };
    int opt, opt_idx;
    while ((opt = getopt_long(argc, argv, "123456", long_option, &opt_idx)) != -1) {
        if (isdigit(opt))
                continue;
        switch (opt) {
            case ARG_BS:
                args.log_bs = (args.n < (1 << atoi(optarg))) ? (log2(std::__bit_floor(args.n))) : (atoi(optarg));
                args.nb = args.n >> args.log_bs;
                break;
            case ARG_THRESHOLD:
                if (optarg != NULL) {
                    args.log_build_threshold = atoi(optarg);
                }
                break;
            case ARG_NB:
            #if IS_LONG == 1
                args.nb = min(args.n, atol(optarg));
            #else
                args.nb = min(args.n, atoi(optarg));
            #endif
                args.log_bs = log2(std::__bit_ceil(args.n / args.nb));
                break;
            case ARG_REPS:
                args.reps = atoi(optarg);
                break;
            case ARG_DEV:
                args.dev = atoi(optarg);
                break;
            case ARG_NT: 
                args.nt = atoi(optarg);
                break;
            case ARG_SEED:
                args.seed = atoi(optarg);
                break;
            case ARG_CHECK:
                args.check = 1;
                args.trivialCheck = 0;
                args.randTrivialCheck = 0;
                break;
            case ARG_TRIVIALCHECK:
                args.trivialCheck = 1;
                args.check = 0;
                args.randTrivialCheck = 0;
                break;
            case ARG_RAND_TRIVIALCHECK:
                args.randTrivialCheck = 1;
                args.check = 0;
                args.trivialCheck = 0;
                break;
            case ARG_SAVE_INPUTDATA:
                args.save_input_data = 1;
                break;
            case ARG_TIME:
                args.save_time = 1;
                if (optarg != NULL)
                    args.time_file = optarg;
                break;
            case ARG_POWER:
                args.save_power = 1;
                if (optarg != NULL)
                    args.power_file = optarg;
                break;
            default:
                break;
        }
    }

    if (args.alg != ALG_GPU_RTX_BLOCKS &&
            args.alg != ALG_GPU_RTX_BLOCKS_IDX &&
            args.alg != ALG_GPU_RTX_SER &&
            args.alg != ALG_XXX &&
            args.alg != ALG_INTERLEAVED &&
            args.alg != ALG_INTERLEAVED2 &&
            args.alg != ALG_INTERLEAVED_IN_CUDA &&
            args.alg != ALG_HIERARCHICAL_VECTOR_LOAD &&
            args.alg != ALG_INTERLEAVED_IN_OPTIX &&
            args.alg != ALG_XXX_WITHOUT_SHUFFLES &&
            args.alg != ALG_XXX_MULTI_LOAD
        ) {
        args.log_bs = 0;
        args.nb = 0;
    }

    const char * algAsStr;
    if (args.alg < 100) {
        algAsStr = algStr[args.alg];
    } else {
        algAsStr = algStrIdx[args.alg%10];
    }
    
    printf( "Params:\n"
            "   reps = %i\n"
            "   seed = %i\n"
            "   dev  = %i\n"
            AC_GREEN "   n    = %ld (~%f GB, float)\n" AC_RESET
            "   bs   = %ld\n"
            "   nb   = %ld\n"
            AC_GREEN "   q    = %i (~%f GB, int2)\n" AC_RESET
            "   lr   = %i\n"
            "   nt   = %i CPU threads\n"
            "   alg  = %i (%s)\n\n",
            args.reps, args.seed, args.dev, args.n, sizeof(float)*args.n/1e9, (1 << args.log_bs), args.nb, args.q,
            sizeof(int2)*args.q/1e9, args.lr, args.nt, args.alg, algAsStr);

    return args;
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

template <typename nType, typename n2Type>
bool check_result(float *hA, n2Type *hQ, uint32_t q, float *expected, float *result, nType *indices){
    bool pass = true;
    uint32_t count = 0;
    for (uint32_t i = 0; i < q; ++i) {
        if (!is_equal(expected[i], result[i])) {
            printf("Error on %i-th query: got %.10f, expected %.10f at idx %ld\n", i, result[i], expected[i], indices[i]);
            printf("  [%ld,%ld]\n", hQ[i].x, hQ[i].y);
            printf("Value in array at position %ld: %f\n", indices[i], hA[indices[i]]);
            pass = false;
            count++;
        }
    }
    if (!pass)
        printf("%.2f%% wrong.\n", count / static_cast<float>(q) * 100);
    return pass;
}

template <typename nType, typename n2Type>
bool check_result_idx(float *hA, n2Type *hQ, uint32_t q, nType *expected, nType *result){
    bool pass = true;
    for (uint32_t i = 0; i < q; ++i) {
        if (expected[i] != result[i]) {
            printf("Error on %i-th query: got %ld, expected %ld\n", i, result[i], expected[i]);
            printf("  [%ld,%ld]\n", hQ[i].x, hQ[i].y);
            printf("  got min %.7f, expected %.7f\n", hA[result[i]], hA[expected[i]]);
            pass = false;
        }
    }
    return pass;
}

template <typename nType, typename n2Type>
bool check_result_rand_sample(float *hA, n2Type *hQ, uint32_t q, uint32_t sampled_q, uint32_t* sampled_indices, float *expected, float *result, nType *indices){
    bool pass = true;
    uint32_t count = 0;
    for (uint32_t i = 0; i < sampled_q; ++i) {
        if (!is_equal(expected[i], result[sampled_indices[i]])) {
            printf("Error on %i-th query: got %.10f, expected %.10f at idx %ld\n", sampled_indices[i], result[sampled_indices[i]], expected[i], indices[i]);
            printf("  [%ld,%ld]\n", hQ[sampled_indices[i]].x, hQ[sampled_indices[i]].y);
            printf("Value in array at position %ld: %f\n", indices[i], hA[indices[i]]);
            pass = false;
            count++;
        }
    }
    if (!pass)
        printf("%.2f%% wrong.\n", count / static_cast<float>(sampled_q) * 100);
    return pass;
}

template <typename nType, typename n2Type>
bool check_result_idx_rand_sample(float *hA, n2Type *hQ, uint32_t q, uint32_t sampled_q, uint32_t* sampled_indices, nType *expected, nType *result){
    bool pass = true;
    for (uint32_t i = 0; i < sampled_q; ++i) {
        if (expected[i] != result[sampled_indices[i]]) {
            printf("Error on %i-th query: got %ld, expected %ld\n", sampled_indices[i], result[sampled_indices[i]], expected[i]);
            printf("  [%ld,%ld]\n", hQ[sampled_indices[i]].x, hQ[sampled_indices[i]].y);
            printf("  got min %.7f, expected %.7f\n", hA[result[sampled_indices[i]]], hA[expected[i]]);
            pass = false;
        }
    }
    return pass;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Memory:                       %f GB\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %s\n", prop.concurrentKernels == 1? "yes" : "no");
    printf("  Memory Clock Rate:            %d MHz\n", prop.memoryClockRate);
    printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth:        %f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

template <typename nType>
void write_results(int dev, int alg, nType n, nType log_bs, nType nb, uint32_t q, int lr, int reps, int nt, CmdArgs args) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    char *device = prop.name;
    int GPUBlockSize = BSIZE;
    if (alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ || alg == ALG_CPU_BASE_IDX || alg == ALG_CPU_HRMQ_IDX) {
        strcpy(device, "CPU ");
        char hostname[50];
        gethostname(hostname, 50);
        strcat(device, hostname);
        GPUBlockSize = 0;
    }
    
    int CGTileSize = 0;
    int memoryAlignmentFactor = 0;
    
    const char * algAsStr;
    if (args.alg < 100) {
        algAsStr = algStr[args.alg];
    } else {
        algAsStr = algStrIdx[args.alg%10];
    }

    // get timestamp to save in csv
    std::time_t timestamp = std::time(nullptr);
    std::tm * time_ptr = std::localtime(&timestamp);

    char time_str[32];
    std::strftime(time_str, 32, "%Y-%m-%dT%H:%M:%SZ", time_ptr); 

    char lr_string[50];
    snprintf(lr_string, sizeof(lr_string), "%d", lr);


    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    if (alg == ALG_XXX || alg == ALG_INTERLEAVED || alg == ALG_INTERLEAVED2 || alg == ALG_INTERLEAVED_IN_CUDA || alg == ALG_HIERARCHICAL_VECTOR_LOAD || alg == ALG_INTERLEAVED_IN_OPTIX || alg == ALG_XXX_WITHOUT_SHUFFLES || alg == ALG_XXX_MULTI_LOAD) {
        fprintf(fp, "%s,%s,%i,%i,%ld,%i,%ld,%i,%s,%i,%i,%i,%s,%i,%i,%i,",
            device,
            algAsStr,
            nt,
            reps,
            n,
            1,
            0,
            q,
            lr_string,
            GPUBlockSize,
            CGTileSize,
            memoryAlignmentFactor,
            time_str,
            (1 << args.log_bs),
            (alg != ALG_HIERARCHICAL_VECTOR_LOAD) ? XXX_CG_SIZE_LOG : 0,
            XXX_CG_AMOUNT_LOG
        );
    } else {
        fprintf(fp, "%s,%s,%i,%i,%ld,%ld,%ld,%i,%s,%i,%i,%i,%s,",
            device,
            algAsStr,
            nt,
            reps,
            n,
            (1 << args.log_bs),
            nb,
            q,
            lr_string,
            GPUBlockSize,
            CGTileSize,
            memoryAlignmentFactor,
            time_str);
    }
    fclose(fp);
}

void write_results(float time_ms, uint32_t q, float construction_time, int reps, CmdArgs args) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    // get free GPU memory
    size_t free_memory;
    size_t total_memory;
    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    int cudaMemInfoCorrect = 1;
    
    if (cuda_status != cudaSuccess) {
        cudaMemInfoCorrect = 0;
    }

    float time_it = time_ms/reps;
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, "%f,%f,%f,%f,0,0,%f",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q,
            construction_time,
            cudaMemInfoCorrect,
            free_memory/1048576.0);
    fclose(fp);
}
void write_results(float time_ms, uint32_t q, float construction_time, int reps, CmdArgs args, VBHMem mem) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    // get free GPU memory
    size_t free_memory;
    size_t total_memory;
    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    int cudaMemInfoCorrect = 1;

    if (cuda_status != cudaSuccess) {
        cudaMemInfoCorrect = 0;
    }

    float time_it = time_ms/reps;
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%i,%f",
            time_ms/1000.0,
            (double)q/(time_it/1000.0),
            (double)time_it*1e6/q,
            construction_time,
            mem.out_buffer/1e6,
            mem.temp_buffer/1e6,
            cudaMemInfoCorrect,
            free_memory/1048576.0);
    fclose(fp);
}

void write_check_result(int check_result, CmdArgs args) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    if (args.trivialCheck) {
        check_result += 2;
    } else if (args.randTrivialCheck) {
        check_result += 4;
    }
    
    FILE *fp;
    fp = fopen(filename.c_str(), "a");
    fprintf(fp, ",%i\n", check_result);
    fclose(fp);
}

void create_header(std::string filename) {
    if (fopen(filename.c_str(), "r") == nullptr) {
        FILE *fp;
        fp = fopen(filename.c_str(), "a");
        fprintf(fp, "dev,alg,nt,reps,n,bs,nb,q,lr,GPU_BSIZE,CG_GROUP_SIZE,CG_MEM_ALIGNMENT,timestamp,t,q/s,ns/q,construction,outbuffer,tempbuffer,freeGPUMemCorrect,freeGPUMem,checkResult\n");
        fflush(fp);
    }
}

void create_XXX_header(std::string filename) {
    if (fopen(filename.c_str(), "r") == nullptr) {
        FILE *fp;
        fp = fopen(filename.c_str(), "a");
        fprintf(fp, "dev,alg,nt,reps,n,bs,nb,q,lr,GPU_BSIZE,CG_GROUP_SIZE,CG_MEM_ALIGNMENT,timestamp,scan_threshold,XXX_CG_SIZE_LOG,XXX_CG_AMOUNT_LOG,t,q/s,ns/q,construction,outbuffer,tempbuffer,freeGPUMemCorrect,freeGPUMem,checkResult\n");
        fflush(fp);
    }
}

template<typename nType, typename arrayType>
void save_array(const std::string& filename, const arrayType* data, nType n)
{   
    static_assert(std::is_trivially_copyable<arrayType>::value, "arrayType must be trivially copyable");
    
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open file for writing");

    std::cout << "\n" << "Writing data in file " << filename << std::endl;
    out.write(reinterpret_cast<const char*>(data), n * sizeof(arrayType));
}

template<typename nType, typename arrayType>
arrayType* load_array(const std::string& filename, nType n)
{
    static_assert(std::is_trivially_copyable<arrayType>::value, "arrayType must be trivially copyable");

    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file for reading");

    arrayType* data = new arrayType[n];

    std::cout << "\n" << "Reading data in file " << filename << std::endl;
    in.read(reinterpret_cast<char*>(data), n * sizeof(arrayType));

    return data;
}

inline bool file_exists(const std::string& filename)
{
    std::ifstream f(filename.c_str(), std::ios::binary);
    return f.good();
}

template<typename nType>
std::string make_filename(const std::string &prefix, nType n, uint32_t q, int lr, int seed, bool trivialCheck, bool randTrivialCheck) {
    std::stringstream s;
    char lr_string[50];
    snprintf(lr_string, sizeof(lr_string), "%d", lr);
    s << prefix << "_n-" << n << "_q-" << q << "_lr-" << lr_string << "_seed-" << seed << "_trivCheck-" << trivialCheck << "_randTrivCheck-" << randTrivialCheck << ".bin";
    return s.str();
}

template<typename nType>
std::string make_filename_sampled(const std::string &prefix, nType n, uint32_t ori_q, uint32_t q, int lr, int seed, bool trivialCheck, bool randTrivialCheck) {
    std::stringstream s;
    char lr_string[50];
    snprintf(lr_string, sizeof(lr_string), "%d", lr);
    s << prefix << "_n-" << n << "_ori_q-" << ori_q << "_q-" << q << "_lr-" << lr_string << "_seed-" << seed << "_trivCheck-" << trivialCheck << "_randTrivCheck-" << randTrivialCheck << ".bin";
    return s.str();
}

template<typename nType>
std::string make_array_filename(const std::string &prefix, nType n, int seed) {
    std::stringstream s;
    s << prefix << "_n-" << n << "_seed-" << seed << ".bin";
    return s.str();
}

template<typename nType, typename n2Type>
n2Type* get_or_generate_queries(uint32_t q, int lr, nType n, int nt, int seed, CmdArgs args) {
    std::string filename = make_filename<nType>(directory_save_aux_data + "queries", n, q, lr, seed, args.trivialCheck, args.randTrivialCheck);
    
    if (file_exists(filename)) {
        return load_array<uint32_t, n2Type>(filename, q);
    }

    n2Type* data = random_queries_par_cpu<nType, n2Type>(q, lr, n, nt, seed, args.trivialCheck);

    if (args.save_input_data) {
        save_array<uint32_t, n2Type>(filename, data, q);
    }
    return data;
}

template<typename nType>
float* get_or_generate_array(uint32_t q, int lr, nType n, int nt, int seed, CmdArgs args) {
    std::string filename = make_array_filename<nType>(directory_save_aux_data + "array", n, seed);
    
    if (file_exists(filename)) {
        return load_array<nType, float>(filename, n);
    }

    float* data = random_array_par_cpu<nType>(n, seed);

    if (args.save_input_data) {
        save_array<nType, float>(filename, data, n);
    }
    return data;
}


