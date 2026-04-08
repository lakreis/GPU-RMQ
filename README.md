# GPU-RMQ
This repository provides an RMQ implementation based on NVIDIA OptiX and RT cores. It accompanies our work, *GPU-RMQ: Accelerating Range Minimum Queries on Modern GPUs*, and builds upon [rtxrmq](https://github.com/temporal-hpc/rtxrmq) by Meneses et al., from *Accelerating Range Minimum Queries with Ray Tracing Cores*.

DOI: TODO 

The algorithms mentioned in the paper correspond to:
- HRMQ (CPU): 1
- Full GPU Scan: 2
- RTXRMQ: 5
- GPU-RMQ (CL): 16 (XXX)
- GPU-RMQ (VL): 20 (Hierarchical_vector_load)
- GPU-RMQ (CL) w/o warp intrinsics: 19 (Interleaved_in_CUDA)
- GPU-RMQ (CL) w/o warp intrinsics in OptiX w/o RT: 21 (Interleaved_in_OptiX)
- GPU-RMQ (CL) w/o warp intrinsics in OptiX w/ RT: 18 (Interleaved2)
- GPU-RMW (CL) multi load: 24 (XXX_multiload)

We adapted RTXRMQ from [rtxrmq](https://github.com/temporal-hpc/rtxrmq) (MIT License) to OptiX 8. For HRMQ, the implementation [hrmq](https://github.com/hferrada/rmq) (GNU GPL) must be included in this repository. Therefore, create a directory `hrmq`, include the files from the mentioned repository and build the library (see README in the directory).
For the algorithm LCA we additionally used the implementation [euler-meets-cuda-rmq](https://github.com/temporal-hpc/euler-meets-cuda-rmq) (no license), which itself is based on [euler-meets-cuda](https://github.com/stobis/euler-meets-cuda) (no license), and evaluated it separately.
  
## Dependencies
- CUDA 12.9 or later
- NVIDIA OptiX 8 (compatibility with OptiX 9 has not been tested)

## Compile and run
```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=<PATH-TO-OPTIX-MAIN-DIR>
make
./rtxrmq <n> <q> <lr> <alg>

n   = num elements
q   = num RMQ querys
lr  = length of range; min 1, max n
  >0 -> value
  -1 -> uniform distribution (large values)
  -2 -> lognormal distribution (medium values)
  -3 -> lognormal distribution (small values)
  -6 -> mixed distribution (large, medium and small values)
alg = algorithm
   0 -> [CPU] BASE
   1 -> [CPU] HRMQ
   2 -> [GPU] BASE
   5 -> [GPU] RTX_blocks (RTXRMQ)
   10 -> [GPU] RTX_ser
   16 -> [GPU] XXX
   17 -> [GPU] Interleaved
   18 -> [GPU] Interleaved2
   19 -> [GPU] Interleaved in CUDA
   20 -> [GPU] BASIC VECTOR LOAD
   21 -> [GPU] Interleaved_in_OptiX
   23 -> [GPU] XXX without shuffles
   24 -> [GPU] XXX multi load

   100, 101, 102, 105 -> algs 0 1 2 5 returning indices

Options:
   --log_bs <block size>     block size for RTX_blocks and scan threshold t for the XXX algorithms (16, 17, 18, 19, 20, 21, 23, 24), both given as logarithm with base 2 (default: 2^15)
   --nb <#blocks>            number of blocks for RTX_blocks (overrides --log_bs)
   --reps <repetitions>      RMQ repeats for the avg time (default: 10)
   --dev <device ID>         device ID (default: 0)
   --nt  <thread num>        number of CPU threads
   --seed <seed>             seed for PRNG
   --check                   check correctness
   --save-time=<file>
   --save-power=<file>
   --randTrivialCheck         alternative for --check where only 2^15 random queries are checked (to reduce overall run time)
   --trivialCheck             other alternative for --check where an extra test run with the same 2^10 queries repeated (to reduce overall run time)  
   --save-input-data          if set arrays, queries and results for arrays > 2^24 are saved in a specified folder to reduce generation and result checking times 
```

Some run scripts are available in the `scripts/` directory.

## References
[1] Lara Kreis, Justus Henneberg, Valentin Henkys, Felix Schuhknecht, Bertil Schmidt, GPU-RMQ: Accelerating Range Minimum Queries on Modern GPUs (2026)

[2] E. Meneses, C. Navarro, H. Ferrada, F. Quezada, Accelerating range minimum queries with ray tracing cores, Future Generation Computer Systems 157 (2024) 98-111

[3] H. Ferrada, G. Navarro, Improved range minimum queries, J. Discrete Algorithms 43 (2017) 72–80

[4] A. Polak, A. Siwiec, M. Stobierski, Euler meets GPU: Practical graph algorithms with theoretical guarantees, in: 2021 IEEE International Parallel and Distributed Processing Symposium, IPDPS, IEEE, 2021, pp. 233–244
