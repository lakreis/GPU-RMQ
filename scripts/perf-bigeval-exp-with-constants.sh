#!/bin/bash

cd ..

recompile=false
# TODO@newUser: write your local OptiX path here
OptiX_path=/your/OptiX/path/here

if [ "$recompile" = true ]; then
    rm -r build/
    cmake -S . -B build -DOPTIX_HOME=$OptiX_path
fi

cmake --build build
cd scripts/

# for sizes larger than 2^24
large_sizes_save_input_data=0

dev=${1}
nt=${2}
name=${3}

ALGS=($((16)) $((20)))

N1=$((15))
N2=$((31))

Q1=$((26))
Q2=$((26))

REA=$((4))
REPS=$((1))

SEED_RANDOM=$((0))

# threshold for the XXX algorithms
T1=$((5))
T2=$((7))
T_STEP_SIZE=$((1))
SMALLEST_T=$((0))

# bs for the RTXRMQ algorithms
B1=$((6))
B2=$((18))

# 1 to check normally, 2 to check trivially (not a normal run)
# 3 to check randomly trivally, 0 = no check at all
check=3

XXX_CG_SIZE_LOGS=(4)
XXX_CG_AMOUNT_LOGS=(1)
GPU_BSIZES=(1024)

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"
for GPU_BSIZE in "${GPU_BSIZES[@]}"; do
    for i in "${!XXX_CG_SIZE_LOGS[@]}"; do
        XXX_CG_SIZE_LOG="${XXX_CG_SIZE_LOGS[$i]}"
        for XXX_CG_AMOUNT_LOG in "${XXX_CG_AMOUNT_LOGS[@]}"; do
            for IS_LONG in 0 1; do
                cd ..
                echo "\nConfiguring..."
                cmake -S . -B build \
                        -DOPTIX_HOME=OptiX_path \
                        -DIS_LONG=$IS_LONG \
                        -DXXX_CG_SIZE_LOG=$XXX_CG_SIZE_LOG \
                        -DXXX_CG_AMOUNT_LOG=$XXX_CG_AMOUNT_LOG \
                        -DBSIZE=$GPU_BSIZE

                echo "Building..."
                cmake --build build
                cd scripts/
                for(( n=$N1; n<=$N2; n++ )); do
                    # Skip invalid combinations
                    if [ "$IS_LONG" -eq 0 ] && [ "$n" -gt 30 ]; then
                        continue
                    fi
                    if [ "$IS_LONG" -eq 1 ] && [ "$n" -le 30 ]; then
                        continue
                    fi
                    if [ "$n" -lt 25 ]; then
                        save_input_data=0
                    else
                        save_input_data=$large_sizes_save_input_data
                    fi
                    for(( q=$Q1; q<=$Q2; q++ ))
                    do  
                        for lr in {-1,-2,-3,-6}
                        do  
                            for alg in "${ALGS[@]}"
                            do  
                                # the CPU algorithms
                                if [ "$alg" -lt 2 ]; then
                                    old_name="$name"
                                    name="3990X"
                                    ./perf-benchmark-exp.sh $dev $nt $alg $REA $REPS $n $q "" 0 $lr $name $check $SEED_RANDOM $save_input_data
                                    name="$old_name"
                                # the exhaustive GPU algorithm
                                elif [ "$alg" -eq 2 ]; then
                                    ./perf-benchmark-exp.sh $dev $nt $alg $REA $REPS $n $q "" 0 $lr $name $check $SEED_RANDOM $save_input_data
                                # the XXX algorithm with scan threshold
                                elif [ "$alg" -eq 16 ] || [ "$alg" -eq 17 ] || [ "$alg" -eq 18 ] || [ "$alg" -eq 19 ] || [ "$alg" -eq 20 ] || [ "$alg" -eq 21 ] || [ "$alg" -eq 23 ] || [ "$alg" -eq 24 ]; then
                                    if [[ $alg -eq 20 && $i -ne 0 ]]; then
                                        continue
                                    fi
                                    if [[ $SMALLEST_T -eq 1 && $alg -ne 20 ]]; then
                                        T1=$XXX_CG_SIZE_LOG+$XXX_CG_AMOUNT_LOG+1
                                        T2=$XXX_CG_SIZE_LOG+$XXX_CG_AMOUNT_LOG+1
                                    elif [[ $SMALLEST_T -eq 1 && $alg -eq 20 ]]; then
                                        T1=$XXX_CG_SIZE_LOG+$XXX_CG_AMOUNT_LOG
                                        T2=$XXX_CG_SIZE_LOG+$XXX_CG_AMOUNT_LOG
                                    fi
                                    for(( e=$T1; e<=$T2; e+=T_STEP_SIZE ))
                                    do  
                                        ./perf-benchmark-exp.sh $dev $nt $alg $REA $REPS $n $q "bs" $e $lr $name $check $SEED_RANDOM $save_input_data
                                    done
                                # the relevant RTXRMQ algorithms
                                elif [ "$alg" -eq 5 ] || [ "$alg" -eq 10 ]; then
                                    if [ $n -lt $B1 ]; then
                                        echo "no benchmark for algo ${alg} since n is smaller than the smallest given bs!"      
                                    else
                                        for(( bs=$B1; bs<=$B2; bs++ ))
                                        do  
                                            if [ $n -lt $bs ]; then
                                                break
                                            fi
                                            nb=$((n - $bs))
                                            if [ $nb -ge $((23)) ]; then
                                                echo "skipped benchmark for algo ${alg} with n_log = ${n}, bs_log = ${bs} and nb_log = ${nb}"
                                                continue
                                            fi
                                            ./perf-benchmark-exp.sh $dev $nt $alg $REA $REPS $n $q "bs" $bs $lr $name $check $SEED_RANDOM $save_input_data
                                        done
                                    fi
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"