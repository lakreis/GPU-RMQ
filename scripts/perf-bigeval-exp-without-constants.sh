#!/bin/bash

cd ..

recompile=false

if [ "$recompile" = true ]; then
    rm -r build/
    # TODO@newUser: write your local OptiX path here
    cmake -S . -B build -DOPTIX_HOME=/your/OptiX/path/here
fi

cmake --build build
cd scripts/

#for sizes larger than 2^24
large_sizes_save_input_data=0

dev=${1}
nt=${2}
name=${3}

ALGS=($((16)))

N1=$((15))
N2=$((31))

Q1=$((26))
Q2=$((26))

REA=$((4))
REPS=$((1))

SEED_RANDOM=$((0))

# threshold for the XXX algorithms
T1=$((6))
T2=$((6))

# bs for the RTXRMQ algorithms
B1=$((6))
B2=$((6))

# 1 to check normally, 2 to check trivially (not a normal run)
# 3 to check randomly trivally, 0 = no check at all
check=3

DATEBEGIN=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "START #DATE = ${DATEBEGIN}"

for(( n=$N1; n<=$N2; n++ ))
do  
    if [ "$n" -lt 25 ]; then
        save_input_data=0
    else
        save_input_data=$large_sizes_save_input_data
    fi
    for(( q=$Q1; q<=$Q2; q++ ))
    do  
        for lr in {-1,-2,-3,-6}
        do  
            for alg in ${ALGS[@]}
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
                    for(( e=$T1; e<=$T2; e++ ))
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

DATEEND=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
printf "perf-benchmark.sh FINISHED:\n"
printf "\tBEGIN: ${DATEBEGIN}\n\tEND: ${DATEEND}\n\n"