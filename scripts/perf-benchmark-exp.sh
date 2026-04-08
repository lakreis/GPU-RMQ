#!/bin/bash

cd ..

dev=${1}
nt=${2}
alg=${3}
rea=${4}
reps=${5}
n=${6}
q=${7}
bs_or_nb=${8}
bsize=${9}
lr=${10}
name=${11}
check=${12}
SEED_RANDOM=${13}
save_input_data=${14}

SEEDS=(27722 833241 913 97528 127319876 26 992737 47203 3684 63912901 3683 2470 534263 462456)

binary=build/rtxrmq
outfile_path=data/perfexp-${name}-ALG${alg}.csv

#[ ! -f ${outfile_path} ] && echo "dev,alg,nt,reps,n,bs,nb,q,lr,GPU_BSIZE,CG_GROUP_SIZE,CG_MEM_ALIGNMENT,timestamp,t,q/s,ns/q,construction,outbuffer,tempbuffer,freeGPUMemCorrect,freeGPUMem,checkResult" > ${outfile_path}

if [ "$save_input_data" -eq 1 ]; then
    suffix="--save-input-data"
else
    suffix=""
fi

printf "\n\n\n\n\n\n\n\n"
for(( R=1; R<=$rea; R++ ))
do  
    printf "\n\n\n"
    if [ "$SEED_RANDOM" -eq 1 ] || [ "$R" -ge ${#SEEDS[@]} ]; then
        SEED=${RANDOM}
    else
        SEED=${SEEDS[$R-1]}
    fi
    printf "REALIZATION $R of algorithm $alg -> n=$n q=$q\n"
    if [ $check -eq 1 ]; then
        if [ "${bs_or_nb}" = "bs" ]; then
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --check --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --check --seed ${SEED} ${suffix} \n
        else
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --check --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --check --seed ${SEED} ${suffix} \n
        fi
    elif [ $check -eq 3 ]; then
        if [ "${bs_or_nb}" = "bs" ]; then
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --randTrivialCheck --seed ${SEED} ${suffix}\n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --randTrivialCheck --seed ${SEED} ${suffix} \n
        else
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --randTrivialCheck --seed ${SEED} ${suffix}\n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --randTrivialCheck --seed ${SEED} ${suffix} \n
        fi
    elif [ $R -eq 1 ] && [ $check -eq 2 ]; then
        if [ "${bs_or_nb}" = "bs" ]; then
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --trivialCheck --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --trivialCheck --seed ${SEED} ${suffix} \n
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix}\n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix}\n
        else
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --trivialCheck --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --trivialCheck --seed ${SEED} ${suffix} \n
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n
        fi
    else
        if [ "${bs_or_nb}" = "bs" ]; then
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --log_bs $((${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n
        else
            printf "${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n"
                    ${binary} $((2**$n)) $((2**$q)) ${lr} ${alg} --nb $((2**${bsize})) --reps $reps --nt $nt --dev $dev --save-time=${outfile_path} --seed ${SEED} ${suffix} \n
        fi
    fi
done

cd scripts/