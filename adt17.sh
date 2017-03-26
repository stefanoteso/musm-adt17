#!/bin/bash

#MC_LIST=('2 1')
MC_LIST=('10 1' '10 2' '10 5')

for i in `seq 0 $((${#MC_LIST[*]} - 1))`; do
    M=`echo ${MC_LIST[$i]} | cut -d' ' -f1`
    C=`echo ${MC_LIST[$i]} | cut -d' ' -f2`
    for d in 0.5; do
        for K in 2; do

            echo "RUNNING INDEPENDENT $M $C $d $K"
            ./adt17.py synthetic -N 10 -M $M -C $C -T 50 -K $K -d $d --min-regret 0.1 -F indep

            for S in best; do
                for L in 0.25 0.5 0.75; do
                    echo "RUNNING SUMCOV $M $C $d $K $S $L"
                    ./adt17.py synthetic -N 10 -M $M -C $C -T 50 -K $K -d $d --min-regret 0.1 -F sumcov -S $S -L $L
                done
            done

        done
    done
done
