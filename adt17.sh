#!/bin/bash

N=10
T=200
S=best

MC_LIST=('10 1' '10 2' '10 5')

for i in `seq 0 $((${#MC_LIST[*]} - 1))`; do
    M=`echo ${MC_LIST[$i]} | cut -d' ' -f1`
    C=`echo ${MC_LIST[$i]} | cut -d' ' -f2`
    for d in 0.5; do
        for K in 2; do
            for F in indep sumcov varsumvarcov; do
                echo "RUNNING RF $M $C $d $K"
                ./adt17.py synthetic -N $N -M $M -C $C -T $T -K $K -d $d --min-regret 0.1 -F $F -v
            done
        done
    done
done
