#!/bin/bash

N=10
T=250
S=best

MC_LIST=('10 1' '10 2' '10 5')

for K in 2 3; do
    for d in 0.5 1.0; do
        for problem in synthetic pc; do
            for i in `seq 0 $((${#MC_LIST[*]} - 1))`; do
                M=`echo ${MC_LIST[$i]} | cut -d' ' -f1`
                C=`echo ${MC_LIST[$i]} | cut -d' ' -f2`
                for F in indep varsumvarcov; do
                    echo "RUNNING $problem $M $C $d $K"
                    ./adt17.py $problem -N $N -M $M -C $C -T $T -K $K -d $d --min-regret 0.1 -F $F -v
                done
            done
        done
    done
done
