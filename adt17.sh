#!/bin/bash

N=20
T=50

MC_LIST=('10 1' '10 5')

for K in 2 3; do
    for d in 0.2 1.0; do
        for problem in synthetic; do
            for i in `seq 0 $((${#MC_LIST[*]} - 1))`; do
                M=`echo ${MC_LIST[$i]} | cut -d' ' -f1`
                C=`echo ${MC_LIST[$i]} | cut -d' ' -f2`

                GROUPS_PATH="users/groups_${problem}_${M}_${C}_${d}"

                echo "RUNNING $problem $M $C $d $K"
                ./adt17.py $problem -N $N -M $M -C $C -G $GROUPS_PATH -d $d -T $T -K $K --min-regret 0.1 -F indep

                 for t in 50 10 5 1; do
                     echo "RUNNING $problem $M $C $d $K $t"
                     ./adt17.py $problem -N $N -M $M -C $C -G $GROUPS_PATH -d $d -T $T -K $K -t $t --min-regret 0.1 -F varsumvarcov
                 done
            done
        done
    done
done
