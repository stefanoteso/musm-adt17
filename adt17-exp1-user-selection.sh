#!/bin/bash

N=20
T=200

MC_LIST=('20 1' '20 2' '20 5')

for K in 2 3; do
    for d in 0.2 1.0; do
        for u in normal uniform; do
            for problem in synthetic; do
                for i in `seq 0 $((${#MC_LIST[*]} - 1))`; do
                    M=`echo ${MC_LIST[$i]} | cut -d' ' -f1`
                    C=`echo ${MC_LIST[$i]} | cut -d' ' -f2`

                    GROUPS_PATH="users/groups_${problem}_${M}_${C}_${u}_${d}"

                    ./adt17.py $problem -N $N -M $M -C $C -G $GROUPS_PATH -u $u -d $d -T $T -K $K --min-regret 0.1 -F indep

                    for P in random numqueries maxvar pseudoregret regret; do
                        for t in 2 5; do
                            ./adt17.py $problem -N $N -M $M -C $C -G $GROUPS_PATH -u $u -d $d -T $T -K $K -F varsumvarcov -t $t --min-regret 0.1 -P $P
                        done
                    done
                done
            done
        done
    done
done
