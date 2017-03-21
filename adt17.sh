#!/bin/bash

for k in 3 4; do
    for c in 1 2 5; do
        for d in 0.5 1.0; do
            for f in indep sumcov varsumvarcov; do
                ./adt17.py synthetic -N 20 -M 10 -C $c -K $k -T 100 -d $d --min-regret 0.1 -F $f
            done
            ./draw.py debug_k${k}_10on${c}_density\=$d results/synthetic_20_${c}_10_100_${k}_*_False_0.1_normal_${d}_pl_1_0.pickle
        done
    done
done
