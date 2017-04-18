#!/bin/bash

for C in 1 2 5 20; do
    for K in 2; do
        for u in normal uniform; do
            for d in 0.2; do
                ./draw.py pc-exp3-20in${C}-K${K}-${u}-${d} \
                    results/pc_20_${C}_20_200_${K}_random_indep_0_0.25_False_0.1_${u}_${d}_pl_1_0.pickle \
                    results/pc_20_${C}_20_200_${K}_numqueries_varsumvarcov_0_2.0_False_0.1_${u}_${d}_pl_1_0.pickle \
                    results/pc_20_${C}_20_200_${K}_regret_varsumvarcov_0_2.0_False_0.1_${u}_${d}_pl_1_0.pickle
            done
        done
    done
done
