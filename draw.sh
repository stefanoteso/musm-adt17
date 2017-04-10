#!/bin/bash

for C in 1 2 5 10 20; do
    for K in 2 3; do
        for d in 0.2 1.0; do
            ./draw.py synthetic-20in${C}-K${K}-${d} \
                results/synthetic_20_${C}_20_200_${K}_maxvar_indep_*_${d}_pl_* \
                results/synthetic_20_${C}_20_200_${K}_maxvar_varsumvarcov_{2.0,5.0}_*_${d}_pl_*
        done
    done
done
