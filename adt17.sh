#!/bin/bash

for c in 1 2 5; do
    for d in 1 0.5; do
        for f in indep sumcov varsumvarcov; do
            ./adt17.py synthetic -N 20 -M 10 -C $c -T 100 -d $d --min-regret 0.1 -F $f
        done
    done
done
