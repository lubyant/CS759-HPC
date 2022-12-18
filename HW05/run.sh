#!/usr/bin/env bash

# for n in {16..32}; do
#     ./task2 $((2**14)) $((n))
# done > task2_op.txt

# for n in {5..14}; do
#     ./task2 $((2**n)) 32
# done > task2_32.txt

for n in {5..14}; do
    ./task2 $((2**n)) 16
done > task2_16.txt