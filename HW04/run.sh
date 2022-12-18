#!/usr/bin/env bash

# for n in {5..14}; do
#     ./matmul $((2**n)) 
# done > matmul.txt

for n in {10..29}; do
    ./stencil $((2**n)) 128
done > stencil.txt