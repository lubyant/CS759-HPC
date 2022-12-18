#!/usr/bin/env bash
make task2

for n in {10..29}; do
    ./task2 $((2**n)) 128 256
done > task2_256.txt

for n in {10..29}; do
    ./task2 $((2**n)) 128 128
done > task2_128.txt

for n in {10..29}; do
    ./task2 $((2**n)) 128 512
done > task2_512.txt

for n in {10..29}; do
    ./task2 $((2**n)) 128 1024
done > task2_1024.txt