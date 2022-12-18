make build

# for n in {1..20}; do
#     ./task1 1024 $((n))
# done > task1.txt

# for n in {1..20}; do
#     ./task2 1024 $((n))
# done > task2.txt

for n in {1..10}; do
    ./task3 1000000 8 $((2**n))
done > task3_ts.txt