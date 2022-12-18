for n in {10..29}; do
    ./task3 $((2**n)) 
done > result.txt