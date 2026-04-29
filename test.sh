#!/bin/bash

# Ensure the executable exists
EXECUTABLE="./build/rope_benchmark"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable $EXECUTABLE not found. Please compile the project first by running 'cd build && make'."
    exit 1
fi

# Define 5 different combinations of seq-len and head-dim
# Note: head-dim must be an even number
declare -a seq_lens=(512 1024 2048 4096 8192 16384)
declare -a head_dims=(64 64 128 128 256 256)

NUM_TESTS=${#seq_lens[@]}

echo "Starting benchmark suite with $NUM_TESTS configurations..."
echo "================================================="

for i in "${!seq_lens[@]}"; do
    seq_len=${seq_lens[$i]}
    head_dim=${head_dims[$i]}
    
    echo -e "\n\n---> Running Test $((i+1))/$NUM_TESTS: --seq-len $seq_len --head-dim $head_dim"
    $EXECUTABLE --seq-len $seq_len --head-dim $head_dim --iterations 5
done

echo -e "\n\n================================================="
echo "Benchmark suite completed."
