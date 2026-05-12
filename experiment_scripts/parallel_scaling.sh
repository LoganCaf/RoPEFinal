#!/bin/bash

# Ensure the executable exists
EXECUTABLE="./build/rope_benchmark"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable $EXECUTABLE not found. Please compile the project first by running 'cd build && make'."
    exit 1
fi

# Test higher maximum values for sequence lengths
declare -a seq_lens=(512 1024 2048 4096 8192 16384 32768)

NUM_TESTS=${#seq_lens[@]}
HEAD_DIM=64
NUM_ITERATIONS=25

echo "Starting parallel scaling benchmark suite with $NUM_TESTS configurations..."
echo "================================================="

for i in "${!seq_lens[@]}"; do
    seq_len=${seq_lens[$i]}
    
    echo -e "\n\n---> Running Test $((i+1))/$NUM_TESTS: --seq-len $seq_len --head-dim $HEAD_DIM"
    if [ "$#" -eq 1 ]; then
        $EXECUTABLE --mode parallel-only --seq-len $seq_len --head-dim $HEAD_DIM --iterations $NUM_ITERATIONS --csv-output "$1"
    else
        $EXECUTABLE --mode parallel-only --seq-len $seq_len --head-dim $HEAD_DIM --iterations $NUM_ITERATIONS
    fi
done

echo -e "\n\n================================================="
echo "Parallel scaling benchmark suite completed."