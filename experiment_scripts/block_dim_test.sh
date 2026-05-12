#!/bin/bash

# Ensure the executable exists
EXECUTABLE="./build/rope_benchmark"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable $EXECUTABLE not found. Please compile the project first by running 'cd build && make'."
    exit 1
fi

declare -a threads_per_blocks=(64 128 256 512 1024)

NUM_TESTS=${#threads_per_blocks[@]}
SEQ_LEN=4096
HEAD_DIM=64
NUM_ITERATIONS=25


echo "Starting runtime scaling benchmark suite with $NUM_TESTS configurations..."
echo "================================================="

for i in "${!threads_per_blocks[@]}"; do
    thread_per_block=${threads_per_blocks[$i]}
    
    echo -e "\n\n---> Running Test $((i+1))/$NUM_TESTS: --seq-len $SEQ_LEN --head-dim $HEAD_DIM --threads-per-block $thread_per_block"
    if [ "$#" -eq 1 ]; then
        $EXECUTABLE --mode parallel-only --seq-len $SEQ_LEN --head-dim $HEAD_DIM --iterations $NUM_ITERATIONS --threads-per-block $thread_per_block --csv-output "$1"
    else
        $EXECUTABLE --mode parallel-only --seq-len $SEQ_LEN --head-dim $HEAD_DIM --iterations $NUM_ITERATIONS --threads-per-block $thread_per_block
    fi
done

echo -e "\n\n================================================="
echo "Scaling benchmark suite completed."