#!/bin/bash

# Default values
language=""
type=""
student=""
benchmark=""
install=""


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--language)
            language="$2"
            shift 2
            ;;
        -t|--type)
            type="$2"
            shift 2
            ;;
        -s|--student)
            student="true"
            shift
            ;;
        -b|--benchmark)
            benchmark="true"
            shift
            ;;
        -h|--help)
            display_usage
            ;;
        -i|--install)
            install="true"
            shift
            ;;
        *)
            echo "Invalid argument: $1"
            display_usage
            ;;
    esac
done


# Run benchmark if requested
if [[ "$benchmark" == "true" ]]; then
    python ./src/python-tensorflow/benchmark.py
    exit 0
fi

# Check if language and type are provided
if [[ -z "$language" || -z "$type" ]]; then
    echo "Both language and type are required."
    display_usage
fi

# Set GPU partition based on student option
if [[ "$student" == "true" ]]; then
    gpu_partition="gpu.stu"
else
    gpu_partition="gpu"
fi

# Run the selected script
if [[ "$language" == "julia" ]]; then
    if [[ "$install" == "true" ]]; then
        julia ./src/julia-flux/requirement.jl
    fi
    if [[ "$type" == "gpu" ]]; then
        srun -p "$gpu_partition" --gres gpu:1 julia ./src/julia-flux/gpu.jl
    else
        julia ./src/julia-flux/cpu.jl
    fi
elif [[ "$language" == "python" ]]; then
    if [[ "$install" == "true" ]]; then
        pip install -r ./src/python-tensorflow/requirement.txt
    fi
    if [[ "$type" == "gpu" ]]; then
        srun -p "$gpu_partition" --gres gpu:1 python ./src/python-tensorflow/gpu.py
    else
        python ./src/python-tensorflow/cpu.py
    fi
else
    echo "Invalid language: $language"
    display_usage
fi

# Function to display script usage
function display_usage {
    echo "Usage: $0 [-l|--language python|julia] [-t|--type gpu|cpu] [-i|--install] [-s|--student] [-b|--benchmark] [-h|--help]"
    exit 1
}