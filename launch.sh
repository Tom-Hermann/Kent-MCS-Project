#!/bin/bash

# Default values
language=""
type=""
student=""
benchmark=""
install=""


# Function to display script usage
function display_usage {
    echo "Usage: $0 [-l|--language python|julia] [-t|--type gpu|cpu] [-i|install] [-s|--student] [-b|--benchmark] [-h|--help]"
    echo ""
    echo "Description:"
    echo "This script is designed to run either Python or Julia scripts for CPU or GPU, with optional benchmarking and student GPU partition."
    echo "It provides flexibility in choosing the language, execution type, and other options."
    echo ""
    echo "Options:"
    echo "-l, --language    Choose the scripting language (python or julia)."
    echo "-t, --type        Specify the execution type (gpu or cpu)."
    echo "-i, --install     Install the required packages before script execution (optional)."
    echo "-s, --student     Use the student GPU partition (optional)."
    echo "-b, --benchmark   Run benchmark after script execution (optional)."
    echo "-h, --help        Display this usage message and exit."
    echo ""
    echo "Examples:"
    echo "1. Run Python script on GPU partition:"
    echo "   $0 -l python -t gpu"
    echo ""
    echo "2. Run Julia script on CPU:"
    echo "   $0 -l julia -t cpu"
    echo ""
    echo "3. Run Python script on student GPU partition with benchmark:"
    echo "   $0 -l python -t gpu -s -b"
    echo ""
    echo "4. Display script usage information:"
    echo "   $0 -h"
    exit 1
}

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
