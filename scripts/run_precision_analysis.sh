#!/bin/bash

# Batch script to run precision analysis on sentiment-bert experiments
# This script helps analyze existing experiment results to find optimal configurations

set -e

# Default values
BASE_DIR="outputs"
TARGET="sentiment"
MODEL="bert"
OUTPUT_DIR="analysis_results"
METRIC="precision_max"
PLOT_FLAG=""

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Analyze precision performance for sentiment-bert experiments across different methods and configurations.

OPTIONS:
    -d, --base-dir DIR      Base directory containing experiment results (default: outputs)
    -t, --target TARGET     Target dataset name (default: sentiment)
    -m, --model MODEL       Model name (default: bert)
    -o, --output-dir DIR    Output directory for reports (default: analysis_results)
    -M, --metric METRIC     Metric for best performance (default: precision_max)
                           Options: precision_max, precision_mean, f1_max, f1_mean
    -p, --plot             Generate visualization plots
    -h, --help             Show this help message

EXAMPLES:
    # Basic analysis
    $0 -d outputs/sentiment_experiments

    # Analysis with plots
    $0 -d outputs/sentiment_experiments -p

    # Custom metric and output directory
    $0 -d /path/to/experiments -M f1_max -o /path/to/results

    # Analyze different target/model combination
    $0 -d outputs -t imdb -m bert

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -M|--metric)
            METRIC="$2"
            shift 2
            ;;
        -p|--plot)
            PLOT_FLAG="--plot"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory '$BASE_DIR' does not exist!"
    exit 1
fi

# Print configuration
echo "🔍 Precision Performance Analysis"
echo "=================================="
echo "Base Directory: $BASE_DIR"
echo "Target: $TARGET"
echo "Model: $MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Metric: $METRIC"
echo "Plot: $([ -n "$PLOT_FLAG" ] && echo "Yes" || echo "No")"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/analyze_precision_performance.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python analysis script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the Python analysis script
echo "Running analysis..."
python "$PYTHON_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --target "$TARGET" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --metric "$METRIC" \
    $PLOT_FLAG

echo ""
echo "✅ Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"

# List output files
if [[ -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
fi

echo ""
echo "📊 To view the detailed report:"
echo "   cat $OUTPUT_DIR/precision_analysis_report.txt"
echo ""
echo "📈 To view detailed results:"
echo "   head -20 $OUTPUT_DIR/detailed_results.csv"