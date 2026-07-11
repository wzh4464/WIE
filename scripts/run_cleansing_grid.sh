#!/bin/bash

# Batch script wrapper for influence cleansing grid experiments
# Mimics the pattern of run_sentiment_grid.py but for cleansing experiments

set -e

# Default values
GRID_FILE="configs/influence_cleansing_grid.json"
OUTPUT_ROOT="outputs"
SAVE_DIR_PREFIX="influence_cleansing"
TARGET="sentiment"
MODEL="bert"
GPUS="0,1,2"
COMPUTE_PRECISION="True"
COMPUTE_RETRAINING="True"
DRY_RUN=""
RUN_ANALYSIS=""
LOG_LEVEL="INFO"

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run influence cleansing grid experiments across multiple GPUs.

OPTIONS:
    -g, --grid-file FILE        Grid configuration JSON file (default: $GRID_FILE)
    -o, --output-root DIR       Output root directory (default: $OUTPUT_ROOT)
    -p, --prefix PREFIX         Save directory prefix (default: $SAVE_DIR_PREFIX)
    -t, --target TARGET         Target dataset (default: $TARGET)
    -m, --model MODEL           Model type (default: $MODEL)
    --gpus GPU_LIST             Comma-separated GPU IDs (default: $GPUS)
    --compute-precision BOOL    Compute precision metrics (default: $COMPUTE_PRECISION)
    --compute-retraining BOOL   Compute retraining losses (default: $COMPUTE_RETRAINING)
    --log-level LEVEL           Logging level (default: $LOG_LEVEL)
    --dry-run                   Print commands without executing
    --run-analysis              Run precision analysis after completion
    -h, --help                  Show this help message

EXAMPLES:
    # Basic grid search with default parameters
    $0

    # Dry run to see what would be executed
    $0 --dry-run

    # Use small configuration for testing
    $0 -g configs/sentiment_cleansing_small.json --dry-run

    # Run with precision analysis
    $0 --run-analysis

    # Custom configuration
    $0 -g my_grid.json -o /path/to/outputs -t sentiment -m bert --gpus 0,1,2,3

    # Only compute precision (faster)
    $0 --compute-retraining False --run-analysis

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--grid-file)
            GRID_FILE="$2"
            shift 2
            ;;
        -o|--output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -p|--prefix)
            SAVE_DIR_PREFIX="$2"
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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --compute-precision)
            COMPUTE_PRECISION="$2"
            shift 2
            ;;
        --compute-retraining)
            COMPUTE_RETRAINING="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --run-analysis)
            RUN_ANALYSIS="--run-analysis"
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

# Convert comma-separated GPU list to space-separated
GPU_ARRAY=(${GPUS//,/ })

# Validate grid file exists
if [[ ! -f "$GRID_FILE" ]]; then
    echo "Error: Grid file '$GRID_FILE' does not exist!"
    echo "Available grid files:"
    find configs/ -name "*.json" -type f 2>/dev/null || echo "  (none found in configs/)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Print configuration
echo "🚀 Influence Cleansing Grid Search"
echo "=================================="
echo "Grid File: $GRID_FILE"
echo "Output Root: $OUTPUT_ROOT"
echo "Save Dir Prefix: $SAVE_DIR_PREFIX"
echo "Target: $TARGET"
echo "Model: $MODEL"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Compute Precision: $COMPUTE_PRECISION"
echo "Compute Retraining: $COMPUTE_RETRAINING"
echo "Log Level: $LOG_LEVEL"
echo "Dry Run: $([ -n "$DRY_RUN" ] && echo "Yes" || echo "No")"
echo "Run Analysis: $([ -n "$RUN_ANALYSIS" ] && echo "Yes" || echo "No")"
echo ""

# Preview grid configuration
echo "📋 Grid Configuration Preview:"
echo "------------------------------"
if command -v jq &> /dev/null; then
    jq '.' "$GRID_FILE"
else
    cat "$GRID_FILE"
fi
echo ""

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_influence_cleansing_grid.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python grid script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Build and run the command
echo "🔧 Building command..."

PYTHON_CMD=(
    python "$PYTHON_SCRIPT"
    --grid-file "$GRID_FILE"
    --output-root "$OUTPUT_ROOT"
    --save-dir-prefix "$SAVE_DIR_PREFIX"
    --target "$TARGET"
    --model "$MODEL"
    --compute-precision "$COMPUTE_PRECISION"
    --compute-retraining-loss "$COMPUTE_RETRAINING"
    --log-level "$LOG_LEVEL"
    --gpus "${GPU_ARRAY[@]}"
)

# Add optional flags
if [[ -n "$DRY_RUN" ]]; then
    PYTHON_CMD+=("$DRY_RUN")
fi

if [[ -n "$RUN_ANALYSIS" ]]; then
    PYTHON_CMD+=("$RUN_ANALYSIS")
fi

echo "Command: ${PYTHON_CMD[*]}"
echo ""

# Execute the command
echo "▶️ Executing grid search..."
"${PYTHON_CMD[@]}"

# Summary
if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "✅ Grid search completed!"
    echo ""
    echo "📁 Results location: $OUTPUT_ROOT"
    echo ""
    echo "🔍 To analyze results manually:"
    echo "   python scripts/analyze_precision_performance.py --base_dir $OUTPUT_ROOT --target $TARGET --model $MODEL --plot"
    echo ""
    echo "📊 Generated experiments can be found in:"
    echo "   $OUTPUT_ROOT/$SAVE_DIR_PREFIX*/"
fi