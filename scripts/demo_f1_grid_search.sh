#!/bin/bash

# Demo script showing complete workflow for F1 score comparison grid search

echo "🎯 F1 Score Comparison Grid Search Demo"
echo "========================================"
echo ""

echo "📋 Available Grid Configurations:"
echo "-----------------------------------"
find configs/ -name "*.json" -type f | while read file; do
    echo "📄 $file"
    if command -v jq &> /dev/null; then
        echo "   Experiments: $(jq -r '[.methods, .keep_ratios, .relabel_percentages, .seeds] | map(length) | .[0] * .[1] * .[2] * .[3]' "$file")"
        echo "   Methods: $(jq -r '.methods | length' "$file") ($(jq -r '.methods | join(", ")' "$file"))"
        echo "   Keep Ratios: $(jq -r '.keep_ratios | join(", ")' "$file")"
        echo "   Relabel %: $(jq -r '.relabel_percentages | join(", ")' "$file")"
    fi
    echo ""
done

echo "🔍 Testing F1 Comparison Grid (Dry Run):"
echo "----------------------------------------"
echo "Grid: configs/f1_comparison_grid.json"
echo "Keep Ratios: 100, 95, 90, 85, 80"
echo "Relabel Percentages: 5, 10, 15, 20"
echo "Total Experiments: 240"
echo ""

# Show first few commands that would be executed
echo "📝 Sample Commands (first 5):"
bash scripts/run_cleansing_grid.sh -g configs/f1_comparison_grid.json --dry-run 2>/dev/null | grep "python scripts/epoch_wise_keep_ratio.py" | head -5 | while read cmd; do
    echo "   $cmd"
done

echo ""
echo "🚀 How to Run the Full F1 Comparison Grid:"
echo "-------------------------------------------"

cat << 'EOF'
# 1. Run the complete grid search (240 experiments)
bash scripts/run_cleansing_grid.sh -g configs/f1_comparison_grid.json --run-analysis

# 2. For testing, use a smaller grid first
bash scripts/run_cleansing_grid.sh -g configs/sentiment_cleansing_small.json --run-analysis

# 3. Only compute precision metrics (faster, no retraining)
bash scripts/run_cleansing_grid.sh -g configs/f1_comparison_grid.json \
  --compute-retraining False --run-analysis

# 4. Custom GPU allocation (if you have more GPUs)
bash scripts/run_cleansing_grid.sh -g configs/f1_comparison_grid.json \
  --gpus 0,1,2,3,4,5 --run-analysis

# 5. Run F1-specific analysis on existing results
python scripts/analyze_f1_comparison.py --base_dir outputs \
  --target sentiment --model bert --plot
EOF

echo ""
echo "📊 Analysis Features:"
echo "---------------------"
echo "✅ F1 Score focused analysis"
echo "✅ Method performance ranking"
echo "✅ Best configuration identification"
echo "✅ Statistical significance testing"
echo "✅ Comprehensive visualizations"
echo "✅ Automated report generation"

echo ""
echo "🎁 Expected Output Files:"
echo "-------------------------"
echo "📄 f1_analysis_report.txt - Comprehensive F1 analysis"
echo "📊 f1_comparison_plots.png - Performance visualizations"
echo "📋 f1_summary_table.csv - Detailed statistical summary"
echo "📈 precision_analysis_report.txt - General precision analysis"

echo ""
echo "⏱️ Estimated Runtime:"
echo "----------------------"
echo "🔸 Small grid (8 experiments): ~30-60 minutes"
echo "🔸 F1 grid (240 experiments): ~8-12 hours"
echo "🔸 Precision-only mode: ~50% faster"

echo ""
echo "💡 Pro Tips:"
echo "-------------"
echo "• Start with a dry run to verify commands"
echo "• Use precision-only mode for faster iterations"
echo "• Monitor GPU utilization during experiments"
echo "• Check individual experiment logs if failures occur"
echo "• Run analysis separately if grid search completes successfully"