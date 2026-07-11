#!/bin/bash
# Cleanup script to remove all .pt checkpoint files from experiment directories
# Usage: ./scripts/cleanup_checkpoints.sh [directory_pattern]

set -e

# Default pattern
PATTERN="${1:-bert_custom_*}"

echo "🔍 Searching for .pt files in: $PATTERN"
echo ""

# Find and count .pt files
PT_FILES=$(find . -maxdepth 3 -type d -name "$PATTERN" -exec find {} -name "*.pt" -type f \; 2>/dev/null)
COUNT=$(echo "$PT_FILES" | grep -c ".pt" || echo "0")

if [ "$COUNT" -eq 0 ]; then
    echo "ℹ️  No .pt files found matching pattern: $PATTERN"
    exit 0
fi

# Calculate total size
TOTAL_SIZE=0
while IFS= read -r file; do
    if [ -n "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    fi
done <<< "$PT_FILES"

TOTAL_MB=$((TOTAL_SIZE / 1024 / 1024))

echo "📊 Found $COUNT .pt files (Total: ${TOTAL_MB} MB)"
echo ""
echo "Files to delete:"
echo "$PT_FILES" | head -20
if [ "$COUNT" -gt 20 ]; then
    echo "... and $((COUNT - 20)) more files"
fi
echo ""

# Ask for confirmation
read -p "❓ Delete these files? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

# Delete files
echo "🧹 Deleting .pt files..."
DELETED=0
while IFS= read -r file; do
    if [ -n "$file" ]; then
        rm -f "$file"
        DELETED=$((DELETED + 1))
        echo "  Deleted: $file"
    fi
done <<< "$PT_FILES"

echo ""
echo "✅ Cleanup complete: Deleted $DELETED files, freed ${TOTAL_MB} MB"
