#!/bin/bash

# Clear All Plots Script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_DIR="$SCRIPT_DIR/../plot"

echo "=== Clearing All Plot Files (Force) ==="
echo "Plot Directory: $PLOT_DIR"
echo "Start Time: $(date)"
echo "========================================"

# Check if plot directory exists
if [ ! -d "$PLOT_DIR" ]; then
    echo "Error: Plot directory does not exist: $PLOT_DIR"
    exit 1
fi

# Count and delete PDF files
pdf_count=$(find "$PLOT_DIR" -name "*.pdf" -type f | wc -l)

if [ $pdf_count -eq 0 ]; then
    echo "No PDF files found in plot directory and its subdirectories."
else
    echo "Found $pdf_count PDF file(s). Deleting..."
    
    # List files being deleted
    echo "Deleting the following PDF files:"
    find "$PLOT_DIR" -name "*.pdf" -type f
    
    # Delete all PDF files
    find "$PLOT_DIR" -name "*.pdf" -type f -delete
    echo "Successfully deleted $pdf_count PDF file(s)."
fi

echo ""
echo "========================================"
echo "Clear operation completed!"
echo "End Time: $(date)"