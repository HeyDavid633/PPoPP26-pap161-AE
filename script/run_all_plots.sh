#!/bin/bash

# Batch Plot Script Executor

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_DIR="$SCRIPT_DIR/../plot"

echo "=== Starting All Plot Scripts ==="
echo "Plot Directory: $PLOT_DIR"
echo "Start Time: $(date)"
echo "========================================"

# Check if plot directory exists
if [ ! -d "$PLOT_DIR" ]; then
    echo "Error: Plot directory does not exist: $PLOT_DIR"
    exit 1
fi

# Function to run Python script from its own directory
run_python_script() {
    local script_dir="$1"
    local script_name="$2"
    local figure_name="$3"
    
    echo ""
    echo "[INFO] Starting $figure_name plot..."
    
    local current_dir=$(pwd)

    cd "$script_dir" || {
        echo "[ERROR] Cannot change to directory: $script_dir"
        return 1
    }
    
    python "$script_name"
    local result=$?
    
    cd "$current_dir"
    
    if [ $result -eq 0 ]; then
        echo "[$figure_name] Plot finished successfully!"
    else
        echo "[ERROR] $figure_name plot execution failed!"
    fi
    
    return $result
}

# Execute Figure 9-10
run_python_script "$PLOT_DIR/fig9-10" "fig9-10.py" "Figure 9-10"

# Execute Figure 11
run_python_script "$PLOT_DIR/fig11" "fig11.py" "Figure 11"

# Execute Figure 11 single device 
run_python_script "$PLOT_DIR/fig11" "fig11_single_device.py" "Figure 11"

# Execute Figure 12
run_python_script "$PLOT_DIR/fig12" "fig12.py" "Figure 12"

# Execute Figure 13
run_python_script "$PLOT_DIR/fig13" "fig13.py" "Figure 13"

echo ""
echo "========================================"
echo "All plot scripts execution completed!"
echo "End Time: $(date)"
echo ""
echo "Data Information:"
echo "Original data located in:"
echo "  - data/MHA_Performance: Figure 9-10"
echo "  - data/End2End_Performance: Figure 11" 
echo "  - data/Ablation_Study: Figure 12"
echo "  - data/Overhead_Analysis: Figure 13"