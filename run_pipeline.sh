#!/bin/bash

# =============================================================================
# Image Alignment Pipeline Runner
# Runs all 5 scripts in sequence automatically
# =============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "IMAGE ALIGNMENT PIPELINE"
echo "=========================================="
echo ""
echo "This script will run all 5 steps:"
echo "  1. Particle Selection (interactive)"
echo "  2. Drift Analysis (with progress bars)"
echo "  3. Interactive Trajectory Selector"
echo "  4. Image Alignment by Drift"
echo "  5. Validation (generate trajectory PNGs)"
echo ""
echo "=========================================="
echo ""

# Check if scripts directory exists
if [ ! -d "scripts" ]; then
    echo "Error: 'scripts/' directory not found!"
    echo "Please run this script from the image_alignmenetv2 directory."
    exit 1
fi

# Step 1: Particle Selection (Interactive)
echo ""
echo "=========================================="
echo "STEP 1: Particle Selection"
echo "=========================================="
echo "Starting interactive particle selection..."
echo ""

cd scripts
STEP1_OUTPUT=$(python3 1_select_particles_for_drift.py)
STEP1_EXIT_CODE=$?
echo "$STEP1_OUTPUT"
cd ..

if [ $STEP1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Particle selection failed or was cancelled."
    exit 1
fi

# Extract JSON path from output
JSON_PATH=$(echo "$STEP1_OUTPUT" | grep "^JSON_PATH=" | cut -d'=' -f2)

if [ -z "$JSON_PATH" ]; then
    echo ""
    echo "Error: Could not find JSON file path in Step 1 output."
    exit 1
fi

echo ""
echo "Using JSON file: $JSON_PATH"

# Step 2: Drift Analysis
echo ""
echo "=========================================="
echo "STEP 2: Drift Analysis"
echo "=========================================="
echo ""

cd scripts
python3 2_drift_analysis.py "$JSON_PATH"
STEP2_EXIT_CODE=$?
cd ..

if [ $STEP2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Drift analysis failed."
    exit 1
fi

# Step 3: Interactive Trajectory Selector
echo ""
echo "=========================================="
echo "STEP 3: Interactive Trajectory Selector"
echo "=========================================="
echo ""

cd scripts
python3 3_interactive_trajectory_selector.py "$JSON_PATH"
STEP3_EXIT_CODE=$?
cd ..

if [ $STEP3_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Interactive trajectory selector failed."
    exit 1
fi

# Step 4: Image Alignment
echo ""
echo "=========================================="
echo "STEP 4: Image Alignment by Drift"
echo "=========================================="
echo ""

cd scripts
python3 4_align_images_by_drift.py "$JSON_PATH"
STEP4_EXIT_CODE=$?
cd ..

if [ $STEP4_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Image alignment failed."
    exit 1
fi

# Step 5: Validation
echo ""
echo "=========================================="
echo "STEP 5: Validation"
echo "=========================================="
echo ""

cd scripts
python3 5_validation.py "$JSON_PATH"
STEP5_EXIT_CODE=$?
cd ..

if [ $STEP5_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Validation failed."
    exit 1
fi

# Success!
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "All steps completed successfully!"
echo ""
echo "Results are in the script_output folder (located next to your image folder):"
echo "  - Configuration: script_output/particle_selections.json"
echo "  - Particle tracking: script_output/particles_tracking/"
echo "  - Drift analysis: script_output/drift_analysis/"
echo "  - Aligned images: script_output/aligned/"
echo "  - Validation PNGs: script_output/validation/"
echo "  - Log file: script_output/log.txt"
echo ""
echo "=========================================="
