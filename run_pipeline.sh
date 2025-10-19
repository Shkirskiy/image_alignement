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
echo "  2. Parallel Drift Analysis"
echo "  3. Aggregate Particle Drift"
echo "  4. Batch GPU Alignment"
echo "  5. Validation"
echo ""
echo "=========================================="
echo ""

# Check if scripts directory exists
if [ ! -d "scripts" ]; then
    echo "Error: 'scripts/' directory not found!"
    echo "Please run this script from the image_alignement directory."
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
python3 1_select_particles_for_drift.py
STEP1_EXIT_CODE=$?
cd ..

if [ $STEP1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Particle selection failed or was cancelled."
    exit 1
fi

# Find the JSON file created by step 1
JSON_FILE=$(find scripts_output -maxdepth 1 -name "particle_selections.json" -type f 2>/dev/null | head -n 1)

if [ -z "$JSON_FILE" ]; then
    echo ""
    echo "Error: No JSON file found in scripts_output/"
    echo "Particle selection may have been cancelled."
    exit 1
fi

echo ""
echo "Found JSON file: $JSON_FILE"
echo ""

# Step 2: Parallel Drift Analysis
echo ""
echo "=========================================="
echo "STEP 2: Parallel Drift Analysis"
echo "=========================================="
echo ""

cd scripts
python3 2_parallel_drift_analysis.py "../$JSON_FILE"
STEP2_EXIT_CODE=$?
cd ..

if [ $STEP2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Parallel drift analysis failed."
    exit 1
fi

# Step 3: Aggregate Particle Drift
echo ""
echo "=========================================="
echo "STEP 3: Aggregate Particle Drift"
echo "=========================================="
echo ""

cd scripts
python3 3_aggregate_particle_drift.py "../$JSON_FILE"
STEP3_EXIT_CODE=$?
cd ..

if [ $STEP3_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Aggregate particle drift failed."
    exit 1
fi

# Step 4: Batch GPU Alignment
echo ""
echo "=========================================="
echo "STEP 4: Batch GPU Alignment"
echo "=========================================="
echo ""

cd scripts
python3 4_batch_gpu_alignment.py "../$JSON_FILE"
STEP4_EXIT_CODE=$?
cd ..

if [ $STEP4_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Batch GPU alignment failed."
    exit 1
fi

# Step 5: Validation
echo ""
echo "=========================================="
echo "STEP 5: Validation"
echo "=========================================="
echo ""

cd scripts
python3 5_validation.py "../$JSON_FILE"
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
echo "Results:"
echo "  - JSON file: $JSON_FILE"
echo "  - Particle tracking CSV: scripts_output/particles_tracking/"
echo "  - Drift analysis CSV: scripts_output/particle_drift/"
echo "  - Aligned images: (sibling to input folders with _aligned suffix)"
echo "  - Validation results: scripts_output/validation/"
echo ""
echo "=========================================="
