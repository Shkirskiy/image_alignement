"""
Aggregate Particle Drift Data
Converts per-particle drift CSV files to per-image aggregated drift data
compatible with gpu_image_alignment.py

This script reads CSV files produced by parallel_drift_analysis.py and aggregates
drift values across all particles for each image. Processes multiple CSV files
in parallel for improved performance.

Requirements:
pip install pandas numpy tqdm matplotlib

Usage:
python aggregate_particle_drift.py [optional: path_to_json_file]

Examples:
# Auto-discover JSON in scripts_output/ folder (must be only one JSON file)
python aggregate_particle_drift.py

# Manually specify JSON file path
python aggregate_particle_drift.py particle_selections_20251016_100525.json
python aggregate_particle_drift.py scripts_output/particle_selections_20251016_100525.json

Output:
- CSV files saved to: scripts_output/particle_drift/
- Filename format: drift_{original_csv_name} (e.g., drift_example_data1_xn9.csv)
- Plots saved to same directory
- JSON file updated with 'drift_csv_file_path' and 'drift_csv_file_name' for each image set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import sys
import json
import multiprocessing as mp
from multiprocessing import Pool
from typing import Dict, Any, Tuple, Optional, List
from logging_utils import setup_logger, log_worker_message, log_exception

# Setup centralized logger
logger = setup_logger('Step3_AggregateDrift')

def plot_drift_data(df_agg, output_dir, folder_name):
    """
    Plot drift data (X, Y, rotation) over image sequence.

    Args:
        df_agg: Aggregated drift DataFrame
        output_dir: Directory to save plot
        folder_name: Name for output file
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Create image index for X axis
    image_indices = np.arange(len(df_agg))

    # Plot 1: X drift
    ax1 = axes[0]
    scatter1 = ax1.scatter(image_indices, df_agg['dx_pixels'],
                          c=df_agg['fit_rmse'], cmap='viridis',
                          s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.plot(image_indices, df_agg['dx_pixels'], 'b-', alpha=0.3, linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('X Drift (pixels)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Drift Analysis: {folder_name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('RMSE (px)', fontsize=9)

    # Add statistics text
    mean_x = df_agg['dx_pixels'].mean()
    std_x = df_agg['dx_pixels'].std()
    ax1.text(0.02, 0.98, f'Mean: {mean_x:.2f} px\nStd: {std_x:.2f} px',
            transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Y drift
    ax2 = axes[1]
    scatter2 = ax2.scatter(image_indices, df_agg['dy_pixels'],
                          c=df_agg['fit_rmse'], cmap='viridis',
                          s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.plot(image_indices, df_agg['dy_pixels'], 'g-', alpha=0.3, linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Y Drift (pixels)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('RMSE (px)', fontsize=9)

    # Add statistics text
    mean_y = df_agg['dy_pixels'].mean()
    std_y = df_agg['dy_pixels'].std()
    ax2.text(0.02, 0.98, f'Mean: {mean_y:.2f} px\nStd: {std_y:.2f} px',
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Rotation
    ax3 = axes[2]
    scatter3 = ax3.scatter(image_indices, df_agg['rotation_degrees'],
                          c=df_agg['fit_rmse'], cmap='viridis',
                          s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.plot(image_indices, df_agg['rotation_degrees'], 'm-', alpha=0.3, linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Image Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Rotation (degrees)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('RMSE (px)', fontsize=9)

    # Add statistics text
    mean_rot = df_agg['rotation_degrees'].mean()
    std_rot = df_agg['rotation_degrees'].std()
    ax3.text(0.02, 0.98, f'Mean: {mean_rot:.4f}°\nStd: {std_rot:.4f}°',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / f"drift_plot_{folder_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_file

def calculate_rotation_from_positions(
    ref_positions: np.ndarray,
    curr_positions: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate rotation angle between two sets of particle positions using rigid transformation.

    Uses a simplified Procrustes-like approach:
    1. Center both point sets at origin
    2. Calculate optimal rotation using SVD
    3. Return rotation angle in radians

    Args:
        ref_positions: Nx2 array of reference positions [[x1, y1], [x2, y2], ...]
        curr_positions: Nx2 array of current positions [[x1, y1], [x2, y2], ...]

    Returns:
        Tuple of (rotation_radians, rotation_degrees, dx, dy)
        - rotation_radians: rotation angle in radians
        - rotation_degrees: rotation angle in degrees
        - dx, dy: translation after removing rotation
    """
    if len(ref_positions) < 2 or len(curr_positions) < 2:
        # Need at least 2 points for rotation
        return 0.0, 0.0, 0.0, 0.0

    if len(ref_positions) != len(curr_positions):
        return 0.0, 0.0, 0.0, 0.0

    # Center both point sets
    ref_center = np.mean(ref_positions, axis=0)
    curr_center = np.mean(curr_positions, axis=0)

    ref_centered = ref_positions - ref_center
    curr_centered = curr_positions - curr_center

    # Calculate cross-covariance matrix H = curr_centered^T * ref_centered
    H = curr_centered.T @ ref_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Optimal rotation matrix R = U * V^T
    R = U @ Vt

    # Ensure proper rotation (det(R) = 1, not -1 for reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Extract rotation angle from rotation matrix
    # R = [[cos(θ), -sin(θ)],
    #      [sin(θ),  cos(θ)]]
    rotation_radians = np.arctan2(R[1, 0], R[0, 0])
    rotation_degrees = np.degrees(rotation_radians)

    # Calculate translation (difference in centroids)
    dx = curr_center[0] - ref_center[0]
    dy = curr_center[1] - ref_center[1]

    return rotation_radians, rotation_degrees, dx, dy

def aggregate_particle_drift(args: Tuple[str, str, Path, int]) -> Optional[Dict[str, Any]]:
    """
    Aggregate per-particle drift data to per-image drift data.
    Designed to work with multiprocessing.

    Args:
        args: Tuple of (csv_file_path, csv_file_name, output_dir, set_index)

    Returns:
        Dictionary with output info {'output_csv_path', 'output_csv_name', 'set_index'} or None if failed
    """
    csv_file_path, csv_file_name, output_dir, set_index = args
    csv_file = Path(csv_file_path)

    # Print start message with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_worker_message(f"[{timestamp}] Set {set_index}: Processing {csv_file_name[:50]}...")

    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Check required columns
        required_cols = ['filename', 'particle_id', 'image_index', 'success',
                        'drift_x', 'drift_y', 'rotation_angle']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Error - Missing columns: {missing_cols}")
            return None

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Loaded {len(df)} records, {df['image_index'].max() + 1} images, {df['particle_id'].nunique()} particles")

        # Filter only successful fits
        df_success = df[df['success'] == True].copy()
        success_rate = len(df_success) / len(df) * 100 if len(df) > 0 else 0
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Successful fits: {len(df_success)}/{len(df)} ({success_rate:.1f}%)")

        if len(df_success) == 0:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Error - No successful fits found!")
            return None

        aggregated_data = []

        # Get unique filenames in order
        unique_files = df['filename'].unique()

        # Store reference frame particle positions (first image with successful fits)
        reference_positions = None
        reference_particle_ids = None

        for idx, filename in enumerate(unique_files):
            # Get all successful particle measurements for this image
            img_data = df_success[df_success['filename'] == filename]

            if len(img_data) == 0:
                # No successful particles for this image - skip or use NaN
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_worker_message(f"[{timestamp}] Set {set_index}: Warning - No successful fits for image {idx} ({filename})")
                continue

            # Count particles used
            n_particles = len(img_data)

            # Mark first image as reference frame
            is_reference = (idx == 0)

            # Check minimum particle count for reliable statistics
            if n_particles < 2:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_worker_message(f"[{timestamp}] Set {set_index}: Warning - Only {n_particles} particle(s) in image {idx}, rotation may be unreliable")

            # Extract particle positions for rotation calculation
            particle_ids = img_data['particle_id'].values
            current_positions = img_data[['center_x', 'center_y']].values

            # Initialize reference frame on first successful image
            if reference_positions is None:
                reference_positions = current_positions.copy()
                reference_particle_ids = particle_ids.copy()

                # First frame: no drift or rotation
                median_drift_x = 0.0
                median_drift_y = 0.0
                rotation_degrees = 0.0
                rotation_radians = 0.0
                rmse = 0.0
            else:
                # Find matching particles between reference and current frame
                matched_ref_pos = []
                matched_curr_pos = []

                for pid in particle_ids:
                    if pid in reference_particle_ids:
                        ref_idx = np.where(reference_particle_ids == pid)[0][0]
                        curr_idx = np.where(particle_ids == pid)[0][0]
                        matched_ref_pos.append(reference_positions[ref_idx])
                        matched_curr_pos.append(current_positions[curr_idx])

                if len(matched_ref_pos) >= 2:
                    # Calculate rotation from matched particle positions
                    matched_ref_pos = np.array(matched_ref_pos)
                    matched_curr_pos = np.array(matched_curr_pos)

                    rotation_radians, rotation_degrees, dx_from_rot, dy_from_rot = \
                        calculate_rotation_from_positions(matched_ref_pos, matched_curr_pos)

                    # Use translation from rotation calculation (more accurate)
                    median_drift_x = dx_from_rot
                    median_drift_y = dy_from_rot

                    # Calculate RMSE as measure of drift consistency
                    # (residual after rigid transformation)
                    # Apply rotation to reference positions
                    cos_theta = np.cos(rotation_radians)
                    sin_theta = np.sin(rotation_radians)
                    rot_matrix = np.array([[cos_theta, -sin_theta],
                                          [sin_theta, cos_theta]])

                    ref_center = np.mean(matched_ref_pos, axis=0)
                    curr_center = np.mean(matched_curr_pos, axis=0)

                    # Transform reference positions
                    ref_centered = matched_ref_pos - ref_center
                    ref_rotated = (rot_matrix @ ref_centered.T).T
                    ref_transformed = ref_rotated + curr_center

                    # Calculate residuals
                    residuals = matched_curr_pos - ref_transformed
                    rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

                else:
                    # Not enough matched particles - fall back to simple median
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_worker_message(f"[{timestamp}] Set {set_index}: Warning - Only {len(matched_ref_pos)} matched particle(s) in image {idx}, using simple drift")
                    median_drift_x = img_data['drift_x'].median()
                    median_drift_y = img_data['drift_y'].median()
                    rotation_degrees = 0.0
                    rotation_radians = 0.0

                    # Calculate RMSE from drift spread
                    drift_residuals_x = img_data['drift_x'] - median_drift_x
                    drift_residuals_y = img_data['drift_y'] - median_drift_y
                    rmse = np.sqrt(np.mean(drift_residuals_x**2 + drift_residuals_y**2))

            aggregated_data.append({
                'filename': filename,
                'dx_pixels': median_drift_x,
                'dy_pixels': median_drift_y,
                'rotation_degrees': rotation_degrees,
                'is_reference_frame': is_reference,
                'n_particles_used': n_particles,
                'fit_rmse': rmse
            })

        # Create aggregated DataFrame
        df_agg = pd.DataFrame(aggregated_data)

        # Generate output filename: drift_{original_csv_name}
        # E.g., example_data1_xn9.csv -> drift_example_data1_xn9.csv
        output_csv_name = f"drift_{csv_file_name}"
        output_file = output_dir / output_csv_name

        # Extract base name without extension for plot filename
        base_name = csv_file.stem  # e.g., example_data1_xn9

        # Save to CSV
        df_agg.to_csv(output_file, index=False)

        # Print statistics
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Saved CSV: {output_file.name}")
        log_worker_message(f"[{timestamp}] Set {set_index}: Images: {len(df_agg)}, Avg particles/image: {df_agg['n_particles_used'].mean():.1f}")
        log_worker_message(f"[{timestamp}] Set {set_index}: Drift X: {df_agg['dx_pixels'].min():.2f} to {df_agg['dx_pixels'].max():.2f} px")
        log_worker_message(f"[{timestamp}] Set {set_index}: Drift Y: {df_agg['dy_pixels'].min():.2f} to {df_agg['dy_pixels'].max():.2f} px")

        # Generate drift plot with updated naming
        try:
            plot_file = plot_drift_data(df_agg, output_dir, base_name)
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Saved plot: {plot_file.name}")
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Warning - Could not create plot: {e}")

        # Return dictionary with output info
        return {
            'output_csv_path': str(output_file.absolute()),
            'output_csv_name': output_csv_name,
            'set_index': set_index
        }

    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_summary_all_sets(csv_files, output_dir):
    """
    Create a summary plot showing drift from all image sets.

    Args:
        csv_files: List of paths to drift_analysis_*.csv files
        output_dir: Directory to save summary plot
    """
    if not csv_files:
        return None

    num_sets = len(csv_files)
    fig, axes = plt.subplots(num_sets, 3, figsize=(18, 5 * num_sets))

    # Handle case of single set
    if num_sets == 1:
        axes = axes.reshape(1, -1)

    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            folder_name = csv_file.stem.replace('drift_analysis_', '')
            image_indices = np.arange(len(df))

            # X drift plot
            ax_x = axes[idx, 0]
            ax_x.plot(image_indices, df['dx_pixels'], 'b-', linewidth=1.5, alpha=0.7)
            ax_x.fill_between(image_indices, df['dx_pixels'], 0, alpha=0.2)
            ax_x.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax_x.set_ylabel('X Drift (px)', fontsize=10, fontweight='bold')
            ax_x.set_title(f'{folder_name[:40]}', fontsize=10)
            ax_x.grid(True, alpha=0.3)

            # Y drift plot
            ax_y = axes[idx, 1]
            ax_y.plot(image_indices, df['dy_pixels'], 'g-', linewidth=1.5, alpha=0.7)
            ax_y.fill_between(image_indices, df['dy_pixels'], 0, alpha=0.2, color='green')
            ax_y.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax_y.set_ylabel('Y Drift (px)', fontsize=10, fontweight='bold')
            ax_y.set_title(f'Set {idx}', fontsize=10)
            ax_y.grid(True, alpha=0.3)

            # Rotation plot
            ax_rot = axes[idx, 2]
            ax_rot.plot(image_indices, df['rotation_degrees'], 'm-', linewidth=1.5, alpha=0.7)
            ax_rot.fill_between(image_indices, df['rotation_degrees'], 0, alpha=0.2, color='magenta')
            ax_rot.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax_rot.set_ylabel('Rotation (°)', fontsize=10, fontweight='bold')
            ax_rot.set_title(f'{len(df)} images', fontsize=10)
            ax_rot.grid(True, alpha=0.3)

            # Add X labels only to bottom row
            if idx == num_sets - 1:
                ax_x.set_xlabel('Image Index', fontsize=10, fontweight='bold')
                ax_y.set_xlabel('Image Index', fontsize=10, fontweight='bold')
                ax_rot.set_xlabel('Image Index', fontsize=10, fontweight='bold')

        except Exception as e:
            print(f"  Warning: Could not plot {csv_file.name}: {e}")
            continue

    plt.suptitle('Drift Summary - All Image Sets', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save summary plot
    summary_file = output_dir / "drift_summary_all_sets.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()

    return summary_file

def main():
    """Main function to run aggregate particle drift analysis."""
    logger.info("=== Aggregate Particle Drift Data ===\n")

    # Determine JSON file location (identical logic to parallel_drift_analysis.py)
    json_file = None

    if len(sys.argv) >= 2:
        # User provided JSON path manually
        json_file = Path(sys.argv[1])
        if not json_file.exists():
            print(f"Error: File not found: {json_file}")
            sys.exit(1)
        print(f"Using user-specified JSON file: {json_file}")
    else:
        # Try to auto-discover JSON in scripts_output folder (parent directory)
        scripts_output_dir = Path('../scripts_output')

        if not scripts_output_dir.exists():
            print("Error: 'scripts_output' folder not found!")
            print("\nPlease either:")
            print("  1. Create the 'scripts_output' folder and place your JSON file there")
            print("  2. Pass the JSON file path manually:")
            print("     python aggregate_particle_drift.py <path_to_json_file>")
            sys.exit(1)

        # Find all JSON files in scripts_output
        json_files = list(scripts_output_dir.glob('*.json'))

        if len(json_files) == 0:
            print(f"Error: No JSON files found in '{scripts_output_dir}/'")
            print("\nPlease pass the JSON file path manually:")
            print("  python aggregate_particle_drift.py <path_to_json_file>")
            sys.exit(1)
        elif len(json_files) == 1:
            json_file = json_files[0]
            print(f"Auto-discovered JSON file: {json_file.name}")
        else:
            print(f"Error: Multiple JSON files found in '{scripts_output_dir}/':")
            for jf in json_files:
                print(f"  - {jf.name}")
            print("\nPlease specify which JSON file to use:")
            print("  python aggregate_particle_drift.py <path_to_json_file>")
            sys.exit(1)

    # Load JSON
    logger.info(f"Loading selections from: {json_file}")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    image_sets = data.get('image_sets', [])

    if not image_sets:
        print("No image sets found in JSON file!")
        sys.exit(1)

    # Extract CSV file paths from JSON
    csv_info_list = []
    for i, img_set in enumerate(image_sets):
        csv_file_path = img_set.get('csv_file_path')
        csv_file_name = img_set.get('csv_file_name')

        if not csv_file_path or not csv_file_name:
            print(f"Warning: Image set {i} missing CSV file information, skipping...")
            continue

        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_file_path}, skipping...")
            continue

        csv_info_list.append({
            'csv_file_path': csv_file_path,
            'csv_file_name': csv_file_name,
            'set_index': i
        })

    if not csv_info_list:
        print("Error: No valid CSV files found in JSON!")
        sys.exit(1)

    logger.info(f"Found {len(csv_info_list)} CSV file(s) to process\n")

    # Print summary
    for info in csv_info_list:
        logger.info(f"Set {info['set_index']}: {info['csv_file_name']}")

    # Create output directory (parent directory)
    output_dir = Path('../scripts_output') / 'particle_drift'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\n" + "="*60)
    logger.info("Starting parallel processing...")
    logger.info("="*60 + "\n")

    # Create argument tuples for parallel processing
    process_args = [
        (info['csv_file_path'], info['csv_file_name'], output_dir, info['set_index'])
        for info in csv_info_list
    ]

    # Determine number of workers
    num_workers = min(len(csv_info_list), mp.cpu_count())
    logger.info(f"Using {num_workers} parallel workers\n")

    # Run parallel processing with centralized progress bar
    import time
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = []
        with tqdm(total=len(csv_info_list), desc="Aggregating drift data", unit="file", ncols=100, file=sys.stderr) as pbar:
            for result in pool.imap_unordered(aggregate_particle_drift, process_args):
                results.append(result)
                pbar.update(1)

    end_time = time.time()

    logger.info("\n" + "="*60)
    logger.info("Parallel processing complete!")
    logger.info("="*60 + "\n")

    # Separate successful and failed results
    successful_outputs = [r for r in results if r is not None]
    failed_count = len(results) - len(successful_outputs)

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Files processed: {len(results)}")
    logger.info(f"Successful: {len(successful_outputs)}")
    logger.info(f"Failed: {failed_count}\n")

    if successful_outputs:
        logger.info("Generated drift analysis files:")
        for result in successful_outputs:
            logger.info(f"  ✓ {result['output_csv_name']}")

    # Create summary plot if multiple sets were processed
    if len(successful_outputs) > 1:
        logger.info(f"\nCreating summary plot for all {len(successful_outputs)} sets...")
        try:
            # Get list of output CSV paths for plotting
            output_csv_paths = [Path(r['output_csv_path']) for r in successful_outputs]
            summary_plot = plot_summary_all_sets(output_csv_paths, output_dir)
            if summary_plot:
                logger.info(f"  ✓ Summary plot saved: {summary_plot.name}")
        except Exception as e:
            logger.error(f"  Warning: Could not create summary plot: {e}")
            log_exception(logger, e, "Summary plot error")

    # Update JSON file with drift CSV information
    if successful_outputs:
        logger.info("\nUpdating JSON file with drift CSV information...")
        try:
            # Map output info by set_index
            output_info_by_index = {r['set_index']: r for r in successful_outputs}

            # Update image_sets with drift CSV info
            for img_set in data['image_sets']:
                set_idx = data['image_sets'].index(img_set)
                if set_idx in output_info_by_index:
                    output_info = output_info_by_index[set_idx]
                    img_set['drift_csv_file_path'] = output_info['output_csv_path']
                    img_set['drift_csv_file_name'] = output_info['output_csv_name']

            # Save updated JSON back to file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"✓ JSON file updated: {json_file.name}")
            logger.info("  Added 'drift_csv_file_path' and 'drift_csv_file_name' to each image set")

        except Exception as e:
            logger.error(f"Warning: Could not update JSON file: {e}")
            log_exception(logger, e, "JSON update error")

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\n=== DONE ===")

    return failed_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
