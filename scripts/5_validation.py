"""
Validation of Aligned Images
Validates alignment quality by re-running drift analysis on aligned images.

This script processes aligned images to measure residual drift after alignment.
It performs the same Gaussian fitting and drift calculation as scripts 2 & 3,
but on the aligned images, allowing validation that alignment was successful.

Requirements:
pip install pandas numpy scipy tifffile tqdm matplotlib

Usage:
python 5_validation.py [optional: path_to_json_file]

Examples:
# Auto-discover JSON in scripts_output/ folder (must be only one JSON file)
python 5_validation.py

# Manually specify JSON file path
python 5_validation.py particle_selections_20251016_112754.json
python 5_validation.py scripts_output/particle_selections_20251016_112754.json

Output:
- Per-particle CSV: scripts_output/validation/{folder_name}_{3char}.csv
- Aggregated CSV: scripts_output/validation/drift_{folder_name}_{3char}.csv
- Comparison plots: scripts_output/validation/validation_plot_{folder_name}.png
- Summary plot: scripts_output/validation/validation_summary_all_sets.png
- JSON file updated with validation file paths

Expected Results:
- Residual drift should be near zero (< 0.1 pixels typically)
- Residual rotation should be near zero (< 0.01 degrees typically)
- This validates that alignment was successful!
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
from scipy.optimize import curve_fit
import tifffile
from logging_utils import setup_logger, log_worker_message, log_exception

# Setup centralized logger
logger = setup_logger('Step5_Validation')
import random
import string

# ============================================================================
# CORE FUNCTIONS (Copied from Scripts 2 & 3)
# ============================================================================

def gaussian_2d(coords: Tuple[np.ndarray, np.ndarray],
               amp: float, x0: float, y0: float,
               sigma_x: float, sigma_y: float,
               theta: float, offset: float) -> np.ndarray:
    """
    2D Gaussian model with rotation.
    Identical to script 2.
    """
    X, Y = coords

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_rot = (X - x0) * cos_theta + (Y - y0) * sin_theta
    y_rot = -(X - x0) * sin_theta + (Y - y0) * cos_theta

    gaussian = amp * np.exp(-(x_rot**2)/(2*sigma_x**2) - (y_rot**2)/(2*sigma_y**2)) + offset

    return gaussian.ravel()

def fit_gaussian_to_region(region: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
    """
    Fit a 2D Gaussian to the given image region.
    Identical to script 2.
    """
    h, w = region.shape
    if h < 5 or w < 5:
        return None

    # Create coordinate meshgrids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # Initial parameter guesses
    background = np.median(region)
    min_val = np.min(region)
    max_val = np.max(region)

    # Check if dark or bright spot
    if (background - min_val) > (max_val - background):
        amp_guess = min_val - background
        offset_guess = background
    else:
        amp_guess = max_val - background
        offset_guess = background

    # Find approximate center
    if amp_guess < 0:
        center_idx = np.unravel_index(np.argmin(region), region.shape)
    else:
        center_idx = np.unravel_index(np.argmax(region), region.shape)

    y0_guess, x0_guess = center_idx
    sigma_guess = min(w, h) / 6.0

    # Initial parameters
    p0 = [amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0.0, offset_guess]

    # Parameter bounds
    bounds = (
        [-np.inf, 0, 0, 0.5, 0.5, -np.pi, -np.inf],
        [np.inf, w-1, h-1, w*2, h*2, np.pi, np.inf]
    )

    try:
        popt, pcov = curve_fit(
            gaussian_2d, (X, Y), region.ravel(),
            p0=p0, bounds=bounds, maxfev=2000
        )

        amp, x0, y0, sigma_x, sigma_y, theta, offset = popt

        # Calculate R²
        fitted_data = gaussian_2d((X, Y), *popt).reshape(region.shape)
        residuals = region - fitted_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((region - np.mean(region))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Convert to global coordinates
        global_x = x0 + bbox[0]
        global_y = y0 + bbox[1]

        return {
            'success': True,
            'amplitude': amp,
            'center_x': global_x,
            'center_y': global_y,
            'local_center_x': x0,
            'local_center_y': y0,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'rotation_angle': theta,
            'background': offset,
            'r_squared': r_squared,
            'bbox': bbox
        }

    except Exception as e:
        return None

def calculate_rotation_from_positions(
    ref_positions: np.ndarray,
    curr_positions: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate rotation angle between two sets of particle positions using rigid transformation.
    Identical to script 3.

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

# ============================================================================
# VALIDATION-SPECIFIC FUNCTIONS
# ============================================================================

def validate_aligned_image_set(args: Tuple[int, Dict[str, Any], Optional[mp.Queue]]) -> Dict[str, Any]:
    """
    Validate a single aligned image set by re-running Gaussian fitting and drift analysis.
    Similar to process_single_image_set from script 2, but processes aligned images.

    Args:
        args: (set_index, image_set_data, progress_queue)
            - set_index: Index of this image set
            - image_set_data: Dictionary with aligned_folder_path and selected_particles
            - progress_queue: Optional queue for progress updates

    Returns:
        Dictionary with validation results and metadata
    """
    set_index, image_set_data, progress_queue = args

    aligned_folder_path = Path(image_set_data['aligned_folder_path'])
    selected_particles = image_set_data['selected_particles']
    folder_name = Path(image_set_data['folder_path']).name

    # Get all TIF files from aligned folder
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []
    for pattern in tif_patterns:
        tif_files.extend(list(aligned_folder_path.glob(pattern)))
    tif_files.sort()

    if not tif_files:
        return {
            'set_index': set_index,
            'folder_name': folder_name,
            'success': False,
            'error': f'No TIF files found in {aligned_folder_path}'
        }

    # Store results for all images
    all_results = []
    total_operations = len(tif_files) * len(selected_particles)

    # Store first frame positions for drift calculation
    first_frame_positions = {}

    # Print start message with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_worker_message(f"[{timestamp}] Set {set_index} (Validation: {folder_name[:40]}): Starting {total_operations} fits...")

    # Process aligned images ONE BY ONE (memory efficient)
    for img_idx, tif_file in enumerate(tif_files):
        try:
            # Load single aligned image
            image = tifffile.imread(str(tif_file))

            # Handle images with singleton channel dimension (e.g., shape (500, 500, 1))
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze()  # Remove singleton dimension
            elif image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze()  # Handle channel-first format

            if image.ndim != 2:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_worker_message(f"[{timestamp}] Set {set_index}: Skipping {tif_file.name} - not grayscale (shape: {image.shape})")
                continue

            # Process each particle in this image
            for particle in selected_particles:
                bbox = tuple(particle['bbox'])  # [x0, y0, x1, y1]
                particle_id = particle['particle_id']

                x0, y0, x1, y1 = bbox

                # Ensure coordinates are within bounds
                x0 = max(0, int(x0))
                y0 = max(0, int(y0))
                x1 = min(image.shape[1], int(x1))
                y1 = min(image.shape[0], int(y1))

                if x1 <= x0 or y1 <= y0:
                    continue

                # Extract region
                region = image[y0:y1, x0:x1].copy()

                # Fit Gaussian
                fit_result = fit_gaussian_to_region(region, (x0, y0, x1, y1))

                # Build result record
                result = {
                    'filename': tif_file.name,
                    'particle_id': particle_id,
                    'image_index': img_idx,
                    'bbox': bbox,
                    'success': fit_result is not None and fit_result.get('success', False)
                }

                if result['success']:
                    result.update(fit_result)

                    # Track first frame position for drift calculation
                    if img_idx == 0:
                        first_frame_positions[particle_id] = {
                            'center_x': fit_result['center_x'],
                            'center_y': fit_result['center_y']
                        }

                    # Calculate residual drift (should be near zero after alignment)
                    if particle_id in first_frame_positions:
                        dx = fit_result['center_x'] - first_frame_positions[particle_id]['center_x']
                        dy = fit_result['center_y'] - first_frame_positions[particle_id]['center_y']
                        drift_magnitude = np.sqrt(dx**2 + dy**2)

                        result['drift_x'] = dx
                        result['drift_y'] = dy
                        result['drift_magnitude'] = drift_magnitude
                    else:
                        result['drift_x'] = None
                        result['drift_y'] = None
                        result['drift_magnitude'] = None

                all_results.append(result)

            # Image processing done - image will be garbage collected

        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Error processing {tif_file.name}: {e}")
            continue

    # Calculate statistics
    successful_fits = sum(1 for r in all_results if r['success'])
    success_rate = (successful_fits / len(all_results) * 100) if all_results else 0

    # Print completion message with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_worker_message(f"[{timestamp}] Set {set_index} (Validation: {folder_name[:40]}): Completed {successful_fits}/{len(all_results)} fits ({success_rate:.1f}%)")

    return {
        'set_index': set_index,
        'folder_name': folder_name,
        'aligned_folder_path': str(aligned_folder_path),
        'success': True,
        'results': all_results,
        'total_images': len(tif_files),
        'total_particles': len(selected_particles),
        'total_fits_attempted': len(all_results),
        'successful_fits': successful_fits,
        'success_rate': success_rate,
        'first_frame_positions': first_frame_positions
    }

def export_validation_to_csv(val_result: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """
    Export validation results for one image set to CSV.
    Similar to export_results_to_csv from script 2.

    Args:
        val_result: Result dictionary from validate_aligned_image_set
        output_dir: Directory to save CSV file (scripts_output/validation/)

    Returns:
        Dictionary with 'csv_file_path' and 'csv_file_name', or None if failed
    """
    if not val_result['success']:
        print(f"Cannot export - validation failed for set {val_result['set_index']}")
        return None

    # Create output directory: scripts_output/validation/
    output_subdir = output_dir / 'validation'
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Get folder name from image set
    folder_name = val_result['folder_name']

    # Generate unique 3-character identifier
    unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=3))

    # Generate filename: foldername_uniqueID.csv
    csv_filename = output_subdir / f"{folder_name}_{unique_id}.csv"

    # Ensure uniqueness (very unlikely collision, but check anyway)
    while csv_filename.exists():
        unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
        csv_filename = output_subdir / f"{folder_name}_{unique_id}.csv"

    # CSV headers
    headers = [
        'filename', 'particle_id', 'image_index',
        'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1',
        'success', 'center_x', 'center_y',
        'drift_x', 'drift_y', 'drift_magnitude',
        'amplitude', 'sigma_x', 'sigma_y',
        'rotation_angle', 'background', 'r_squared'
    ]

    try:
        import csv
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for result in val_result['results']:
                row = [
                    result['filename'],
                    result['particle_id'],
                    result['image_index'],
                    result['bbox'][0],  # x0
                    result['bbox'][1],  # y0
                    result['bbox'][2],  # x1
                    result['bbox'][3],  # y1
                    result['success']
                ]

                if result['success']:
                    row.extend([
                        f"{result['center_x']:.3f}",
                        f"{result['center_y']:.3f}",
                        f"{result['drift_x']:.3f}" if result['drift_x'] is not None else '',
                        f"{result['drift_y']:.3f}" if result['drift_y'] is not None else '',
                        f"{result['drift_magnitude']:.3f}" if result['drift_magnitude'] is not None else '',
                        f"{result['amplitude']:.3f}",
                        f"{result['sigma_x']:.3f}",
                        f"{result['sigma_y']:.3f}",
                        f"{result['rotation_angle']:.6f}",
                        f"{result['background']:.3f}",
                        f"{result['r_squared']:.6f}"
                    ])
                else:
                    row.extend(['', '', '', '', '', '', '', '', '', '', ''])

                writer.writerow(row)

        print(f"[Set {val_result['set_index']}] Validation CSV: {csv_filename.name}")

        # Return dictionary with both path and name
        return {
            'csv_file_path': str(csv_filename.absolute()),
            'csv_file_name': csv_filename.name,
            'unique_id': unique_id
        }

    except Exception as e:
        print(f"Error exporting validation results for set {val_result['set_index']}: {e}")
        return None

def aggregate_validation_drift(args: Tuple[str, str, Path, int]) -> Optional[Dict[str, Any]]:
    """
    Aggregate per-particle validation data to per-image drift data.
    Similar to aggregate_particle_drift from script 3.

    Args:
        args: Tuple of (csv_file_path, csv_file_name, output_dir, set_index)

    Returns:
        Dictionary with output info or None if failed
    """
    csv_file_path, csv_file_name, output_dir, set_index = args
    csv_file = Path(csv_file_path)

    # Print start message with timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_worker_message(f"[{timestamp}] Set {set_index}: Aggregating validation data for {csv_file_name[:50]}...")

    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Check required columns
        required_cols = ['filename', 'particle_id', 'image_index', 'success',
                        'drift_x', 'drift_y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_worker_message(f"[{timestamp}] Set {set_index}: Error - Missing columns: {missing_cols}")
            return None

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Loaded {len(df)} records, {df['image_index'].max() + 1} images")

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

        # Store reference frame particle positions
        reference_positions = None
        reference_particle_ids = None

        for idx, filename in enumerate(unique_files):
            # Get all successful particle measurements for this image
            img_data = df_success[df_success['filename'] == filename]

            if len(img_data) == 0:
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
                log_worker_message(f"[{timestamp}] Set {set_index}: Warning - Only {n_particles} particle(s) in image {idx}")

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

                    # Use translation from rotation calculation
                    median_drift_x = dx_from_rot
                    median_drift_y = dy_from_rot

                    # Calculate RMSE as residual after rigid transformation
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
                    log_worker_message(f"[{timestamp}] Set {set_index}: Warning - Only {len(matched_ref_pos)} matched particle(s)")
                    median_drift_x = img_data['drift_x'].median()
                    median_drift_y = img_data['drift_y'].median()
                    rotation_degrees = 0.0

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
        output_csv_name = f"drift_{csv_file_name}"
        output_file = output_dir / output_csv_name

        # Extract base name without extension for plot filename
        base_name = csv_file.stem

        # Save to CSV
        df_agg.to_csv(output_file, index=False)

        # Print statistics
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Aggregated CSV: {output_file.name}")
        log_worker_message(f"[{timestamp}] Set {set_index}: Images: {len(df_agg)}, Avg particles/image: {df_agg['n_particles_used'].mean():.1f}")
        log_worker_message(f"[{timestamp}] Set {set_index}: Residual Drift X: {df_agg['dx_pixels'].min():.4f} to {df_agg['dx_pixels'].max():.4f} px")
        log_worker_message(f"[{timestamp}] Set {set_index}: Residual Drift Y: {df_agg['dy_pixels'].min():.4f} to {df_agg['dy_pixels'].max():.4f} px")
        log_worker_message(f"[{timestamp}] Set {set_index}: Residual Rotation: {df_agg['rotation_degrees'].min():.6f} to {df_agg['rotation_degrees'].max():.6f}°")

        # Return dictionary with output info
        return {
            'output_csv_path': str(output_file.absolute()),
            'output_csv_name': output_csv_name,
            'set_index': set_index,
            'df_agg': df_agg,
            'base_name': base_name
        }

    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_worker_message(f"[{timestamp}] Set {set_index}: Error aggregating {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_validation_comparison(original_csv_path: Path, validation_csv_path: Path,
                               output_dir: Path, folder_name: str) -> Optional[Path]:
    """
    Create comparison plot showing drift before and after alignment.

    Args:
        original_csv_path: Path to original drift CSV (from script 3)
        validation_csv_path: Path to validation drift CSV
        output_dir: Directory to save plot
        folder_name: Name for output file

    Returns:
        Path to saved plot file or None if failed
    """
    try:
        # Load both CSV files
        df_original = pd.read_csv(original_csv_path)
        df_validation = pd.read_csv(validation_csv_path)

        # Create figure with 3 rows x 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        image_indices_orig = np.arange(len(df_original))
        image_indices_val = np.arange(len(df_validation))

        # Row 1: X Drift
        # Before alignment
        ax1 = axes[0, 0]
        ax1.plot(image_indices_orig, df_original['dx_pixels'], 'b-o', linewidth=2, markersize=6, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax1.set_ylabel('X Drift (pixels)', fontsize=12, fontweight='bold')
        ax1.set_title('BEFORE Alignment: X Drift', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        mean_x_orig = df_original['dx_pixels'].mean()
        std_x_orig = df_original['dx_pixels'].std()
        max_x_orig = df_original['dx_pixels'].abs().max()
        ax1.text(0.02, 0.98, f'Mean: {mean_x_orig:.3f} px\nStd: {std_x_orig:.3f} px\nMax: {max_x_orig:.3f} px',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # After alignment
        ax2 = axes[0, 1]
        ax2.plot(image_indices_val, df_validation['dx_pixels'], 'g-o', linewidth=2, markersize=6, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_ylabel('X Drift (pixels)', fontsize=12, fontweight='bold')
        ax2.set_title('AFTER Alignment: X Drift (Validation)', fontsize=13, fontweight='bold', color='green')
        ax2.grid(True, alpha=0.3)
        mean_x_val = df_validation['dx_pixels'].mean()
        std_x_val = df_validation['dx_pixels'].std()
        max_x_val = df_validation['dx_pixels'].abs().max()
        improvement_x = (1 - max_x_val/max_x_orig) * 100 if max_x_orig > 0 else 100
        ax2.text(0.02, 0.98, f'Mean: {mean_x_val:.4f} px\nStd: {std_x_val:.4f} px\nMax: {max_x_val:.4f} px\nImprovement: {improvement_x:.1f}%',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Row 2: Y Drift
        # Before alignment
        ax3 = axes[1, 0]
        ax3.plot(image_indices_orig, df_original['dy_pixels'], 'b-o', linewidth=2, markersize=6, alpha=0.7)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax3.set_ylabel('Y Drift (pixels)', fontsize=12, fontweight='bold')
        ax3.set_title('BEFORE Alignment: Y Drift', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        mean_y_orig = df_original['dy_pixels'].mean()
        std_y_orig = df_original['dy_pixels'].std()
        max_y_orig = df_original['dy_pixels'].abs().max()
        ax3.text(0.02, 0.98, f'Mean: {mean_y_orig:.3f} px\nStd: {std_y_orig:.3f} px\nMax: {max_y_orig:.3f} px',
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # After alignment
        ax4 = axes[1, 1]
        ax4.plot(image_indices_val, df_validation['dy_pixels'], 'g-o', linewidth=2, markersize=6, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax4.set_ylabel('Y Drift (pixels)', fontsize=12, fontweight='bold')
        ax4.set_title('AFTER Alignment: Y Drift (Validation)', fontsize=13, fontweight='bold', color='green')
        ax4.grid(True, alpha=0.3)
        mean_y_val = df_validation['dy_pixels'].mean()
        std_y_val = df_validation['dy_pixels'].std()
        max_y_val = df_validation['dy_pixels'].abs().max()
        improvement_y = (1 - max_y_val/max_y_orig) * 100 if max_y_orig > 0 else 100
        ax4.text(0.02, 0.98, f'Mean: {mean_y_val:.4f} px\nStd: {std_y_val:.4f} px\nMax: {max_y_val:.4f} px\nImprovement: {improvement_y:.1f}%',
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Row 3: Rotation
        # Before alignment
        ax5 = axes[2, 0]
        ax5.plot(image_indices_orig, df_original['rotation_degrees'], 'b-o', linewidth=2, markersize=6, alpha=0.7)
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax5.set_xlabel('Image Index', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Rotation (degrees)', fontsize=12, fontweight='bold')
        ax5.set_title('BEFORE Alignment: Rotation', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        mean_rot_orig = df_original['rotation_degrees'].mean()
        std_rot_orig = df_original['rotation_degrees'].std()
        max_rot_orig = df_original['rotation_degrees'].abs().max()
        ax5.text(0.02, 0.98, f'Mean: {mean_rot_orig:.4f}°\nStd: {std_rot_orig:.4f}°\nMax: {max_rot_orig:.4f}°',
                transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # After alignment
        ax6 = axes[2, 1]
        ax6.plot(image_indices_val, df_validation['rotation_degrees'], 'g-o', linewidth=2, markersize=6, alpha=0.7)
        ax6.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax6.set_xlabel('Image Index', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Rotation (degrees)', fontsize=12, fontweight='bold')
        ax6.set_title('AFTER Alignment: Rotation (Validation)', fontsize=13, fontweight='bold', color='green')
        ax6.grid(True, alpha=0.3)
        mean_rot_val = df_validation['rotation_degrees'].mean()
        std_rot_val = df_validation['rotation_degrees'].std()
        max_rot_val = df_validation['rotation_degrees'].abs().max()
        improvement_rot = (1 - max_rot_val/max_rot_orig) * 100 if max_rot_orig > 0 else 100
        ax6.text(0.02, 0.98, f'Mean: {mean_rot_val:.6f}°\nStd: {std_rot_val:.6f}°\nMax: {max_rot_val:.6f}°\nImprovement: {improvement_rot:.1f}%',
                transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.suptitle(f'Validation: {folder_name}\nBefore vs. After Alignment',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_file = output_dir / f"validation_plot_{folder_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_file

    except Exception as e:
        print(f"Error creating validation comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run validation analysis."""
    logger.info("=== Validation of Aligned Images ===\n")

    # Determine JSON file location (identical logic to other scripts)
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
            print("\nPlease pass the JSON file path manually:")
            print("  python 5_validation.py <path_to_json_file>")
            sys.exit(1)

        # Find all JSON files in scripts_output
        json_files = list(scripts_output_dir.glob('*.json'))

        if len(json_files) == 0:
            print(f"Error: No JSON files found in '{scripts_output_dir}/'")
            print("\nPlease pass the JSON file path manually:")
            print("  python 5_validation.py <path_to_json_file>")
            sys.exit(1)
        elif len(json_files) == 1:
            json_file = json_files[0]
            print(f"Auto-discovered JSON file: {json_file.name}")
        else:
            print(f"Error: Multiple JSON files found in '{scripts_output_dir}/':")
            for jf in json_files:
                print(f"  - {jf.name}")
            print("\nPlease specify which JSON file to use:")
            print("  python 5_validation.py <path_to_json_file>")
            sys.exit(1)

    # Load JSON
    logger.info(f"Loading configuration from: {json_file}")
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

    # Check if aligned folders exist
    valid_sets = []
    for i, img_set in enumerate(image_sets):
        aligned_folder = img_set.get('aligned_folder_path')
        if not aligned_folder:
            print(f"Warning: Image set {i} missing 'aligned_folder_path', skipping...")
            continue

        aligned_path = Path(aligned_folder)
        if not aligned_path.exists():
            print(f"Warning: Aligned folder not found: {aligned_folder}, skipping...")
            continue

        valid_sets.append((i, img_set))

    if not valid_sets:
        print("Error: No valid aligned image sets found!")
        print("\nPlease run script 4 (batch_gpu_alignment.py) first to create aligned images.")
        sys.exit(1)

    logger.info(f"Found {len(valid_sets)} aligned image set(s) to validate\n")

    # Print summary
    for i, img_set in valid_sets:
        print(f"Set {i}: {Path(img_set['folder_path']).name}")
        print(f"  Aligned folder: {img_set['aligned_folder_path']}")
        print(f"  Particles: {len(img_set['selected_particles'])}")

    logger.info("\n" + "="*60)
    logger.info("Step 1: Validating aligned images (Gaussian fitting)...")
    logger.info("="*60 + "\n")

    # Create argument tuples for parallel processing
    process_args = [
        (i, img_set, None)
        for i, img_set in valid_sets
    ]

    # Determine number of workers
    num_workers = min(len(valid_sets), mp.cpu_count())
    logger.info(f"Using {num_workers} parallel workers\n")

    # Run parallel validation processing with centralized progress bar
    import time
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        val_results = []
        with tqdm(total=len(valid_sets), desc="Validating aligned images", unit="set", ncols=100, file=sys.stderr) as pbar:
            for result in pool.imap_unordered(validate_aligned_image_set, process_args):
                val_results.append(result)
                pbar.update(1)

    end_time = time.time()

    logger.info("\n" + "="*60)
    logger.info("Step 1 complete: Gaussian fitting on aligned images")
    logger.info("="*60 + "\n")

    # Export validation results to CSV (parent directory)
    output_dir = Path('../scripts_output')
    validation_csv_info_list = []

    for val_result in val_results:
        if val_result['success']:
            csv_info = export_validation_to_csv(val_result, output_dir)
            if csv_info:
                csv_info['set_index'] = val_result['set_index']
                validation_csv_info_list.append(csv_info)

    logger.info("\n" + "="*60)
    logger.info("Step 2: Aggregating validation drift data...")
    logger.info("="*60 + "\n")

    # Create argument tuples for aggregation
    aggregation_args = [
        (info['csv_file_path'], info['csv_file_name'],
         output_dir / 'validation', info['set_index'])
        for info in validation_csv_info_list
    ]

    # Run parallel aggregation with centralized progress bar
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        agg_results = []
        with tqdm(total=len(aggregation_args), desc="Aggregating validation drift", unit="file", ncols=100, file=sys.stderr) as pbar:
            for result in pool.imap_unordered(aggregate_validation_drift, aggregation_args):
                agg_results.append(result)
                pbar.update(1)

    logger.info("\n" + "="*60)
    logger.info("Step 2 complete: Aggregation finished")
    logger.info("="*60 + "\n")

    # Filter successful aggregation results
    successful_agg = [r for r in agg_results if r is not None]

    # Generate validation comparison plots
    if successful_agg:
        print("=" * 60)
        print("Step 3: Generating validation comparison plots...")
        print("=" * 60 + "\n")

        for agg_result in successful_agg:
            set_index = agg_result['set_index']

            # Find corresponding original drift CSV
            img_set_data = dict(valid_sets)[set_index]
            original_drift_csv = img_set_data.get('drift_csv_file_path')

            if not original_drift_csv or not Path(original_drift_csv).exists():
                print(f"Warning: Original drift CSV not found for set {set_index}, skipping plot...")
                continue

            folder_name = agg_result['base_name']
            validation_csv = Path(agg_result['output_csv_path'])

            plot_file = plot_validation_comparison(
                Path(original_drift_csv),
                validation_csv,
                output_dir / 'validation',
                folder_name
            )

            if plot_file:
                print(f"[Set {set_index}] ✓ Validation plot: {plot_file.name}")

    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60 + "\n")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Image sets validated: {len(val_results)}")
    logger.info(f"Successful validations: {len(successful_agg)}\n")

    for val_result in val_results:
        if val_result['success']:
            print(f"Set {val_result['set_index']}: {val_result['folder_name']}")
            print(f"  Total fits attempted: {val_result['total_fits_attempted']}")
            print(f"  Successful fits: {val_result['successful_fits']}")
            print(f"  Success rate: {val_result['success_rate']:.1f}%")
        else:
            print(f"Set {val_result['set_index']}: ERROR - {val_result.get('error', 'Unknown error')}")
        print()

    # Update JSON file with validation information
    if successful_agg:
        print("Updating JSON file with validation file paths...")
        try:
            # Map validation info by set_index
            validation_info_by_index = {}
            for val_csv_info in validation_csv_info_list:
                set_idx = val_csv_info['set_index']
                validation_info_by_index[set_idx] = val_csv_info

            agg_info_by_index = {r['set_index']: r for r in successful_agg}

            # Update image_sets with validation info
            for img_set in data['image_sets']:
                set_idx = data['image_sets'].index(img_set)
                if set_idx in validation_info_by_index and set_idx in agg_info_by_index:
                    val_info = validation_info_by_index[set_idx]
                    agg_info = agg_info_by_index[set_idx]

                    img_set['validation_csv_file_path'] = val_info['csv_file_path']
                    img_set['validation_csv_file_name'] = val_info['csv_file_name']
                    img_set['validation_drift_csv_file_path'] = agg_info['output_csv_path']
                    img_set['validation_drift_csv_file_name'] = agg_info['output_csv_name']

            # Save updated JSON back to file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ JSON file updated: {json_file.name}")
            print("  Added validation file paths to each image set")

        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")

    logger.info(f"\nValidation files saved to: {output_dir / 'validation'}")
    logger.info("\n=== VALIDATION COMPLETE ===")
    logger.info("\nIf residual drift is near zero (< 0.1 px) and rotation is near zero (< 0.01°),")
    logger.info("then the alignment was successful!")

if __name__ == "__main__":
    main()
