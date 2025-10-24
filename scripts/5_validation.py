"""
Validation Script - Re-fit Gaussians on Aligned Images (Parallelized)
Performs Gaussian fitting on aligned images to validate drift correction effectiveness.

Requirements:
pip install tifffile numpy scipy pandas tqdm matplotlib

Usage:
python3 5_validation.py <json_path>

The script loads all information from the JSON file and validates alignment by:
1. Re-fitting Gaussians on ONLY selected particles in aligned images
2. Comparing drift before/after alignment
3. Generating validation plots and statistics

Output:
- CSV: script_output/validation/aligned_particles_tracking.csv
- PNGs: Comparison plots for each particle
- Summary: Drift reduction statistics
"""

# Limit threading in numerical libraries to prevent CPU overload
# Each worker process should use only 1 thread
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import tifffile
from pathlib import Path
from scipy.optimize import curve_fit
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import sys
import time
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from logging_utils import setup_logger, log_exception

# Logger will be set up after finding JSON file
logger = None

def gaussian_2d(coords: Tuple[np.ndarray, np.ndarray],
               amp: float, x0: float, y0: float,
               sigma_x: float, sigma_y: float,
               theta: float, offset: float) -> np.ndarray:
    """
    2D Gaussian model with rotation.
    Same as Step 2.
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
    Same logic as Step 2.
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


# Global cache for loaded images (used by worker processes)
_image_cache = {}
_cache_max_size = 20  # Keep last 20 images in memory per worker (optimized for HDD)

def _load_image_cached(image_path: str) -> np.ndarray:
    """
    Load image with simple LRU-style caching.
    Each worker process maintains its own cache.
    """
    global _image_cache
    
    if image_path in _image_cache:
        return _image_cache[image_path]
    
    # Load new image
    image = tifffile.imread(image_path)
    
    # Simple cache management: if too many cached, remove oldest
    if len(_image_cache) >= _cache_max_size:
        # Remove first (oldest) item
        oldest_key = next(iter(_image_cache))
        del _image_cache[oldest_key]
    
    _image_cache[image_path] = image
    return image


def process_single_fit(work_item: Tuple[str, int, str, Dict, int]) -> Dict[str, Any]:
    """
    Worker function: process a single Gaussian fit.
    
    Args:
        work_item: (image_path, image_index, filename, particle, particle_index)
    
    Returns:
        Dictionary with fit results
    """
    image_path, image_index, filename, particle, particle_idx = work_item
    
    try:
        # Load image (with caching)
        image = _load_image_cached(image_path)
        
        if image.ndim != 2:
            return {
                'filename': filename,
                'particle_id': particle['particle_id'],
                'image_index': image_index,
                'bbox': tuple(particle['bbox']),
                'success': False,
                'error': 'Not grayscale'
            }
        
        bbox = tuple(particle['bbox'])
        particle_id = particle['particle_id']
        
        x0, y0, x1, y1 = bbox
        
        # Ensure coordinates are within bounds
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(image.shape[1], int(x1))
        y1 = min(image.shape[0], int(y1))
        
        if x1 <= x0 or y1 <= y0:
            return {
                'filename': filename,
                'particle_id': particle_id,
                'image_index': image_index,
                'bbox': bbox,
                'success': False,
                'error': 'Invalid bbox'
            }
        
        # Extract region
        region = image[y0:y1, x0:x1].copy()
        
        # Fit Gaussian
        fit_result = fit_gaussian_to_region(region, (x0, y0, x1, y1))
        
        # Build result record
        result = {
            'filename': filename,
            'particle_id': particle_id,
            'image_index': image_index,
            'bbox': bbox,
            'success': fit_result is not None and fit_result.get('success', False)
        }
        
        if result['success']:
            result.update(fit_result)
        
        return result
        
    except Exception as e:
        return {
            'filename': filename,
            'particle_id': particle['particle_id'],
            'image_index': image_index,
            'bbox': tuple(particle['bbox']),
            'success': False,
            'error': str(e)
        }


def export_validation_csv(all_results: List[Dict], output_file: Path,
                          selected_particle_ids: List[int]) -> bool:
    """
    Export validation results to CSV (same format as Step 2).

    Args:
        all_results: List of result dictionaries
        output_file: Path to save CSV
        selected_particle_ids: List of particle IDs for reference

    Returns:
        True if successful, False otherwise
    """
    headers = [
        'filename', 'particle_id', 'image_index',
        'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1',
        'success', 'center_x', 'center_y',
        'drift_x', 'drift_y', 'drift_magnitude',
        'amplitude', 'sigma_x', 'sigma_y',
        'rotation_angle', 'background', 'r_squared'
    ]

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for result in all_results:
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

        logger.info(f"✓ Validation CSV saved: {output_file.name}")
        return True

    except Exception as e:
        logger.error(f"Error exporting validation CSV: {e}")
        log_exception(logger, e, "CSV export error")
        return False


def plot_particle_comparison(original_df: pd.DataFrame, validation_df: pd.DataFrame,
                             particle_id: int, output_file: Path):
    """
    Plot before/after comparison for a single particle.

    Args:
        original_df: Original tracking data from Step 2
        validation_df: Validation tracking on aligned images
        particle_id: ID of particle to plot
        output_file: Path to save PNG
    """
    # Get data for this particle
    orig_data = original_df[original_df['particle_id'] == particle_id].sort_values('image_index')
    val_data = validation_df[validation_df['particle_id'] == particle_id].sort_values('image_index')

    if len(orig_data) == 0 or len(val_data) == 0:
        logger.warning(f"No data for particle {particle_id} in comparison")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: Drift trajectories comparison
    ax1.plot(orig_data['drift_x'], orig_data['drift_y'],
             'o-', linewidth=2, markersize=3, alpha=0.7, color='red',
             label='Original (before alignment)')
    ax1.plot(val_data['drift_x'], val_data['drift_y'],
             'o-', linewidth=2, markersize=3, alpha=0.7, color='green',
             label='After alignment')

    # Mark origin
    ax1.plot(0, 0, 'k*', markersize=20, markeredgecolor='black',
             markeredgewidth=2, label='Start (0,0)', zorder=10)

    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Drift X (pixels)', fontsize=14)
    ax1.set_ylabel('Drift Y (pixels)', fontsize=14)
    ax1.set_title(f'Particle {particle_id}: Drift Comparison', fontsize=15, weight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # RIGHT: Drift magnitude over time
    orig_mag = np.sqrt(orig_data['drift_x']**2 + orig_data['drift_y']**2)
    val_mag = np.sqrt(val_data['drift_x']**2 + val_data['drift_y']**2)

    ax2.plot(orig_data['image_index'], orig_mag,
             'o-', linewidth=2, markersize=3, alpha=0.7, color='red',
             label='Original')
    ax2.plot(val_data['image_index'], val_mag,
             'o-', linewidth=2, markersize=3, alpha=0.7, color='green',
             label='After alignment')

    ax2.set_xlabel('Frame Index', fontsize=14)
    ax2.set_ylabel('Drift Magnitude (pixels)', fontsize=14)
    ax2.set_title(f'Particle {particle_id}: Drift Magnitude Over Time', fontsize=15, weight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    orig_max = orig_mag.max()
    val_max = val_mag.max()
    reduction = ((orig_max - val_max) / orig_max * 100) if orig_max > 0 else 0

    stats_text = f"Original max drift: {orig_max:.2f} px\n"
    stats_text += f"Aligned max drift: {val_max:.2f} px\n"
    stats_text += f"Reduction: {reduction:.1f}%"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file.name}")


def plot_summary_comparison(original_df: pd.DataFrame, validation_df: pd.DataFrame,
                           selected_particle_ids: List[int], output_file: Path):
    """
    Plot overall drift reduction summary for all selected particles.

    Args:
        original_df: Original tracking data
        validation_df: Validation tracking data
        selected_particle_ids: List of particle IDs
        output_file: Path to save PNG
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_particle_ids)))

    # TOP LEFT: All original trajectories
    for i, pid in enumerate(selected_particle_ids):
        orig_data = original_df[original_df['particle_id'] == pid].sort_values('image_index')
        if len(orig_data) > 0:
            ax1.plot(orig_data['drift_x'], orig_data['drift_y'],
                    'o-', color=colors[i], markersize=2, linewidth=1.5,
                    alpha=0.7, label=f'Particle {pid}')

    ax1.plot(0, 0, 'k*', markersize=20, markeredgecolor='black', markeredgewidth=2, label='Start', zorder=10)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('Drift X (pixels)', fontsize=12)
    ax1.set_ylabel('Drift Y (pixels)', fontsize=12)
    ax1.set_title('BEFORE Alignment - Original Drift', fontsize=13, weight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # TOP RIGHT: All aligned trajectories
    for i, pid in enumerate(selected_particle_ids):
        val_data = validation_df[validation_df['particle_id'] == pid].sort_values('image_index')
        if len(val_data) > 0:
            ax2.plot(val_data['drift_x'], val_data['drift_y'],
                    'o-', color=colors[i], markersize=2, linewidth=1.5,
                    alpha=0.7, label=f'Particle {pid}')

    ax2.plot(0, 0, 'k*', markersize=20, markeredgecolor='black', markeredgewidth=2, label='Start', zorder=10)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Drift X (pixels)', fontsize=12)
    ax2.set_ylabel('Drift Y (pixels)', fontsize=12)
    ax2.set_title('AFTER Alignment - Residual Drift', fontsize=13, weight='bold', color='green')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # BOTTOM LEFT: Drift magnitude comparison
    for i, pid in enumerate(selected_particle_ids):
        orig_data = original_df[original_df['particle_id'] == pid].sort_values('image_index')
        val_data = validation_df[validation_df['particle_id'] == pid].sort_values('image_index')

        if len(orig_data) > 0:
            orig_mag = np.sqrt(orig_data['drift_x']**2 + orig_data['drift_y']**2)
            ax3.plot(orig_data['image_index'], orig_mag,
                    'o', color=colors[i], markersize=2, linewidth=1,
                    alpha=0.5, linestyle='--', label=f'P{pid} original')

        if len(val_data) > 0:
            val_mag = np.sqrt(val_data['drift_x']**2 + val_data['drift_y']**2)
            ax3.plot(val_data['image_index'], val_mag,
                    'o-', color=colors[i], markersize=2, linewidth=2,
                    alpha=0.8, label=f'P{pid} aligned')

    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Drift Magnitude (pixels)', fontsize=12)
    ax3.set_title('Drift Magnitude Over Time', fontsize=13, weight='bold')
    ax3.legend(fontsize=9, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3)

    # BOTTOM RIGHT: Statistics table
    ax4.axis('off')

    # Calculate statistics
    stats_data = []
    for pid in selected_particle_ids:
        orig_data = original_df[original_df['particle_id'] == pid]
        val_data = validation_df[validation_df['particle_id'] == pid]

        if len(orig_data) > 0 and len(val_data) > 0:
            orig_mag = np.sqrt(orig_data['drift_x']**2 + orig_data['drift_y']**2)
            val_mag = np.sqrt(val_data['drift_x']**2 + val_data['drift_y']**2)

            orig_max = orig_mag.max()
            val_max = val_mag.max()
            reduction = ((orig_max - val_max) / orig_max * 100) if orig_max > 0 else 0

            stats_data.append([
                f"Particle {pid}",
                f"{orig_max:.2f}",
                f"{val_max:.2f}",
                f"{reduction:.1f}%"
            ])

    # Create table
    table_data = [['Particle', 'Original\nMax (px)', 'Aligned\nMax (px)', 'Reduction']] + stats_data
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax4.set_title('Drift Reduction Summary', fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Summary plot saved: {output_file.name}")


def main():
    """Main function to validate alignment by re-fitting Gaussians on aligned images."""
    # Check if JSON path was provided as command-line argument
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
        if not json_file.exists():
            print(f"Error: Provided JSON file does not exist: {json_file}")
            sys.exit(1)
    else:
        # Find JSON file automatically (fallback)
        current_dir = Path.cwd()

        possible_locations = [
            current_dir.parent / "script_output",  # If running from scripts/
            current_dir / "script_output",  # If running from parent
        ]

        json_file = None
        for location in possible_locations:
            if location.exists():
                json_files = list(location.glob('particle_selections.json'))
                if json_files:
                    json_file = json_files[0]
                    break

        if not json_file or not json_file.exists():
            print("Error: Cannot find particle_selections.json!")
            print("Please run scripts 1-4 first.")
            sys.exit(1)

    # Setup logger
    global logger
    log_dir = json_file.parent
    logger = setup_logger('Step5_Validation', log_dir=str(log_dir))

    logger.info("=== VALIDATION: Re-fit Gaussians on Aligned Images (Parallelized) ===\n")
    logger.info(f"Loading configuration from: {json_file}")

    # Load JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        sys.exit(1)

    # Get image set
    image_set = data.get('image_set')
    if not image_set:
        logger.error("No image set found in JSON!")
        sys.exit(1)

    # Extract required information
    selected_particles = image_set.get('selected_particles')
    selected_particles_for_drift = image_set.get('selected_particles_for_drift')
    aligned_folder_path = image_set.get('aligned_folder_path')
    csv_file_path = image_set.get('csv_file_path')

    # Validate all required data exists
    if not selected_particles:
        logger.error("No selected_particles in JSON! Run script 1 first.")
        sys.exit(1)

    if not selected_particles_for_drift:
        logger.error("No selected_particles_for_drift in JSON! Run script 3 first.")
        sys.exit(1)

    if not aligned_folder_path:
        logger.error("No aligned_folder_path in JSON! Run script 4 first.")
        sys.exit(1)

    if not csv_file_path:
        logger.error("No csv_file_path in JSON! Run script 2 first.")
        sys.exit(1)

    aligned_folder = Path(aligned_folder_path)
    if not aligned_folder.exists():
        logger.error(f"Aligned folder not found: {aligned_folder}")
        sys.exit(1)

    csv_file = Path(csv_file_path)
    if not csv_file.exists():
        logger.error(f"Original tracking CSV not found: {csv_file}")
        sys.exit(1)

    # Filter selected_particles to only include those chosen for drift
    selected_particles_info = [p for p in selected_particles
                               if p['particle_id'] in selected_particles_for_drift]

    logger.info(f"\nConfiguration:")
    logger.info(f"  Selected particles for validation: {selected_particles_for_drift}")
    logger.info(f"  Aligned images folder: {aligned_folder.name}")
    logger.info(f"  Original tracking CSV: {csv_file.name}")

    # Get all aligned TIF files
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    aligned_files = []
    for pattern in tif_patterns:
        aligned_files.extend(list(aligned_folder.glob(pattern)))
    aligned_files.sort()

    if not aligned_files:
        logger.error("No TIF files found in aligned folder!")
        sys.exit(1)

    logger.info(f"  Total aligned images: {len(aligned_files)}")
    
    # Determine number of workers (use 1/4 of CPU cores for HDD read operations)
    total_cores = mp.cpu_count()
    num_workers = max(1, total_cores // 4)
    logger.info(f"\nUsing {num_workers} workers (1/4 of {total_cores} CPU cores) - optimized for HDD")

    # Create output directory
    output_dir = json_file.parent / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput directory: {output_dir}")

    logger.info("\n" + "="*60)
    logger.info("Starting parallel Gaussian fitting on aligned images...")
    logger.info("="*60 + "\n")

    # Create all work items (image, particle combinations)
    work_items = []
    for img_idx, tif_file in enumerate(aligned_files):
        for particle_idx, particle in enumerate(selected_particles_info):
            work_items.append((
                str(tif_file),           # image_path
                img_idx,                  # image_index
                tif_file.name,           # filename
                particle,                 # particle dict
                particle_idx              # particle_index
            ))
    
    total_work_items = len(work_items)
    logger.info(f"Total Gaussian fits to perform: {total_work_items}")
    logger.info(f"  ({len(aligned_files)} images × {len(selected_particles_info)} particles)\n")

    # Process in parallel with progress bar
    start_time = time.time()
    
    all_results = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for streaming results with progress tracking
        with tqdm(total=total_work_items, desc="Fitting Gaussians", unit="fit", ncols=100) as pbar:
            for result in pool.imap_unordered(process_single_fit, work_items, chunksize=10):
                all_results.append(result)
                pbar.update(1)
    
    end_time = time.time()

    logger.info("\n" + "="*60)
    logger.info("Parallel processing complete!")
    logger.info("="*60 + "\n")

    # Calculate first frame positions and drift
    logger.info("Calculating drift from first aligned frame...")
    first_frame_positions = {}
    
    # Sort results by image_index and particle_id for organized processing
    all_results.sort(key=lambda x: (x['image_index'], x['particle_id']))
    
    # First pass: collect first frame positions
    for result in all_results:
        if result['image_index'] == 0 and result['success']:
            particle_id = result['particle_id']
            first_frame_positions[particle_id] = {
                'center_x': result['center_x'],
                'center_y': result['center_y']
            }
    
    # Second pass: calculate drift for all frames
    for result in all_results:
        if result['success']:
            particle_id = result['particle_id']
            if particle_id in first_frame_positions:
                dx = result['center_x'] - first_frame_positions[particle_id]['center_x']
                dy = result['center_y'] - first_frame_positions[particle_id]['center_y']
                drift_magnitude = np.sqrt(dx**2 + dy**2)
                
                result['drift_x'] = dx
                result['drift_y'] = dy
                result['drift_magnitude'] = drift_magnitude
            else:
                result['drift_x'] = None
                result['drift_y'] = None
                result['drift_magnitude'] = None
        else:
            result['drift_x'] = None
            result['drift_y'] = None
            result['drift_magnitude'] = None

    # Calculate statistics
    successful_fits = sum(1 for r in all_results if r['success'])
    success_rate = (successful_fits / len(all_results) * 100) if all_results else 0

    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total fits attempted: {len(all_results)}")
    logger.info(f"Successful fits: {successful_fits}")
    logger.info(f"Success rate: {success_rate:.1f}%")

    # Export validation results to CSV
    logger.info("\n" + "="*60)
    logger.info("Exporting validation results...")
    logger.info("="*60 + "\n")

    csv_output = output_dir / 'aligned_particles_tracking.csv'
    if export_validation_csv(all_results, csv_output, selected_particles_for_drift):
        logger.info(f"✓ Validation tracking CSV saved to: {csv_output.name}")

    # Load original tracking data for comparison
    logger.info("\n" + "="*60)
    logger.info("Generating comparison plots...")
    logger.info("="*60 + "\n")

    original_df = pd.read_csv(csv_file)
    original_df = original_df[original_df['success'] == True].copy()

    validation_df = pd.DataFrame(all_results)
    validation_df = validation_df[validation_df['success'] == True].copy()

    # Generate individual particle comparison plots
    logger.info(f"Generating individual particle comparison plots...")
    for particle_id in selected_particles_for_drift:
        output_file = output_dir / f"particle_{particle_id}_comparison.png"
        plot_particle_comparison(original_df, validation_df, particle_id, output_file)

    # Generate summary comparison plot
    logger.info(f"\nGenerating overall summary plot...")
    summary_output = output_dir / "drift_reduction_summary.png"
    plot_summary_comparison(original_df, validation_df, selected_particles_for_drift, summary_output)

    # Calculate and display overall statistics
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60 + "\n")

    for particle_id in selected_particles_for_drift:
        orig_data = original_df[original_df['particle_id'] == particle_id]
        val_data = validation_df[validation_df['particle_id'] == particle_id]

        if len(orig_data) > 0 and len(val_data) > 0:
            orig_mag = np.sqrt(orig_data['drift_x']**2 + orig_data['drift_y']**2)
            val_mag = np.sqrt(val_data['drift_x']**2 + val_data['drift_y']**2)

            orig_max = orig_mag.max()
            orig_mean = orig_mag.mean()
            val_max = val_mag.max()
            val_mean = val_mag.mean()

            max_reduction = ((orig_max - val_max) / orig_max * 100) if orig_max > 0 else 0
            mean_reduction = ((orig_mean - val_mean) / orig_mean * 100) if orig_mean > 0 else 0

            orig_r2_mean = orig_data['r_squared'].mean()
            val_r2_mean = val_data['r_squared'].mean()

            logger.info(f"Particle {particle_id}:")
            logger.info(f"  Original - Max drift: {orig_max:.2f} px, Mean drift: {orig_mean:.2f} px, Mean R²: {orig_r2_mean:.4f}")
            logger.info(f"  Aligned  - Max drift: {val_max:.2f} px, Mean drift: {val_mean:.2f} px, Mean R²: {val_r2_mean:.4f}")
            logger.info(f"  Reduction - Max: {max_reduction:.1f}%, Mean: {mean_reduction:.1f}%")
            logger.info("")

    # Update JSON with validation info
    logger.info("Updating JSON with validation results...")
    try:
        data['image_set']['validation_folder'] = str(output_dir.absolute())
        data['image_set']['validation_csv_path'] = str(csv_output.absolute())
        data['image_set']['validation_csv_name'] = csv_output.name
        data['image_set']['validation_completed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✓ JSON file updated: {json_file.name}")

    except Exception as e:
        logger.error(f"Warning: Could not update JSON file: {e}")
        log_exception(logger, e, "JSON update error")

    logger.info("\n" + "="*60)
    logger.info("VALIDATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nAll validation files saved to: {output_dir}/")
    logger.info(f"  - Tracking CSV: {csv_output.name}")
    logger.info(f"  - Individual plots: particle_*_comparison.png")
    logger.info(f"  - Summary plot: drift_reduction_summary.png")
    logger.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
