"""
Drift Analysis
Processes single image set using particle selections from select_particles_for_drift.py
Performs Gaussian fitting across all images to track particle positions and calculate drift.

Requirements:
pip install tifffile numpy scipy tqdm

Usage:
python 2_parallel_drift_analysis.py

The script automatically finds the JSON file in the script_output folder (sibling to image folder).

Output:
- CSV file saved to: script_output/particles_tracking/
- Filename format: {folder_name}_{3char_id}.csv (e.g., exported_xPf.csv)
- JSON file updated with 'csv_file_path' and 'csv_file_name'
"""

import numpy as np
import tifffile
from pathlib import Path
from scipy.optimize import curve_fit
import json
import csv
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import sys
import time
from tqdm import tqdm
import random
import string
from logging_utils import setup_logger, log_exception

# Logger will be set up after finding JSON file
logger = None

def gaussian_2d(coords: Tuple[np.ndarray, np.ndarray],
               amp: float, x0: float, y0: float,
               sigma_x: float, sigma_y: float,
               theta: float, offset: float) -> np.ndarray:
    """
    2D Gaussian model with rotation.
    Same as original script.
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
    Same logic as original script.
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

def export_results_to_csv(set_result: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """
    Export results for one image set to CSV.

    Args:
        set_result: Result dictionary from process_single_image_set
        output_dir: Directory to save CSV file (will use scripts_output/particles_tracking/)

    Returns:
        Dictionary with 'csv_file_path' and 'csv_file_name', or None if failed
    """
    if not set_result['success']:
        print(f"Cannot export - processing failed for set {set_result['set_index']}")
        return None

    # Create output directory: scripts_output/particles_tracking/
    output_subdir = output_dir / 'particles_tracking'
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Get folder name from image set
    folder_name = set_result['folder_name']

    # Use parent folder name if current folder is just "exported"
    if folder_name == 'exported':
        parent_name = Path(set_result['folder_path']).parent.name
        descriptive_name = parent_name
    else:
        descriptive_name = folder_name

    # Generate unique 3-character identifier
    unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=3))

    # Generate filename: foldername_uniqueID.csv
    csv_filename = output_subdir / f"{descriptive_name}_{unique_id}.csv"

    # Ensure uniqueness (very unlikely collision, but check anyway)
    while csv_filename.exists():
        unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
        csv_filename = output_subdir / f"{descriptive_name}_{unique_id}.csv"

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
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for result in set_result['results']:
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

        print(f"[Set {set_result['set_index']}] Exported to: {csv_filename.name}")

        # Return dictionary with both path and name
        return {
            'csv_file_path': str(csv_filename.absolute()),
            'csv_file_name': csv_filename.name
        }

    except Exception as e:
        print(f"Error exporting results for set {set_result['set_index']}: {e}")
        return None

def main():
    """Main function to run drift analysis on single image set."""
    # Check if JSON path was provided as command-line argument
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
        if not json_file.exists():
            print(f"Error: Provided JSON file does not exist: {json_file}")
            sys.exit(1)
    else:
        # Find JSON file automatically (fallback)
        # Look for JSON file in parent directory's script_output folder
        current_dir = Path.cwd()

        # Try to find script_output folder
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
            print("Please run script 1 first to create particle selections.")
            sys.exit(1)
    
    # Setup logger to use same directory as JSON
    global logger
    log_dir = json_file.parent
    logger = setup_logger('Step2_DriftAnalysis', log_dir=str(log_dir))
    
    logger.info("=== Drift Analysis ===\n")
    logger.info(f"Loading selections from: {json_file}")
    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        sys.exit(1)

    # Get single image set
    image_set = data.get('image_set')
    if not image_set:
        logger.error("No image set found in JSON file!")
        sys.exit(1)

    folder_path = Path(image_set['folder_path'])
    selected_particles = image_set['selected_particles']
    
    logger.info(f"\nProcessing image set:")
    logger.info(f"  Folder: {folder_path.name}")
    logger.info(f"  Full path: {folder_path}")
    logger.info(f"  Particles: {len(selected_particles)}")

    # Get all TIF files
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []
    for pattern in tif_patterns:
        tif_files.extend(list(folder_path.glob(pattern)))
    tif_files.sort()

    if not tif_files:
        logger.error("No TIF files found in folder!")
        sys.exit(1)
    
    logger.info(f"  Total images: {len(tif_files)}")

    logger.info("\n" + "="*60)
    logger.info("Starting drift analysis...")
    logger.info("="*60 + "\n")

    # Process sequentially with progress bars
    start_time = time.time()
    
    all_results = []
    first_frame_positions = {}
    
    # Outer progress bar for images
    for img_idx, tif_file in enumerate(tqdm(tif_files, desc="Processing images", unit="img")):
        try:
            # Load single image
            image = tifffile.imread(str(tif_file))
            if image.ndim != 2:
                logger.warning(f"Skipping {tif_file.name} - not grayscale")
                continue

            # Inner progress bar for particles in this image
            for particle in tqdm(selected_particles, desc=f"  Image {img_idx+1}/{len(tif_files)}", 
                                unit="particle", leave=False):
                bbox = tuple(particle['bbox'])
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

                    # Track first frame position
                    if img_idx == 0:
                        first_frame_positions[particle_id] = {
                            'center_x': fit_result['center_x'],
                            'center_y': fit_result['center_y']
                        }

                    # Calculate drift
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

        except Exception as e:
            logger.error(f"Error processing {tif_file.name}: {e}")
            continue

    end_time = time.time()

    # Calculate statistics
    successful_fits = sum(1 for r in all_results if r['success'])
    success_rate = (successful_fits / len(all_results) * 100) if all_results else 0

    logger.info("\n" + "="*60)
    logger.info("Processing complete!")
    logger.info("="*60 + "\n")

    # Create result dictionary
    set_result = {
        'set_index': 0,
        'folder_path': str(folder_path),
        'folder_name': folder_path.name,
        'success': True,
        'results': all_results,
        'total_images': len(tif_files),
        'total_particles': len(selected_particles),
        'total_fits_attempted': len(all_results),
        'successful_fits': successful_fits,
        'success_rate': success_rate,
        'first_frame_positions': first_frame_positions
    }

    # Export results to CSV
    output_dir = json_file.parent
    csv_info = export_results_to_csv(set_result, output_dir)

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total fits attempted: {set_result['total_fits_attempted']}")
    logger.info(f"Successful fits: {set_result['successful_fits']}")
    logger.info(f"Success rate: {set_result['success_rate']:.1f}%\n")

    if csv_info:
        logger.info(f"CSV file saved: {csv_info['csv_file_name']}")

        # Update JSON file with CSV information
        logger.info("\nUpdating JSON file with CSV information...")
        try:
            data['image_set']['csv_file_path'] = csv_info['csv_file_path']
            data['image_set']['csv_file_name'] = csv_info['csv_file_name']

            # Save updated JSON back to file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"✓ JSON file updated: {json_file.name}")
            logger.info("  Added 'csv_file_path' and 'csv_file_name'")

        except Exception as e:
            logger.error(f"Warning: Could not update JSON file: {e}")
            log_exception(logger, e, "JSON update error")

    logger.info("\n=== DONE ===")

if __name__ == "__main__":
    main()
