"""
Drift Analysis with Parallelization
Processes single image set using particle selections from select_particles_for_drift.py
Performs parallel Gaussian fitting across all images to track particle positions and calculate drift.

Requirements:
pip install tifffile numpy scipy tqdm

Usage:
python3 2_drift_analysis.py

The script automatically finds the JSON file in the script_output folder (sibling to image folder).

Output:
- CSV file saved to: script_output/particles_tracking/
- Filename format: {folder_name}_{3char_id}.csv (e.g., exported_xPf.csv)
- JSON file updated with 'csv_file_path' and 'csv_file_name'
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
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import sys
import time
from tqdm import tqdm
import random
import string
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


def clip_bbox_to_bounds(bbox: Tuple[float, float, float, float], 
                       image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Clip bounding box coordinates to ensure they stay within image bounds.
    
    Args:
        bbox: (x0, y0, x1, y1) bounding box coordinates
        image_shape: (height, width) of the image
    
    Returns:
        Clipped bounding box as (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    
    # Ensure box stays within bounds
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(width, int(x1))
    y1 = min(height, int(y1))
    
    # Ensure box is valid (has positive dimensions)
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    
    return (x0, y0, x1, y1)


def process_single_particle_with_tracking(work_item: Tuple[Dict, List[str], List[str]]) -> List[Dict[str, Any]]:
    """
    Worker function: process a single particle through all frames sequentially with adaptive bbox tracking.
    
    This function tracks the particle position across frames by adjusting the bounding box
    based on the fitted center from the previous frame. This ensures the particle stays
    roughly centered in the fitting region even when significant drift occurs.
    
    Args:
        work_item: (particle_dict, list_of_image_paths, list_of_filenames)
    
    Returns:
        List of dictionaries with fit results for all frames of this particle
    """
    particle, image_paths, filenames = work_item
    particle_id = particle['particle_id']
    
    results = []
    
    # Start with the manually selected bbox from step 1
    current_bbox = list(particle['bbox'])  # [x0, y0, x1, y1]
    bbox_width = current_bbox[2] - current_bbox[0]
    bbox_height = current_bbox[3] - current_bbox[1]
    
    # Track previous fitted center for shift calculation
    prev_fitted_center = None
    
    for img_idx, (image_path, filename) in enumerate(zip(image_paths, filenames)):
        try:
            # Load image (with caching)
            image = _load_image_cached(image_path)
            
            if image.ndim != 2:
                results.append({
                    'filename': filename,
                    'particle_id': particle_id,
                    'image_index': img_idx,
                    'bbox': tuple(current_bbox),
                    'success': False,
                    'error': 'Not grayscale'
                })
                continue
            
            # Clip bbox to image bounds
            x0, y0, x1, y1 = clip_bbox_to_bounds(current_bbox, image.shape)
            
            if x1 <= x0 or y1 <= y0:
                results.append({
                    'filename': filename,
                    'particle_id': particle_id,
                    'image_index': img_idx,
                    'bbox': (x0, y0, x1, y1),
                    'success': False,
                    'error': 'Invalid bbox'
                })
                continue
            
            # Extract region
            region = image[y0:y1, x0:x1].copy()
            
            # Fit Gaussian
            fit_result = fit_gaussian_to_region(region, (x0, y0, x1, y1))
            
            # Build result record
            result = {
                'filename': filename,
                'particle_id': particle_id,
                'image_index': img_idx,
                'bbox': (x0, y0, x1, y1),
                'success': fit_result is not None and fit_result.get('success', False)
            }
            
            if result['success']:
                result.update(fit_result)
                
                # Adaptive bbox tracking: adjust bbox for next frame based on fitted center
                if img_idx < len(image_paths) - 1:  # Don't adjust after last frame
                    fitted_center_x = fit_result['center_x']
                    fitted_center_y = fit_result['center_y']
                    
                    # Calculate current bbox center
                    bbox_center_x = (x0 + x1) / 2.0
                    bbox_center_y = (y0 + y1) / 2.0
                    
                    # Calculate shift
                    shift_x = fitted_center_x - bbox_center_x
                    shift_y = fitted_center_y - bbox_center_y
                    
                    # Adjust bbox for next frame, keeping size constant
                    current_bbox = [
                        x0 + shift_x,
                        y0 + shift_y,
                        x0 + shift_x + bbox_width,
                        y0 + shift_y + bbox_height
                    ]
                    
                    # Log significant shifts (for debugging)
                    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
                    if shift_magnitude > 5.0:  # More than 5 pixels
                        result['bbox_shift_magnitude'] = shift_magnitude
            else:
                # If fit failed, keep the same bbox for next frame
                # (don't propagate failure by shifting to a bad position)
                pass
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': filename,
                'particle_id': particle_id,
                'image_index': img_idx,
                'bbox': tuple(current_bbox),
                'success': False,
                'error': str(e)
            })
    
    return results


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
    """Main function to run drift analysis on single image set with parallelization."""
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
    
    logger.info("=== Drift Analysis (Parallelized) ===\n")
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
    
    # Determine number of workers (use 1/4 of CPU cores for HDD optimization)
    total_cores = mp.cpu_count()
    num_workers = max(1, total_cores // 4)
    logger.info(f"\nUsing {num_workers} workers (1/4 of {total_cores} CPU cores) - optimized for HDD")

    logger.info("\n" + "="*60)
    logger.info("Starting parallel drift analysis with adaptive bbox tracking...")
    logger.info("="*60 + "\n")

    # Create work items: one per particle (each contains all image paths)
    # This allows sequential processing within each particle for bbox tracking
    image_paths = [str(f) for f in tif_files]
    filenames = [f.name for f in tif_files]
    
    work_items = []
    for particle in selected_particles:
        work_items.append((
            particle,      # particle dict
            image_paths,   # all image paths (processed sequentially per particle)
            filenames      # all filenames
        ))
    
    total_particles = len(selected_particles)
    total_fits = len(tif_files) * len(selected_particles)
    logger.info(f"Total particles to track: {total_particles}")
    logger.info(f"Total Gaussian fits to perform: {total_fits}")
    logger.info(f"  ({len(tif_files)} images × {len(selected_particles)} particles)")
    logger.info(f"\nAdaptive bbox tracking: Each particle tracked sequentially through all frames")
    logger.info(f"  Bbox adjusts after each fit to keep particle centered\n")

    # Process in parallel (one worker per particle)
    start_time = time.time()
    
    all_results = []
    
    # Adjust worker count if we have fewer particles than workers
    actual_workers = min(num_workers, total_particles)
    
    with Pool(processes=actual_workers) as pool:
        # Use imap for ordered results with progress tracking by particle
        with tqdm(total=total_particles, desc="Processing particles", unit="particle", ncols=100) as pbar:
            for particle_results in pool.imap(process_single_particle_with_tracking, work_items):
                all_results.extend(particle_results)  # Each result is a list of fits for one particle
                pbar.update(1)
    
    end_time = time.time()


    logger.info("\n" + "="*60)
    logger.info("Parallel processing complete!")
    logger.info("="*60 + "\n")

    # Calculate first frame positions and drift
    logger.info("Calculating drift from first frame...")
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
    logger.info(f"Success rate: {set_result['success_rate']:.1f}%")
    logger.info(f"Average time per fit: {(end_time - start_time) / total_fits * 1000:.2f} ms\n")

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
