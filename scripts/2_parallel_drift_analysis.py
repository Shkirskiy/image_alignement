"""
Parallel Drift Analysis
Processes multiple image sets in parallel using particle selections from select_particles_for_drift.py
Performs Gaussian fitting across all images to track particle positions and calculate drift.

Requirements:
pip install tifffile numpy scipy tqdm

Usage:
python parallel_drift_analysis.py [optional: path_to_selection_json]

Examples:
# Auto-discover JSON in scripts_output/ folder (must be only one JSON file)
python parallel_drift_analysis.py

# Manually specify JSON file path
python parallel_drift_analysis.py particle_selections_20251015_143000.json
python parallel_drift_analysis.py scripts_output/particle_selections_20251015_143000.json

Output:
- CSV files saved to: scripts_output/particles_tracking/
- Filename format: {folder_name}_{3char_id}.csv (e.g., example_data1_xPf.csv)
- JSON file updated with 'csv_file_path' and 'csv_file_name' for each image set
"""

import numpy as np
import tifffile
from pathlib import Path
from scipy.optimize import curve_fit
import json
import csv
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import multiprocessing as mp
from multiprocessing import Pool
import sys
import time
from tqdm import tqdm
import random
import string

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

def process_single_image_set(args: Tuple[int, Dict[str, Any], Optional[mp.Queue]]) -> Dict[str, Any]:
    """
    Process a single image set: load images ONE BY ONE and track particles.
    This function runs in a separate process.

    Args:
        args: (set_index, image_set_data, progress_queue)
            - set_index: Index of this image set
            - image_set_data: Dictionary with folder_path and selected_particles
            - progress_queue: Optional queue for progress updates

    Returns:
        Dictionary with processing results and metadata
    """
    set_index, image_set_data, progress_queue = args

    folder_path = Path(image_set_data['folder_path'])
    selected_particles = image_set_data['selected_particles']

    # Get all TIF files
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []
    for pattern in tif_patterns:
        tif_files.extend(list(folder_path.glob(pattern)))
    tif_files.sort()

    if not tif_files:
        return {
            'set_index': set_index,
            'folder_path': str(folder_path),
            'success': False,
            'error': 'No TIF files found'
        }

    # Store results for all images
    all_results = []
    total_operations = len(tif_files) * len(selected_particles)
    operation_count = 0

    # Store first frame positions for drift calculation
    first_frame_positions = {}

    # Create progress bar for this worker
    pbar = tqdm(
        total=total_operations,
        desc=f"Set {set_index}: {folder_path.name[:30]}",
        position=set_index,
        leave=True,
        unit="fit",
        ncols=100
    )

    # Process images ONE BY ONE (memory efficient)
    for img_idx, tif_file in enumerate(tif_files):
        try:
            # Load single image
            image = tifffile.imread(str(tif_file))
            if image.ndim != 2:
                print(f"[Set {set_index}]   Skipping {tif_file.name} - not grayscale")
                continue

            # Process each particle in this image
            for particle in selected_particles:
                operation_count += 1

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

                    # Calculate drift (displacement from first frame)
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

                # Update progress bar
                pbar.update(1)

            # Image processing done - image will be garbage collected

        except Exception as e:
            pbar.write(f"[Set {set_index}] Error processing {tif_file.name}: {e}")
            continue

    # Close progress bar
    pbar.close()

    # Calculate statistics
    successful_fits = sum(1 for r in all_results if r['success'])
    success_rate = (successful_fits / len(all_results) * 100) if all_results else 0

    return {
        'set_index': set_index,
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
    """Main function to run parallel drift analysis."""
    print("=== Parallel Drift Analysis ===\n")

    # Determine JSON file location
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
            print("     python parallel_drift_analysis.py <path_to_json_file>")
            sys.exit(1)

        # Find all JSON files in scripts_output
        json_files = list(scripts_output_dir.glob('*.json'))

        if len(json_files) == 0:
            print(f"Error: No JSON files found in '{scripts_output_dir}/'")
            print("\nPlease pass the JSON file path manually:")
            print("  python parallel_drift_analysis.py <path_to_json_file>")
            sys.exit(1)
        elif len(json_files) == 1:
            json_file = json_files[0]
            print(f"Auto-discovered JSON file: {json_file.name}")
        else:
            print(f"Error: Multiple JSON files found in '{scripts_output_dir}/':")
            for jf in json_files:
                print(f"  - {jf.name}")
            print("\nPlease specify which JSON file to use:")
            print("  python parallel_drift_analysis.py <path_to_json_file>")
            sys.exit(1)

    # Load JSON
    print(f"Loading selections from: {json_file}")
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

    print(f"Found {len(image_sets)} image set(s) to process\n")

    # Print summary
    for i, img_set in enumerate(image_sets):
        print(f"Set {i}: {Path(img_set['folder_path']).name}")
        print(f"  Particles: {len(img_set['selected_particles'])}")
        print(f"  Folder: {img_set['folder_path']}")

    print("\n" + "="*60)
    print("Starting parallel processing...")
    print("="*60 + "\n")

    # Create argument tuples for each image set (no progress queue needed with tqdm)
    process_args = [
        (i, img_set, None)
        for i, img_set in enumerate(image_sets)
    ]

    # Determine number of workers (one per image set, but cap at CPU count)
    num_workers = min(len(image_sets), mp.cpu_count())
    print(f"Using {num_workers} parallel workers")
    print(f"Progress bars will appear below...\n")

    # Run parallel processing
    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image_set, process_args)

    end_time = time.time()

    print("\n" + "="*60)
    print("Parallel processing complete!")
    print("="*60 + "\n")

    # Export results to CSV
    # Use scripts_output as base directory (parent directory)
    if json_file.parent.name == 'scripts_output':
        output_dir = json_file.parent
    else:
        output_dir = Path('../scripts_output')
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_info_list = []  # Store CSV info dicts with set_index

    for result in results:
        if result['success']:
            csv_info = export_results_to_csv(result, output_dir)
            if csv_info:
                # Add set_index to the info dict
                csv_info['set_index'] = result['set_index']
                csv_info_list.append(csv_info)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Image sets processed: {len(results)}")
    print(f"CSV files created: {len(csv_info_list)}\n")

    for result in results:
        print(f"Set {result['set_index']}: {result['folder_name']}")
        if result['success']:
            print(f"  Total fits attempted: {result['total_fits_attempted']}")
            print(f"  Successful fits: {result['successful_fits']}")
            print(f"  Success rate: {result['success_rate']:.1f}%")
        else:
            print(f"  ERROR: {result.get('error', 'Unknown error')}")
        print()

    print("All CSV files saved to:")
    for csv_info in csv_info_list:
        print(f"  {csv_info['csv_file_name']}")

    # Update JSON file with CSV information
    if csv_info_list:
        print("\nUpdating JSON file with CSV information...")
        try:
            # Map CSV info by set_index for easy lookup
            csv_info_by_index = {info['set_index']: info for info in csv_info_list}

            # Update image_sets with CSV info
            for img_set in data['image_sets']:
                set_idx = data['image_sets'].index(img_set)
                if set_idx in csv_info_by_index:
                    csv_info = csv_info_by_index[set_idx]
                    img_set['csv_file_path'] = csv_info['csv_file_path']
                    img_set['csv_file_name'] = csv_info['csv_file_name']

            # Save updated JSON back to file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ JSON file updated: {json_file.name}")
            print("  Added 'csv_file_path' and 'csv_file_name' to each image set")

        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
