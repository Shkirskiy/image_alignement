"""
Align Images by Particle Drift (Parallelized)
Aligns a series of images based on the average drift of selected particles.
Uses particles selected in script 3 to calculate average displacement and applies inverse shift.

Requirements:
pip install pandas numpy scipy tifffile tqdm

Usage:
python3 4_align_images_by_drift.py

The script automatically finds all required information from the JSON file.

Input:
- Reads particle_selections.json for all configuration
- Uses selected_particles_for_drift from JSON
- Reads drift CSV file path from JSON

Output:
- Aligned images saved to: script_output/aligned/
- Alignment log CSV with shifts applied per frame
- JSON updated with alignment folder path
"""

# Limit threading in numerical libraries to prevent CPU overload
# Each worker process should use only 1 thread
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import tifffile
from scipy.ndimage import shift as ndimage_shift
from scipy.interpolate import CubicSpline, interp1d
from pathlib import Path
import json
from tqdm import tqdm
import sys
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import multiprocessing as mp
from multiprocessing import Pool

def validate_particle_quality(df: pd.DataFrame, particle_ids: List[int],
                               max_failure_rate: float = 0.10,
                               max_consecutive_gap: int = 4) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate particle tracking quality before alignment.

    Args:
        df: DataFrame with particle tracking data (must include all rows, not just success=True)
        particle_ids: List of particle IDs to validate
        max_failure_rate: Maximum allowed failure rate (default: 0.10 = 10%)
        max_consecutive_gap: Maximum allowed consecutive frames without detection (default: 4)

    Returns:
        Tuple of (passed: bool, stats: Dict with detailed statistics per particle)
    """
    # Determine total number of frames
    total_frames = df['image_index'].max() + 1

    all_passed = True
    particle_stats = {}

    for particle_id in particle_ids:
        # Filter data for this particle
        particle_df = df[df['particle_id'] == particle_id].copy()
        particle_df = particle_df.sort_values('image_index').reset_index(drop=True)

        # Count successful fits
        successful_fits = particle_df['success'].sum()
        failure_rate = (total_frames - successful_fits) / total_frames

        # Find gaps (frames where particle was not successfully detected)
        successful_indices = set(particle_df[particle_df['success'] == True]['image_index'].values)
        all_indices = set(range(total_frames))
        missing_indices = sorted(all_indices - successful_indices)

        # Find consecutive gaps
        gap_locations = []
        max_gap = 0

        if missing_indices:
            current_gap_start = missing_indices[0]
            current_gap_end = missing_indices[0]

            for i in range(1, len(missing_indices)):
                if missing_indices[i] == missing_indices[i-1] + 1:
                    # Consecutive gap continues
                    current_gap_end = missing_indices[i]
                else:
                    # Gap ends, record it
                    gap_length = current_gap_end - current_gap_start + 1
                    gap_locations.append((current_gap_start, current_gap_end, gap_length))
                    max_gap = max(max_gap, gap_length)

                    # Start new gap
                    current_gap_start = missing_indices[i]
                    current_gap_end = missing_indices[i]

            # Record final gap
            gap_length = current_gap_end - current_gap_start + 1
            gap_locations.append((current_gap_start, current_gap_end, gap_length))
            max_gap = max(max_gap, gap_length)

        # Check if particle passes thresholds
        particle_passed = (failure_rate <= max_failure_rate) and (max_gap < max_consecutive_gap)
        all_passed = all_passed and particle_passed

        # Build failure reasons list
        failure_reasons = []
        if failure_rate > max_failure_rate:
            failure_reasons.append(f"Failure rate {failure_rate*100:.2f}% exceeds {max_failure_rate*100:.2f}%")
        if max_gap >= max_consecutive_gap:
            failure_reasons.append(f"Max consecutive gap {max_gap} exceeds threshold {max_consecutive_gap-1}")

        # Store statistics
        particle_stats[particle_id] = {
            'total_frames': total_frames,
            'successful_fits': successful_fits,
            'failure_rate': failure_rate,
            'max_consecutive_gap': max_gap,
            'gap_locations': gap_locations,
            'passed': particle_passed,
            'failure_reasons': failure_reasons
        }

    return all_passed, {'passed': all_passed, 'particle_stats': particle_stats}

def interpolate_drift_trajectory(df: pd.DataFrame, particle_id: int,
                                  total_frames: int,
                                  method: str = 'cubic') -> pd.DataFrame:
    """
    Create complete frame-by-frame trajectory for a single particle using interpolation.

    Args:
        df: DataFrame with particle tracking data (all rows for this particle)
        particle_id: Particle ID to interpolate
        total_frames: Total number of frames in sequence
        method: Interpolation method ('cubic' or 'linear')

    Returns:
        DataFrame with complete trajectory (all frames) including interpolated values
        Columns: image_index, filename, drift_x, drift_y, source
    """
    # Filter for this particle and successful fits only
    particle_df = df[(df['particle_id'] == particle_id) & (df['success'] == True)].copy()
    particle_df = particle_df.sort_values('image_index').reset_index(drop=True)

    if len(particle_df) == 0:
        raise ValueError(f"No successful fits found for particle {particle_id}")

    # Extract valid data points
    valid_indices = particle_df['image_index'].values
    drift_x_values = particle_df['drift_x'].values
    drift_y_values = particle_df['drift_y'].values
    filenames = particle_df['filename'].values

    # Create filename lookup
    filename_map = dict(zip(valid_indices, filenames))

    # Generate all frame indices
    all_indices = np.arange(total_frames)

    # Interpolate drift values
    if len(valid_indices) == 1:
        # Only one point - use constant value
        interp_drift_x = np.full(total_frames, drift_x_values[0])
        interp_drift_y = np.full(total_frames, drift_y_values[0])
    elif method == 'cubic' and len(valid_indices) >= 4:
        # Use cubic spline (requires at least 4 points)
        try:
            cs_x = CubicSpline(valid_indices, drift_x_values, bc_type='clamped')
            cs_y = CubicSpline(valid_indices, drift_y_values, bc_type='clamped')
            interp_drift_x = cs_x(all_indices)
            interp_drift_y = cs_y(all_indices)
        except Exception:
            # Fall back to linear if cubic fails
            f_x = interp1d(valid_indices, drift_x_values, kind='linear', fill_value='extrapolate')
            f_y = interp1d(valid_indices, drift_y_values, kind='linear', fill_value='extrapolate')
            interp_drift_x = f_x(all_indices)
            interp_drift_y = f_y(all_indices)
    else:
        # Use linear interpolation
        f_x = interp1d(valid_indices, drift_x_values, kind='linear', fill_value='extrapolate')
        f_y = interp1d(valid_indices, drift_y_values, kind='linear', fill_value='extrapolate')
        interp_drift_x = f_x(all_indices)
        interp_drift_y = f_y(all_indices)

    # Build complete trajectory DataFrame
    trajectory_data = []
    for idx in all_indices:
        # Determine source (measured vs interpolated)
        is_measured = idx in valid_indices

        # Get filename (use measured if available, otherwise infer from pattern)
        if idx in filename_map:
            filename = filename_map[idx]
        else:
            # Infer filename from pattern (use nearest neighbor's pattern)
            if len(filename_map) > 0:
                nearest_idx = min(filename_map.keys(), key=lambda x: abs(x - idx))
                nearest_filename = filename_map[nearest_idx]
                # Try to infer pattern (e.g., cropped_0000.tif -> cropped_XXXX.tif)
                import re
                match = re.search(r'(\d+)', nearest_filename)
                if match:
                    num_digits = len(match.group(1))
                    filename = re.sub(r'\d+', str(idx).zfill(num_digits), nearest_filename)
                else:
                    filename = f"frame_{idx:04d}.tif"
            else:
                filename = f"frame_{idx:04d}.tif"

        trajectory_data.append({
            'image_index': idx,
            'filename': filename,
            'drift_x': interp_drift_x[idx],
            'drift_y': interp_drift_y[idx],
            'source': 'measured' if is_measured else 'interpolated'
        })

    return pd.DataFrame(trajectory_data)

def load_drift_data(csv_path: Path, particle_ids: list) -> pd.DataFrame:
    """
    Load drift data for specified particles (includes all rows, not just successful fits).

    Args:
        csv_path: Path to tracking CSV file
        particle_ids: List of particle IDs to include (e.g., [1, 3, 6])

    Returns:
        DataFrame with drift data for selected particles (including success and failure rows)
    """
    print(f"Loading drift data from: {csv_path.name}")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Filter for specified particles only (load ALL rows, including failures, for gap analysis)
    df = df[df['particle_id'].isin(particle_ids)].copy()

    successful_fits = df['success'].sum()
    total_rows = len(df)
    print(f"Loaded {total_rows} rows for particles {particle_ids}")
    print(f"  Successful fits: {successful_fits}")
    print(f"  Failed fits: {total_rows - successful_fits}")
    if len(df) > 0:
        print(f"  Frame range: {df['image_index'].min()} to {df['image_index'].max()}")

    return df

def calculate_drift_with_interpolation(df: pd.DataFrame,
                                       particle_ids: List[int],
                                       total_frames: int,
                                       interpolation_method: str = 'cubic') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate drift with hybrid approach: use measured data when available, interpolate when missing.

    Args:
        df: DataFrame with particle tracking data (all rows, including failures)
        particle_ids: List of particle IDs to use
        total_frames: Total number of frames in sequence
        interpolation_method: 'cubic' or 'linear'

    Returns:
        Tuple of (drift_summary DataFrame, interpolation_metadata Dict)
    """
    print("\n=== INTERPOLATION ===")
    print(f"Interpolating drift trajectories (method: {interpolation_method})...")

    # Create complete trajectories per particle
    particle_trajectories = {}
    interpolation_stats = {}

    for particle_id in particle_ids:
        trajectory = interpolate_drift_trajectory(df, particle_id, total_frames, interpolation_method)
        particle_trajectories[particle_id] = trajectory

        # Count interpolated frames for this particle
        num_interpolated = (trajectory['source'] == 'interpolated').sum()
        interpolation_stats[particle_id] = {
            'total_frames': total_frames,
            'interpolated_frames': num_interpolated,
            'measured_frames': total_frames - num_interpolated
        }
        print(f"  Particle {particle_id}: {num_interpolated} frames interpolated ({num_interpolated/total_frames*100:.2f}%)")

    print("\nCalculating hybrid drift per frame...")

    # Combine frame-by-frame
    drift_summary_list = []
    frames_averaged = 0
    frames_single_particle = 0
    frames_interpolated = 0

    for frame_idx in range(total_frames):
        # Collect drift values from all particles for this frame
        drift_x_values = []
        drift_y_values = []
        sources = []
        filename = None

        for particle_id in particle_ids:
            traj = particle_trajectories[particle_id]
            frame_row = traj[traj['image_index'] == frame_idx].iloc[0]

            drift_x_values.append(frame_row['drift_x'])
            drift_y_values.append(frame_row['drift_y'])
            sources.append(frame_row['source'])

            if filename is None:
                filename = frame_row['filename']

        # Calculate average
        avg_drift_x = np.mean(drift_x_values)
        avg_drift_y = np.mean(drift_y_values)

        # Categorize frame
        num_measured = sources.count('measured')
        num_interpolated = sources.count('interpolated')

        if num_measured >= 2:
            drift_source = 'averaged'
            frames_averaged += 1
        elif num_measured == 1:
            drift_source = 'single_particle'
            frames_single_particle += 1
        else:
            drift_source = 'interpolated'
            frames_interpolated += 1

        drift_summary_list.append({
            'image_index': frame_idx,
            'filename': filename,
            'avg_drift_x': avg_drift_x,
            'avg_drift_y': avg_drift_y,
            'num_particles_measured': num_measured,
            'num_particles_interpolated': num_interpolated,
            'num_particles': len(particle_ids),
            'drift_source': drift_source
        })

    drift_summary = pd.DataFrame(drift_summary_list)

    # Print statistics
    print(f"  Frames with ≥2 particles (averaged): {frames_averaged} ({frames_averaged/total_frames*100:.2f}%)")
    print(f"  Frames with 1 particle (single): {frames_single_particle} ({frames_single_particle/total_frames*100:.2f}%)")
    print(f"  Frames with 0 particles (interpolated): {frames_interpolated} ({frames_interpolated/total_frames*100:.2f}%)")

    print(f"\n  Mean drift X: {drift_summary['avg_drift_x'].mean():.3f} pixels")
    print(f"  Mean drift Y: {drift_summary['avg_drift_y'].mean():.3f} pixels")
    print(f"  Max drift X: {drift_summary['avg_drift_x'].abs().max():.3f} pixels")
    print(f"  Max drift Y: {drift_summary['avg_drift_y'].abs().max():.3f} pixels")

    # Create metadata (convert numpy types to Python types for JSON serialization)
    interpolation_metadata = {
        'frames_averaged': int(frames_averaged),
        'frames_single_particle': int(frames_single_particle),
        'frames_interpolated': int(frames_interpolated),
        'total_frames': int(total_frames),
        'interpolation_method': interpolation_method,
        'particle_stats': {
            pid: {
                'total_frames': int(stats['total_frames']),
                'interpolated_frames': int(stats['interpolated_frames']),
                'measured_frames': int(stats['measured_frames'])
            }
            for pid, stats in interpolation_stats.items()
        }
    }

    return drift_summary, interpolation_metadata

def align_image(image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """
    Align an image by applying inverse drift shift.

    Args:
        image: Input image array (2D)
        shift_x: Drift in X direction (will be inverted)
        shift_y: Drift in Y direction (will be inverted)

    Returns:
        Aligned image array
    """
    # Apply inverse shift to compensate for drift
    # Drift is the measured displacement, so we shift by negative to align
    aligned = ndimage_shift(
        image,
        shift=(-shift_y, -shift_x),  # Note: (row, col) = (y, x)
        order=1,  # Bilinear interpolation
        mode='nearest',  # Use nearest pixel for edges
        prefilter=True
    )

    return aligned


def process_single_image(work_item: Tuple[Path, Path, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Worker function: process a single image alignment.
    
    Args:
        work_item: (source_path, output_path, drift_data_or_None)
            - source_path: Path to source TIF file
            - output_path: Path to save aligned TIF file
            - drift_data: Dict with avg_drift_x, avg_drift_y, num_particles (or None if no drift data)
    
    Returns:
        Dictionary with alignment results
    """
    source_path, output_path, drift_data = work_item
    filename = source_path.name
    
    try:
        # Load image
        image = tifffile.imread(str(source_path))
        
        # If no drift data, just save original image
        if drift_data is None:
            tifffile.imwrite(str(output_path), image)
            return {
                'filename': filename,
                'avg_drift_x': 0.0,
                'avg_drift_y': 0.0,
                'num_particles': 0,
                'aligned': False,
                'status': 'no_drift_data'
            }
        
        # Check if image is 2D grayscale
        if image.ndim != 2:
            # Not 2D, just save original
            tifffile.imwrite(str(output_path), image)
            return {
                'filename': filename,
                'avg_drift_x': drift_data['avg_drift_x'],
                'avg_drift_y': drift_data['avg_drift_y'],
                'num_particles': drift_data['num_particles'],
                'aligned': False,
                'status': 'not_2d'
            }
        
        # Get shift values
        shift_x = drift_data['avg_drift_x']
        shift_y = drift_data['avg_drift_y']
        
        # Align image
        aligned_image = align_image(image, shift_x, shift_y)
        
        # Ensure output has same dtype as input
        aligned_image = aligned_image.astype(image.dtype)
        
        # Save aligned image
        tifffile.imwrite(str(output_path), aligned_image)
        
        return {
            'filename': filename,
            'avg_drift_x': shift_x,
            'avg_drift_y': shift_y,
            'num_particles': drift_data['num_particles'],
            'aligned': True,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'avg_drift_x': drift_data['avg_drift_x'] if drift_data else 0.0,
            'avg_drift_y': drift_data['avg_drift_y'] if drift_data else 0.0,
            'num_particles': drift_data['num_particles'] if drift_data else 0,
            'aligned': False,
            'status': f'error: {str(e)}'
        }


def align_images_parallel(source_folder: Path, drift_summary: pd.DataFrame, output_folder: Path,
                         num_workers: int) -> pd.DataFrame:
    """
    Align all images based on drift data using parallel processing.

    Args:
        source_folder: Folder containing original TIF images
        drift_summary: DataFrame with avg_drift_x, avg_drift_y per frame
        output_folder: Folder to save aligned images
        num_workers: Number of parallel workers

    Returns:
        DataFrame with alignment log
    """
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nAligning images in parallel...")
    print(f"  Source: {source_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Workers: {num_workers}")

    # Get all TIF files from source folder
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []
    for pattern in tif_patterns:
        tif_files.extend(list(source_folder.glob(pattern)))
    tif_files.sort()

    print(f"  Found {len(tif_files)} TIF files in source folder")

    # Create lookup dict for fast drift access by filename
    drift_dict = {}
    for _, row in drift_summary.iterrows():
        drift_dict[row['filename']] = {
            'avg_drift_x': row['avg_drift_x'],
            'avg_drift_y': row['avg_drift_y'],
            'num_particles': row['num_particles']
        }

    # Create work items for all images
    work_items = []
    for tif_file in tif_files:
        filename = tif_file.name
        output_file = output_folder / filename
        drift_data = drift_dict.get(filename, None)
        
        work_items.append((tif_file, output_file, drift_data))

    # Process in parallel with progress bar
    alignment_results = []
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(work_items), desc="Aligning images", unit="img", ncols=100) as pbar:
            for result in pool.imap_unordered(process_single_image, work_items, chunksize=1):
                alignment_results.append(result)
                pbar.update(1)
    
    # Create DataFrame from results
    alignment_log = pd.DataFrame(alignment_results)
    
    # Print summary
    aligned_count = alignment_log['aligned'].sum()
    no_drift_count = (alignment_log['status'] == 'no_drift_data').sum()
    
    print(f"\n  Successfully aligned: {aligned_count} images")
    print(f"  No drift data (copied): {no_drift_count} images")
    print(f"  Failed/Skipped: {len(alignment_log) - aligned_count - no_drift_count} images")

    return alignment_log


def main():
    """Main function to align images by drift."""
    from logging_utils import setup_logger, log_exception

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
            print("Please run scripts 1-3 first.")
            sys.exit(1)
    
    # Setup logger
    log_dir = json_file.parent
    logger = setup_logger('Step4_ImageAlignment', log_dir=str(log_dir))
    
    logger.info("=== Image Alignment by Particle Drift (Parallelized) ===\n")
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

    # Get paths and selected particles from JSON
    source_folder = Path(image_set['folder_path'])
    csv_file_path = image_set.get('csv_file_path')
    selected_particles = image_set.get('selected_particles_for_drift')
    
    if not csv_file_path:
        logger.error("No CSV file path in JSON! Run script 2 first.")
        sys.exit(1)
    
    if not selected_particles:
        logger.error("No selected particles in JSON! Run script 3 first.")
        sys.exit(1)
    
    csv_file = Path(csv_file_path)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Source folder: {source_folder}")
    logger.info(f"  CSV file: {csv_file.name}")
    logger.info(f"  Selected particles for drift: {selected_particles}")

    # Verify paths exist
    if not source_folder.exists():
        logger.error(f"Source folder not found: {source_folder}")
        sys.exit(1)

    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)

    # Determine number of workers (fixed at 2 for HDD with read+write operations)
    # This conservative setting prevents HDD write queue saturation
    total_cores = mp.cpu_count()
    num_workers = 2
    logger.info(f"\nUsing {num_workers} workers (fixed, out of {total_cores} CPU cores) - optimized for HDD with read+write")

    # Load drift data
    drift_data = load_drift_data(csv_file, selected_particles)

    if len(drift_data) == 0:
        logger.error("No drift data found for selected particles!")
        sys.exit(1)

    # Validate particle quality
    logger.info("\n=== PRE-FLIGHT VALIDATION ===")
    logger.info("Validating particle quality...")

    passed, validation_result = validate_particle_quality(
        drift_data,
        selected_particles,
        max_failure_rate=0.10,
        max_consecutive_gap=4
    )

    # Print validation results
    for pid, stats in validation_result['particle_stats'].items():
        logger.info(f"\nParticle {pid}:")
        logger.info(f"  Total frames: {stats['total_frames']}")
        logger.info(f"  Successful fits: {stats['successful_fits']} ({(1-stats['failure_rate'])*100:.2f}%)")
        logger.info(f"  Failure rate: {stats['failure_rate']*100:.2f}% (threshold: 10.00%)")
        logger.info(f"  Max consecutive gap: {stats['max_consecutive_gap']} frames (threshold: < 4)")

        if stats['passed']:
            logger.info(f"  ✓ PASS")
        else:
            logger.error(f"  ✗ FAIL")
            for reason in stats['failure_reasons']:
                logger.error(f"    - {reason}")

    if not validation_result['passed']:
        logger.error("\n✗ VALIDATION FAILED - Quality thresholds not met")
        logger.error("Consider selecting different particles or adjusting thresholds")
        sys.exit(1)

    logger.info("\n✓ ALL PARTICLES PASSED QUALITY VALIDATION")

    # Calculate drift with interpolation (hybrid approach)
    total_frames = drift_data['image_index'].max() + 1
    drift_summary, interpolation_metadata = calculate_drift_with_interpolation(
        drift_data,
        selected_particles,
        total_frames,
        interpolation_method='cubic'
    )

    # Define output folder - use script_output/aligned
    output_folder = json_file.parent / 'aligned'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save complete drift trajectory with interpolation flags
    drift_csv_path = output_folder / 'drift_trajectory_complete.csv'
    drift_summary.to_csv(drift_csv_path, index=False)
    logger.info(f"\nComplete drift trajectory saved to: {drift_csv_path}")

    # Align images in parallel
    alignment_log = align_images_parallel(source_folder, drift_summary, output_folder, num_workers)

    # Save alignment log to CSV
    log_file = output_folder / 'alignment_log.csv'
    alignment_log.to_csv(log_file, index=False)
    logger.info(f"\nAlignment log saved to: {log_file}")

    # Print summary statistics
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total images processed: {len(alignment_log)}")
    logger.info(f"Successfully aligned: {alignment_log['aligned'].sum()}")
    logger.info(f"Failed/Skipped: {(~alignment_log['aligned']).sum()}")

    # Print drift correction sources
    logger.info(f"\nDrift correction sources:")
    logger.info(f"  - Averaged (≥2 particles): {interpolation_metadata['frames_averaged']} frames")
    logger.info(f"  - Single particle: {interpolation_metadata['frames_single_particle']} frames")
    logger.info(f"  - Interpolated: {interpolation_metadata['frames_interpolated']} frames")

    # Print drift statistics
    aligned_only = alignment_log[alignment_log['aligned'] == True]
    if len(aligned_only) > 0:
        logger.info(f"\nDrift correction applied:")
        logger.info(f"  Mean shift X: {aligned_only['avg_drift_x'].mean():.3f} pixels")
        logger.info(f"  Mean shift Y: {aligned_only['avg_drift_y'].mean():.3f} pixels")
        logger.info(f"  Max shift X: {aligned_only['avg_drift_x'].abs().max():.3f} pixels")
        logger.info(f"  Max shift Y: {aligned_only['avg_drift_y'].abs().max():.3f} pixels")

    logger.info(f"\nAligned images saved to: {output_folder}/")

    # Update JSON with alignment results
    logger.info("\nUpdating JSON with alignment results...")
    try:
        data['image_set']['aligned_folder_path'] = str(output_folder.absolute())
        data['image_set']['alignment_log_path'] = str(log_file.absolute())
        data['image_set']['alignment_completed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data['image_set']['interpolation_metadata'] = interpolation_metadata
        data['image_set']['drift_trajectory_complete_path'] = str(drift_csv_path.absolute())

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✓ JSON file updated: {json_file.name}")
        logger.info(f"  Aligned folder: {output_folder}")

    except Exception as e:
        logger.error(f"Warning: Could not update JSON file: {e}")
        log_exception(logger, e, "JSON update error")
    
    logger.info("\n=== DONE ===")

if __name__ == "__main__":
    main()
