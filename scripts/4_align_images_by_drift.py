"""
Align Images by Particle Drift
Aligns a series of images based on the average drift of selected particles.
Uses particles selected in script 3 to calculate average displacement and applies inverse shift.

Requirements:
pip install pandas numpy scipy tifffile tqdm

Usage:
python 4_align_images_by_drift.py

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

import pandas as pd
import numpy as np
import tifffile
from scipy.ndimage import shift as ndimage_shift
from pathlib import Path
import json
from tqdm import tqdm
import sys
from datetime import datetime

def load_drift_data(csv_path: Path, particle_ids: list) -> pd.DataFrame:
    """
    Load drift data for specified particles only.

    Args:
        csv_path: Path to tracking CSV file
        particle_ids: List of particle IDs to include (e.g., [1, 3, 6])

    Returns:
        DataFrame with drift data for selected particles
    """
    print(f"Loading drift data from: {csv_path.name}")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Filter for specified particles and successful fits only
    df = df[(df['particle_id'].isin(particle_ids)) & (df['success'] == True)].copy()

    print(f"Loaded {len(df)} successful fits for particles {particle_ids}")
    print(f"Frame range: {df['image_index'].min()} to {df['image_index'].max()}")

    return df

def calculate_average_drift(df: pd.DataFrame, particle_ids: list) -> pd.DataFrame:
    """
    Calculate average drift across selected particles for each frame.

    Args:
        df: DataFrame with particle tracking data
        particle_ids: List of particle IDs used for averaging

    Returns:
        DataFrame with columns: image_index, filename, avg_drift_x, avg_drift_y, num_particles
    """
    print("\nCalculating average drift per frame...")

    # Group by image_index and filename, calculate mean drift
    drift_summary = df.groupby(['image_index', 'filename']).agg({
        'drift_x': 'mean',
        'drift_y': 'mean',
        'particle_id': 'count'  # Count how many particles contributed
    }).reset_index()

    # Rename columns for clarity
    drift_summary.rename(columns={
        'drift_x': 'avg_drift_x',
        'drift_y': 'avg_drift_y',
        'particle_id': 'num_particles'
    }, inplace=True)

    # Filter out frames with too few particles (need at least 2 for robust average)
    initial_count = len(drift_summary)
    drift_summary = drift_summary[drift_summary['num_particles'] >= 2].copy()
    filtered_count = initial_count - len(drift_summary)

    if filtered_count > 0:
        print(f"  Warning: Filtered out {filtered_count} frames with <2 particles")

    print(f"  Average drift calculated for {len(drift_summary)} frames")
    print(f"  Mean drift X: {drift_summary['avg_drift_x'].mean():.3f} pixels")
    print(f"  Mean drift Y: {drift_summary['avg_drift_y'].mean():.3f} pixels")
    print(f"  Max drift X: {drift_summary['avg_drift_x'].abs().max():.3f} pixels")
    print(f"  Max drift Y: {drift_summary['avg_drift_y'].abs().max():.3f} pixels")

    return drift_summary

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

def align_images(source_folder: Path, drift_summary: pd.DataFrame, output_folder: Path,
                 skip_existing: bool = False) -> pd.DataFrame:
    """
    Align all images based on drift data.

    Args:
        source_folder: Folder containing original TIF images
        drift_summary: DataFrame with avg_drift_x, avg_drift_y per frame
        output_folder: Folder to save aligned images
        skip_existing: If True, skip already processed images

    Returns:
        DataFrame with alignment log
    """
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nAligning images...")
    print(f"  Source: {source_folder}")
    print(f"  Output: {output_folder}")

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

    # Process images with progress bar
    alignment_log = []
    skipped_count = 0
    no_drift_count = 0
    aligned_count = 0

    with tqdm(total=len(tif_files), desc="Aligning images", unit="img", ncols=100) as pbar:
        for tif_file in tif_files:
            filename = tif_file.name
            output_file = output_folder / filename

            # Skip if already exists and skip_existing is True
            if skip_existing and output_file.exists():
                skipped_count += 1
                pbar.update(1)
                continue

            # Check if we have drift data for this file
            if filename not in drift_dict:
                # No drift data - just copy the original image
                try:
                    image = tifffile.imread(str(tif_file))
                    tifffile.imwrite(str(output_file), image)
                    no_drift_count += 1

                    alignment_log.append({
                        'filename': filename,
                        'avg_drift_x': 0.0,
                        'avg_drift_y': 0.0,
                        'num_particles': 0,
                        'aligned': False,
                        'status': 'no_drift_data'
                    })
                except Exception as e:
                    print(f"\n  Error copying {filename}: {e}")
                    alignment_log.append({
                        'filename': filename,
                        'avg_drift_x': 0.0,
                        'avg_drift_y': 0.0,
                        'num_particles': 0,
                        'aligned': False,
                        'status': f'error: {e}'
                    })

                pbar.update(1)
                continue

            # Get drift data
            drift_data = drift_dict[filename]
            shift_x = drift_data['avg_drift_x']
            shift_y = drift_data['avg_drift_y']

            try:
                # Load image
                image = tifffile.imread(str(tif_file))

                # Check if image is 2D grayscale
                if image.ndim != 2:
                    print(f"\n  Warning: {filename} is not 2D grayscale, skipping alignment")
                    tifffile.imwrite(str(output_file), image)
                    alignment_log.append({
                        'filename': filename,
                        'avg_drift_x': shift_x,
                        'avg_drift_y': shift_y,
                        'num_particles': drift_data['num_particles'],
                        'aligned': False,
                        'status': 'not_2d'
                    })
                    pbar.update(1)
                    continue

                # Align image
                aligned_image = align_image(image, shift_x, shift_y)

                # Ensure output has same dtype as input
                aligned_image = aligned_image.astype(image.dtype)

                # Save aligned image
                tifffile.imwrite(str(output_file), aligned_image)

                aligned_count += 1

                alignment_log.append({
                    'filename': filename,
                    'avg_drift_x': shift_x,
                    'avg_drift_y': shift_y,
                    'num_particles': drift_data['num_particles'],
                    'aligned': True,
                    'status': 'success'
                })

            except Exception as e:
                print(f"\n  Error processing {filename}: {e}")
                alignment_log.append({
                    'filename': filename,
                    'avg_drift_x': shift_x,
                    'avg_drift_y': shift_y,
                    'num_particles': drift_data['num_particles'],
                    'aligned': False,
                    'status': f'error: {e}'
                })

            pbar.update(1)

    print(f"\n  Aligned: {aligned_count} images")
    print(f"  No drift data (copied): {no_drift_count} images")
    print(f"  Skipped (already exists): {skipped_count} images")

    return pd.DataFrame(alignment_log)

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
    
    logger.info("=== Image Alignment by Particle Drift ===\n")
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

    # Load drift data
    drift_data = load_drift_data(csv_file, selected_particles)

    if len(drift_data) == 0:
        logger.error("No drift data found for selected particles!")
        sys.exit(1)

    # Calculate average drift
    drift_summary = calculate_average_drift(drift_data, selected_particles)

    # Define output folder - use script_output/aligned
    output_folder = json_file.parent / 'aligned'

    # Align images
    alignment_log = align_images(source_folder, drift_summary, output_folder, skip_existing=False)

    # Save alignment log to CSV
    log_file = output_folder / 'alignment_log.csv'
    alignment_log.to_csv(log_file, index=False)
    logger.info(f"\nAlignment log saved to: {log_file}")

    # Print summary statistics
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total images processed: {len(alignment_log)}")
    logger.info(f"Successfully aligned: {alignment_log['aligned'].sum()}")
    logger.info(f"Failed/Skipped: {(~alignment_log['aligned']).sum()}")

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
