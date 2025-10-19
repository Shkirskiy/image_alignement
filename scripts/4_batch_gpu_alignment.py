"""
Batch GPU Image Alignment
Processes multiple image sets sequentially using GPU-accelerated drift correction.

Reads JSON file (auto-discovered or specified) to get folder paths and drift CSV
file locations, then performs GPU-accelerated alignment for all image sets.

Requirements:
pip install torch torchvision tifffile pandas numpy tqdm pillow

Usage:
python batch_gpu_alignment.py [optional: path_to_json_file] [options]

Examples:
# Auto-discover JSON in scripts_output/ folder (must be only one JSON file)
python batch_gpu_alignment.py

# Manually specify JSON file path
python batch_gpu_alignment.py particle_selections_20251016_100525.json
python batch_gpu_alignment.py scripts_output/particle_selections_20251016_100525.json

# With optional parameters
python batch_gpu_alignment.py --batch_size 16 --interpolation bicubic --device cuda

# Manual JSON + options
python batch_gpu_alignment.py my_selections.json --batch_size 8 --interpolation bilinear

Output:
- Aligned images saved to: {original_folder}_aligned/ (sibling to input folder)
- JSON file updated with 'aligned_folder_path' for each image set
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Import GPU alignment module
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import tifffile
from tqdm import tqdm
from logging_utils import setup_logger, log_exception

# Setup centralized logger
logger = setup_logger('Step4_GPUAlignment')

class BatchGPUAligner:
    """
    Batch processor for multiple image sets using GPU alignment.
    """

    def __init__(self, json_file, batch_size=8,
                 interpolation='bilinear', device=None, suffix='_aligned'):
        """
        Initialize batch GPU aligner.

        Args:
            json_file: Path to particle_selections JSON
            batch_size: Batch size for GPU processing
            interpolation: Interpolation method ('bilinear' or 'bicubic')
            device: torch device (None for auto)
            suffix: Suffix for output folders
        """
        self.json_file = Path(json_file)
        self.batch_size = batch_size
        self.interpolation = interpolation
        self.suffix = suffix

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load configuration
        self.image_sets = []
        self.processing_tasks = []

    def load_configuration(self):
        """Load JSON and read drift CSV file paths from image sets."""
        logger.info(f"\nLoading configuration from: {self.json_file.name}")

        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")

        with open(self.json_file, 'r') as f:
            data = json.load(f)

        self.image_sets = data.get('image_sets', [])

        if not self.image_sets:
            raise ValueError("No image sets found in JSON!")

        logger.info(f"Found {len(self.image_sets)} image set(s) in JSON")

        # Read drift CSV paths directly from JSON
        for idx, img_set in enumerate(self.image_sets):
            folder_path = Path(img_set['folder_path'])
            folder_name = folder_path.name

            # Get drift CSV path from JSON (added by aggregate_particle_drift.py)
            drift_csv_path = img_set.get('drift_csv_file_path')
            drift_csv_name = img_set.get('drift_csv_file_name')

            if not drift_csv_path or not drift_csv_name:
                logger.warning(f"  Warning: Set {idx} missing drift CSV information, skipping...")
                logger.warning(f"    Make sure to run aggregate_particle_drift.py first!")
                continue

            drift_csv = Path(drift_csv_path)
            if not drift_csv.exists():
                logger.warning(f"  Warning: Drift CSV not found: {drift_csv}, skipping...")
                continue

            logger.info(f"  Set {idx}: {drift_csv_name}")

            # Create output directory (sibling to input folder with suffix)
            output_dir = folder_path.parent / f"{folder_name}{self.suffix}"

            task = {
                'index': idx,
                'folder_name': folder_name,
                'input_dir': folder_path,
                'output_dir': output_dir,
                'drift_csv': drift_csv,
                'total_images': img_set.get('total_images', 0),
                'particles': len(img_set.get('selected_particles', []))
            }

            self.processing_tasks.append(task)

            logger.info(f"\n  Set {idx}: {folder_name}")
            logger.info(f"    Input: {folder_path}")
            logger.info(f"    Output: {output_dir}")
            logger.info(f"    Drift CSV: {drift_csv.name}")
            logger.info(f"    Images: {task['total_images']}, Particles: {task['particles']}")

        if not self.processing_tasks:
            raise ValueError("No valid processing tasks found!")

        logger.info(f"\nReady to process {len(self.processing_tasks)} image set(s)")

        return True

    def process_single_set(self, task):
        """
        Process a single image set with GPU alignment.

        Args:
            task: Processing task dictionary

        Returns:
            Processing statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Set {task['index']}: {task['folder_name']}")
        logger.info(f"{'='*70}")

        start_time = time.time()

        # Create output directory
        task['output_dir'].mkdir(parents=True, exist_ok=True)

        # Load drift data
        logger.info(f"Loading drift data from: {task['drift_csv'].name}")
        try:
            drift_df = pd.read_csv(task['drift_csv'])

            # Validate columns
            required_cols = ['filename', 'dx_pixels', 'dy_pixels',
                           'rotation_degrees', 'is_reference_frame']
            missing = [c for c in required_cols if c not in drift_df.columns]
            if missing:
                logger.error(f"  Error: Missing columns in CSV: {missing}")
                return None

            logger.info(f"  Loaded drift data for {len(drift_df)} images")

            # Find reference frame
            ref_frames = drift_df[drift_df['is_reference_frame'] == True]
            if len(ref_frames) > 0:
                ref_frame = ref_frames.iloc[0]['filename']
                logger.info(f"  Reference frame: {ref_frame}")
            else:
                logger.warning(f"  Warning: No reference frame marked")

            # Create drift mapping
            drift_data = {}
            for _, row in drift_df.iterrows():
                drift_data[row['filename']] = {
                    'dx': row['dx_pixels'],
                    'dy': row['dy_pixels'],
                    'rotation_deg': row['rotation_degrees'],
                    'rotation_rad': np.radians(row['rotation_degrees'])
                }

        except Exception as e:
            logger.error(f"  Error loading drift CSV: {e}")
            log_exception(logger, e, "Drift CSV loading error")
            return None

        # Find available images
        input_dir = task['input_dir']
        tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        tif_files = []
        for pattern in tif_patterns:
            tif_files.extend(list(input_dir.glob(pattern)))
        tif_files.sort()

        if not tif_files:
            logger.error(f"  Error: No TIF files found in {input_dir}")
            return None

        logger.info(f"  Found {len(tif_files)} TIF files")

        # Match images with drift data
        images_to_process = []
        for tif_file in tif_files:
            if tif_file.name in drift_data:
                images_to_process.append((tif_file, drift_data[tif_file.name]))

        logger.info(f"  Matched {len(images_to_process)} images with drift data")

        if len(images_to_process) == 0:
            logger.error(f"  Error: No images matched with drift data!")
            return None

        # Process images in batches
        stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        with tqdm(total=len(images_to_process),
                 desc=f"  Set {task['index']}: {task['folder_name'][:30]}",
                 unit="img",
                 position=1,
                 leave=True) as pbar:

            for i in range(0, len(images_to_process), self.batch_size):
                batch_items = images_to_process[i:i + self.batch_size]

                try:
                    self.process_batch_gpu(batch_items, task['output_dir'], stats)
                    pbar.update(len(batch_items))

                except Exception as e:
                    logger.error(f"\n  Error processing batch: {e}")
                    log_exception(logger, e, "Batch processing error")
                    stats['errors'] += len(batch_items)
                    pbar.update(len(batch_items))
                    continue

        processing_time = time.time() - start_time

        # Print statistics
        logger.info(f"\n  Set {task['index']} Complete:")
        logger.info(f"    Processed: {stats['processed']}")
        logger.info(f"    Skipped: {stats['skipped']}")
        logger.info(f"    Errors: {stats['errors']}")
        logger.info(f"    Time: {processing_time:.1f}s")
        if stats['processed'] > 0:
            logger.info(f"    Speed: {stats['processed']/processing_time:.1f} img/s")

        return {
            'task': task,
            'stats': stats,
            'processing_time': processing_time
        }

    def process_batch_gpu(self, batch_items, output_dir, stats):
        """Process a batch of images on GPU."""
        image_tensors = []
        valid_items = []

        # Load images
        for img_path, drift_info in batch_items:
            tensor = self.load_image_as_tensor(img_path)
            if tensor is not None:
                image_tensors.append(tensor)
                valid_items.append((img_path, drift_info))
            else:
                stats['skipped'] += 1

        if not image_tensors:
            return

        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0)
        batch_size, channels, height, width = batch_tensor.shape

        # Create transformation matrices
        theta_batch = []
        for _, drift_info in valid_items:
            theta = self.create_transformation_matrix(
                drift_info['dx'],
                drift_info['dy'],
                drift_info['rotation_rad'],
                height, width
            )
            theta_batch.append(theta)

        theta_batch = torch.cat(theta_batch, dim=0)

        # Apply transformations
        grid = F.affine_grid(theta_batch, batch_tensor.size(), align_corners=False)
        aligned_batch = F.grid_sample(
            batch_tensor, grid,
            mode=self.interpolation,
            padding_mode='border',
            align_corners=False
        )

        # Save aligned images
        for i, (img_path, drift_info) in enumerate(valid_items):
            aligned_tensor = aligned_batch[i:i+1]
            output_name = f"{img_path.stem}_aligned{img_path.suffix}"
            output_path = output_dir / output_name

            self.save_aligned_image(aligned_tensor, output_path, img_path)
            stats['processed'] += 1

        # Clean up GPU memory
        del batch_tensor, aligned_batch, grid, theta_batch
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def create_transformation_matrix(self, dx, dy, rotation_rad, height, width):
        """Create affine transformation matrix."""
        cos_r = torch.cos(torch.tensor(rotation_rad))
        sin_r = torch.sin(torch.tensor(rotation_rad))

        norm_dx = 2.0 * dx / width
        norm_dy = 2.0 * dy / height

        theta = torch.tensor([
            [cos_r, -sin_r, norm_dx],
            [sin_r,  cos_r, norm_dy]
        ], dtype=torch.float32, device=self.device)

        return theta.unsqueeze(0)

    def load_image_as_tensor(self, image_path):
        """Load image as PyTorch tensor."""
        try:
            if image_path.suffix.lower() in ['.tif', '.tiff']:
                image = tifffile.imread(str(image_path))

                if image.ndim == 2:
                    image = np.expand_dims(image, axis=0)
                elif image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                    pass
                elif image.ndim == 3:
                    image = np.transpose(image, (2, 0, 1))
                else:
                    return None

                if image.dtype in [np.uint8, np.uint16]:
                    image = image.astype(np.float32)
                    if image.max() > 1.0:
                        image = image / (255.0 if image.dtype == np.uint8 else 65535.0)

                tensor = torch.from_numpy(image).unsqueeze(0)
            else:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                tensor = TF.to_tensor(image).unsqueeze(0)

            return tensor.to(self.device)

        except Exception as e:
            return None

    def save_aligned_image(self, tensor, output_path, original_path):
        """Save aligned image tensor to file."""
        try:
            tensor = tensor.cpu().squeeze(0)

            if tensor.dim() == 3:
                image = tensor.permute(1, 2, 0).numpy()
            else:
                image = tensor.numpy()

            if original_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    original = tifffile.imread(str(original_path))
                    if original.dtype == np.uint8:
                        image = (image * 255).astype(np.uint8)
                    elif original.dtype == np.uint16:
                        image = (image * 65535).astype(np.uint16)
                    else:
                        image = image.astype(np.float32)
                except:
                    image = (image * 255).astype(np.uint8)

                tifffile.imwrite(str(output_path), image)
            else:
                if image.ndim == 3 and image.shape[2] == 1:
                    image = image.squeeze(2)

                image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                pil_image.save(output_path)

        except Exception as e:
            pass

    def process_all_sets(self):
        """Process all image sets sequentially."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING BATCH GPU ALIGNMENT")
        logger.info(f"{'='*70}")
        logger.info(f"Total image sets: {len(self.processing_tasks)}")
        logger.info(f"Processing mode: Sequential (one set at a time)")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Interpolation: {self.interpolation}")

        total_start = time.time()
        results = []

        # Add overall progress bar for all image sets
        for task in tqdm(self.processing_tasks,
                        desc="Overall Progress",
                        unit="set",
                        position=0,
                        leave=True):
            result = self.process_single_set(task)
            if result:
                results.append(result)

        total_time = time.time() - total_start

        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH ALIGNMENT COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"\nSummary by image set:")

        total_processed = 0
        total_skipped = 0
        total_errors = 0

        for result in results:
            task = result['task']
            stats = result['stats']
            logger.info(f"\n  Set {task['index']}: {task['folder_name']}")
            logger.info(f"    Processed: {stats['processed']}")
            logger.info(f"    Skipped: {stats['skipped']}")
            logger.info(f"    Errors: {stats['errors']}")
            logger.info(f"    Output: {task['output_dir']}")

            total_processed += stats['processed']
            total_skipped += stats['skipped']
            total_errors += stats['errors']

        print(f"\nOverall totals:")
        print(f"  Total images processed: {total_processed}")
        print(f"  Total skipped: {total_skipped}")
        print(f"  Total errors: {total_errors}")

        if total_processed > 0:
            print(f"  Average speed: {total_processed/total_time:.1f} images/second")

        # Update JSON file with aligned folder paths
        if results:
            print("\nUpdating JSON file with aligned folder paths...")
            try:
                # Load JSON data
                with open(self.json_file, 'r') as f:
                    data = json.load(f)

                # Update each image set with aligned folder path
                for result in results:
                    set_idx = result['task']['index']
                    output_dir = str(result['task']['output_dir'].absolute())

                    # Add aligned_folder_path to the image set
                    if set_idx < len(data['image_sets']):
                        data['image_sets'][set_idx]['aligned_folder_path'] = output_dir

                # Save updated JSON back to file
                with open(self.json_file, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"✓ JSON file updated: {self.json_file.name}")
                print("  Added 'aligned_folder_path' to each processed image set")

            except Exception as e:
                print(f"Warning: Could not update JSON file: {e}")

        print(f"\n{'='*70}")
        print("All image sets have been aligned!")
        print(f"{'='*70}\n")

        return results

def main():
    """Main function to run batch GPU alignment."""
    print("=== Batch GPU Image Alignment ===\n")

    # Parse optional arguments first
    parser = argparse.ArgumentParser(
        description="Batch GPU image alignment for multiple image sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help manually
    )

    parser.add_argument('--batch_size', '-b', type=int, default=8,
                       help='Batch size for GPU processing (default: 8)')
    parser.add_argument('--interpolation', choices=['bilinear', 'bicubic'], default='bilinear',
                       help='Interpolation method (default: bilinear)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device to use (default: auto)')
    parser.add_argument('--suffix', type=str, default='_aligned',
                       help='Suffix for output folders (default: _aligned)')
    parser.add_argument('--help', '-h', action='store_true',
                       help='Show this help message and exit')

    # Parse known args to allow positional JSON argument
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    # Determine JSON file location (identical logic to parallel_drift_analysis.py)
    json_file = None

    # Check if first positional argument is provided (not starting with --)
    if remaining and not remaining[0].startswith('--'):
        # User provided JSON path manually
        json_file = Path(remaining[0])
        if not json_file.exists():
            logger.error(f"Error: File not found: {json_file}")
            sys.exit(1)
        logger.info(f"Using user-specified JSON file: {json_file}")
    else:
        # Try to auto-discover JSON in scripts_output folder (parent directory)
        scripts_output_dir = Path('../scripts_output')

        if not scripts_output_dir.exists():
            print("Error: 'scripts_output' folder not found!")
            print("\nPlease either:")
            print("  1. Create the 'scripts_output' folder and place your JSON file there")
            print("  2. Pass the JSON file path manually:")
            print("     python batch_gpu_alignment.py <path_to_json_file>")
            sys.exit(1)

        # Find all JSON files in scripts_output
        json_files = list(scripts_output_dir.glob('*.json'))

        if len(json_files) == 0:
            print(f"Error: No JSON files found in '{scripts_output_dir}/'")
            print("\nPlease pass the JSON file path manually:")
            print("  python batch_gpu_alignment.py <path_to_json_file>")
            sys.exit(1)
        elif len(json_files) == 1:
            json_file = json_files[0]
            logger.info(f"Auto-discovered JSON file: {json_file.name}")
        else:
            print(f"Error: Multiple JSON files found in '{scripts_output_dir}/':")
            for jf in json_files:
                print(f"  - {jf.name}")
            print("\nPlease specify which JSON file to use:")
            print("  python batch_gpu_alignment.py <path_to_json_file>")
            sys.exit(1)

    device = None if args.device == 'auto' else args.device

    try:
        # Create batch aligner
        aligner = BatchGPUAligner(
            json_file=json_file,
            batch_size=args.batch_size,
            interpolation=args.interpolation,
            device=device,
            suffix=args.suffix
        )

        # Load configuration
        aligner.load_configuration()

        # Process all sets
        aligner.process_all_sets()

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
