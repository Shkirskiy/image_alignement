# Image Alignment Pipeline

This pipeline is designed to align large sequences of 16-bit .tif images from high-resolution monochromatic cameras. Image misalignment may result from vibrations, thermal drift, or other artifacts. It has been tested on the example datasets provided in this repository and validated with real experimental data published in scientific articles [1-4].

## References

1. https://doi.org/10.1016/j.corsci.2025.113184
2. https://doi.org/10.3390/met15080821
3. https://doi.org/10.1021/acs.analchem.3c04160
4. https://doi.org/10.1002/smtd.202300214

## Method Overview

The core methodology is based on manual selection of particles, followed by Gaussian fitting to precisely determine and track their locations. The pipeline then applies drift corrections to transpose images based on the measured particle movements. The approach assumes your images have: rigid body motion (translation + rotation, no shearing or scaling), multiple trackable particles that move together, small incremental drift between consecutive frames. The transformation treats the image as a rigid plane that has shifted and rotated slightly, and the alignment corrects this by applying the inverse transformation to bring everything back into a common reference frame.

The following example shows typical results from `example_data1` before and after alignment (center mark added for visual reference):

### Before Alignment

![Before Alignment](example_data/only_for_illustration/cropped_before.png)

### After Alignment

![After Alignment](example_data/only_for_illustration/cropped_after.png)

A detailed technical description of the pipeline is provided below.

## Project Structure

```
image_alignmenetv2/
├── scripts/                          # All Python scripts
│   ├── 1_select_particles_for_drift.py
│   ├── 2_drift_analysis.py
│   ├── 3_interactive_trajectory_selector.py
│   ├── 4_align_images_by_drift.py
│   ├── 5_validation.py
│   └── logging_utils.py             # Centralized logging utility
├── run_pipeline.sh                   # Automated pipeline runner
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

# Output Structure (created next to your image folder)
/path/to/your/images/
├── exported/                         # Your original images
└── script_output/                    # All pipeline outputs (created automatically)
    ├── particle_selections.json     # Configuration (single image set)
    ├── log.txt                      # Centralized log file with timestamps
    ├── particles_tracking/          # Per-particle drift CSVs
    ├── drift_analysis/              # Averaged drift trajectories
    ├── aligned/                     # Aligned images
    └── validation/                  # Validation PNGs
```

## Quick Start

### Option 1: Run Entire Pipeline (Recommended)

```bash
./run_pipeline.sh
```

This will automatically run all 5 steps in sequence.

### Option 2: Run Scripts Individually

```bash
cd scripts

# Step 1: Interactive particle selection
python3 1_select_particles_for_drift.py

# Step 2: Drift analysis
python3 2_drift_analysis.py

# Step 3: Interactive trajectory selector
python3 3_interactive_trajectory_selector.py

# Step 4: Image alignment
python3 4_align_images_by_drift.py

# Step 5: Validation
python3 5_validation.py
```

## Pipeline Steps

### 1. Particle Selection (`1_select_particles_for_drift.py`)

- **Interactive** napari GUI for selecting particles to track
- Validates each selection with real-time Gaussian fitting
- Green markers = successful fits (saved), Red X = failed fits (rejected)
- Single image set workflow (simplified)
- Saves selections to JSON in `script_output/` (next to image folder)
- Logs all activity to console and `log.txt`

### 2. Drift Analysis (`2_drift_analysis.py`)

- **Parallelized** processing for improved performance
- Uses multiprocessing to perform Gaussian fitting across multiple CPU cores
- Processes all (image, particle) combinations in parallel
- Worker count: **1/4 of CPU cores** (optimized for HDD read operations)
- Each worker limited to single-threaded NumPy/SciPy operations
- Implements image caching (20 images per worker) to minimize disk I/O
- Real-time progress tracking with tqdm progress bar
- Calculates drift (displacement from first frame) for each particle
- Outputs per-particle tracking CSV to `script_output/particles_tracking/`
- Updates JSON with CSV file paths
- Logs progress and statistics

### 3. Interactive Trajectory Selector (`3_interactive_trajectory_selector.py`)

- **Interactive** GUI to visualize and select particles for drift correction
- Shows absolute trajectories and drift preview
- Select which particles to use for averaging
- Generates averaged drift trajectory and individual trajectory PNGs
- Outputs to `script_output/drift_analysis/`
- Updates JSON with selected particle IDs and drift file paths

### 4. Image Alignment (`4_align_images_by_drift.py`)

- Applies drift correction using selected particles
- CPU-based alignment with scipy's ndimage.shift
- Reads configuration automatically from JSON
- Creates aligned images in `script_output/aligned/`
- Saves alignment log with applied shifts
- Updates JSON with aligned folder path

### 5. Validation (`5_validation.py`)

- **Parallelized** processing for improved performance (same as Script 2)
- Uses multiprocessing to perform Gaussian fitting across multiple CPU cores
- Processes all (aligned_image, selected_particle) combinations in parallel
- Worker count: **1/4 of CPU cores** (optimized for HDD read operations)
- Each worker limited to single-threaded NumPy/SciPy operations
- Implements image caching (20 images per worker) to minimize disk I/O
- Real-time progress tracking with tqdm progress bar
- **Re-fits Gaussians on aligned images** to validate drift correction effectiveness
- Processes ONLY selected particles from Step 3
- Exports validation tracking CSV (same format as Step 2)
- Generates before/after comparison plots for each particle
- Creates summary plot with drift reduction statistics and table
- Calculates quantitative drift reduction percentages
- **Proves alignment worked** by demonstrating reduced drift in aligned images
- Outputs to `script_output/validation/`
- Updates JSON with validation folder and CSV paths

## Requirements

### Installation

Install all dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install napari[all] tifffile numpy scipy scikit-image magicgui
pip install pandas tqdm matplotlib
```

### System Requirements

- Python 3.8 or higher
- 8+ GB RAM recommended
- No GPU required (CPU-based processing)

## How It Works

### Automatic Configuration

All scripts automatically discover and load configuration:

- **Script 1** creates `script_output/` folder next to your image folder and outputs JSON path
- **When using pipeline runner**: JSON path passed automatically from Script 1 to Scripts 2-5
- **When running individually**: Scripts 2-5 auto-discover the JSON file location
- No manual path configuration required from user
- All outputs saved in `script_output/` structure

### JSON Evolution

The `particle_selections.json` file evolves through the pipeline:

1. **After Script 1**: Contains selected particles and image folder path
2. **After Script 2**: Adds CSV file paths for particle tracking
3. **After Script 3**: Adds selected particle IDs for drift correction + drift CSV paths
4. **After Script 4**: Adds aligned images folder path + alignment log paths
5. **After Script 5**: Adds validation folder path + validation CSV paths

## Output Files

### script_output/ (created next to your images)

- `particle_selections.json` - Central configuration (single image set)
- `log.txt` - Centralized log with timestamps from all steps
- `particles_tracking/{folder}_{id}.csv` - Per-particle tracking data
- `drift_analysis/averaged_drift_trajectory.csv` - Averaged drift data
- `drift_analysis/averaged_drift_trajectory.png` - Drift visualization
- `drift_analysis/absolute_trajectories.png` - Absolute position plots
- `aligned/{image_files}.tif` - Aligned images
- `aligned/alignment_log.csv` - Applied shifts per frame
- `validation/aligned_particles_tracking.csv` - Re-fitted positions on aligned images
- `validation/particle_{id}_comparison.png` - Before/after comparison plots
- `validation/drift_reduction_summary.png` - Overall validation with statistics table

## Usage Notes

1. **Run from project root** or use `./run_pipeline.sh`
2. **Single image set workflow** - process one folder at a time
3. **Interactive steps**: Scripts 1 and 3 require user interaction
4. **Automatic steps**: Scripts 2, 4, and 5 run automatically
5. **All activity logged** to `script_output/log.txt` with timestamps
6. **No manual path passing** - scripts auto-discover everything from JSON

### Running with Pipeline vs Standalone

- **Using `./run_pipeline.sh`**: JSON path automatically passed between scripts via shell
- **Running scripts individually**: Scripts auto-discover JSON file location
- Both methods work identically from user perspective
- Pipeline runner recommended for full workflow execution

## Performance & Optimization

### Parallelization Strategy

Scripts 2 and 4 use **multiprocessing** to significantly speed up processing:

#### Script 2: Drift Analysis (Gaussian Fitting)
- **Worker count**: 1/4 of CPU cores (e.g., 12 cores → 3 workers)
- **Operations**: Only reads images from disk
- **Bottleneck**: CPU-intensive Gaussian fitting
- **Image caching**: 20 images per worker to minimize repeated disk reads
- **Why 1/4?**: Balances parallelization with HDD read contention

#### Script 4: Image Alignment (Read + Write)
- **Worker count**: Fixed at 4 workers (regardless of CPU count)
- **Operations**: Reads original images AND writes aligned images to disk
- **Bottleneck**: HDD write queue saturation
- **Why fixed 4?**: Conservative setting prevents disk thrashing with simultaneous read+write operations

### Thread Limiting

Both parallelized scripts limit NumPy/SciPy to **single-threaded operations** per worker:

```python
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
```

**Why?** Without this, each worker spawns multiple threads internally, leading to:
- Total threads = num_workers × threads_per_worker
- Example: 6 workers × 2 threads = 12 threads → all CPUs loaded
- Result: System becomes unresponsive for other tasks

**With thread limiting**: 6 workers × 1 thread = 6 threads → half of CPUs remain free

### HDD vs SSD Considerations

**Current settings optimized for HDD:**
- HDDs struggle with parallel I/O due to physical read/write head movement
- Multiple workers reading/writing simultaneously cause disk thrashing
- Symptoms: slowdowns after initial fast period, unusual disk noises

**If you have an SSD:**
You can increase worker counts for better performance:

**Script 2** (`scripts/2_drift_analysis.py`):
```python
# Change from:
num_workers = max(1, total_cores // 4)

# To (for SSD):
num_workers = max(1, total_cores // 2)  # Use half of cores
```

**Script 4** (`scripts/4_align_images_by_drift.py`):
```python
# Change from:
num_workers = 4

# To (for SSD):
num_workers = 8  # Or more, experiment to find optimal
```

### Hardware-Specific Adjustments

**The default settings (1/4 cores for Script 2, 4 workers for Script 4) work well for:**
- Standard HDDs (5400-7200 RPM)
- Systems with 8-32 CPU cores
- Mixed workload scenarios (running other tasks simultaneously)

**You may need to adjust if you have:**

1. **Very high core count (>64 cores)**:
   - Script 2's "1/4 cores" may still overwhelm HDD
   - Consider using fixed worker count instead of fractional

2. **NVMe SSD or RAID array**:
   - Much higher I/O capacity
   - Can safely increase worker counts significantly
   - Experiment with 2x-4x the default values

3. **Low RAM (<8 GB)**:
   - Reduce image cache size in Script 2:
   ```python
   _cache_max_size = 10  # Instead of 20
   ```

4. **Need maximum single-task performance**:
   - Script 2: Increase to 1/2 or 3/4 of cores
   - Script 4: Increase to 6-8 workers
   - Remove thread limits to use multi-threaded NumPy/SciPy

### Performance Monitoring

**Signs of optimal performance:**
- Consistent processing speed throughout
- CPU usage at expected level (e.g., 25% for Script 2)
- No disk thrashing sounds
- System remains responsive

**Signs of over-parallelization:**
- Fast start, then dramatic slowdown
- Disk making excessive seeking noises
- CPU usage drops despite work remaining
- System becomes unresponsive

**Recommended approach:**
1. Start with default settings
2. Monitor system performance during first run
3. Adjust worker counts if needed
4. Test and iterate to find optimal settings for your hardware

## Logging System

All scripts use centralized logging (`logging_utils.py`):

- Writes to **both console and `log.txt`**
- Includes timestamps for all operations
- Log file persists across runs (appends)
- Useful for debugging and tracking progress
- Located in `script_output/` next to your images

## Troubleshooting

**"Error: Cannot find particle_selections.json!"**

- Run script 1 first to create the configuration
- Make sure you selected a valid image folder

**"No TIF files found"**

- Check that your folder contains .tif or .tiff files
- Ensure file permissions allow reading

**"No selected particles in JSON"**

- You need to run script 3 to select particles for drift correction
- Script 3 must complete successfully (save, not cancel)

**Running individual scripts**

- All scripts can run from `scripts/` directory
- Use: `cd scripts && python3 script_name.py`
- Or use the pipeline runner: `./run_pipeline.sh`

## Expected Results

After successful completion:

- **Script 1**: JSON created with selected particles
- **Script 2**: CSV with particle positions over time (with progress bars)
- **Script 3**: Selected particles for drift, averaged trajectory
- **Script 4**: Aligned images in `script_output/aligned/`
- **Script 5**: Validation CSV + comparison plots proving drift reduction (quantitative)

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International License). You are free to share and adapt this work for any purpose, even commercially, as long as you provide appropriate credit.

## Author

**Viacheslav (Slava) Shkirskiy**  
Website: [https://slava-shkirskiy.org/](https://slava-shkirskiy.org/)  
Contact: viacheslava.shkirskiy@cnrs.fr

For support, questions, or feedback, please contact via email.
