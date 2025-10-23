"""
Interactive Trajectory Selector
Opens an interactive GUI window to visualize all particle trajectories and select which ones to use for drift correction.

Requirements:
pip install pandas matplotlib numpy

Usage:
python 3_interactive_trajectory_selector.py

The script automatically finds the CSV file from the JSON configuration.

Features:
- Interactive plot with zoom/pan capabilities
- Checkboxes to select which particles to include in drift calculation
- Preview averaged DRIFT trajectory before saving (all start at 0,0)
- Saves PNG plot and CSV file with averaged drift trajectory
- Updates JSON with selected particle IDs

Output:
- PNG: script_output/drift_analysis/averaged_drift_trajectory.png
- CSV: script_output/drift_analysis/averaged_drift_trajectory.csv
- JSON updated with selected particle IDs
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

class TrajectorySelector:
    """Interactive GUI for selecting and averaging particle drift trajectories."""

    def __init__(self, df: pd.DataFrame, particle_ids: list, output_dir: Path):
        """
        Initialize the trajectory selector.

        Args:
            df: DataFrame with particle tracking data
            particle_ids: List of all particle IDs
            output_dir: Directory to save outputs
        """
        self.df = df
        self.particle_ids = sorted(particle_ids)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize selection state (all particles selected by default)
        self.selected_particles = {pid: True for pid in self.particle_ids}

        # Store trajectory data for each particle
        self.trajectories = {}
        for pid in self.particle_ids:
            particle_data = df[df['particle_id'] == pid].sort_values('image_index')
            self.trajectories[pid] = particle_data

        # Color map for particles
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.particle_ids)))
        self.color_map = {pid: self.colors[i] for i, pid in enumerate(self.particle_ids)}

        # Store plot elements for main plot
        self.trajectory_lines = {}
        self.start_markers = {}
        self.particle_labels = {}

        # Store plot elements for drift preview plot
        self.drift_lines = {}
        self.avg_drift_line = None

        # Create the GUI
        self.create_gui()

    def create_gui(self):
        """Create the interactive GUI window."""
        # Create figure with custom layout - horizontal arrangement
        self.fig = plt.figure(figsize=(20, 8))
        self.fig.canvas.manager.set_window_title('Interactive Trajectory Selector - Drift Analysis')

        # Three-column layout:
        # Left: Absolute trajectories (40% width)
        # Middle: Drift preview (40% width)
        # Right: Control buttons (20% width)

        # Left plot: Absolute trajectories
        self.ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=4, rowspan=10)

        # Middle plot: Averaged drift preview
        self.ax_drift = plt.subplot2grid((10, 10), (0, 4), colspan=4, rowspan=10)

        # Create control panel axes (right side)
        ax_checkbox = plt.subplot2grid((10, 10), (0, 8), colspan=2, rowspan=6)
        ax_preview_btn = plt.subplot2grid((10, 10), (6, 8), colspan=2, rowspan=1)
        ax_save_btn = plt.subplot2grid((10, 10), (7, 8), colspan=2, rowspan=1)
        ax_cancel_btn = plt.subplot2grid((10, 10), (8, 8), colspan=2, rowspan=1)
        ax_status = plt.subplot2grid((10, 10), (9, 8), colspan=2, rowspan=1)

        # Remove axes for control panels
        ax_checkbox.axis('off')
        ax_preview_btn.axis('off')
        ax_save_btn.axis('off')
        ax_cancel_btn.axis('off')
        ax_status.axis('off')

        # Plot all trajectories (absolute positions)
        self.plot_trajectories()

        # Initialize drift preview plot (empty initially)
        self.plot_drift_preview()

        # Create checkboxes for particle selection
        checkbox_labels = [f'Particle {pid}' for pid in self.particle_ids]
        checkbox_states = [self.selected_particles[pid] for pid in self.particle_ids]

        # Position checkboxes in the checkbox axis
        self.check_buttons = CheckButtons(
            ax_checkbox,
            checkbox_labels,
            checkbox_states
        )

        # Color checkbox labels to match trajectory colors (LARGER FONT)
        for i, label in enumerate(self.check_buttons.labels):
            label.set_color(self.color_map[self.particle_ids[i]])
            label.set_fontsize(14)  # Increased from 10
            label.set_weight('bold')

        # Connect checkbox callback
        self.check_buttons.on_clicked(self.on_checkbox_clicked)

        # Create buttons (LARGER FONTS)
        self.preview_button = Button(ax_preview_btn, 'Update Preview',
                                     color='lightblue', hovercolor='skyblue')
        self.preview_button.on_clicked(self.on_preview_clicked)
        self.preview_button.label.set_fontsize(13)  # Increased from 10

        self.save_button = Button(ax_save_btn, 'Save Drift Trajectory',
                                  color='lightgreen', hovercolor='green')
        self.save_button.on_clicked(self.on_save_clicked)
        self.save_button.label.set_fontsize(13)  # Increased from 10

        self.cancel_button = Button(ax_cancel_btn, 'Cancel',
                                    color='lightcoral', hovercolor='red')
        self.cancel_button.on_clicked(self.on_cancel_clicked)
        self.cancel_button.label.set_fontsize(12)  # Increased from 9

        # Status text (LARGER FONT)
        self.status_text = ax_status.text(
            0.5, 0.5,
            f'Selected: {sum(self.selected_particles.values())}/{len(self.particle_ids)} particles',
            ha='center', va='center', fontsize=12, weight='bold'  # Increased from 9
        )

        # Adjust layout
        plt.tight_layout()

    def plot_trajectories(self):
        """Plot all particle trajectories (absolute positions) on the main axis."""
        self.ax_main.clear()

        # Plot each trajectory
        for pid in self.particle_ids:
            particle_data = self.trajectories[pid]

            if len(particle_data) == 0:
                continue

            # Plot trajectory
            line, = self.ax_main.plot(
                particle_data['center_x'],
                particle_data['center_y'],
                'o-',
                color=self.color_map[pid],
                markersize=2,
                linewidth=1.5,
                alpha=0.7,
                label=f'Particle {pid}',
                picker=5
            )
            self.trajectory_lines[pid] = line

            # Mark start point
            start_x = particle_data.iloc[0]['center_x']
            start_y = particle_data.iloc[0]['center_y']
            marker, = self.ax_main.plot(
                start_x, start_y, 'o',
                color=self.color_map[pid],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1.5,
                zorder=5
            )
            self.start_markers[pid] = marker

            # Add particle label near the end of trajectory (LARGER FONT)
            end_x = particle_data.iloc[-1]['center_x']
            end_y = particle_data.iloc[-1]['center_y']
            label_text = self.ax_main.text(
                end_x, end_y,
                f'P{pid}',
                color=self.color_map[pid],
                fontsize=14,  # Large font for visibility
                weight='bold',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=self.color_map[pid], alpha=0.8)
            )
            self.particle_labels[pid] = label_text

        # LARGER FONTS for axes
        self.ax_main.set_xlabel('X Position (pixels)', fontsize=16)
        self.ax_main.set_ylabel('Y Position (pixels)', fontsize=16)
        self.ax_main.set_title(
            f'Particle Trajectories (Absolute Positions)\n'
            f'n={len(self.particle_ids)} | Zoom/Pan to inspect',
            fontsize=14, weight='bold'  # Decreased from 18
        )
        self.ax_main.legend(fontsize=12, loc='best')  # Increased from 9
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal', adjustable='box')
        self.ax_main.tick_params(labelsize=12)  # Larger tick labels

    def plot_drift_preview(self):
        """Initialize the drift preview plot (empty at start)."""
        self.ax_drift.clear()
        self.ax_drift.set_xlabel('Drift X (pixels)', fontsize=16)
        self.ax_drift.set_ylabel('Drift Y (pixels)', fontsize=16)
        self.ax_drift.set_title(
            'Averaged Drift Trajectory\n(Click "Update Preview")',
            fontsize=14, weight='bold'  # Decreased from 18
        )
        self.ax_drift.grid(True, alpha=0.3)
        self.ax_drift.set_aspect('equal', adjustable='box')
        self.ax_drift.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax_drift.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax_drift.tick_params(labelsize=12)

    def on_checkbox_clicked(self, label):
        """Handle checkbox click events."""
        # Extract particle ID from label
        particle_id = int(label.split()[-1])

        # Toggle selection
        self.selected_particles[particle_id] = not self.selected_particles[particle_id]

        # Update trajectory appearance
        is_selected = self.selected_particles[particle_id]
        self.trajectory_lines[particle_id].set_alpha(0.8 if is_selected else 0.2)
        self.trajectory_lines[particle_id].set_linewidth(2 if is_selected else 0.5)
        self.start_markers[particle_id].set_alpha(1.0 if is_selected else 0.3)
        self.particle_labels[particle_id].set_alpha(1.0 if is_selected else 0.3)

        # Update status text
        num_selected = sum(self.selected_particles.values())
        self.status_text.set_text(f'Selected: {num_selected}/{len(self.particle_ids)} particles')

        # Redraw
        self.fig.canvas.draw_idle()

    def calculate_averaged_drift(self):
        """
        Calculate averaged drift trajectory from selected particles.
        Each particle's drift starts at (0,0) relative to its own first frame.

        Returns:
            DataFrame with averaged drift data
        """
        selected_ids = [pid for pid, selected in self.selected_particles.items() if selected]

        if len(selected_ids) == 0:
            print("Warning: No particles selected!")
            return None

        print(f"Calculating averaged drift from particles: {selected_ids}")

        # Filter data for selected particles only
        selected_data = self.df[self.df['particle_id'].isin(selected_ids)].copy()

        # Group by image_index and filename, calculate mean and std of drift
        avg_drift = selected_data.groupby(['image_index', 'filename']).agg({
            'drift_x': ['mean', 'std'],
            'drift_y': ['mean', 'std'],
            'particle_id': 'count'
        }).reset_index()

        # Flatten column names
        avg_drift.columns = [
            'image_index', 'filename',
            'avg_drift_x', 'std_drift_x',
            'avg_drift_y', 'std_drift_y',
            'num_particles_averaged'
        ]

        # Calculate drift magnitude
        avg_drift['avg_drift_magnitude'] = np.sqrt(
            avg_drift['avg_drift_x']**2 + avg_drift['avg_drift_y']**2
        )

        # Fill NaN std with 0 (happens when only 1 particle selected)
        avg_drift['std_drift_x'].fillna(0, inplace=True)
        avg_drift['std_drift_y'].fillna(0, inplace=True)

        return avg_drift

    def on_preview_clicked(self, event):
        """Handle preview button click - show averaged drift trajectory."""
        # Calculate averaged drift
        avg_drift = self.calculate_averaged_drift()

        if avg_drift is None or len(avg_drift) == 0:
            print("No data to preview!")
            return

        # Clear drift plot
        self.ax_drift.clear()

        # Get selected particle IDs
        selected_ids = [pid for pid, selected in self.selected_particles.items() if selected]

        # Plot individual drift trajectories (faded)
        self.drift_lines = {}
        for pid in selected_ids:
            particle_data = self.trajectories[pid]
            if len(particle_data) == 0:
                continue

            line, = self.ax_drift.plot(
                particle_data['drift_x'],
                particle_data['drift_y'],
                'o-',
                color=self.color_map[pid],
                markersize=1.5,
                linewidth=1,
                alpha=0.4,
                label=f'P{pid} drift'
            )
            self.drift_lines[pid] = line

        # Plot averaged drift trajectory (prominent)
        self.avg_drift_line, = self.ax_drift.plot(
            avg_drift['avg_drift_x'],
            avg_drift['avg_drift_y'],
            'ko-',
            linewidth=3,
            markersize=2,
            alpha=0.9,
            label=f'Averaged Drift (n={len(selected_ids)})',
            zorder=10
        )

        # Mark origin (0,0)
        self.ax_drift.plot(0, 0, 'r*', markersize=20, markeredgecolor='yellow',
                          markeredgewidth=2, label='Start (0,0)', zorder=11)

        # Styling
        self.ax_drift.set_xlabel('Drift X (pixels)', fontsize=16)
        self.ax_drift.set_ylabel('Drift Y (pixels)', fontsize=16)
        self.ax_drift.set_title(
            f'Drift Trajectories (start at 0,0)\n'
            f'Averaged: {selected_ids}',
            fontsize=14, weight='bold'  # Decreased from 18
        )
        self.ax_drift.legend(fontsize=12, loc='best')
        self.ax_drift.grid(True, alpha=0.3)
        self.ax_drift.set_aspect('equal', adjustable='box')
        self.ax_drift.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax_drift.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax_drift.tick_params(labelsize=12)

        # Redraw
        self.fig.canvas.draw_idle()

        print("Drift preview updated!")

    def on_save_clicked(self, event):
        """Handle save button click - save averaged drift trajectory and plot."""
        # Calculate averaged drift
        avg_drift = self.calculate_averaged_drift()

        if avg_drift is None or len(avg_drift) == 0:
            print("ERROR: No data to save! Please select at least one particle.")
            return

        selected_ids = [pid for pid, selected in self.selected_particles.items() if selected]

        print(f"\nSaving averaged drift trajectory from particles: {selected_ids}")

        # Save CSV file
        csv_file = self.output_dir / 'averaged_drift_trajectory.csv'
        avg_drift.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")

        # ========== PNG 1: Drift trajectories (side-by-side) ==========
        fig_drift, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # LEFT: Individual drift trajectories
        for pid in selected_ids:
            particle_data = self.trajectories[pid]
            if len(particle_data) == 0:
                continue

            ax1.plot(
                particle_data['drift_x'],
                particle_data['drift_y'],
                'o-',
                color=self.color_map[pid],
                markersize=2,
                linewidth=1.5,
                alpha=0.6,
                label=f'Particle {pid}'
            )

            # Mark start (0,0) for each
            ax1.plot(0, 0, 'o', color=self.color_map[pid], markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=5)

        ax1.set_xlabel('Drift X (pixels)', fontsize=16)
        ax1.set_ylabel('Drift Y (pixels)', fontsize=16)
        ax1.set_title(
            f'Individual Drift Trajectories\n'
            f'Particles: {selected_ids}',
            fontsize=14, weight='bold'
        )
        ax1.legend(fontsize=12, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax1.tick_params(labelsize=13)

        # RIGHT: Averaged drift trajectory
        ax2.plot(
            avg_drift['avg_drift_x'],
            avg_drift['avg_drift_y'],
            'ko-',
            linewidth=3,
            markersize=3,
            alpha=0.9,
            label=f'Averaged Drift (n={len(selected_ids)})'
        )

        # Mark origin
        ax2.plot(0, 0, 'r*', markersize=25, markeredgecolor='yellow',
                markeredgewidth=2, label='Start (0,0)', zorder=11)

        # Add error bars (std deviation) every 100 frames for clarity
        sample_indices = avg_drift.index[::100]
        ax2.errorbar(
            avg_drift.loc[sample_indices, 'avg_drift_x'],
            avg_drift.loc[sample_indices, 'avg_drift_y'],
            xerr=avg_drift.loc[sample_indices, 'std_drift_x'],
            yerr=avg_drift.loc[sample_indices, 'std_drift_y'],
            fmt='none',
            ecolor='gray',
            alpha=0.5,
            label='Std Dev (sampled)'
        )

        ax2.set_xlabel('Drift X (pixels)', fontsize=16)
        ax2.set_ylabel('Drift Y (pixels)', fontsize=16)
        ax2.set_title(
            f'Averaged Drift Trajectory\n'
            f'From particles: {selected_ids}',
            fontsize=14, weight='bold'
        )
        ax2.legend(fontsize=12, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.tick_params(labelsize=13)

        # Save drift plot
        png_drift = self.output_dir / 'averaged_drift_trajectory.png'
        plt.tight_layout()
        plt.savefig(png_drift, dpi=150, bbox_inches='tight')
        plt.close(fig_drift)
        print(f"  Saved drift PNG: {png_drift}")

        # ========== PNG 2: Absolute trajectories (as shown in GUI) ==========
        fig_abs = plt.figure(figsize=(12, 12))
        ax_abs = fig_abs.add_subplot(111)

        # Plot all particle trajectories with labels (same as GUI)
        for pid in self.particle_ids:
            particle_data = self.trajectories[pid]
            if len(particle_data) == 0:
                continue

            # Determine if selected (for styling)
            is_selected = self.selected_particles[pid]
            alpha = 0.8 if is_selected else 0.3
            linewidth = 2 if is_selected else 0.8

            # Plot trajectory
            ax_abs.plot(
                particle_data['center_x'],
                particle_data['center_y'],
                'o-',
                color=self.color_map[pid],
                markersize=2,
                linewidth=linewidth,
                alpha=alpha,
                label=f'Particle {pid}{"*" if is_selected else ""}'
            )

            # Mark start point
            start_x = particle_data.iloc[0]['center_x']
            start_y = particle_data.iloc[0]['center_y']
            ax_abs.plot(
                start_x, start_y, 'o',
                color=self.color_map[pid],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1.5,
                alpha=1.0 if is_selected else 0.4,
                zorder=5
            )

            # Add particle label at end of trajectory
            end_x = particle_data.iloc[-1]['center_x']
            end_y = particle_data.iloc[-1]['center_y']
            ax_abs.text(
                end_x, end_y,
                f'P{pid}',
                color=self.color_map[pid],
                fontsize=14,
                weight='bold',
                ha='center',
                va='bottom',
                alpha=1.0 if is_selected else 0.4,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=self.color_map[pid], alpha=0.8 if is_selected else 0.4)
            )

        ax_abs.set_xlabel('X Position (pixels)', fontsize=16)
        ax_abs.set_ylabel('Y Position (pixels)', fontsize=16)
        ax_abs.set_title(
            f'Absolute Particle Trajectories\n'
            f'Selected particles (* = used for averaging): {selected_ids}',
            fontsize=14, weight='bold'
        )
        ax_abs.legend(fontsize=12, loc='best')
        ax_abs.grid(True, alpha=0.3)
        ax_abs.set_aspect('equal', adjustable='box')
        ax_abs.tick_params(labelsize=13)

        # Save absolute trajectory plot
        png_abs = self.output_dir / 'absolute_trajectories.png'
        plt.tight_layout()
        plt.savefig(png_abs, dpi=150, bbox_inches='tight')
        plt.close(fig_abs)
        print(f"  Saved absolute PNG: {png_abs}")

        # Print summary statistics
        print(f"\n=== SUMMARY ===")
        print(f"Particles used: {selected_ids}")
        print(f"Total frames: {len(avg_drift)}")
        print(f"Average drift: X={avg_drift['avg_drift_x'].mean():.3f}, Y={avg_drift['avg_drift_y'].mean():.3f}")
        print(f"Max drift magnitude: {avg_drift['avg_drift_magnitude'].max():.3f} pixels")
        print(f"Final drift position: X={avg_drift['avg_drift_x'].iloc[-1]:.3f}, Y={avg_drift['avg_drift_y'].iloc[-1]:.3f}")
        print(f"\nFiles saved to: {self.output_dir}/")

        # Close the GUI window
        plt.close(self.fig)
        print("\nGUI closed. Done!")

    def on_cancel_clicked(self, event):
        """Handle cancel button click - close without saving."""
        print("Cancelled. Closing GUI without saving.")
        plt.close(self.fig)

    def show(self):
        """Display the GUI window."""
        plt.show()


def load_particle_data(csv_path: Path) -> pd.DataFrame:
    """
    Load particle tracking data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with particle tracking data
    """
    print(f"Loading data from: {csv_path.name}")
    print(f"File size: {csv_path.stat().st_size / (1024*1024):.1f} MB")

    # Read CSV file
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df)}")

    # Filter only successful fits
    df = df[df['success'] == True].copy()

    print(f"Successful fits: {len(df)}")
    print(f"Unique particles: {df['particle_id'].nunique()}")

    return df


def main():
    """Main function to launch interactive trajectory selector."""
    import json
    from logging_utils import setup_logger

    # Check if JSON path was provided as command-line argument
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
        if not json_file.exists():
            print(f"Error: Provided JSON file does not exist: {json_file}")
            sys.exit(1)
    else:
        # Find JSON file automatically (fallback)
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
            print("Please run scripts 1 and 2 first.")
            sys.exit(1)
    
    # Setup logger
    log_dir = json_file.parent
    logger = setup_logger('Step3_TrajectorySelector', log_dir=str(log_dir))
    
    logger.info("=== Interactive Drift Trajectory Selector ===\n")
    logger.info(f"Loading configuration from: {json_file}")
    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        sys.exit(1)

    # Get CSV path from JSON
    image_set = data.get('image_set')
    if not image_set:
        logger.error("No image set found in JSON!")
        sys.exit(1)
    
    csv_file_path = image_set.get('csv_file_path')
    if not csv_file_path:
        logger.error("No CSV file path in JSON! Please run script 2 first.")
        sys.exit(1)
    
    csv_file = Path(csv_file_path)
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)
    
    logger.info(f"Loading particle data from: {csv_file.name}")

    # Load data
    df = load_particle_data(csv_file)

    if len(df) == 0:
        logger.error("No successful fits found in CSV file!")
        sys.exit(1)

    # Get unique particle IDs (convert from numpy int64 to Python int for JSON compatibility)
    particle_ids = [int(pid) for pid in sorted(df['particle_id'].unique())]
    logger.info(f"Found {len(particle_ids)} particles: {particle_ids}\n")

    # Output directory - use drift_analysis subfolder
    output_dir = json_file.parent / 'drift_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and show interactive selector
    logger.info("Opening interactive GUI window...")
    logger.info("\nLayout:")
    logger.info("  LEFT: Absolute trajectories (zoom/pan enabled)")
    logger.info("  MIDDLE: Drift trajectory preview (all start at 0,0)")
    logger.info("  RIGHT: Control buttons")
    logger.info("\nInstructions:")
    logger.info("  1. Check/uncheck particles in right panel")
    logger.info("  2. Click 'Update Preview' to see averaged drift in middle panel")
    logger.info("  3. Click 'Save Drift Trajectory' when satisfied")
    logger.info("  4. Click 'Cancel' to exit without saving\n")

    selector = TrajectorySelector(df, particle_ids, output_dir)
    selector.show()
    
    # After GUI closes, check if data was saved and update JSON
    # Check if the output files exist (means user saved)
    csv_output = output_dir / 'averaged_drift_trajectory.csv'
    if csv_output.exists():
        logger.info("\n" + "="*60)
        logger.info("Updating JSON with selected particles...")
        logger.info("="*60 + "\n")
        
        # Get selected particles (convert to Python int for JSON compatibility)
        selected_ids = [int(pid) for pid, selected in selector.selected_particles.items() if selected]

        try:
            # Update JSON with selected particles and drift file paths
            data['image_set']['selected_particles_for_drift'] = selected_ids
            data['image_set']['drift_csv_file_path'] = str(csv_output.absolute())
            data['image_set']['drift_csv_file_name'] = csv_output.name
            data['image_set']['drift_analysis_folder'] = str(output_dir.absolute())
            
            # Save updated JSON
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✓ JSON file updated: {json_file.name}")
            logger.info(f"  Selected particles for drift: {selected_ids}")
            logger.info(f"  Drift CSV: {csv_output.name}")
            logger.info(f"  Output folder: {output_dir}")
            
        except Exception as e:
            logger.error(f"Warning: Could not update JSON file: {e}")
    else:
        logger.info("\nNo data was saved (user cancelled or closed window).")
    
    logger.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
