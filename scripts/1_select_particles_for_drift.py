"""
Particle Selection Tool for Drift Analysis
Select particles interactively with real-time Gaussian fit validation.
Saves validated bounding boxes to JSON for later batch processing.

Requirements:
pip install napari[all] tifffile numpy scipy scikit-image magicgui

Usage:
python select_particles_for_drift.py

Instructions:
1. Select directory containing TIF files
2. First image opens in napari viewer
3. Draw rectangles around particles
4. Each rectangle is validated with Gaussian fitting
5. Green points = successful fits (saved)
6. Red X = failed fits (rejected)
7. Click "Save Selection" to save to JSON
8. Option to add more image sets or exit
"""

import napari
import numpy as np
import tifffile
from pathlib import Path
from scipy.optimize import curve_fit
import json
from typing import Optional, Tuple, Dict, Any, List
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog
from qtpy.QtCore import QTimer
import sys
from datetime import datetime
from magicgui import magic_factory
from logging_utils import setup_logger, log_exception

# Logger will be set up after directory selection to use correct path
logger = None

class ParticleSelector:
    """
    Interactive particle selector with Gaussian fit validation.
    Saves validated selections to JSON for batch processing.
    """

    def __init__(self):
        self.directory_path = None
        self.tif_files = []
        self.current_image = None
        self.current_image_path = None
        self.viewer = None
        self.detection_layer = None
        self.points_layer = None
        self.failed_points_layer = None

        self.validated_particles = []  # Store validated bounding boxes
        self.analyzed_rectangles = set()  # Track analyzed rectangles
        self.particle_counter = 0

        # Store all image sets
        self.all_image_sets = []

        # Store save button widget for dynamic updates
        self.save_widget = None

        # Store JSON file path for output at end
        self.json_file_path = None

    def select_directory(self) -> bool:
        """Select directory containing TIF files."""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        directory = QFileDialog.getExistingDirectory(
            None,
            "Select directory containing TIF files",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not directory:
            return False

        self.directory_path = Path(directory)

        # Find all TIF files
        tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        self.tif_files = []

        for pattern in tif_patterns:
            self.tif_files.extend(list(self.directory_path.glob(pattern)))

        if not self.tif_files:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("No TIF files found in selected directory!")
            msg.exec_()
            return False

        self.tif_files.sort()
        
        # Setup logger now that we know the directory location
        global logger
        log_dir = self.directory_path.parent / "script_output"
        logger = setup_logger('Step1_ParticleSelection', log_dir=str(log_dir))
        
        logger.info(f"\nFound {len(self.tif_files)} TIF files in:")
        logger.info(f"  {self.directory_path}")
        for f in self.tif_files[:5]:
            logger.info(f"  {f.name}")
        if len(self.tif_files) > 5:
            logger.info(f"  ... and {len(self.tif_files) - 5} more")

        return True

    def load_first_image(self) -> bool:
        """Load the first image for particle selection."""
        if not self.tif_files:
            print("No TIF files available!")
            return False

        self.current_image_path = self.tif_files[0]

        try:
            self.current_image = tifffile.imread(str(self.current_image_path))
            if self.current_image.ndim != 2:
                print("Error: Image must be grayscale (2D)")
                return False

            logger.info(f"\nLoaded first image: {self.current_image_path.name}")
            logger.info(f"  Shape: {self.current_image.shape}")
            logger.info(f"  Value range: {self.current_image.min()} - {self.current_image.max()}")
            return True

        except Exception as e:
            print(f"Error loading first image: {e}")
            return False

    def gaussian_2d(self, coords: Tuple[np.ndarray, np.ndarray],
                   amp: float, x0: float, y0: float,
                   sigma_x: float, sigma_y: float,
                   theta: float, offset: float) -> np.ndarray:
        """2D Gaussian model with rotation (same as original script)."""
        X, Y = coords

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_rot = (X - x0) * cos_theta + (Y - y0) * sin_theta
        y_rot = -(X - x0) * sin_theta + (Y - y0) * cos_theta

        gaussian = amp * np.exp(-(x_rot**2)/(2*sigma_x**2) - (y_rot**2)/(2*sigma_y**2)) + offset

        return gaussian.ravel()

    def fit_gaussian_to_region(self, region: np.ndarray,
                              bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Fit a 2D Gaussian to validate particle selection.
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

        p0 = [amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0.0, offset_guess]

        bounds = (
            [-np.inf, 0, 0, 0.5, 0.5, -np.pi, -np.inf],
            [np.inf, w-1, h-1, w*2, h*2, np.pi, np.inf]
        )

        try:
            popt, pcov = curve_fit(
                self.gaussian_2d, (X, Y), region.ravel(),
                p0=p0, bounds=bounds, maxfev=2000
            )

            amp, x0, y0, sigma_x, sigma_y, theta, offset = popt

            # Calculate R²
            fitted_data = self.gaussian_2d((X, Y), *popt).reshape(region.shape)
            residuals = region - fitted_data
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((region - np.mean(region))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Convert to global coordinates
            global_x = x0 + bbox[0]
            global_y = y0 + bbox[1]

            return {
                'success': True,
                'center_x': global_x,
                'center_y': global_y,
                'r_squared': r_squared,
                'amplitude': amp,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y
            }

        except Exception as e:
            return None

    def validate_region(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Validate a rectangular region with Gaussian fitting.
        Only save if fit is successful.
        """
        x0, y0, x1, y1 = bbox

        # Ensure coordinates are within image bounds
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(self.current_image.shape[1], int(x1))
        y1 = min(self.current_image.shape[0], int(y1))

        if x1 <= x0 or y1 <= y0:
            print("Invalid region coordinates")
            return

        # Extract region
        region = self.current_image[y0:y1, x0:x1].copy()

        self.particle_counter += 1
        logger.info(f"\n=== Validating Particle #{self.particle_counter} ===")
        logger.info(f"  Region: ({x0}, {y0}) to ({x1}, {y1})")
        logger.info(f"  Size: {region.shape}")

        # Fit Gaussian
        result = self.fit_gaussian_to_region(region, (x0, y0, x1, y1))

        if result and result['success']:
            # Successful fit - SAVE IT
            particle_data = {
                'bbox': [x0, y0, x1, y1],
                'particle_id': self.particle_counter,
                'test_fit_r_squared': float(result['r_squared']),
                'test_center_x': float(result['center_x']),
                'test_center_y': float(result['center_y'])
            }
            self.validated_particles.append(particle_data)

            logger.info(f"  ✓ FIT SUCCESSFUL - SAVED")
            logger.info(f"    Center: ({result['center_x']:.2f}, {result['center_y']:.2f})")
            logger.info(f"    R²: {result['r_squared']:.4f}")

            # Add GREEN point to show success
            if self.points_layer is not None:
                new_point = np.array([[result['center_y'], result['center_x']]])
                current_points = self.points_layer.data
                if len(current_points) > 0:
                    self.points_layer.data = np.vstack([current_points, new_point])
                else:
                    self.points_layer.data = new_point

            # Update button text with new particle count
            self.update_button_text()
        else:
            # Failed fit - REJECT IT
            logger.info(f"  ✗ FIT FAILED - REJECTED (not saved)")

            # Add RED X to show failure
            if self.failed_points_layer is not None:
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                new_point = np.array([[center_y, center_x]])
                current_points = self.failed_points_layer.data
                if len(current_points) > 0:
                    self.failed_points_layer.data = np.vstack([current_points, new_point])
                else:
                    self.failed_points_layer.data = new_point

    def on_rectangle_added(self, event):
        """Callback when user draws a rectangle."""
        if len(self.detection_layer.data) > 0:
            last_shape = self.detection_layer.data[-1]

            if len(last_shape) >= 4:
                corners = np.array(last_shape)
                y_coords = corners[:, 0]
                x_coords = corners[:, 1]

                x0, x1 = np.min(x_coords), np.max(x_coords)
                y0, y1 = np.min(y_coords), np.max(y_coords)

                bbox_tuple = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

                # Check for duplicates
                if bbox_tuple not in self.analyzed_rectangles:
                    self.analyzed_rectangles.add(bbox_tuple)
                    self.validate_region(bbox_tuple)

    def save_current_selection(self):
        """Save current image set selection to JSON."""
        if not self.validated_particles:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("No Particles Selected")
            msg.setText("No validated particles to save!")
            msg.exec_()
            return False

        # Create image set data
        image_set_data = {
            'folder_path': str(self.directory_path.absolute()),
            'selected_particles': self.validated_particles.copy(),
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            'first_image': self.current_image_path.name,
            'total_images': len(self.tif_files),
            'image_shape': list(self.current_image.shape)
        }

        self.all_image_sets.append(image_set_data)

        logger.info(f"\n=== Current Selection Saved ===")
        logger.info(f"  Folder: {self.directory_path.name}")
        logger.info(f"  Validated particles: {len(self.validated_particles)}")
        logger.info(f"  Total images in set: {len(self.tif_files)}")

        return True

    def update_button_text(self):
        """Update save button text with current particle count."""
        if self.save_widget is not None:
            count = len(self.validated_particles)
            particle_word = "particle" if count == 1 else "particles"
            self.save_widget.call_button.text = f"Save & Exit ({count} {particle_word})"

    def create_save_button(self):
        """Create save button widget."""
        @magic_factory(call_button="Save & Exit (0 particles)")
        def save_callback(selector=self):
            if selector.save_current_selection():
                # Export to JSON and exit (deferred to avoid widget deletion errors)
                selector.export_all_to_json()
                if selector.viewer:
                    # Use QTimer to defer the close until after callback completes
                    QTimer.singleShot(0, selector.viewer.close)

        widget = save_callback()
        self.save_widget = widget
        return widget

    def create_undo_button(self):
        """Create undo last particle button widget."""
        @magic_factory(call_button="Undo Last Particle")
        def undo_callback(selector=self):
            if not selector.validated_particles:
                print("\nNo particles to undo!")
                return

            # Remove last particle from list
            removed_particle = selector.validated_particles.pop()
            particle_id = removed_particle['particle_id']

            print(f"\n=== Undoing Particle #{particle_id} ===")
            print(f"  Removed from selection")
            print(f"  Remaining particles: {len(selector.validated_particles)}")

            # Remove corresponding green point from points layer
            if selector.points_layer is not None and len(selector.points_layer.data) > 0:
                # Remove the last point (most recently added)
                current_points = selector.points_layer.data
                if len(current_points) > 0:
                    selector.points_layer.data = current_points[:-1]

            # Update save button text
            selector.update_button_text()

        widget = undo_callback()
        return widget

    def export_all_to_json(self, auto_save=False):
        """Export single image set to JSON file.
        
        Overwrites any existing file with the current selection.

        Args:
            auto_save: If True, this is an automatic save on window close
        """
        if not self.all_image_sets:
            if not auto_save:
                print("No image set to export!")
            return None

        # Create script_output folder next to the image directory
        # e.g., if images are in /path/to/images/exported/, create /path/to/images/script_output/
        output_dir = self.directory_path.parent / "script_output"
        output_dir.mkdir(exist_ok=True)

        # Use fixed filename (no timestamp)
        filename = output_dir / "particle_selections.json"

        # For single image set workflow, just use the first (and only) set
        image_set_data = self.all_image_sets[0]

        output_data = {
            'image_set': image_set_data,
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_modified': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)

            if auto_save:
                logger.info(f"\n=== AUTO-SAVE SUCCESSFUL ===")
                logger.info(f"  Your work has been saved automatically!")
            else:
                logger.info(f"\n=== EXPORT SUCCESSFUL ===")

            logger.info(f"  Saved to: {filename}")
            logger.info(f"  Image folder: {Path(image_set_data['folder_path']).name}")
            logger.info(f"  Selected particles: {len(image_set_data['selected_particles'])}")

            # Store the JSON file path for output at end
            self.json_file_path = str(filename.absolute())

            return filename

        except Exception as e:
            logger.error(f"\n=== EXPORT FAILED ===")
            log_exception(logger, e, "Export error")
            if not auto_save:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Export Failed")
                msg.setText(f"Error: {e}")
                msg.exec_()
            return None

    def setup_napari_viewer(self):
        """Set up napari viewer with layers and widgets."""
        self.viewer = napari.Viewer(title="Particle Selection for Drift Analysis")

        # Override closeEvent to add auto-save functionality
        # Store reference to original closeEvent
        original_closeEvent = self.viewer.window._qt_window.closeEvent

        # Create custom closeEvent that auto-saves before closing
        def custom_close_event(event):
            # Call our auto-save handler
            if self.all_image_sets:
                logger.info("\n=== Window closing - Auto-saving your work... ===")
                self.export_all_to_json(auto_save=True)
            else:
                logger.info("\n=== Window closing - No data to save ===")

            # Call original napari closeEvent to maintain default behavior
            original_closeEvent(event)

        # Replace the closeEvent method with our custom one
        self.viewer.window._qt_window.closeEvent = custom_close_event

        # Add image
        self.viewer.add_image(
            self.current_image,
            name=f"image_{self.current_image_path.name}",
            colormap="gray",
            contrast_limits=[np.percentile(self.current_image, 1),
                           np.percentile(self.current_image, 99)]
        )

        # Add shapes layer for drawing rectangles
        self.detection_layer = self.viewer.add_shapes(
            name="detection_boxes",
            shape_type="rectangle",
            edge_color="red",
            face_color="transparent",
            edge_width=2
        )

        # Add points layer for successful fits (GREEN)
        self.points_layer = self.viewer.add_points(
            name="validated_centers",
            size=8,
            face_color="green",
            border_color="yellow",
            border_width=2,
            border_width_is_relative=False
        )

        # Add points layer for failed fits (RED)
        self.failed_points_layer = self.viewer.add_points(
            name="rejected_centers",
            size=10,
            symbol="x",
            face_color="red",
            border_color="darkred",
            border_width=2,
            border_width_is_relative=False
        )

        # Add save button
        save_widget = self.create_save_button()
        self.viewer.window.add_dock_widget(save_widget, area='right', name='Save Selection')

        # Add undo button
        undo_widget = self.create_undo_button()
        self.viewer.window.add_dock_widget(undo_widget, area='right', name='Undo Last')

        # Connect event handler
        self.detection_layer.events.data.connect(self.on_rectangle_added)

        logger.info("\n=== INSTRUCTIONS ===")
        logger.info("1. Select 'detection_boxes' layer")
        logger.info("2. Draw rectangles around particles")
        logger.info("3. GREEN points = successful fit (SAVED)")
        logger.info("4. RED X = failed fit (REJECTED)")
        logger.info("5. Click 'Undo Last Particle' to remove most recent selection")
        logger.info("6. Click 'Save & Exit' when done with particle selection")
        logger.info("\nNOTE: Your work will be auto-saved if you close the window!")

    def run_selection_session(self):
        """Run a single selection session for one image set."""
        # Reset for new session
        self.validated_particles = []
        self.analyzed_rectangles = set()
        self.particle_counter = 0

        # Select directory
        if not self.select_directory():
            # User cancelled directory selection
            print("\nDirectory selection cancelled.")
            # If there's already saved data, export it before exiting
            if self.all_image_sets:
                print("You have unsaved work from previous selections.")
                self.export_all_to_json(auto_save=True)
            return False

        # Load first image
        if not self.load_first_image():
            print("\nFailed to load first image.")
            # If there's already saved data, export it before exiting
            if self.all_image_sets:
                print("You have unsaved work from previous selections.")
                self.export_all_to_json(auto_save=True)
            return False

        # Setup viewer
        self.setup_napari_viewer()

        # Start napari
        napari.run()

        return True

    def run(self):
        """Main entry point."""
        print("=== Particle Selection for Drift Analysis ===")
        print("This tool helps you select particles for drift tracking.")
        print("Only validated selections (successful Gaussian fits) are saved.\n")

        self.run_selection_session()

        # Output JSON path for shell script to capture
        if self.json_file_path:
            print(f"JSON_PATH={self.json_file_path}")

def main():
    """Main function."""
    try:
        import magicgui
    except ImportError:
        print("Error: magicgui package not found!")
        print("Please install: pip install magicgui")
        return

    selector = ParticleSelector()
    selector.run()

if __name__ == "__main__":
    main()
