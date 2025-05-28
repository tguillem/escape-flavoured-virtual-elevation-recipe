import csv

# from folium.plugins import BeautifulifyIcon
import io
import os
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.virtual_elevation import VirtualElevation


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class AnalysisResult(QMainWindow):
    """Window for displaying virtual elevation analysis results"""

    def __init__(self, fit_file, settings, selected_laps, params):
        super().__init__()
        self.fit_file = fit_file
        self.settings = settings
        self.selected_laps = selected_laps
        self.params = params
        self.result_dir = settings.result_dir

        # Prepare merged lap data
        self.prepare_merged_data()

        # Create VE calculator
        self.ve_calculator = VirtualElevation(self.merged_data, self.params)

        # Get lap combination ID for settings
        self.lap_combo_id = "_".join(map(str, sorted(self.selected_laps)))

        # Try to load saved trim values for this lap combination
        file_settings = self.settings.get_file_settings(self.fit_file.filename)
        trim_settings = file_settings.get("trim_settings", {})
        saved_trim = trim_settings.get(self.lap_combo_id, {})

        # Initialize UI values
        if saved_trim and "trim_start" in saved_trim and "trim_end" in saved_trim:
            # Use saved trim values if available
            self.trim_start = saved_trim["trim_start"]
            self.trim_end = saved_trim["trim_end"]
        else:
            # Use defaults
            self.trim_start = 0
            self.trim_end = len(self.merged_data)

        self.current_cda = self.params.get("cda")
        self.current_crr = self.params.get("crr")

        # If CdA or Crr are None (to be optimized), set initial values to middle of range
        if self.current_cda is None:
            if saved_trim and "cda" in saved_trim:
                self.current_cda = saved_trim["cda"]
            else:
                self.current_cda = (
                    self.params.get("cda_min", 0.15) + self.params.get("cda_max", 0.5)
                ) / 2

        if self.current_crr is None:
            if saved_trim and "crr" in saved_trim:
                self.current_crr = saved_trim["crr"]
            else:
                self.current_crr = (
                    self.params.get("crr_min", 0.001) + self.params.get("crr_max", 0.03)
                ) / 2

        # Setup UI
        self.initUI()

        # Calculate and plot initial VE
        self.calculate_ve()
        self.update_plots()

    def prepare_merged_data(self):
        """Extract and merge data for selected laps"""
        # Get records for selected laps
        self.merged_data = self.fit_file.get_records_for_laps(self.selected_laps)

        # Check if we have enough data
        if len(self.merged_data) < 30:
            raise ValueError("Not enough data points (less than 30 seconds)")

        # Get lap info for display
        self.lap_info = []
        all_laps = self.fit_file.get_lap_data()

        for lap in all_laps:
            if lap["lap_number"] in self.selected_laps:
                self.lap_info.append(lap)

        # Calculate distance, duration, etc. for the merged lap
        self.total_distance = sum(lap["distance"] for lap in self.lap_info)
        self.total_duration = sum(lap["duration"] for lap in self.lap_info)
        self.avg_power = np.mean(self.merged_data["power"].dropna())
        self.avg_speed = (
            self.total_distance / self.total_duration
        ) * 3600  # Convert to km/h

        # Extract GPS coordinates for map
        if (
            "position_lat" in self.merged_data.columns
            and "position_long" in self.merged_data.columns
        ):
            self.has_gps = True
            # Filter out missing coordinates
            valid_coords = self.merged_data.dropna(
                subset=["position_lat", "position_long"]
            )
            if not valid_coords.empty:
                self.start_lat = valid_coords["position_lat"].iloc[0]
                self.start_lon = valid_coords["position_long"].iloc[0]
                self.end_lat = valid_coords["position_lat"].iloc[-1]
                self.end_lon = valid_coords["position_long"].iloc[-1]

                # Extract all route points
                self.route_points = list(
                    zip(valid_coords["position_lat"], valid_coords["position_long"])
                )

                # Store the timestamps to ensure correct mapping of trim indices to route points
                self.route_timestamps = valid_coords["timestamp"].tolist()

                # Initial trim values should correspond to the valid coordinates
                self.trim_start = 0
                self.trim_end = len(self.route_points) - 1
            else:
                self.has_gps = False
        else:
            self.has_gps = False

    def initUI(self):
        """Initialize the UI components"""
        self.setWindowTitle(
            f'Virtual Elevation Analysis - Laps {", ".join(map(str, self.selected_laps))}'
        )
        self.setGeometry(50, 50, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create a splitter for adjustable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Map and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Map
        if self.has_gps:
            self.map_widget = QWebEngineView()
            self.create_map()
            left_layout.addWidget(self.map_widget, 2)
        else:
            no_gps_label = QLabel("No GPS data available")
            no_gps_label.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(no_gps_label, 2)

        # Parameter display
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.update_config_text()
        param_layout.addRow("Configuration:", self.config_text)

        # Configuration name input
        self.config_name = QLineEdit("Test")
        param_layout.addRow("Save as:", self.config_name)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group, 1)

        # Control buttons
        button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back to Lap Selection")
        self.back_button.clicked.connect(self.back_to_selection)

        self.close_button = QPushButton("Close App")
        self.close_button.clicked.connect(self.close)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setStyleSheet(f"background-color: #4363d8; color: white;")

        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.close_button)
        button_layout.addWidget(self.save_button)

        left_layout.addLayout(button_layout)

        # Right side - Plots and sliders
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create plots
        self.fig_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.fig_canvas)

        right_layout.addWidget(plot_widget, 3)

        # Sliders
        slider_group = QGroupBox("Adjust Parameters")
        slider_layout = QFormLayout()

        # Trim start slider
        self.trim_start_slider = QSlider(Qt.Horizontal)
        self.trim_start_slider.setMinimum(0)
        self.trim_start_slider.setMaximum(len(self.merged_data) - 30)
        self.trim_start_slider.setValue(self.trim_start)  # Use the loaded trim value
        self.trim_start_slider.valueChanged.connect(self.on_trim_start_changed)

        self.trim_start_label = QLabel(f"{self.trim_start} s")
        trim_start_layout = QHBoxLayout()
        trim_start_layout.addWidget(self.trim_start_slider)
        trim_start_layout.addWidget(self.trim_start_label)

        slider_layout.addRow("Trim Start:", trim_start_layout)

        # Trim end slider
        self.trim_end_slider = QSlider(Qt.Horizontal)
        self.trim_end_slider.setMinimum(30)
        self.trim_end_slider.setMaximum(len(self.merged_data))
        self.trim_end_slider.setValue(self.trim_end)  # Use the loaded trim value
        self.trim_end_slider.valueChanged.connect(self.on_trim_end_changed)

        self.trim_end_label = QLabel(f"{self.trim_end} s")
        trim_end_layout = QHBoxLayout()
        trim_end_layout.addWidget(self.trim_end_slider)
        trim_end_layout.addWidget(self.trim_end_label)

        slider_layout.addRow("Trim End:", trim_end_layout)

        # CdA slider
        self.cda_slider = QSlider(Qt.Horizontal)
        self.cda_slider.setMinimum(int(self.params.get("cda_min", 0.15) * 1000))
        self.cda_slider.setMaximum(int(self.params.get("cda_max", 0.5) * 1000))
        self.cda_slider.setValue(int(self.current_cda * 1000))
        self.cda_slider.valueChanged.connect(self.on_cda_changed)
        self.cda_slider.setEnabled(self.params.get("cda") is None)

        self.cda_label = QLabel(f"{self.current_cda:.3f}")
        cda_layout = QHBoxLayout()
        cda_layout.addWidget(self.cda_slider)
        cda_layout.addWidget(self.cda_label)

        slider_layout.addRow("CdA:", cda_layout)

        # Crr slider
        self.crr_slider = QSlider(Qt.Horizontal)
        self.crr_slider.setMinimum(int(self.params.get("crr_min", 0.001) * 10000))
        self.crr_slider.setMaximum(int(self.params.get("crr_max", 0.03) * 10000))
        self.crr_slider.setValue(int(self.current_crr * 10000))
        self.crr_slider.valueChanged.connect(self.on_crr_changed)
        self.crr_slider.setEnabled(self.params.get("crr") is None)

        self.crr_label = QLabel(f"{self.current_crr:.4f}")
        crr_layout = QHBoxLayout()
        crr_layout.addWidget(self.crr_slider)
        crr_layout.addWidget(self.crr_label)

        slider_layout.addRow("Crr:", crr_layout)

        slider_group.setLayout(slider_layout)
        right_layout.addWidget(slider_group, 1)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])  # Set initial sizes

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_map(self):
        """Create the map showing the merged track with start/end markers and trimmed portion"""
        if not self.has_gps or not self.route_points:
            return

        # Calculate center point
        center_lat = (self.start_lat + self.end_lat) / 2
        center_lon = (self.start_lon + self.end_lon) / 2

        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Track styling:
        # 1. First draw parts before trim_start and after trim_end with dashed opacity
        # 2. Then draw the selected portion with solid blue

        try:
            # Map time indices to route indices
            if not hasattr(self, "route_timestamps") or not self.route_timestamps:
                # Fall back to simple index mapping if no timestamps available
                total_points = len(self.route_points)
                total_records = len(self.merged_data)

                if total_points > 0 and total_records > 0:
                    route_trim_start = min(
                        int(self.trim_start * total_points / total_records),
                        total_points - 1,
                    )
                    route_trim_end = min(
                        int(self.trim_end * total_points / total_records),
                        total_points - 1,
                    )
                else:
                    route_trim_start = 0
                    route_trim_end = len(self.route_points) - 1
            else:
                # Map using timestamps
                route_trim_start = self.map_time_to_route_index(self.trim_start)
                route_trim_end = self.map_time_to_route_index(self.trim_end)

                # Fall back if mapping fails
                if route_trim_start is None or route_trim_end is None:
                    total_points = len(self.route_points)
                    total_records = len(self.merged_data)

                    if total_points > 0 and total_records > 0:
                        route_trim_start = min(
                            int(self.trim_start * total_points / total_records),
                            total_points - 1,
                        )
                        route_trim_end = min(
                            int(self.trim_end * total_points / total_records),
                            total_points - 1,
                        )
                    else:
                        route_trim_start = 0
                        route_trim_end = len(self.route_points) - 1

            # Make sure indices are valid
            route_trim_start = max(0, min(route_trim_start, len(self.route_points) - 1))
            route_trim_end = max(
                route_trim_start, min(route_trim_end, len(self.route_points) - 1)
            )

            # 1. Draw the parts before trim_start with dashed blue line (if exists)
            if route_trim_start > 0:
                pre_trim_route = self.route_points[: route_trim_start + 1]
                folium.PolyLine(
                    pre_trim_route,
                    color="#4363d8",  # Blue color
                    weight=3,
                    opacity=0.5,
                    dash_array="5,10",  # Dashed line
                    popup="Pre-selected portion",
                ).add_to(m)

            # 2. Draw the parts after trim_end with dashed blue line (if exists)
            if route_trim_end < len(self.route_points) - 1:
                post_trim_route = self.route_points[route_trim_end:]
                folium.PolyLine(
                    post_trim_route,
                    color="#4363d8",  # Blue color
                    weight=3,
                    opacity=0.5,
                    dash_array="5,10",  # Dashed line
                    popup="Post-selected portion",
                ).add_to(m)

            # 3. Draw the selected portion with solid blue line
            trimmed_route = self.route_points[route_trim_start : route_trim_end + 1]
            if len(trimmed_route) > 1:
                folium.PolyLine(
                    trimmed_route,
                    color="#4363d8",  # Blue color
                    weight=5,  # Slightly thicker
                    opacity=1.0,  # Full opacity
                    popup="Selected portion",
                ).add_to(m)

                # Add trim markers
                folium.Marker(
                    location=trimmed_route[0],
                    popup="Trim Start",
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                ).add_to(m)

                folium.Marker(
                    location=trimmed_route[-1],
                    popup="Trim End",
                    icon=folium.Icon(color="red", icon="stop", prefix="fa"),
                ).add_to(m)

        except Exception as e:
            print(f"Error highlighting trimmed route: {e}")

        # Calculate bounds for automatic zoom
        try:
            if self.route_points:
                lats = [p[0] for p in self.route_points]
                lons = [p[1] for p in self.route_points]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)

                # Add some padding (5%)
                lat_padding = (max_lat - min_lat) * 0.05
                lon_padding = (max_lon - min_lon) * 0.05
                bounds = [
                    [min_lat - lat_padding, min_lon - lon_padding],
                    [max_lat + lat_padding, max_lon + lon_padding],
                ]
                m.fit_bounds(bounds)
        except Exception as e:
            print(f"Error fitting map bounds: {e}")

        # Add wind arrow AFTER map bounds have been set
        wind_speed = self.params.get("wind_speed")
        wind_dir = self.params.get("wind_direction")

        if wind_speed not in [None, 0] and wind_dir is not None:
            # Create a custom HTML element for the wind arrow
            wind_html = f"""
            <div id="wind-arrow" style="position: absolute; top: 10px; right: 10px; 
                    background-color: rgba(255, 255, 255, 0.9); padding: 10px; 
                    border-radius: 5px; border: 1px solid #4363d8; z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <div style="font-weight: bold; text-align: center; margin-bottom: 5px; color: #4363d8;">Wind</div>
                <div style="text-align: center;">
                    <div style="font-size: 28px; transform: rotate({wind_dir + 180}deg); color: #4363d8;">↑</div>
                    <div style="font-size: 12px; margin-top: 5px;">{wind_speed} m/s</div>
                    <div style="font-size: 12px;">{wind_dir}°</div>
                </div>
            </div>
            """

            # Add the HTML element to the map
            m.get_root().html.add_child(folium.Element(wind_html))

        # Save map to HTML and load into QWebEngineView
        data = io.BytesIO()
        m.save(data, close_file=False)
        html_content = data.getvalue().decode()
        self.map_widget.setHtml(html_content)

    def update_config_text(self):
        """Update the configuration text display"""
        lap_str = ", ".join(map(str, self.selected_laps))
        distance_str = f"{self.total_distance:.2f} km"
        duration_str = f"{self.total_duration:.0f} s"
        power_str = f"{self.avg_power:.0f} W"
        speed_str = f"{self.avg_speed:.2f} km/h"

        config_text = f"Selected Laps: {lap_str}\n"
        config_text += f"Distance: {distance_str}\n"
        config_text += f"Duration: {duration_str}\n"
        config_text += f"Avg Power: {power_str}\n"
        config_text += f"Avg Speed: {speed_str}\n"
        config_text += f"System Mass: {self.params.get('system_mass', 90)} kg\n"
        rho_low = self.params.get('rho_low', None)
        rho_high = self.params.get('rho_high', None)
        if rho_low and rho_high:
            config_text += f"Rho (air density): {rho_low} - {rho_high} kg/m³\n"
        else:
            config_text += f"Rho (air density): {self.params.get('rho', 1.2)} kg/m³\n"
        config_text += f"Eta (drivetrain eff.): {self.params.get('eta', 0.98)}\n"
        config_text += f"Current CdA: {self.current_cda:.3f}\n"
        config_text += f"Current Crr: {self.current_crr:.4f}\n"

        if self.params.get("wind_speed") not in [None, 0]:
            config_text += f"Wind Speed: {self.params.get('wind_speed')} m/s\n"

        if self.params.get("wind_direction") is not None:
            config_text += f"Wind Direction: {self.params.get('wind_direction')}°"

        self.config_text.setText(config_text)

    def calculate_ve(self):
        """Calculate virtual elevation with current parameters"""
        # Extract actual elevation if available
        self.actual_elevation = None
        if (
            "altitude" in self.merged_data.columns
            and not self.merged_data["altitude"].isna().all()
        ):
            self.actual_elevation = self.merged_data["altitude"].values

        # Calculate virtual elevation
        self.virtual_elevation = self.ve_calculator.calculate_ve(
            self.current_cda, self.current_crr
        )

        # Calculate metrics
        if self.actual_elevation is not None:
            # Ensure same length
            min_len = min(len(self.virtual_elevation), len(self.actual_elevation))
            ve_trim = self.virtual_elevation[:min_len]
            elev_trim = self.actual_elevation[:min_len]

            # Calibrate to match at trim start
            trim_start_idx = self.trim_start
            if trim_start_idx < min_len:
                # Calculate offset to make virtual elevation match actual at trim start
                offset = elev_trim[trim_start_idx] - ve_trim[trim_start_idx]
                ve_calibrated = ve_trim + offset
                self.virtual_elevation_calibrated = ve_calibrated
            else:
                self.virtual_elevation_calibrated = ve_trim

            # Calculate metrics in trimmed region
            trim_indices = np.where(
                (np.arange(len(ve_trim)) >= self.trim_start)
                & (np.arange(len(ve_trim)) <= self.trim_end)
            )[0]

            if len(trim_indices) > 2:  # Need at least 3 points for correlation
                ve_trim_region = self.virtual_elevation_calibrated[trim_indices]
                elev_trim_region = elev_trim[trim_indices]

                # R² calculation
                corr = np.corrcoef(ve_trim_region, elev_trim_region)[0, 1]
                self.r2 = corr**2

                # RMSE calculation
                self.rmse = np.sqrt(np.mean((ve_trim_region - elev_trim_region) ** 2))

                # Calculate elevation gain (difference between end and start)
                # Make sure we don't exceed array bounds
                safe_trim_end = min(self.trim_end, len(ve_trim) - 1)
                safe_trim_start = min(self.trim_start, safe_trim_end)

                # Elevation differences
                self.ve_elevation_diff = (
                    ve_calibrated[safe_trim_end] - ve_calibrated[safe_trim_start]
                )
                self.actual_elevation_diff = (
                    elev_trim[safe_trim_end] - elev_trim[safe_trim_start]
                )
            else:
                self.r2 = 0
                self.rmse = 0
                self.ve_elevation_diff = 0
                self.actual_elevation_diff = 0
        else:
            # If no actual elevation data, still create a calibrated version
            self.virtual_elevation_calibrated = self.virtual_elevation.copy()
            self.ve_elevation_diff = (
                self.virtual_elevation_calibrated[
                    min(self.trim_end, len(self.virtual_elevation_calibrated) - 1)
                ]
                - self.virtual_elevation_calibrated[
                    min(self.trim_start, len(self.virtual_elevation_calibrated) - 1)
                ]
            )

    def update_plots(self):
        """Update the virtual elevation plots"""
        # Clear previous plots
        self.fig_canvas.axes.clear()

        # Create figure with two subplots
        self.fig_canvas.fig.clear()
        gs = self.fig_canvas.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = self.fig_canvas.fig.add_subplot(gs[0])
        ax2 = self.fig_canvas.fig.add_subplot(gs[1])

        # Create time array (assuming 1s intervals)
        time = np.arange(len(self.virtual_elevation))

        # Plot virtual elevation with FULL OPACITY in trimmed region, REDUCED OPACITY elsewhere
        # First plot full curve with reduced opacity
        ax1.plot(
            time,
            self.virtual_elevation_calibrated,
            color="blue",
            alpha=0.3,
            linewidth=1,
            label="_nolegend_",
        )

        # Then plot just the trimmed region with full opacity
        # Special handling for edge cases when trim_start=0 or trim_end=max
        trim_start = self.trim_start
        trim_end = min(self.trim_end, len(self.virtual_elevation_calibrated) - 1)

        # Ensure we have a valid range
        if trim_start <= trim_end:
            trim_time = time[trim_start : trim_end + 1]
            trim_ve = self.virtual_elevation_calibrated[trim_start : trim_end + 1]
            ax1.plot(
                trim_time,
                trim_ve,
                color="blue",
                alpha=1.0,
                linewidth=2,
                label="Virtual Elevation",
            )

        # Mark trimmed region with higher opacity - use lower opacity (0.1) for excluded regions
        ax1.axvspan(0, self.trim_start, alpha=0.1, color="gray")
        ax1.axvspan(self.trim_end, len(self.virtual_elevation), alpha=0.1, color="gray")

        # Add vertical lines at trim points WITHOUT adding to legend
        ax1.axvline(
            x=self.trim_start, color="green", linestyle="--", label="_nolegend_"
        )
        ax1.axvline(x=self.trim_end, color="red", linestyle="--", label="_nolegend_")

        # Add grid lines
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Plot actual elevation if available
        if self.actual_elevation is not None:
            # Ensure same length
            min_len = min(
                len(self.virtual_elevation_calibrated), len(self.actual_elevation)
            )
            time_trim = time[:min_len]
            ve_trim = self.virtual_elevation_calibrated[:min_len]
            elev_trim = self.actual_elevation[:min_len]

            # Plot actual elevation with REDUCED OPACITY outside trim region
            # First plot full curve with reduced opacity
            ax1.plot(
                time_trim,
                elev_trim,
                color="black",
                alpha=0.3,
                linewidth=1,
                label="_nolegend_",
            )

            # Then plot just the trimmed region with full opacity
            trim_end_safe = min(trim_end, min_len - 1)

            # Ensure we have a valid range
            if trim_start <= trim_end_safe:
                trim_time = time_trim[trim_start : trim_end_safe + 1]
                trim_elev = elev_trim[trim_start : trim_end_safe + 1]
                ax1.plot(
                    trim_time,
                    trim_elev,
                    color="black",
                    alpha=1.0,
                    linewidth=2,
                    label="Actual Elevation",
                )

            # Plot residuals in the second subplot
            residuals = ve_trim - elev_trim

            # First plot full residuals with reduced opacity
            ax2.plot(time_trim, residuals, color="gray", alpha=0.3, linewidth=1)

            # Then plot just the trimmed region with full opacity
            if trim_start <= trim_end_safe:
                trim_time = time_trim[trim_start : trim_end_safe + 1]
                trim_residuals = residuals[trim_start : trim_end_safe + 1]
                ax2.plot(
                    trim_time, trim_residuals, color="gray", alpha=1.0, linewidth=2
                )

            ax2.axhline(y=0, color="black", linestyle="-")

            # Mark trimmed region in residuals - use lower opacity (0.1) for excluded regions
            ax2.axvspan(0, self.trim_start, alpha=0.1, color="gray")
            ax2.axvspan(self.trim_end, min_len, alpha=0.1, color="gray")

            # Add vertical lines at trim points
            ax2.axvline(x=self.trim_start, color="green", linestyle="--")
            ax2.axvline(x=self.trim_end, color="red", linestyle="--")

            # Add grid to residuals plot
            ax2.grid(True, linestyle="--", alpha=0.3)

            # Set titles and labels
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Residuals (m)")
            ax2.set_title("Residuals (Virtual - Actual)")

        # Set titles and labels for the main plot
        ax1.set_ylabel("Elevation (m)")
        ax1.set_title("Virtual Elevation Profile")
        ax1.legend()

        # Add text with CdA and Crr values
        cda_str = f"CdA: {self.current_cda:.3f}"
        crr_str = f"Crr: {self.current_crr:.4f}"
        ax1.text(
            0.02,
            0.95,
            cda_str + "\n" + crr_str,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Add R², RMSE and elevation gain if calculated
        if hasattr(self, "r2") and hasattr(self, "rmse"):
            r2_str = f"R²: {self.r2:.3f}"
            rmse_str = f"RMSE: {self.rmse:.3f} m"

            # Add elevation gain differences
            if hasattr(self, "ve_elevation_diff") and hasattr(
                self, "actual_elevation_diff"
            ):
                ve_gain_str = f"VE Gain: {self.ve_elevation_diff:.1f} m"
                actual_gain_str = f"Actual Gain: {self.actual_elevation_diff:.1f} m"
                diff_str = f"Gain Diff: {self.ve_elevation_diff - self.actual_elevation_diff:.1f} m"

                metrics_text = f"{r2_str}\n{rmse_str}\n{ve_gain_str}\n{actual_gain_str}\n{diff_str}"
            else:
                metrics_text = f"{r2_str}\n{rmse_str}"

            ax1.text(
                0.78,
                0.95,
                metrics_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        self.fig_canvas.fig.tight_layout()
        self.fig_canvas.draw()

    def on_trim_start_changed(self, value):
        """Handle trim start slider value change"""
        # Ensure trim_start < trim_end - 30
        if value >= self.trim_end_slider.value() - 30:
            self.trim_start_slider.setValue(self.trim_end_slider.value() - 30)
            return

        self.trim_start = value
        self.trim_start_label.setText(f"{value} s")

        # Recalculate VE metrics and update plots
        self.calculate_ve()
        self.update_plots()

        # Update map to show trim points
        self.create_map()

    def on_trim_end_changed(self, value):
        """Handle trim end slider value change"""
        # Ensure trim_end > trim_start + 30
        if value <= self.trim_start_slider.value() + 30:
            self.trim_end_slider.setValue(self.trim_start_slider.value() + 30)
            return

        self.trim_end = value
        self.trim_end_label.setText(f"{value} s")

        # Recalculate VE metrics and update plots
        self.calculate_ve()
        self.update_plots()

        # Update map to show trim points
        self.create_map()

    def on_cda_changed(self, value):
        """Handle CdA slider value change"""
        self.current_cda = value / 1000.0
        self.cda_label.setText(f"{self.current_cda:.3f}")

        # Recalculate VE and update plots
        self.calculate_ve()
        self.update_plots()
        self.update_config_text()

    def on_crr_changed(self, value):
        """Handle Crr slider value change"""
        self.current_crr = value / 10000.0
        self.crr_label.setText(f"{self.current_crr:.4f}")

        # Recalculate VE and update plots
        self.calculate_ve()
        self.update_plots()
        self.update_config_text()

    def _fit_map_to_full_route(self, m):
        """Helper to fit map to full route bounds"""
        if self.route_points:
            lats = [p[0] for p in self.route_points]
            lons = [p[1] for p in self.route_points]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

            # Add some padding (5%)
            lat_padding = (max_lat - min_lat) * 0.05
            lon_padding = (max_lon - min_lon) * 0.05
            bounds = [
                [min_lat - lat_padding, min_lon - lon_padding],
                [max_lat + lat_padding, max_lon + lon_padding],
            ]
            m.fit_bounds(bounds)

    def save_results(self):
        """Save analysis results"""
        # Get config name
        config_name = self.config_name.text().strip()
        if not config_name:
            QMessageBox.warning(
                self, "Missing Information", "Please enter a configuration name."
            )
            return

        # Create results directory if it doesn't exist
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Get file basename
        file_basename = Path(self.fit_file.filename).stem

        # Create CSV file path
        csv_path = os.path.join(self.result_dir, f"{file_basename}_ve_results.csv")

        # Prepare data for CSV
        lap_str = "_".join(map(str, sorted(self.selected_laps)))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        result_row = {
            "timestamp": timestamp,
            "laps": lap_str,
            "config_name": config_name,
            "cda": self.current_cda,
            "crr": self.current_crr,
            "system_mass": self.params.get("system_mass", 90),
            "rho": self.params.get("rho", 1.2),
            "rho_low": self.params.get("rho_low", None),
            "rho_high": self.params.get("rho_high", None),
            "eta": self.params.get("eta", 0.98),
            "wind_speed": self.params.get("wind_speed", 0),
            "wind_direction": self.params.get("wind_direction", 0),
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "duration": self.total_duration,
            "distance": self.total_distance,
            "avg_power": self.avg_power,
            "avg_speed": self.avg_speed * 3.6,  # Convert to km/h
        }

        # Add R² and RMSE if available
        if hasattr(self, "r2"):
            result_row["r2"] = self.r2

        if hasattr(self, "rmse"):
            result_row["rmse"] = self.rmse

        # Save to CSV
        header = list(result_row.keys())

        # Check if file exists
        file_exists = os.path.isfile(csv_path)

        # If file exists, read existing data
        existing_data = []
        if file_exists:
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)

        # Remove existing row with same lap selection if present
        existing_data = [
            row
            for row in existing_data
            if not (
                row.get("laps") == lap_str and row.get("config_name") == config_name
            )
        ]
        # Add new row
        existing_data.append(result_row)

        # Sort by lap selection
        existing_data.sort(key=lambda x: (x.get("laps", ""), x.get("config_name", "")))

        # Write all data back to CSV
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(existing_data)

        # Save plot
        plot_dir = os.path.join(self.result_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_filename = f"{file_basename}_laps_{lap_str}_{config_name}.png"
        plot_path = os.path.join(plot_dir, plot_filename)

        self.fig_canvas.fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Save trim settings to file settings
        file_settings = self.settings.get_file_settings(self.fit_file.filename)

        # Initialize trim_settings if it doesn't exist
        if "trim_settings" not in file_settings:
            file_settings["trim_settings"] = {}

        # Save trim settings for this lap combination
        file_settings["trim_settings"][self.lap_combo_id] = {
            "trim_start": self.trim_start,
            "trim_end": self.trim_end,
            "cda": self.current_cda,
            "crr": self.current_crr,
        }

        # Save updated file settings
        self.settings.save_file_settings(self.fit_file.filename, file_settings)

        # Show success message
        QMessageBox.information(
            self,
            "Results Saved",
            f"Analysis results saved successfully to:\n{csv_path}\n\nPlot saved to:\n{plot_path}",
        )

    def back_to_selection(self):
        """Return to lap selection window"""
        from ui.analysis_window import AnalysisWindow

        self.analysis_window = AnalysisWindow(self.fit_file, self.settings)
        self.analysis_window.show()
        self.close()

    def map_time_to_route_index(self, time_index):
        """
        Map a time index from the full dataset to the corresponding index in route_points

        Parameters:
        -----------
        time_index : int
            Index in the merged_data dataframe

        Returns:
        --------
        int
            Corresponding index in the route_points list, or None if not mappable
        """
        if not hasattr(self, "route_timestamps") or not self.route_timestamps:
            return None

        if time_index < 0 or time_index >= len(self.merged_data):
            return None

        # Get the timestamp at this index
        target_timestamp = self.merged_data["timestamp"].iloc[time_index]

        # Find the closest timestamp in route_timestamps
        if target_timestamp in self.route_timestamps:
            return self.route_timestamps.index(target_timestamp)

        # If not found directly, find the closest one
        for i, ts in enumerate(self.route_timestamps):
            if ts >= target_timestamp:
                return i

        # If we get here, target_timestamp is after all route_timestamps
        return len(self.route_timestamps) - 1
