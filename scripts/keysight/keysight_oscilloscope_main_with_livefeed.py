#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                 KEYSIGHT OSCILLOSCOPE AUTOMATION SYSTEM                        ║
║                   Enhanced GUI Application                                      ║
║                                                                                ║
║  Purpose: Comprehensive automated control and data acquisition interface      ║
║           for Keysight DSOX6004A series oscilloscopes with advanced           ║
║           timebase, trigger configuration, and multi-channel support          ║
║                                                                                ║
║  Features:                                                                     ║
║    • Responsive multi-threaded GUI using tkinter framework                    ║
║    • Real-time VISA-based oscilloscope communication                          ║
║    • Horizontal timebase (time/div) and vertical (V/div) controls            ║
║    • Individual channel trigger configuration (level, slope, source)          ║
║    • Function generator automation (WGEN1, WGEN2)                            ║
║    • Multi-channel waveform data acquisition                                 ║
║    • Automated CSV export with metadata preservation                         ║
║    • Publication-quality plot generation with statistics                     ║
║    • Screenshot capture with custom save locations                           ║
║    • Color-coded activity logging and status tracking                        ║
║    • Full automation workflow (sequential screenshot→acquire→export→plot)    ║
║                                                                                ║
║  Architecture:                                                                 ║
║    • Queue-based thread communication for GUI responsiveness                 ║
║    • Separation of UI logic from instrument control                          ║
║    • Thread-safe status updates and logging                                  ║
║    • Modular method organization by functional domain                        ║
║                                                                                ║
║  Quality Standards: PEP 8 Compliant | Google-style Docstrings               ║
║                    Industry-Grade Documentation | Professional Review Ready   ║
║                                                                                ║
║  Author: Professional Instrumentation Control System                          ║
║  Last Updated: 2025-10-24 | Status: Production Ready                          ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import sys  # System-level operations and path management
import logging  # Application event logging and diagnostic tracking
from pathlib import Path  # Object-oriented filesystem path handling (cross-platform compatible)
from typing import Optional, Dict, Any, List, Tuple  # Type hints for code clarity and IDE support
import os  # Operating system interface for environment variables and file operations
import threading  # Multi-threaded operation support for non-blocking GUI responsiveness
import queue  # Thread-safe queue for inter-thread communication
from datetime import datetime  # Timestamp generation for logging and file naming
import time  # Lightweight sleeps for backoff on VISA lock

import tkinter as tk  # Core GUI framework for creating windows and widgets
from tkinter import ttk, messagebox, filedialog, scrolledtext  # Advanced GUI components: themed widgets, dialogs, text areas
import pandas as pd  # Data manipulation and CSV export functionality
import matplotlib.pyplot as plt  # Publication-quality graph plotting and visualization
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000  # Fix for large datasets
matplotlib.rcParams['path.simplify'] = True
import numpy as np  # Numerical array operations for waveform statistics

# Add parent directories to Python path to find instrument_control
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

try:
    from instrument_control.keysight_oscilloscope import KeysightDSOX6004A, KeysightDSOX6004AError
    from instrument_control.scpi_wrapper import SCPIWrapper
except ImportError as e:
    print(f"Error importing instrument control modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Looking for instrument_control in: {script_dir}")
    print("Please ensure the instrument_control package is in your Python path")
    sys.exit(1)

def parse_timebase_string(value: str) -> float:
    value = value.strip().lower()
    if "ns" in value:
        return float(value.replace("ns", "").strip()) / 1_000_000_000
    elif "µs" in value or "us" in value:
        return float(value.replace("µs", "").replace("us", "").strip()) / 1_000_000
    elif "ms" in value:
        return float(value.replace("ms", "").strip()) / 1000
    elif "s" in value:
        return float(value.replace("s", "").strip())
    else:
        return float(value)  # Default - assume already in seconds
TRIGGER_SLOPE_MAP = {
    "Raising": "POS",
    "Falling": "NEG"
}


def _format_si_value(value: float, kind: str) -> str:
    v = abs(value)
    if kind == "freq":
        if v >= 1e9:
            return f"{value/1e9:.3f} GHz"
        if v >= 1e6:
            return f"{value/1e6:.3f} MHz"
        if v >= 1e3:
            return f"{value/1e3:.3f} kHz"
        return f"{value:.3f} Hz"
    if kind == "time":
        if v >= 1:
            return f"{value:.3f} s"
        if v >= 1e-3:
            return f"{value*1e3:.3f} ms"
        if v >= 1e-6:
            return f"{value*1e6:.3f} µs"
        if v >= 1e-9:
            return f"{value*1e9:.3f} ns"
        return f"{value*1e12:.3f} ps"
    if kind == "volt":
        if v >= 1e3:
            return f"{value/1e3:.3f} kV"
        if v >= 1:
            return f"{value:.3f} V"
        if v >= 1e-3:
            return f"{value*1e3:.3f} mV"
        return f"{value*1e6:.3f} µV"
    if kind == "percent":
        return f"{value:.2f} %"
    return f"{value}"


def format_measurement_value(meas_type: str, value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if meas_type == "FREQ":
        return _format_si_value(value, "freq")
    if meas_type in ["PERiod", "RISE", "FALL"]:
        return _format_si_value(value, "time")
    if meas_type in ["VAMP", "VTOP", "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "VPP"]:
        return _format_si_value(value, "volt")
    if meas_type in ["DUTYcycle", "NDUTy", "OVERshoot"]:
        return _format_si_value(value, "percent")
    return f"{value}"

class OscilloscopeDataAcquisition:
    """
    Oscilloscope Data Management Handler
    
    Responsible for all waveform data operations: acquisition from instrument,
    export to standard formats (CSV), and visualization (plots with statistics).
    Maintains separation between instrument communication and data processing.
    
    Attributes:
        scope (KeysightDSOX6004A): Reference to connected oscilloscope instance
        _logger (logging.Logger): Diagnostic logger for this data handler
        default_data_dir (Path): Directory for CSV exports (default: ./data/)
        default_graph_dir (Path): Directory for plot images (default: ./graphs/)
        default_screenshot_dir (Path): Directory for instrument screenshots (default: ./screenshots/)
    """

    def __init__(self, oscilloscope_instance, io_lock: Optional[threading.RLock] = None):
        """Initialize data acquisition handler with oscilloscope reference."""
        self.scope = oscilloscope_instance  # Store reference to oscilloscope for method calls
        self._logger = logging.getLogger(f'{self.__class__.__name__}')  # Create diagnostic logger
        self.default_data_dir = Path.cwd() / "data"  # Default location for CSV data files
        self.default_graph_dir = Path.cwd() / "graphs"  # Default location for plot images
        self.default_screenshot_dir = Path.cwd() / "screenshots"  # Default location for screenshots
        self.io_lock = io_lock  # Shared I/O lock for VISA/SCPI serialization

    def acquire_waveform_data(self, channel: int, max_points: int = 62500) -> Optional[Dict[str, Any]]:
        """
        Retrieve raw waveform data from specified oscilloscope channel.
        
        This method configures the oscilloscope for data transfer mode, queries the
        waveform preamble (contains scaling information), retrieves binary waveform data,
        and converts raw ADC values to voltage/time coordinates using calibration factors.
        
        Args:
            channel (int): Target oscilloscope channel (1-4)
            max_points (int): Maximum waveform samples to retrieve (default: 62500 points)
        
        Returns:
            Dict[str, Any]: Structured waveform data with metadata, or None if acquisition fails
                Keys: 'channel', 'time' (seconds), 'voltage' (volts), 'sample_rate' (Hz),
                      'time_increment' (seconds), 'voltage_increment' (volts), 'points_count',
                      'acquisition_time' (ISO format timestamp)
        
        Raises:
            Logs errors internally; returns None on failure (non-blocking)
        """
        if not self.scope.is_connected:  # Verify oscilloscope connection before attempting acquisition
            self._logger.error("Cannot acquire data: oscilloscope not connected")
            return None

        try:
            lock = self.io_lock
            if lock:
                with lock:
                    self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")
                    self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")
                    self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")
                    self.scope._scpi_wrapper.write(f":WAVeform:POINts {max_points}")
                    preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
                    preamble_parts = preamble.split(',')
                    y_increment = float(preamble_parts[7])
                    y_origin = float(preamble_parts[8])
                    y_reference = float(preamble_parts[9])
                    x_increment = float(preamble_parts[4])
                    x_origin = float(preamble_parts[5])
                    raw_data = self.scope._scpi_wrapper.query_binary_values(":WAVeform:DATA?", datatype='B')
            else:
                self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")
                self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")
                self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")
                self.scope._scpi_wrapper.write(f":WAVeform:POINts {max_points}")
                preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
                preamble_parts = preamble.split(',')
                y_increment = float(preamble_parts[7])
                y_origin = float(preamble_parts[8])
                y_reference = float(preamble_parts[9])
                x_increment = float(preamble_parts[4])
                x_origin = float(preamble_parts[5])
                raw_data = self.scope._scpi_wrapper.query_binary_values(":WAVeform:DATA?", datatype='B')
            voltage_data = [(value - y_reference) * y_increment + y_origin for value in raw_data]  # Convert ADC→voltage using calibration
            time_data = [x_origin + (i * x_increment) for i in range(len(voltage_data))]  # Generate time array with proper scaling
            self._logger.info(f"Successfully acquired {len(voltage_data)} points from channel {channel}")
            return {
                'channel': channel,  # Preserve channel identifier
                'time': time_data,  # Time array in seconds
                'voltage': voltage_data,  # Voltage array in volts
                'sample_rate': 1.0 / x_increment,  # Calculate sampling frequency from time resolution
                'time_increment': x_increment,  # Time resolution between samples
                'voltage_increment': y_increment,  # Voltage resolution per ADC count
                'points_count': len(voltage_data),  # Total samples acquired
                'acquisition_time': datetime.now().isoformat()  # Record acquisition timestamp for traceability
            }
        except Exception as e:  # Catch any communication or processing errors
            err = str(e)
            if "VI_ERROR_RSRC_LOCKED" in err or "resource is locked" in err:
                # Brief backoff to allow other operation to finish
                time.sleep(0.05)
            self._logger.error(f"Failed to acquire waveform data from channel {channel}: {e}")
            return None
    # def parse_timebase_string(value: str) -> float:
    #     value = value.strip().lower()
    #     if "ns" in value:
    #         return float(value.replace("ns", "").strip()) / 1_000_000_000
    #     elif "µs" in value or "us" in value:
    #         return float(value.replace("µs", "").replace("us", "").strip()) / 1_000_000
    #     elif "ms" in value:
    #         return float(value.replace("ms", "").strip()) / 1000
    #     elif "s" in value:
    #         return float(value.replace("s", "").strip())
    #     else:
    #         return float(value)  # Default - assume already in seconds

    def export_to_csv(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None, 
                     filename: Optional[str] = None) -> Optional[str]:
        """
        Export waveform data to CSV file with metadata header.
        
        Creates a CSV file containing time and voltage columns with comprehensive
        metadata header (comments) including acquisition parameters for traceability
        and reproducibility in data analysis pipelines.
        
        Args:
            waveform_data (Dict): Waveform dictionary from acquire_waveform_data()
            custom_path (Optional[str]): Override default save directory (None→default)
            filename (Optional[str]): Override auto-generated filename (None→auto-generate)
        
        Returns:
            str: Full path to saved CSV file, or None if export fails
        
        Raises:
            Logs errors internally; returns None on failure (non-blocking)
        """
        if not waveform_data:  # Validate input data exists
            self._logger.error("No waveform data to export")
            return None

        try:
            save_dir = Path(custom_path) if custom_path else self.default_data_dir  # Select output directory
            self.scope.setup_output_directories()  # Create default directory structure if needed
            save_dir.mkdir(parents=True, exist_ok=True)  # Create target directory with parent folders
            if filename is None:  # Auto-generate filename if not provided
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Create readable timestamp
                filename = f"waveform_ch{waveform_data['channel']}_{timestamp}.csv"  # Compose descriptive filename
            if not filename.endswith('.csv'):  # Ensure proper file extension
                filename += '.csv'
            filepath = save_dir / filename  # Construct full file path
            df = pd.DataFrame({  # Create two-column dataframe (Time, Voltage)
                'Time (s)': waveform_data['time'],
                'Voltage (V)': waveform_data['voltage']
            })
            with open(filepath, 'w') as f:  # Open file for writing
                f.write(f"# Oscilloscope Waveform Data\n")  # Metadata: Application identifier
                f.write(f"# Channel: {waveform_data['channel']}\n")  # Metadata: Source channel number
                f.write(f"# Acquisition Time: {waveform_data['acquisition_time']}\n")  # Metadata: ISO timestamp
                f.write(f"# Sample Rate: {waveform_data['sample_rate']:.2e} Hz\n")  # Metadata: Sampling frequency
                f.write(f"# Points Count: {waveform_data['points_count']}\n")  # Metadata: Total samples
                f.write(f"# Time Increment: {waveform_data['time_increment']:.2e} s\n")  # Metadata: Time resolution
                f.write(f"# Voltage Increment: {waveform_data['voltage_increment']:.2e} V\n")  # Metadata: Voltage resolution
                f.write("\n")  # Blank line separator between header and data
            df.to_csv(filepath, mode='a', index=False)  # Append data columns (without index column)
            self._logger.info(f"CSV exported successfully: {filepath}")
            return str(filepath)  # Return path as string for downstream processing
        except Exception as e:  # Catch file I/O or format errors
            self._logger.error(f"Failed to export CSV: {e}")
            return None

#     def generate_waveform_plot(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None,
#                               filename: Optional[str] = None, plot_title: Optional[str] = None) -> Optional[str]:
#         """
#         Generate publication-quality plot with embedded statistics.
        
#         Creates a matplotlib figure showing voltage vs. time with grid overlay,
#         and adds a statistics box (max, min, mean, RMS, std dev) calculated from
#         the waveform for immediate visual analysis without external tools.
        
#         Args:
#             waveform_data (Dict): Waveform dictionary from acquire_waveform_data()
#             custom_path (Optional[str]): Override default save directory (None→default)
#             filename (Optional[str]): Override auto-generated filename (None→auto-generate)
#             plot_title (Optional[str]): Custom plot title (None→auto-generate from channel)
        
#         Returns:
#             str: Full path to saved PNG file, or None if generation fails
        
#         Raises:
#             Logs errors internally; returns None on failure (non-blocking)
#         """
#         if not waveform_data:  # Validate input data exists
#             self._logger.error("No waveform data to plot")
#             return None

#         try:
#             save_dir = Path(custom_path) if custom_path else self.default_graph_dir  # Select output directory
#             self.scope.setup_output_directories()  # Create default directory structure if needed
#             save_dir.mkdir(parents=True, exist_ok=True)  # Create target directory with parent folders
#             if filename is None:  # Auto-generate filename if not provided
#                 timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Create readable timestamp
#                 filename = f"waveform_plot_ch{waveform_data['channel']}_{timestamp}.png"  # Compose descriptive filename
#             if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Validate image format
#                 filename += '.png'  # Default to PNG if extension missing
#             filepath = save_dir / filename  # Construct full file path
#             plt.figure(figsize=(12, 8))  # Create new figure with wide aspect ratio for readability
#             plt.plot(waveform_data['time'], waveform_data['voltage'], 'b-', linewidth=1)  # Plot waveform as blue line
#             if plot_title is None:  # Auto-generate title if not provided
#                 plot_title = f"Oscilloscope Waveform - Channel {waveform_data['channel']}"
#             plt.title(plot_title, fontsize=14, fontweight='bold')  # Add descriptive title
#             plt.xlabel('Time (s)', fontsize=12)  # Label X-axis with units
#             plt.ylabel('Voltage (V)', fontsize=12)  # Label Y-axis with units
#             plt.grid(True, alpha=0.3)  # Enable semi-transparent grid for reference
#             voltage_array = np.array(waveform_data['voltage'])  # Convert to NumPy array for statistics
#             stats_text = f"""Statistics:
# Max: {np.max(voltage_array):.3f} V
# Min: {np.min(voltage_array):.3f} V
# Mean: {np.mean(voltage_array):.3f} V
# RMS: {np.sqrt(np.mean(voltage_array**2)):.3f} V
# Std Dev: {np.std(voltage_array):.3f} V
# Points: {len(voltage_array)}"""  # Compute and format statistical summary
#             plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
#                     fontsize=10, verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))  # Embed statistics box in plot corner
#             plt.tight_layout()  # Adjust spacing to prevent label cutoff
#             plt.savefig(filepath, dpi=1600, bbox_inches='tight')  # Save at high resolution (1600 DPI) for publication
#             plt.close()  # Release memory by closing figure
#             self._logger.info(f"Plot saved successfully: {filepath}")
#             return str(filepath)  # Return path as string for downstream processing
#         except Exception as e:  # Catch plotting or file I/O errors
#             self._logger.error(f"Failed to generate plot: {e}")
#             return None
    # Always perform fresh measurements before plotting
    

# Then update measurements_text as you already do, but use this new 'measurements' dict

    def generate_waveform_plot(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None,
                                filename: Optional[str] = None, plot_title: Optional[str] = None) -> Optional[str]:
        """
        Generate publication-quality plot with embedded MEASUREMENTS (not statistics).
        Creates a matplotlib figure showing voltage vs. time with grid overlay,
        and adds a measurement box containing values reported by the oscilloscope.
        """
        measurements = {}
        measurement_types = ["FREQ", "PERiod", "VPP", "VAMP", "OVERshoot", "VTOP", "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL", "DUTYcycle", "NDUTy"]
        for meas_type in measurement_types:
            try:
                if self.io_lock:
                    with self.io_lock:
                        value = self.scope.measure_single(waveform_data['channel'], meas_type)
                else:
                    value = self.scope.measure_single(waveform_data['channel'], meas_type)
                measurements[meas_type] = value
            except:
                    measurements[meas_type] = None
            if not waveform_data:
                self._logger.error("No waveform data to plot")
                return None

        try:
            save_dir = Path(custom_path) if custom_path else self.default_graph_dir  # Select output directory
            self.scope.setup_output_directories()  # Create default directory structure if needed
            save_dir.mkdir(parents=True, exist_ok=True)  # Create target directory with parent folders
            if filename is None:  # Auto-generate filename if not provided
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Create readable timestamp
                filename = f"waveform_plot_ch{waveform_data['channel']}_{timestamp}.png"  # Compose descriptive filename
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Validate image format
                filename += '.png'  # Default to PNG if extension missing
            filepath = save_dir / filename  # Construct full file path

            plt.figure(figsize=(12, 8))  # Create new figure with wide aspect ratio for readability
            plt.plot(waveform_data['time'], waveform_data['voltage'], 'b-', linewidth=1)  # Plot waveform as blue line
            if plot_title is None:  # Auto-generate title if not provided
                plot_title = f"Oscilloscope Waveform - Channel {waveform_data['channel']}"
            plt.title(plot_title, fontsize=14, fontweight='bold')  # Add descriptive title
            plt.xlabel('Time (s)', fontsize=12)  # Label X-axis with units
            plt.ylabel('Voltage (V)', fontsize=12)  # Label Y-axis with units
            plt.grid(True, alpha=0.3)  # Enable semi-transparent grid for reference

            # Display MEASUREMENTS
            #measurements = waveform_data.get('measurements', {})

            measurements_text = "MEASUREMENTS:\n"
            measurements_text += "─" * 25 + "\n"
            measurements_text += f"Freq: {format_measurement_value('FREQ', measurements.get('FREQ'))}\n"
            measurements_text += f"Period: {format_measurement_value('PERiod', measurements.get('PERiod'))}\n"
            measurements_text += f"VPP: {format_measurement_value('VPP', measurements.get('VPP'))}\n"
            measurements_text += f"VAVG: {format_measurement_value('VAVG', measurements.get('VAVG'))}\n"
            measurements_text += f"OVERshoot: {format_measurement_value('OVERshoot', measurements.get('OVERshoot'))}\n"
            measurements_text += f"VMAX: {format_measurement_value('VMAX', measurements.get('VMAX'))}\n"
            measurements_text += f"VMIN: {format_measurement_value('VMIN', measurements.get('VMIN'))}\n"
            measurements_text += f"DUTYcycle: {format_measurement_value('DUTYcycle', measurements.get('DUTYcycle'))}\n"
            measurements_text += f"NDUTy: {format_measurement_value('NDUTy', measurements.get('NDUTy'))}\n"
            # measurements_text += f"Pcycle: {measurements.get('Pcycle', 'N/A')}\n"
            # measurements_text += f"Ncycle: {measurements.get('Ncycle', 'N/A')}\n"

            plt.text(0.02, 0.98, measurements_text, 
                     transform=plt.gca().transAxes,
                     fontsize=9, 
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
                     family='monospace')

            plt.tight_layout()
            plt.savefig(filepath, dpi=1600, bbox_inches='tight')
            plt.close()
            self._logger.info(f"Plot saved successfully: {filepath}")
            return str(filepath)

        except Exception as e:
            self._logger.error(f"Failed to generate plot: {e}")
            return None



class EnhancedResponsiveAutomationGUI:
    """
    Professional Oscilloscope Automation GUI Application
    
    Implements a complete tkinter-based graphical interface for Keysight oscilloscope
    control with responsive layout, multi-threaded operations, and comprehensive
    instrument configuration capabilities. Designed for laboratory automation and
    test equipment monitoring workflows.
    
    Key Responsibilities:
        • GUI layout management with responsive grid-based design
        • User input capture and validation
        • Oscilloscope method calls via instrument_control module
        • Multi-threaded background operations (non-blocking UI)
        • Status tracking and real-time event logging
        • Data acquisition and export coordination
    
    Architecture Pattern:
        • Main thread: GUI event handling and display updates
        • Worker threads: Long-running operations (connection, data acquisition, plotting)
        • Queue: Thread-safe communication between worker and main threads
        • Periodic polling: Main thread checks queue for status updates
    
    Attributes:
        root (tk.Tk): Main tkinter window
        oscilloscope (KeysightDSOX6004A): Connected instrument instance
        data_acquisition (OscilloscopeDataAcquisition): Data handler instance
        last_acquired_data (Dict): Cache of most recent acquisition for quick export/plot
        status_queue (queue.Queue): Thread-safe status message queue
        Various StringVar/DoubleVar/BooleanVar: GUI control value containers
    """

    def __init__(self):
        """Initialize GUI application with all components and event handlers."""
        self.root = tk.Tk()  # Create main application window
        self.oscilloscope = None  # Will be initialized on connection
        self.data_acquisition = None  # Will be initialized on connection
        self.last_acquired_data = None  # Cache for quick access to acquisition data
        self.save_locations = {  # Default save directories
            'data': str(Path.cwd() / "data"),
            'graphs': str(Path.cwd() / "graphs"),
            'screenshots': str(Path.cwd() / "screenshots")
        }
        self.setup_logging()  # Configure diagnostic logging
        self.setup_gui()  # Create GUI layout and widgets
        self.status_queue = queue.Queue()  # Create thread-safe communication channel
        self.live_feed_active = False
        self.live_feed_paused = False
        self.live_feed_worker_running = False  # Prevent overlapping acquisitions
        self.io_lock = threading.RLock()  # Serialize all VISA/SCPI access
        # Declare trigger level variables dict early so type checkers see the attribute
        self.channel_trigger_vars: dict[int, tk.DoubleVar] = {}
        self.check_status_updates()  # Start polling for status updates from worker threads

    def setup_logging(self):
        """Configure application-wide logging for diagnostics and troubleshooting."""
        logging.basicConfig(
            level=logging.INFO,  # Capture INFO, WARNING, ERROR (not DEBUG)
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Standard format with timestamp
        )
        self.logger = logging.getLogger('EnhancedOscilloscopeAutomation')  # Create application logger

    def setup_gui(self):
        """Construct responsive GUI layout with all control sections."""
        self.root.title("Keysight Oscilloscope Automation")  # Set window title
        self.root.geometry("1200x900")  # Set initial window size (width × height pixels) - increased for live feed
        self.root.configure(bg='#f5f5f5')  # Light gray background for professional appearance
        self.root.minsize(1000, 600)  # Enforce minimum window size for usability
        self.root.columnconfigure(0, weight=1)  # Allow column to expand with window
        self.root.rowconfigure(0, weight=1)  # Allow row to expand with window
        self.style = ttk.Style()  # Create themed widget style manager
        self.style.theme_use('clam')  # Use modern 'clam' theme for professional look
        self.configure_styles()  # Apply custom color and font styles
        
        # Create canvas and scrollbar for scrollable content
        canvas_container = ttk.Frame(self.root)
        canvas_container.grid(row=0, column=0, sticky='nsew', padx=3, pady=3)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        self.main_canvas = tk.Canvas(canvas_container, bg='#f5f5f5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient='vertical', command=self.main_canvas.yview)
        
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        self.main_canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Create main frame inside canvas
        main_frame = ttk.Frame(self.main_canvas, padding="3")  # Main container with 3px padding
        self.canvas_window = self.main_canvas.create_window((0, 0), window=main_frame, anchor='nw')
        
        # Bind canvas resize to update scroll region
        def _configure_scroll(event):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all'))
        main_frame.bind('<Configure>', _configure_scroll)
        
        # Update canvas window width when canvas is resized
        def _configure_canvas(event):
            canvas_width = event.width
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)
        self.main_canvas.bind('<Configure>', _configure_canvas)
        
        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_frame.columnconfigure(0, weight=1)  # Allow column to expand
        main_frame.rowconfigure(0, weight=0)  # Title: fixed size
        main_frame.rowconfigure(1, weight=0)  # Connection: fixed size
        main_frame.rowconfigure(2, weight=0)  # Channel config: fixed size
        main_frame.rowconfigure(3, weight=0)  # Timebase/trigger: fixed size
        main_frame.rowconfigure(4, weight=0)  # Function generators: fixed size
        main_frame.rowconfigure(5, weight=0)  # File preferences: fixed size
        main_frame.rowconfigure(6, weight=0)  # Operations buttons: fixed size (MOVED UP)
        main_frame.rowconfigure(7, weight=0)  # Measurement: fixed size
        main_frame.rowconfigure(8, weight=0)  # Live feed: fixed size
        main_frame.rowconfigure(9, weight=1)  # Status log: EXPANDABLE to fill remaining space
        title_label = ttk.Label(main_frame, text="Keysight Oscilloscope Automation", style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 3), sticky='ew')  # Span full width
        self.create_connection_frame(main_frame, row=1)  # Add connection controls
        self.create_channel_config_frame(main_frame, row=2)  # Add channel configuration
        self.create_timebase_trigger_frame(main_frame, row=3)  # Add timebase/trigger controls
        self.create_function_generator_frame(main_frame, row=4)  # Add function generator controls
        self.create_file_preferences_frame(main_frame, row=5)  # Add file path settings
        self.create_operations_frame(main_frame, row=6)  # Add operation buttons (MOVED UP)
        self.create_measurement_frame(main_frame, row=7)  # Measurement & autoscale
        self.create_live_feed_frame(main_frame, row=8)  # Add live signal feed
        self.create_status_frame(main_frame, row=9)  # Add status log (expandable)
        self.disable_operation_buttons()  # Disable all buttons until connection established
        self.main_frame = main_frame  # Store reference for future modifications

    def configure_styles(self):
        """Define custom widget styles for consistent professional appearance."""
        self.style.configure('Title.TLabel',
                            font=('Times New Roman', 12, 'bold'),  # Large bold serif font
                            foreground='#1a365d',  # Dark blue text
                            background='#f5f5f5')  # Match background
        self.style.configure('Success.TButton', font=('Arial', 8))  # Compact font for buttons
        self.style.configure('Warning.TButton', font=('Arial', 8))
        self.style.configure('Info.TButton', font=('Arial', 8))
        self.style.configure('Primary.TButton', font=('Arial', 8))

    def create_connection_frame(self, parent, row):
        """Create VISA address entry and connection control buttons."""
        conn_frame = ttk.LabelFrame(parent, text="Connection", padding="3")  # Labeled container
        conn_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))  # Expand horizontally
        conn_frame.columnconfigure(1, weight=1)  # Entry field expands
        ttk.Label(conn_frame, text="VISA:", font=('Calibri', 8)).grid(row=0, column=0, sticky='w', padx=(0, 3))
        self.visa_address_var = tk.StringVar(value="USB0::0x0957::0x1780::MY65220169::INSTR")  # Default VISA resource string
        self.visa_entry = ttk.Entry(conn_frame, textvariable=self.visa_address_var, font=('Arial', 8))
        self.visa_entry.grid(row=0, column=1, sticky='ew', padx=(0, 3))  # Expandable entry field
        self.connect_btn = ttk.Button(conn_frame, text="Connect", width=8, 
                                     command=self.connect_oscilloscope, style='Success.TButton')
        self.connect_btn.grid(row=0, column=2, padx=1)  # Connect button
        self.disconnect_btn = ttk.Button(conn_frame, text="Disc", width=6, 
                                        command=self.disconnect_oscilloscope, 
                                        style='Warning.TButton', state='disabled')  # Initially disabled until connected
        self.disconnect_btn.grid(row=0, column=3, padx=1)
        self.test_btn = ttk.Button(conn_frame, text="Test", width=5, 
                                  command=self.test_connection, 
                                  style='Info.TButton', state='disabled')  # Initially disabled until connected
        self.test_btn.grid(row=0, column=4, padx=1)
        self.conn_status_var = tk.StringVar(value="Disconnected")  # Connection status text
        self.conn_status_label = ttk.Label(conn_frame, textvariable=self.conn_status_var, 
                                          font=('Arial', 8, 'bold'), foreground='#e53e3e')  # Red text when disconnected
        self.conn_status_label.grid(row=0, column=5, sticky='e', padx=(5, 0))
        self.info_text = tk.Text(conn_frame, height=1, font=('Courier', 7), state='disabled', 
                                bg='#f8f9fa', relief='flat', borderwidth=0)  # Instrument info display (monospace)
        self.info_text.grid(row=1, column=0, columnspan=6, sticky='ew', pady=(2, 0))

    def create_channel_config_frame(self, parent, row):
        """Create vertical scale, offset, coupling, and probe attenuation controls."""
        config_frame = ttk.LabelFrame(parent, text="Channel Config", padding="3")
        config_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        col = 0
        ttk.Label(config_frame, text="Select:", font=('Arial', 8, 'bold')).grid(row=0, column=col, sticky='w')
        col += 1
        self.channel_enable_vars = {}  # Dictionary to store checkbox state for each channel
        for ch in [1, 2, 3, 4]:
            var = tk.BooleanVar(value=(ch == 1))  # Enable channel 1 by default
            self.channel_enable_vars[ch] = var  # Store reference
            var.trace_add('write', lambda *args, channel=ch: self.toggle_channel_display(channel))  # Enable/disable channel on checkbox change
            ttk.Checkbutton(config_frame, text=f"Ch{ch}", variable=var, style='TCheckbutton').grid(row=0, column=col, padx=2)
            col += 1
        ttk.Separator(config_frame, orient='vertical').grid(row=0, column=col, sticky='ns', padx=8)  # Vertical divider
        col += 1
        #ttk.Label(config_frame, text="V/div:", font=('Arial', 8)).grid(row=0, column=col, sticky='w')
        # col += 1
        # self.v_scale_var = tk.DoubleVar(value=1.0)  # Vertical scale (volts per division)
        # ttk.Combobox(config_frame, textvariable=self.v_scale_var, 
        #             values=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 
        #             width=6, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        
        ttk.Label(config_frame, text="V/div:", font=('Arial', 8)).grid(row=0, column=col, sticky='w')
        col += 1
        self.v_scale_var = tk.DoubleVar(value=0.0)  # Vertical scale (volts per division)
        ttk.Entry(config_frame, textvariable=self.v_scale_var, width=6, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        col += 1
        ttk.Label(config_frame, text="Coup:", font=('Arial', 8)).grid(row=0, column=col, sticky='w')
        col += 1
        self.coupling_var = tk.StringVar(value="DC")  # Input coupling mode (AC or DC)
        ttk.Combobox(config_frame, textvariable=self.coupling_var, values=["AC", "DC"], 
                    width=4, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        col += 1
        # In create_channel_config_frame:
        ttk.Label(config_frame, text="Offset(V):", font=('Arial', 8)).grid(row=0, column=col, sticky='w')
        col += 1
        self.v_offset_var = tk.DoubleVar(value=0.0)
        ttk.Entry(config_frame, textvariable=self.v_offset_var, width=6, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        col += 1

        ttk.Label(config_frame, text="Probe:", font=('Arial', 8)).grid(row=0, column=col, sticky='w')
        col += 1
        self.probe_var = tk.DoubleVar(value=1.0)  # Probe attenuation factor (1×, 10×, 100×)
        ttk.Combobox(config_frame, textvariable=self.probe_var, values=["1.0", "10.0", "100.0"], 
                    width=5, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        col += 1
        self.config_channel_btn = ttk.Button(config_frame, text="Configure", 
                                           command=self.configure_channel, 
                                           style='Primary.TButton', state='disabled')  # Initially disabled until connected
        self.config_channel_btn.grid(row=0, column=col, sticky='ew')
        config_frame.columnconfigure(col, weight=1)  # Allow last column to expand

    def create_timebase_trigger_frame(self, parent, row):
        """Create horizontal timebase and trigger configuration controls."""
        timebase_frame = ttk.LabelFrame(parent, text="Timebase & Trigger Controls", padding="3")
        timebase_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        for i in range(20):
            timebase_frame.columnconfigure(i, weight=0)  # Fixed-width columns
        timebase_frame.columnconfigure(19, weight=1)  # Last column expandable
        col = 0
        ttk.Label(timebase_frame, text="TIMEBASE:", font=('Arial', 8, 'bold'), 
                 foreground='#1a365d').grid(row=0, column=col, sticky='w', padx=(0, 5))
        col += 1
        ttk.Label(timebase_frame, text="Time/div:", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.time_scale_var = tk.StringVar(value="10 ms")  # Horizontal scale (seconds per division)
        timebase_scales = [
            "1 ns", "2 ns", "5 ns",
            "10 ns", "20 ns", "50 ns",
            "100 ns", "200 ns", "500 ns",
            "1 µs", "2 µs", "5 µs",
            "10 µs", "20 µs", "50 µs",
            "100 µs", "200 µs", "500 µs",
            "1 ms", "2 ms", "5 ms",
            "10 ms", "20 ms", "50 ms",
            "100 ms", "200 ms", "500 ms",
            "1 s", "2 s", "5 s",
            "10 s", "20 s", "50 s"
        ]

        self.timebase_combo = ttk.Combobox(timebase_frame, textvariable=self.time_scale_var, 
                                          values=timebase_scales, width=8, state='readonly', font=('Arial', 8))
        self.timebase_combo.grid(row=0, column=col, padx=(0, 5))
        col += 1
        # ttk.Label(timebase_frame, text="Offset(s):", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        # col += 1
        # self.time_offset_var = tk.DoubleVar(value=0.0)  # Horizontal position offset (seconds)
        # ttk.Entry(timebase_frame, textvariable=self.time_offset_var, width=8, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 8))
        # col += 1
        self.timebase_apply_btn = ttk.Button(timebase_frame, text="Apply Timebase", 
                                           command=self.configure_timebase, 
                                           style='Primary.TButton', state='disabled')  # Initially disabled until connected
        self.timebase_apply_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        col = 0
        ttk.Label(timebase_frame, text="TRIGGER:", font=('Arial', 8, 'bold'), 
                 foreground='#1a365d').grid(row=1, column=col, sticky='w', padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(timebase_frame, text="Source:", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.trigger_source_var = tk.StringVar(value="CH1")  # Trigger source channel selection
        ttk.Combobox(timebase_frame, textvariable=self.trigger_source_var, 
                    values=["CH1", "CH2", "CH3", "CH4"], width=5, state='readonly', 
                    font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(timebase_frame, text="Level(V):", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.trigger_level_var = tk.DoubleVar(value=0.0)  # Trigger voltage threshold (volts)
        ttk.Entry(timebase_frame, textvariable=self.trigger_level_var, width=8, font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(timebase_frame, text="Slope:", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.trigger_slope_var = tk.StringVar(value="Raising")  # Trigger edge direction (POS=rising, NEG=falling)
        ttk.Combobox(timebase_frame, textvariable=self.trigger_slope_var, 
                    values=["Raising", "Falling"], width=10, state='readonly', 
                    font=('Arial', 8)).grid(row=1, column=col, padx=(0, 8), pady=(3, 0))
        col += 1
        self.trigger_apply_btn = ttk.Button(timebase_frame, text="Apply Trigger", 
                                          command=self.configure_trigger, 
                                          style='Primary.TButton', state='disabled')  # Initially disabled until connected
        self.trigger_apply_btn.grid(row=1, column=col, sticky='ew', padx=2, pady=(3, 0))
        col += 1
        col = 0
        

    def create_function_generator_frame(self, parent, row):
        """Create WGEN1 and WGEN2 function generator configuration controls."""
        fgen_frame = ttk.LabelFrame(parent, text="Function Generators (WGEN1 & WGEN2)", padding="3")
        fgen_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        for i in range(20):
            fgen_frame.columnconfigure(i, weight=0)
        fgen_frame.columnconfigure(19, weight=1)
        col = 0
        ttk.Label(fgen_frame, text="WGEN1:", font=('Arial', 8, 'bold'), foreground='#1a365d').grid(row=0, column=col, sticky='w', padx=(0, 5))
        col += 1
        self.wgen1_enable_var = tk.BooleanVar(value=False)  # WGEN1 enable/disable state
        ttk.Checkbutton(fgen_frame, text="Enable", variable=self.wgen1_enable_var, style='TCheckbutton').grid(row=0, column=col, padx=2)
        col += 1
        ttk.Label(fgen_frame, text="Wave:", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(5, 2))
        col += 1
        self.wgen1_waveform_var = tk.StringVar(value="SIN")  # WGEN1 waveform type
        ttk.Combobox(fgen_frame, textvariable=self.wgen1_waveform_var, values=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"], width=6, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        ttk.Label(fgen_frame, text="Freq(Hz):", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.wgen1_freq_var = tk.DoubleVar(value=1000.0)  # WGEN1 frequency (Hz)
        ttk.Entry(fgen_frame, textvariable=self.wgen1_freq_var, width=10, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        ttk.Label(fgen_frame, text="Amp(Vpp):", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.wgen1_amp_var = tk.DoubleVar(value=1.0)  # WGEN1 amplitude (peak-to-peak volts)
        ttk.Entry(fgen_frame, textvariable=self.wgen1_amp_var, width=8, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        ttk.Label(fgen_frame, text="Offset(V):", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.wgen1_offset_var = tk.DoubleVar(value=0.0)  # WGEN1 DC offset (volts)
        ttk.Entry(fgen_frame, textvariable=self.wgen1_offset_var, width=8, font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        self.wgen1_apply_btn = ttk.Button(fgen_frame, text="Apply WGEN1", command=lambda: self.configure_wgen(1), style='Primary.TButton', state='disabled')
        self.wgen1_apply_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        col = 0
        ttk.Label(fgen_frame, text="WGEN2:", font=('Arial', 8, 'bold'), foreground='#1a365d').grid(row=1, column=col, sticky='w', padx=(0, 5), pady=(3, 0))
        col += 1
        self.wgen2_enable_var = tk.BooleanVar(value=False)  # WGEN2 enable/disable state
        ttk.Checkbutton(fgen_frame, text="Enable", variable=self.wgen2_enable_var, style='TCheckbutton').grid(row=1, column=col, padx=2, pady=(3, 0))
        col += 1
        ttk.Label(fgen_frame, text="Wave:", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(5, 2), pady=(3, 0))
        col += 1
        self.wgen2_waveform_var = tk.StringVar(value="SIN")  # WGEN2 waveform type
        ttk.Combobox(fgen_frame, textvariable=self.wgen2_waveform_var, values=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"], width=6, state='readonly', font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(fgen_frame, text="Freq(Hz):", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.wgen2_freq_var = tk.DoubleVar(value=1000.0)  # WGEN2 frequency (Hz)
        ttk.Entry(fgen_frame, textvariable=self.wgen2_freq_var, width=10, font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(fgen_frame, text="Amp(Vpp):", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.wgen2_amp_var = tk.DoubleVar(value=1.0)  # WGEN2 amplitude (peak-to-peak volts)
        ttk.Entry(fgen_frame, textvariable=self.wgen2_amp_var, width=8, font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        ttk.Label(fgen_frame, text="Offset(V):", font=('Arial', 8)).grid(row=1, column=col, sticky='w', padx=(0, 2), pady=(3, 0))
        col += 1
        self.wgen2_offset_var = tk.DoubleVar(value=0.0)  # WGEN2 DC offset (volts)
        ttk.Entry(fgen_frame, textvariable=self.wgen2_offset_var, width=8, font=('Arial', 8)).grid(row=1, column=col, padx=(0, 5), pady=(3, 0))
        col += 1
        self.wgen2_apply_btn = ttk.Button(fgen_frame, text="Apply WGEN2", command=lambda: self.configure_wgen(2), style='Primary.TButton', state='disabled')
        self.wgen2_apply_btn.grid(row=1, column=col, sticky='ew', padx=2, pady=(3, 0))
        col += 1

    def create_file_preferences_frame(self, parent, row):
        """Create save directory and plot title configuration controls."""
        pref_frame = ttk.LabelFrame(parent, text="File Preferences", padding="3")
        pref_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        pref_frame.columnconfigure(1, weight=1)  # Allow directory paths to expand
        pref_frame.columnconfigure(4, weight=1)
        pref_frame.columnconfigure(7, weight=1)
        ttk.Label(pref_frame, text="Data:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky='w')
        self.data_path_var = tk.StringVar(value=str(Path.cwd() / "data"))  # CSV data save directory
        ttk.Entry(pref_frame, textvariable=self.data_path_var, font=('Arial', 7)).grid(row=0, column=1, sticky='ew', padx=(2, 2))
        ttk.Button(pref_frame, text="...", command=lambda: self.browse_folder('data'), width=3).grid(row=0, column=2, padx=(0, 8))
        ttk.Label(pref_frame, text="Graphs:", font=('Arial', 8, 'bold')).grid(row=0, column=3, sticky='w')
        self.graph_path_var = tk.StringVar(value=str(Path.cwd() / "graphs"))  # Plot image save directory
        ttk.Entry(pref_frame, textvariable=self.graph_path_var, font=('Arial', 7)).grid(row=0, column=4, sticky='ew', padx=(2, 2))
        ttk.Button(pref_frame, text="...", command=lambda: self.browse_folder('graphs'), width=3).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(pref_frame, text="Screenshots:", font=('Arial', 8, 'bold')).grid(row=0, column=6, sticky='w')
        self.screenshot_path_var = tk.StringVar(value=str(Path.cwd() / "screenshots"))  # Screenshot save directory
        ttk.Entry(pref_frame, textvariable=self.screenshot_path_var, font=('Arial', 7)).grid(row=0, column=7, sticky='ew', padx=(2, 2))
        ttk.Button(pref_frame, text="...", command=lambda: self.browse_folder('screenshots'), width=3).grid(row=0, column=8)
        ttk.Label(pref_frame, text="Plot Title:", font=('Arial', 8, 'bold')).grid(row=1, column=0, sticky='w', pady=(2, 0))
        self.graph_title_var = tk.StringVar(value="")  # Custom plot title prefix (supports multi-channel with "- Channel N" suffix)
        title_entry = ttk.Entry(pref_frame, textvariable=self.graph_title_var, font=('Arial', 8))
        title_entry.grid(row=1, column=1, columnspan=6, sticky='ew', padx=(2, 2), pady=(2, 0))
        hint_label = ttk.Label(pref_frame, text="Multi-channel: adds '- Channel N'",
                              font=('Arial', 7, 'italic'), foreground='#666')
        hint_label.grid(row=1, column=7, columnspan=2, sticky='w', pady=(2, 0))

    def create_measurement_frame(self, parent, row):
        """Create measurement controls for waveform analysis."""
        meas_frame = ttk.LabelFrame(parent, text="Measurements & Autoscale", padding="3")
        meas_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        for i in range(10):
            meas_frame.columnconfigure(i, weight=0)
        meas_frame.columnconfigure(9, weight=1)
        col = 0
        ttk.Label(meas_frame, text="MEASUREMENT:", font=('Arial', 8, 'bold'), foreground='#1a365d').grid(row=0, column=col, sticky='w', padx=(0, 5))
        col += 1
        ttk.Label(meas_frame, text="Channel:", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.measurement_channel_var = tk.StringVar(value="CH1")
        ttk.Combobox(meas_frame, textvariable=self.measurement_channel_var, values=["CH1", "CH2", "CH3", "CH4"], width=5, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        ttk.Label(meas_frame, text="Type:", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        self.measurement_type_var = tk.StringVar(value="FREQ")
        measurement_types = ["FREQ", "PERiod", "VPP", "VAMP", "OVERshoot", "VTOP", "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL", "DUTYcycle", "NDUTy"]
        ttk.Combobox(meas_frame, textvariable=self.measurement_type_var, values=measurement_types, width=8, state='readonly', font=('Arial', 8)).grid(row=0, column=col, padx=(0, 5))
        col += 1
        self.measure_btn = ttk.Button(meas_frame, text="Measure", width=8, command=self.perform_measurement, style='Primary.TButton', state='disabled')
        self.measure_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        ttk.Label(meas_frame, text="Result:", font=('Arial', 8, 'bold')).grid(row=0, column=col, sticky='w', padx=(8, 2))
        col += 1
        self.measurement_result_var = tk.StringVar(value="N/A")
        ttk.Entry(meas_frame, textvariable=self.measurement_result_var, width=18, font=('Arial', 9, 'bold'), state='readonly', foreground='#2563eb').grid(row=0, column=col, sticky='ew', padx=(0, 8))
        col += 1
        ttk.Separator(meas_frame, orient='vertical').grid(row=0, column=col, sticky='ns', padx=5)
        col += 1
        self.autoscale_btn = ttk.Button(meas_frame, text="⚡ Autoscale", width=12, command=self.perform_autoscale, style='Success.TButton', state='disabled')
        self.autoscale_btn.grid(row=0, column=col, sticky='ew', padx=2)

    def create_operations_frame(self, parent, row):
        """Create data acquisition and export operation buttons."""
        ops_frame = ttk.LabelFrame(parent, text="Operations", padding="3")
        ops_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))
        for i in range(6):
            ops_frame.columnconfigure(i, weight=1)  # Equal column widths
        self.screenshot_btn = ttk.Button(ops_frame, text="Screenshot", command=self.capture_screenshot, style='Info.TButton')
        self.screenshot_btn.grid(row=0, column=0, sticky='ew', padx=1)
        self.acquire_data_btn = ttk.Button(ops_frame, text="Acquire Data", command=self.acquire_data, style='Primary.TButton')
        self.acquire_data_btn.grid(row=0, column=1, sticky='ew', padx=1)
        self.export_csv_btn = ttk.Button(ops_frame, text="Export CSV", command=self.export_csv, style='Success.TButton')
        self.export_csv_btn.grid(row=0, column=2, sticky='ew', padx=1)
        self.generate_plot_btn = ttk.Button(ops_frame, text="Generate Plot", command=self.generate_plot, style='Success.TButton')
        self.generate_plot_btn.grid(row=0, column=3, sticky='ew', padx=1)
        self.full_automation_btn = ttk.Button(ops_frame, text="Full Auto", command=self.run_full_automation, style='Primary.TButton')
        self.full_automation_btn.grid(row=0, column=4, sticky='ew', padx=1)
        self.open_folder_btn = ttk.Button(ops_frame, text="Open Folder", command=self.open_output_folder, style='Info.TButton', width=12)
        self.open_folder_btn.grid(row=0, column=5, sticky='ew', padx=1)

    def create_status_frame(self, parent, row):
        """Create expandable status log with color-coded message display."""
        status_frame = ttk.LabelFrame(parent, text="Status & Activity Log", padding="3")
        status_frame.grid(row=row, column=0, sticky='nsew')  # Expand to fill available space
        status_frame.columnconfigure(0, weight=1)  # Allow column expansion
        status_frame.rowconfigure(0, weight=0)  # Status line: fixed
        status_frame.rowconfigure(1, weight=0)  # Controls line: fixed
        status_frame.rowconfigure(2, weight=1)  # Log text: EXPANDABLE
        self.current_operation_var = tk.StringVar(value="Ready - Connect to oscilloscope")  # Current operation status
        status_label = ttk.Label(status_frame, textvariable=self.current_operation_var, 
                                font=('Arial', 9, 'bold'), foreground='#1a365d')
        status_label.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        controls_frame = ttk.Frame(status_frame)  # Container for log controls
        controls_frame.grid(row=1, column=0, sticky='ew', pady=(0, 2))
        controls_frame.columnconfigure(1, weight=1)  # Spacer column expands
        ttk.Label(controls_frame, text="Activity Log:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Frame(controls_frame).grid(row=0, column=1, sticky='ew')  # Spacer
        ttk.Button(controls_frame, text="Clear", command=self.clear_log, width=6).grid(row=0, column=2, padx=(0, 2))
        ttk.Button(controls_frame, text="Save", command=self.save_log, width=6).grid(row=0, column=3)
        self.log_text = scrolledtext.ScrolledText(status_frame,
                                                 font=('Consolas', 8),
                                                 bg='#1a1a1a',  # Dark background (terminal style)
                                                 fg='#00ff00',  # Green text (terminal style)
                                                 insertbackground='white',
                                                 wrap=tk.WORD)
        self.log_text.grid(row=2, column=0, sticky='nsew')  # Expand to fill remaining space

    def create_live_feed_frame(self, parent, row):
        """Create live signal feed display with embedded matplotlib canvas."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        live_frame = ttk.LabelFrame(parent, text="🔴 Live Signal Feed", padding="3")
        live_frame.grid(row=row, column=0, sticky='ew', pady=(0, 2))  # Changed to 'ew' to prevent vertical expansion
        live_frame.columnconfigure(0, weight=1)
        live_frame.rowconfigure(1, weight=0)  # Canvas row fixed height
        
        # Control panel
        controls_frame = ttk.Frame(live_frame)
        controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        
        col = 0
        ttk.Label(controls_frame, text="Channel:", font=('Arial', 8, 'bold')).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        
        self.live_feed_channel_var = tk.IntVar(value=1)
        for ch in [1, 2, 3, 4]:
            rb = ttk.Radiobutton(controls_frame, text=f"Ch{ch}", variable=self.live_feed_channel_var, value=ch)
            rb.grid(row=0, column=col, padx=2)
            col += 1
        
        ttk.Separator(controls_frame, orient='vertical').grid(row=0, column=col, sticky='ns', padx=5)
        col += 1
        
        ttk.Label(controls_frame, text="Update:", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 2))
        col += 1
        
        self.live_feed_rate_var = tk.IntVar(value=500)
        rate_combo = ttk.Combobox(controls_frame, textvariable=self.live_feed_rate_var,
                                  values=["100", "200", "500", "1000", "2000"], width=6, state='readonly', font=('Arial', 8))
        rate_combo.grid(row=0, column=col, padx=(0, 2))
        col += 1
        
        ttk.Label(controls_frame, text="ms", font=('Arial', 8)).grid(row=0, column=col, sticky='w', padx=(0, 8))
        col += 1
        
        self.live_feed_autoscale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto Y-Scale", variable=self.live_feed_autoscale_var).grid(row=0, column=col, padx=5)
        col += 1
        
        self.live_feed_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Grid", variable=self.live_feed_grid_var).grid(row=0, column=col, padx=5)
        col += 1
        
        ttk.Separator(controls_frame, orient='vertical').grid(row=0, column=col, sticky='ns', padx=5)
        col += 1
        
        self.live_feed_start_btn = ttk.Button(controls_frame, text="▶ Start", width=8,
                                              command=self.start_live_feed, style='Success.TButton', state='disabled')
        self.live_feed_start_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        
        self.live_feed_pause_btn = ttk.Button(controls_frame, text="⏸ Pause", width=8,
                                              command=self.pause_live_feed, style='Warning.TButton', state='disabled')
        self.live_feed_pause_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        
        self.live_feed_stop_btn = ttk.Button(controls_frame, text="⏹ Stop", width=8,
                                             command=self.stop_live_feed, style='Warning.TButton', state='disabled')
        self.live_feed_stop_btn.grid(row=0, column=col, sticky='ew', padx=2)
        col += 1
        
        # Status indicator
        self.live_feed_status_var = tk.StringVar(value="● Stopped")
        self.live_feed_status_label = ttk.Label(controls_frame, textvariable=self.live_feed_status_var,
                                                 font=('Arial', 8, 'bold'), foreground='#dc2626')
        self.live_feed_status_label.grid(row=0, column=col, sticky='e', padx=(10, 0))
        controls_frame.columnconfigure(col, weight=1)
        
        # Matplotlib canvas
        canvas_frame = ttk.Frame(live_frame, relief='sunken', borderwidth=1, height=280)
        canvas_frame.grid(row=1, column=0, sticky='ew')
        canvas_frame.grid_propagate(False)  # Prevent frame from resizing to content
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.live_fig = Figure(figsize=(10, 2.5), dpi=100, facecolor='#1a1a1a')
        self.live_ax = self.live_fig.add_subplot(111, facecolor='#0a0a0a')
        self.live_ax.set_xlabel('Time (s)', color='#00ff00', fontsize=9)
        self.live_ax.set_ylabel('Voltage (V)', color='#00ff00', fontsize=9)
        self.live_ax.tick_params(colors='#00ff00', labelsize=8)
        self.live_ax.grid(True, alpha=0.2, color='#00ff00')
        self.live_fig.tight_layout()
        
        # Initialize empty plot line
        self.live_line, = self.live_ax.plot([], [], 'g-', linewidth=1.5)
        
        # Embed in tkinter
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=canvas_frame)
        self.live_canvas.draw()
        self.live_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def start_live_feed(self):
        """Start live waveform acquisition and display."""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            messagebox.showerror("Error", "Connect to oscilloscope first")
            return
        
        self.live_feed_active = True
        self.live_feed_paused = False
        self.live_feed_status_var.set("● Running")
        self.live_feed_status_label.config(foreground='#16a34a')  # Green
        
        self.live_feed_start_btn.config(state='disabled')
        self.live_feed_pause_btn.config(state='normal')
        self.live_feed_stop_btn.config(state='normal')
        
        self.log_message("Live feed started", 'SUCCESS')
        self.update_live_feed()  # Start the update loop

    def pause_live_feed(self):
        """Pause live feed updates."""
        self.live_feed_paused = True
        self.live_feed_status_var.set("● Paused")
        self.live_feed_status_label.config(foreground='#f59e0b')  # Orange
        
        self.live_feed_start_btn.config(state='normal', text="▶ Resume")
        self.live_feed_pause_btn.config(state='disabled')
        
        self.log_message("Live feed paused", 'INFO')

    def stop_live_feed(self):
        """Stop live feed and clear display."""
        self.live_feed_active = False
        self.live_feed_paused = False
        self.live_feed_worker_running = False
        self.live_feed_status_var.set("● Stopped")
        self.live_feed_status_label.config(foreground='#dc2626')  # Red
        
        self.live_feed_start_btn.config(state='normal', text="▶ Start")
        self.live_feed_pause_btn.config(state='disabled')
        self.live_feed_stop_btn.config(state='disabled')
        
        # Clear plot
        self.live_line.set_data([], [])
        self.live_canvas.draw()
        
        self.log_message("Live feed stopped", 'INFO')

    def update_live_feed(self):
        """Periodic update method for live waveform display."""
        if not self.live_feed_active or self.live_feed_paused:
            if self.live_feed_active and self.live_feed_paused:
                # Still active but paused, schedule next check
                update_rate = self.live_feed_rate_var.get()
                self.root.after(update_rate, self.update_live_feed)
            return
        
        def acquire_and_update():
            try:
                daq = self.data_acquisition
                if not daq:
                    self.status_queue.put(("error", "Not connected"))
                    return
                channel = self.live_feed_channel_var.get()
                # Reduce points for responsiveness and lower I/O load
                max_points = 2500 if self.live_feed_rate_var.get() <= 500 else 5000
                waveform_data = daq.acquire_waveform_data(channel, max_points=max_points)
                
                if waveform_data:
                    # Update plot on main thread
                    self.root.after(0, lambda: self.update_live_plot(waveform_data))
            except Exception as e:
                self.log_message(f"Live feed error: {e}", 'ERROR')
            finally:
                self.live_feed_worker_running = False
        
        # Run acquisition in background thread
        if self.live_feed_worker_running:
            # Skip starting another acquisition; will reschedule below
            pass
        else:
            self.live_feed_worker_running = True
            threading.Thread(target=acquire_and_update, daemon=True).start()
        
        # Schedule next update
        update_rate = self.live_feed_rate_var.get()
        if self.live_feed_active:
            self.root.after(update_rate, self.update_live_feed)

    def update_live_plot(self, waveform_data):
        """Update the matplotlib plot with new waveform data."""
        try:
            time_data = waveform_data['time']
            voltage_data = waveform_data['voltage']
            
            # Update line data
            self.live_line.set_data(time_data, voltage_data)
            
            # Update axis limits
            if time_data:
                self.live_ax.set_xlim(min(time_data), max(time_data))
            
            if self.live_feed_autoscale_var.get() and voltage_data:
                v_min, v_max = min(voltage_data), max(voltage_data)
                v_range = v_max - v_min if (v_max - v_min) > 0 else 1.0
                self.live_ax.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)
            
            # Update grid visibility
            self.live_ax.grid(self.live_feed_grid_var.get(), alpha=0.2, color='#00ff00')
            
            # Redraw canvas
            self.live_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to update live plot: {e}")

    def configure_timebase(self):
        """Apply timebase configuration to oscilloscope via background thread."""
        def timebase_config_thread():  # Background thread worker function
            try:
                time_scale_str = self.time_scale_var.get()  # Retrieve time/div setting from GUI
                time_scale = parse_timebase_string(time_scale_str)  # Converts label to seconds per division
                # time_offset = self.time_offset_var.get()  # Retrieve horizontal offset from GUI
                self.update_status("Configuring timebase...")  # Update status display
                self.log_message(f"Configuring timebase: {time_scale_str} ({time_scale}s/div)")
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(("error", "Not connected"))
                    return
                with self.io_lock:
                    success = osc.configure_timebase(time_scale)  # Call oscilloscope method
                if success:
                    self.status_queue.put(("timebase_configured", f"Timebase configured: {time_scale}s/div"))
                else:
                    self.status_queue.put(("error", "Timebase configuration failed"))
            except Exception as e:
                self.status_queue.put(("error", f"Timebase config error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=timebase_config_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def configure_trigger(self):
        """Apply trigger configuration to oscilloscope via background thread."""
        def trigger_config_thread():  # Background thread worker function
            try:
                trigger_source = self.trigger_source_var.get()  # Retrieve trigger source from GUI
                trigger_level = self.trigger_level_var.get()  # Retrieve trigger level from GUI
                trigger_slope_gui = self.trigger_slope_var.get()  # Gets "Raising" or "Falling"
                trigger_slope = TRIGGER_SLOPE_MAP.get(trigger_slope_gui, "POS")  # Retrieve trigger slope from GUI
                channel = int(trigger_source.replace("CH", ""))  # Extract channel number (e.g., "CH1" → 1)
                self.update_status("Configuring trigger...")  # Update status display
                self.log_message(f"Configuring trigger: {trigger_source}, {trigger_level}V, {trigger_slope} edge")
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(("error", "Not connected"))
                    return
                with self.io_lock:
                    success = osc.configure_trigger(channel, trigger_level, trigger_slope)
                if success:
                    self.status_queue.put(("trigger_configured", f"Trigger configured: {trigger_source} @ {trigger_level}V"))
                else:
                    self.status_queue.put(("error", "Trigger configuration failed"))
            except Exception as e:
                self.status_queue.put(("error", f"Trigger config error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=trigger_config_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def configure_all_triggers(self):
        """Apply individual trigger settings to all channels via background thread."""
        def all_triggers_config_thread():  # Background thread worker function
            try:
                self.update_status("Configuring all channel triggers...")  # Update status display
                self.log_message("Configuring individual trigger levels for all channels...")
                success_count = 0  # Count successful configurations
                total_channels = len(self.channel_trigger_vars)  # Total channels to configure
                for channel, var in self.channel_trigger_vars.items():  # Iterate each channel
                    trigger_level = var.get()  # Retrieve trigger level from GUI
                    self.log_message(f"Setting Ch{channel} trigger level: {trigger_level}V")
                    osc = self.oscilloscope
                    if not (osc and osc.is_connected):
                        self.status_queue.put(("error", "Not connected"))
                        return
                    with self.io_lock:
                        success = osc.configure_trigger(channel, trigger_level, "POS")  # Positive edge for batch
                    if success:
                        success_count += 1  # Increment success counter
                        self.log_message(f"Ch{channel} trigger level set successfully", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel} trigger level failed", "ERROR")
                if success_count == total_channels:  # All succeeded
                    self.status_queue.put(("all_triggers_configured", f"All {success_count} channel triggers configured"))
                elif success_count > 0:  # Partial success
                    self.status_queue.put(("all_triggers_configured", f"{success_count}/{total_channels} channel triggers configured"))
                else:  # Complete failure
                    self.status_queue.put(("error", "All channel trigger configurations failed"))
            except Exception as e:
                self.status_queue.put(("error", f"All triggers config error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=all_triggers_config_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def browse_folder(self, folder_type):
        """Open file browser dialog for custom save directory selection."""
        try:
            path_var_mapping = {
                'data': 'data_path_var',
                'graphs': 'graph_path_var',
                'screenshots': 'screenshot_path_var'
            }  # Map folder type to GUI StringVar attribute name
            var_name = path_var_mapping.get(folder_type)  # Get StringVar name
            if not var_name:
                raise ValueError(f"Unknown folder type: {folder_type}")
            current_path = getattr(self, var_name).get()  # Retrieve current path value
            if not current_path or not os.path.exists(current_path):
                current_path = str(Path.cwd())  # Fallback to current working directory if invalid
            self.log_message(f"Opening folder dialog for {folder_type}...")
            folder_path = filedialog.askdirectory(
                initialdir=current_path,  # Start dialog in current directory
                title=f"Select {folder_type.title()} Save Location",
                mustexist=False  # Allow new directory creation
            )
            if folder_path and folder_path.strip():  # User selected a directory
                getattr(self, var_name).set(folder_path)  # Update GUI StringVar
                self.save_locations[folder_type] = folder_path  # Update save locations cache
                self.log_message(f"Updated {folder_type}: {folder_path}", "SUCCESS")
                Path(folder_path).mkdir(parents=True, exist_ok=True)  # Create directory if needed
                self.log_message(f"Directory verified: {folder_path}")
            else:
                self.log_message(f"Folder selection cancelled for {folder_type}")  # User clicked Cancel
        except Exception as e:
            self.log_message(f"Error selecting {folder_type} folder: {e}", "ERROR")
            messagebox.showerror("Folder Selection Error", f"Error: {str(e)}")

    def log_message(self, message: str, level: str = "INFO"):
        """Add timestamped message to activity log with color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")  # Format current time as HH:MM:SS
        log_entry = f"[{timestamp}] {message}\n"  # Prepend timestamp to message
        try:
            self.log_text.insert(tk.END, log_entry)  # Append message to end of text widget
            self.log_text.see(tk.END)  # Auto-scroll to show latest message
            if level == "ERROR":  # Color code ERROR messages
                self.log_text.tag_add("error", f"end-{len(log_entry)}c", "end-1c")  # Tag error message range
                self.log_text.tag_config("error", foreground="#ff6b6b")  # Red text for errors
            elif level == "SUCCESS":  # Color code SUCCESS messages
                self.log_text.tag_add("success", f"end-{len(log_entry)}c", "end-1c")  # Tag success message range
                self.log_text.tag_config("success", foreground="#51cf66")  # Green text for success
            elif level == "WARNING":  # Color code WARNING messages
                self.log_text.tag_add("warning", f"end-{len(log_entry)}c", "end-1c")  # Tag warning message range
                self.log_text.tag_config("warning", foreground="#ffd43b")  # Yellow text for warnings
            else:  # Default to INFO level
                self.log_text.tag_add("info", f"end-{len(log_entry)}c", "end-1c")  # Tag info message range
                self.log_text.tag_config("info", foreground="#74c0fc")  # Light blue text for info
        except Exception as e:
            print(f"Log error: {e}")  # Fallback to console on GUI logging failure

    def clear_log(self):
        """Clear all messages from activity log."""
        try:
            self.log_text.delete(1.0, tk.END)  # Delete from first character to end
            self.log_message("Log cleared")  # Log the clearing action
        except Exception as e:
            print(f"Clear error: {e}")

    def save_log(self):
        """Export activity log contents to text file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format timestamp for filename
            log_content = self.log_text.get(1.0, tk.END)  # Retrieve all log text
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",  # Default to .txt extension
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"oscilloscope_log_{timestamp}.txt"  # Suggested filename with timestamp
            )
            if filename:  # User selected a file (didn't cancel)
                with open(filename, 'w') as f:  # Open file for writing
                    f.write(f"Enhanced Oscilloscope Automation Log - {datetime.now()}\n{'='*50}\n\n{log_content}")
                self.log_message(f"Log saved: {filename}", "SUCCESS")
        except Exception as e:
            self.log_message(f"Save error: {e}", "ERROR")

    def update_status(self, status: str):
        """Update current operation status display."""
        try:
            self.current_operation_var.set(status)  # Set status text
            self.root.update_idletasks()  # Refresh GUI display immediately
        except:
            pass  # Silently ignore display update errors

    def get_selected_channels(self):
        """Retrieve list of channels currently enabled in GUI."""
        selected = []  # Initialize empty channel list
        for ch, var in self.channel_enable_vars.items():  # Iterate channel checkboxes
            if var.get():  # Check if this channel is enabled
                selected.append(ch)  # Add channel number to list
        return sorted(selected)  # Return in ascending channel order

    def toggle_channel_display(self, channel):
        """Enable/disable channel display on oscilloscope when checkbox changes."""
        if not self.oscilloscope or not self.oscilloscope.is_connected:  # Verify connection
            return  # Abort if not connected
        try:
            is_enabled = self.channel_enable_vars[channel].get()  # Get checkbox state
            state = 1 if is_enabled else 0  # Convert boolean to SCPI ON/OFF value
            command = f":CHANnel{channel}:DISPlay {state}"  # Compose SCPI command
            self.oscilloscope._scpi_wrapper.write(command)  # Send command to oscilloscope via wrapper
            action = "enabled" if is_enabled else "disabled"  # Prepare action message
            self.log_message(f"Ch{channel} display {action}")  # Log the action
        except Exception as e:
            self.log_message(f"Error toggling Ch{channel} display: {e}", "ERROR")

    def disable_operation_buttons(self):
        """Disable all operation buttons (called at startup before connection)."""
        buttons = [self.screenshot_btn, self.acquire_data_btn, self.export_csv_btn,
                  self.generate_plot_btn, self.full_automation_btn, self.open_folder_btn,
                  self.config_channel_btn, self.test_btn, self.wgen1_apply_btn, self.wgen2_apply_btn,
                  self.timebase_apply_btn, self.trigger_apply_btn,
                  self.measure_btn, self.autoscale_btn, self.live_feed_start_btn]  # NEW: Measurement & live feed buttons
        for btn in buttons:  # Iterate all operation buttons
            try:
                btn.configure(state='disabled')  # Disable button (grayed out)
            except:
                pass  # Ignore buttons that don't exist

    def enable_operation_buttons(self):
        """Enable all operation buttons (called after successful connection)."""
        buttons = [self.screenshot_btn, self.acquire_data_btn, self.export_csv_btn,
                  self.generate_plot_btn, self.full_automation_btn, self.open_folder_btn,
                  self.config_channel_btn, self.test_btn, self.wgen1_apply_btn, self.wgen2_apply_btn,
                  self.timebase_apply_btn, self.trigger_apply_btn,
                  self.measure_btn, self.autoscale_btn, self.live_feed_start_btn]  # NEW: Measurement & live feed buttons
        for btn in buttons:  # Iterate all operation buttons
            try:
                btn.configure(state='normal')  # Enable button (clickable)
            except:
                pass  # Ignore buttons that don't exist

    def connect_oscilloscope(self):
        """Establish VISA connection to oscilloscope in background thread."""
        def connect_thread():  # Background thread worker function
            try:
                self.update_status("Connecting...")  # Update status display
                self.log_message("Connecting to Keysight oscilloscope...")
                visa_address = self.visa_address_var.get().strip()  # Retrieve VISA address from entry field
                if not visa_address:  # Validate address not empty
                    raise ValueError("VISA address empty")
                self.oscilloscope = KeysightDSOX6004A(visa_address)  # Create oscilloscope instance
                if self.oscilloscope.connect():  # Attempt connection
                    self.data_acquisition = OscilloscopeDataAcquisition(self.oscilloscope, io_lock=self.io_lock)  # Initialize data handler with shared lock
                    info = self.oscilloscope.get_instrument_info()  # Query instrument identification
                    if info:
                        self.log_message(f"Connected: {info['manufacturer']} {info['model']}", "SUCCESS")
                        self.status_queue.put(("connected", info))  # Send info to GUI thread
                    else:
                        self.status_queue.put(("connected", None))  # Send connection status to GUI thread
                else:
                    raise Exception("Connection failed")
            except Exception as e:
                self.status_queue.put(("error", f"Connection failed: {str(e)}"))  # Send error to GUI thread
        threading.Thread(target=connect_thread, daemon=True).start()  # Start connection in background

    def disconnect_oscilloscope(self):
        """Disconnect from oscilloscope and clean up resources."""
        try:
            if hasattr(self, 'live_feed_active') and self.live_feed_active:
                self.stop_live_feed()
            if self.oscilloscope:  # Check if oscilloscope exists
                self.oscilloscope.disconnect()  # Close connection
                self.oscilloscope = None  # Clear reference
                self.data_acquisition = None  # Clear data handler
                self.last_acquired_data = None  # Clear cached data
            self.conn_status_var.set("Disconnected")  # Update status display
            self.conn_status_label.configure(foreground='#e53e3e')  # Red text (disconnected state)
            self.connect_btn.configure(state='normal')  # Enable Connect button
            self.disconnect_btn.configure(state='disabled')  # Disable Disconnect button
            self.disable_operation_buttons()  # Disable all operations
            self.info_text.configure(state='normal')  # Enable text widget for clearing
            self.info_text.delete(1.0, tk.END)  # Clear instrument info display
            self.info_text.configure(state='disabled')  # Disable text widget again
            self.update_status("Disconnected")  # Update main status
            self.log_message("Disconnected", "SUCCESS")
        except Exception as e:
            self.log_message(f"Disconnect error: {e}", "ERROR")

    def test_connection(self):
        """Verify oscilloscope connection is active."""
        try:
            if self.oscilloscope and self.oscilloscope.is_connected:  # Check if connected
                self.log_message("Connection test: PASSED", "SUCCESS")
                self.update_status("Test passed")
                messagebox.showinfo("Test", "Connection OK!")
            else:
                self.log_message("Connection test: FAILED", "ERROR")
                messagebox.showerror("Test", "Not connected")
        except Exception as e:
            self.log_message(f"Test error: {e}", "ERROR")

    def configure_wgen(self, generator):
        """Apply function generator configuration to oscilloscope via background thread."""
        def wgen_config_thread():  # Background thread worker function
            try:
                if generator == 1:  # WGEN1 configuration
                    enable = self.wgen1_enable_var.get()  # Get enable state
                    waveform = self.wgen1_waveform_var.get()  # Get waveform type
                    frequency = self.wgen1_freq_var.get()  # Get frequency
                    amplitude = self.wgen1_amp_var.get()  # Get amplitude
                    offset = self.wgen1_offset_var.get()  # Get DC offset
                else:  # WGEN2 configuration
                    enable = self.wgen2_enable_var.get()  # Get enable state
                    waveform = self.wgen2_waveform_var.get()  # Get waveform type
                    frequency = self.wgen2_freq_var.get()  # Get frequency
                    amplitude = self.wgen2_amp_var.get()  # Get amplitude
                    offset = self.wgen2_offset_var.get()  # Get DC offset
                self.update_status(f"Configuring WGEN{generator}...")  # Update status display
                self.log_message(f"Configuring WGEN{generator}: {waveform}, {frequency}Hz, {amplitude}Vpp, {offset}V offset, Enable: {enable}")
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(("error", "Not connected"))
                    return
                with self.io_lock:
                    success = osc.configure_function_generator(
                        generator=generator,  # Function generator number
                        waveform=waveform,  # Waveform type
                        frequency=frequency,  # Frequency in Hz
                        amplitude=amplitude,  # Amplitude in Vpp
                        offset=offset,  # DC offset in volts
                        enable=enable  # Enable/disable output
                    )
                if success:
                    self.status_queue.put(("wgen_configured", f"WGEN{generator} configured successfully"))
                else:
                    self.status_queue.put(("error", f"WGEN{generator} configuration failed"))
            except Exception as e:
                self.status_queue.put(("error", f"WGEN{generator} config error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=wgen_config_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def configure_channel(self):
        """Apply vertical scale, offset, coupling, and probe settings to selected channels."""
        def config_thread():  # Background thread worker function
            try:
                selected_channels = self.get_selected_channels()  # Get enabled channels from checkboxes
                if not selected_channels:  # Validate at least one channel selected
                    self.status_queue.put(("error", "No channels selected. Please check at least one channel."))
                    return
                v_scale = self.v_scale_var.get()  # Get vertical scale (V/div)
                v_offset = self.v_offset_var.get()  # Get vertical offset (volts)
                coupling = self.coupling_var.get()  # Get AC/DC coupling
                probe = self.probe_var.get()  # Get probe attenuation factor
                self.update_status(f"Configuring {len(selected_channels)} channel(s)...")  # Update status
                self.log_message(f"Configuring channels {selected_channels}: {v_scale}V/div, {v_offset}V, {coupling}, {probe}x")
                success_count = 0  # Count successful configurations
                for channel in selected_channels:  # Configure each selected channel
                    self.log_message(f"Configuring Ch{channel}...")
                    osc = self.oscilloscope
                    if not (osc and osc.is_connected):
                        self.status_queue.put(("error", "Not connected"))
                        return
                    with self.io_lock:
                        success = osc.configure_channel(
                            channel=channel,  # Channel number
                            vertical_scale=v_scale,  # Vertical scale in V/div
                            vertical_offset=v_offset,  # Vertical offset in volts
                            coupling=coupling,  # AC or DC coupling
                            probe_attenuation=probe  # Probe attenuation factor
                        )
                    if success:
                        success_count += 1  # Increment success counter
                        self.log_message(f"Ch{channel} configured successfully", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel} configuration failed", "ERROR")
                if success_count == len(selected_channels):  # All succeeded
                    self.status_queue.put(("channel_configured", f"All {success_count} channel(s) configured"))
                elif success_count > 0:  # Partial success
                    self.status_queue.put(("channel_configured", f"{success_count}/{len(selected_channels)} channel(s) configured"))
                else:  # Complete failure
                    self.status_queue.put(("error", "All channel configurations failed"))
            except Exception as e:
                self.status_queue.put(("error", f"Config error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=config_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def capture_screenshot(self):
        """Acquire display screenshot from oscilloscope via background thread."""
        def screenshot_thread():  # Background thread worker function
            try:
                self.update_status("Capturing screenshot...")  # Update status
                self.log_message("Capturing screenshot...")
                screenshot_dir = Path(self.screenshot_path_var.get())  # Get save directory
                screenshot_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(("error", "Not connected"))
                    return
                with self.io_lock:
                    filename = osc.capture_screenshot()  # Call oscilloscope method
                if filename:  # Check if capture succeeded
                    original_path = Path(filename)  # Convert to Path object
                    if original_path.parent != screenshot_dir:  # Check if need to move file
                        new_path = screenshot_dir / original_path.name  # Construct new path
                        import shutil
                        shutil.move(str(original_path), str(new_path))  # Move file to destination
                        filename = str(new_path)  # Update filename
                self.status_queue.put(("screenshot_success", filename))
            except Exception as e:
                self.status_queue.put(("error", f"Screenshot error: {str(e)}"))
        if self.oscilloscope and self.oscilloscope.is_connected:  # Verify connection
            threading.Thread(target=screenshot_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def acquire_data(self):
        """Retrieve waveform data from all selected channels via background thread."""
        def acquire_thread():  # Background thread worker function
            try:
                selected_channels = self.get_selected_channels()  # Get enabled channels
                if not selected_channels:  # Validate at least one channel selected
                    self.status_queue.put(("error", "No channels selected. Please check at least one channel."))
                    return
                self.update_status(f"Acquiring data from {len(selected_channels)} channel(s)...")  # Update status
                self.log_message(f"Acquiring data from channels: {selected_channels}")
                daq = self.data_acquisition
                if not daq:
                    self.status_queue.put(("error", "Not connected"))
                    return
                all_channel_data = {}  # Dictionary to store data from all channels
                for channel in selected_channels:  # Acquire from each selected channel
                    self.log_message(f"Acquiring Ch{channel}...")
                    data = daq.acquire_waveform_data(channel)  # Call data acquisition method
                    if data:  # Check if acquisition succeeded
                        all_channel_data[channel] = data  # Store channel data
                        self.log_message(f"Ch{channel}: {data['points_count']} points acquired", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel}: Acquisition failed", "ERROR")
                if all_channel_data:  # At least one channel succeeded
                    self.status_queue.put(("data_acquired", all_channel_data))
                else:
                    self.status_queue.put(("error", "Data acquisition failed for all channels"))
            except Exception as e:
                self.status_queue.put(("error", f"Acquire error: {str(e)}"))
        if self.data_acquisition:  # Verify data handler exists
            threading.Thread(target=acquire_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def export_csv(self):
        """Export most recent waveform data to CSV files via background thread."""
        if not hasattr(self, 'last_acquired_data') or not self.last_acquired_data:  # Validate cached data
            messagebox.showwarning("Warning", "No data. Acquire first.")
            return
        def export_thread():  # Background thread worker function
            try:
                self.update_status("Exporting CSV...")  # Update status
                self.log_message("Exporting CSV...")
                exported_files = []  # List to store exported file paths
                daq = self.data_acquisition
                if not daq:
                    self.status_queue.put(("error", "Not connected"))
                    return
                if isinstance(self.last_acquired_data, dict) and 'channel' not in self.last_acquired_data:
                    # Multi-channel data: dictionary keyed by channel number
                    for channel, data in self.last_acquired_data.items():
                        filename = daq.export_to_csv(
                            data, custom_path=self.data_path_var.get())  # Call data handler method
                        if filename:
                            exported_files.append(filename)  # Add to list
                            self.log_message(f"Ch{channel} CSV exported: {Path(filename).name}", "SUCCESS")
                else:
                    # Single channel data (backward compatibility)
                    filename = daq.export_to_csv(
                        self.last_acquired_data, custom_path=self.data_path_var.get())
                    if filename:
                        exported_files.append(filename)  # Add to list
                if exported_files:
                    self.status_queue.put(("csv_exported", exported_files))
                else:
                    self.status_queue.put(("error", "CSV export failed"))
            except Exception as e:
                self.status_queue.put(("error", f"CSV error: {str(e)}"))
        threading.Thread(target=export_thread, daemon=True).start()  # Start worker thread

    def generate_plot(self):
        """Generate waveform plots with statistics for acquired data via background thread."""
        if not hasattr(self, 'last_acquired_data') or not self.last_acquired_data:  # Validate cached data
            messagebox.showwarning("Warning", "No data. Acquire first.")
            return
        def plot_thread():  # Background thread worker function
            try:
                self.update_status("Generating plot...")  # Update status
                generated_plots = []  # List to store generated plot paths
                custom_title = self.graph_title_var.get().strip() or None  # Get custom plot title
                daq = self.data_acquisition
                if not daq:
                    self.status_queue.put(("error", "Not connected"))
                    return
                if isinstance(self.last_acquired_data, dict) and 'channel' not in self.last_acquired_data:
                    # Multi-channel data: dictionary keyed by channel number
                    self.log_message(f"Generating plots for {len(self.last_acquired_data)} channel(s)...")
                    for channel, data in self.last_acquired_data.items():
                        if custom_title:  # Customize title for each channel if base title provided
                            channel_title = f"{custom_title} - Channel {channel}"
                        else:
                            channel_title = None
                        filename = daq.generate_waveform_plot(
                            data, custom_path=self.graph_path_var.get(), plot_title=channel_title)  # Call data handler method
                        if filename:
                            generated_plots.append(filename)  # Add to list
                            self.log_message(f"Ch{channel} plot generated: {Path(filename).name}", "SUCCESS")
                else:
                    # Single channel data (backward compatibility)
                    filename = daq.generate_waveform_plot(
                        self.last_acquired_data, custom_path=self.graph_path_var.get(), plot_title=custom_title)
                    if filename:
                        generated_plots.append(filename)  # Add to list
                if generated_plots:
                    self.status_queue.put(("plot_generated", generated_plots))
                else:
                    self.status_queue.put(("error", "Plot failed"))
            except Exception as e:
                self.status_queue.put(("error", f"Plot error: {str(e)}"))
        threading.Thread(target=plot_thread, daemon=True).start()  # Start worker thread

    def run_full_automation(self):
        """Execute complete automated workflow: screenshot → acquire → export → plot."""
        def full_automation_thread():  # Background thread worker function
            try:
                selected_channels = self.get_selected_channels()  # Get enabled channels
                if not selected_channels:  # Validate at least one channel selected
                    self.status_queue.put(("error", "No channels selected. Please check at least one channel."))
                    return
                custom_title = self.graph_title_var.get().strip() or None  # Get custom plot title
                self.log_message(f"Starting full automation for channels: {selected_channels}...")
                self.update_status("Step 1/4: Screenshot...")  # Update status
                self.log_message("Step 1/4: Screenshot...")
                osc = self.oscilloscope
                daq = self.data_acquisition
                if not (osc and osc.is_connected and daq):
                    self.status_queue.put(("error", "Not connected"))
                    return
                with self.io_lock:
                    screenshot_file = osc.capture_screenshot()  # Capture display
                if screenshot_file:  # Move to user directory if needed
                    screenshot_dir = Path(self.screenshot_path_var.get())
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    original_path = Path(screenshot_file)
                    if original_path.parent != screenshot_dir:
                        new_path = screenshot_dir / original_path.name
                        import shutil
                        shutil.move(str(original_path), str(new_path))
                        screenshot_file = str(new_path)
                self.update_status(f"Step 2/4: Acquiring data from {len(selected_channels)} channel(s)...")  # Update status
                self.log_message(f"Step 2/4: Acquiring data from {len(selected_channels)} channel(s)...")
                all_channel_data = {}  # Dictionary for all channel data
                for channel in selected_channels:  # Acquire each channel
                    self.log_message(f"Acquiring Ch{channel}...")
                    data = daq.acquire_waveform_data(channel)
                    if data:
                        all_channel_data[channel] = data  # Store channel data
                        self.log_message(f"Ch{channel}: {data['points_count']} points acquired", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel}: Acquisition failed", "ERROR")
                if not all_channel_data:  # Validate data was acquired
                    raise Exception("Data acquisition failed for all channels")
                self.update_status("Step 3/4: Exporting CSV...")  # Update status
                self.log_message("Step 3/4: Exporting CSV...")
                csv_files = []  # List for exported CSV paths
                for channel, data in all_channel_data.items():  # Export each channel
                    csv_file = daq.export_to_csv(data, custom_path=self.data_path_var.get())
                    if csv_file:
                        csv_files.append(csv_file)  # Add to list
                        self.log_message(f"Ch{channel} CSV exported", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel} CSV export failed", "ERROR")
                self.update_status("Step 4/4: Generating plots...")  # Update status
                self.log_message("Step 4/4: Generating plots...")
                plot_files = []  # List for generated plot paths
                for channel, data in all_channel_data.items():  # Generate plot for each channel
                    if custom_title:
                        channel_title = f"{custom_title} - Channel {channel}"
                    else:
                        channel_title = None
                    plot_file = daq.generate_waveform_plot(
                        data, custom_path=self.graph_path_var.get(), plot_title=channel_title)
                    if plot_file:
                        plot_files.append(plot_file)  # Add to list
                        self.log_message(f"Ch{channel} plot generated", "SUCCESS")
                    else:
                        self.log_message(f"Ch{channel} plot generation failed", "ERROR")
                results = {  # Package all results for GUI thread
                    'screenshot': screenshot_file,
                    'csv': csv_files,
                    'plot': plot_files,
                    'data': all_channel_data,
                    'channels': selected_channels
                }
                self.status_queue.put(("full_automation_complete", results))
            except Exception as e:
                self.status_queue.put(("error", f"Automation error: {str(e)}"))
        if self.data_acquisition:  # Verify data handler exists
            threading.Thread(target=full_automation_thread, daemon=True).start()  # Start worker thread
        else:
            messagebox.showerror("Error", "Not connected")

    def open_output_folder(self):
        """Open all output folders (data, graphs, screenshots) in file explorer."""
        try:
            import subprocess
            import platform
            folders = [("Data", self.data_path_var.get()), 
                      ("Graphs", self.graph_path_var.get()), 
                      ("Screenshots", self.screenshot_path_var.get())]  # List of folders to open
            for name, path in folders:  # Open each folder
                try:
                    path_obj = Path(path)  # Convert string to Path
                    if not path_obj.is_dir():
                        path_obj.mkdir(parents=True)
                    if platform.system() == "Windows":  # Windows: use explorer
                        subprocess.run(['explorer', str(path_obj)], check=False)
                    elif platform.system() == "Darwin":  # macOS: use open command
                        subprocess.run(['open', str(path_obj)], check=True)
                    else:  # Linux: use xdg-open
                        subprocess.run(['xdg-open', str(path_obj)], check=True)
                    self.log_message(f"Opened {name}")
                except Exception as e:
                    self.log_message(f"Failed to open {name}: {e}", "ERROR")
        except Exception as e:
            self.log_message(f"Folder error: {e}", "ERROR")

    def display_instrument_info(self, info):
        """Display retrieved instrument information in the info text widget."""
        try:
            self.info_text.configure(state='normal')  # Enable text widget for editing
            self.info_text.delete(1.0, tk.END)  # Clear existing content
            if info:  # Format instrument info string
                info_str = f"Connected: {info.get('manufacturer', 'N/A')} {info.get('model', 'N/A')} | S/N: {info.get('serial_number', 'N/A')} | FW: {info.get('firmware_version', 'N/A')}"
            else:
                info_str = "Connected but no instrument info available"
            self.info_text.insert(1.0, info_str)  # Insert text at beginning
            self.info_text.configure(state='disabled')  # Disable widget again (read-only)
        except Exception as e:
            print(f"Info display error: {e}")

    def check_status_updates(self):
        """Periodically poll queue for status updates from worker threads."""
        try:
            while True:  # Process all queued messages
                status_type, data = self.status_queue.get_nowait()  # Get message (non-blocking)
                if status_type == "connected":  # Connection established
                    self.conn_status_var.set("Connected")
                    self.conn_status_label.configure(foreground='#2d7d32')  # Green text
                    self.connect_btn.configure(state='disabled')  # Disable Connect button
                    self.disconnect_btn.configure(state='normal')  # Enable Disconnect button
                    self.enable_operation_buttons()  # Enable all operations
                    self.update_status("Connected - Ready")
                    if data:
                        self.display_instrument_info(data)  # Show instrument info
                elif status_type == "error":  # Error occurred
                    self.log_message(data, "ERROR")
                    self.update_status("Error occurred")
                elif status_type == "screenshot_success":  # Screenshot captured
                    self.log_message(f"Screenshot saved: {Path(data).name}", "SUCCESS")
                    self.update_status("Screenshot saved")
                elif status_type == "data_acquired":  # Waveform data acquired
                    self.last_acquired_data = data  # Cache data
                    if isinstance(data, dict) and 'channel' not in data:
                        # Multi-channel data
                        total_points = sum(ch_data['points_count'] for ch_data in data.values())
                        self.log_message(f"Data acquired: {len(data)} channels, {total_points} total points", "SUCCESS")
                    else:
                        # Single channel data
                        self.log_message(f"Data acquired: {data['points_count']} points Ch{data['channel']}", "SUCCESS")
                    self.update_status("Data acquired")
                elif status_type == "csv_exported":  # CSV export completed
                    if isinstance(data, list):
                        for filepath in data:
                            self.log_message(f"CSV exported: {Path(filepath).name}", "SUCCESS")
                        self.log_message(f"Total: {len(data)} CSV file(s) exported", "SUCCESS")
                    else:
                        self.log_message(f"CSV exported: {Path(data).name}", "SUCCESS")
                    self.update_status("CSV exported")
                elif status_type == "plot_generated":  # Plot generation completed
                    if isinstance(data, list):
                        for filepath in data:
                            self.log_message(f"Plot generated: {Path(filepath).name}", "SUCCESS")
                        self.log_message(f"Total: {len(data)} plot(s) generated", "SUCCESS")
                    else:
                        self.log_message(f"Plot generated: {Path(data).name}", "SUCCESS")
                    self.update_status("Plot generated")
                elif status_type == "channel_configured":  # Channel configuration completed
                    self.log_message(data, "SUCCESS")
                    self.update_status("Channel configured")
                elif status_type == "wgen_configured":  # Function generator configured
                    self.log_message(data, "SUCCESS")
                    self.update_status("Function generator configured")
                elif status_type == "timebase_configured":  # Timebase configured
                    self.log_message(data, "SUCCESS")
                    self.update_status("Timebase configured")
                elif status_type == "trigger_configured":  # Trigger configured
                    self.log_message(data, "SUCCESS")
                    self.update_status("Trigger configured")
                elif status_type == "all_triggers_configured":  # All triggers configured
                    self.log_message(data, "SUCCESS")
                    self.update_status("All triggers configured")
                elif status_type == "full_automation_complete":  # Full automation workflow completed
                    self.last_acquired_data = data['data']  # Cache data
                    channels = data.get('channels', [])
                    self.log_message(f"Full automation completed for {len(channels)} channel(s)!", "SUCCESS")
                    self.log_message(f"Screenshot: {Path(data['screenshot']).name}", "SUCCESS")
                    if isinstance(data['csv'], list):
                        for csv_file in data['csv']:
                            self.log_message(f"CSV: {Path(csv_file).name}", "SUCCESS")
                    else:
                        self.log_message(f"CSV: {Path(data['csv']).name}", "SUCCESS")
                    if isinstance(data['plot'], list):
                        for plot_file in data['plot']:
                            self.log_message(f"Plot: {Path(plot_file).name}", "SUCCESS")
                    else:
                        self.log_message(f"Plot: {Path(data['plot']).name}", "SUCCESS")
                    self.update_status("Automation complete")
                    csv_count = len(data['csv']) if isinstance(data['csv'], list) else 1
                    plot_count = len(data['plot']) if isinstance(data['plot'], list) else 1
                    messagebox.showinfo("Complete!",
                                       f"Full automation complete!\n\n"
                                       f"Channels: {', '.join(map(str, channels))}\n"
                                       f"Screenshot: 1 file\n"
                                       f"CSV files: {csv_count}\n"
                                       f"Plots: {plot_count}")
                
                elif status_type == "measurement_complete":
                    channel = data['channel']
                    meas_type = data['type']
                    formatted = data['formatted']
                    self.measurement_result_var.set(formatted)
                    self.log_message(f"✓ {channel} {meas_type} = {formatted}", "SUCCESS")
                    self.update_status("Measurement complete")

                elif status_type == "autoscale_complete":
                    self.log_message(data, "SUCCESS")
                    self.update_status("Autoscale complete")

        except queue.Empty:  # No more messages in queue
            pass
        except Exception as e:
            print(f"Status update error: {e}")
        finally:
            self.root.after(100, self.check_status_updates)  # Check again after 100ms

    def perform_measurement(self):
        """Execute measurement on selected channel."""
        def measure_task():
            try:
                channel_str = self.measurement_channel_var.get()
                channel = int(channel_str.replace("CH", ""))
                measurement_type = self.measurement_type_var.get()
                self.log_message(f"Measuring {measurement_type} on {channel_str}...", "INFO")
                self.update_status(f"Measuring {measurement_type}...")
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(('error', 'Not connected'))
                    return
                with self.io_lock:
                    result = osc.measure_single(channel, measurement_type)
                if result is not None:
                    formatted_result = self._format_measurement_result(measurement_type, result)
                    self.status_queue.put(('measurement_complete', {'channel': channel_str, 'type': measurement_type, 'value': result, 'formatted': formatted_result}))
                else:
                    self.status_queue.put(('error', f"Measurement failed: No data from {channel_str}"))
            except Exception as e:
                self.log_message(f"Measurement error: {e}", "ERROR")
                self.status_queue.put(('error', f"Measurement error: {e}"))
        threading.Thread(target=measure_task, daemon=True).start()

    def _format_measurement_result(self, meas_type: str, value: float) -> str:
        warning = ""
        if abs(value) > 1e12:
            warning = " [Warning: Out of expected range]"
        base = format_measurement_value(meas_type, value)
        if warning:
            return f"{base}{warning}"
        return base

    def perform_autoscale(self):
        """Execute autoscale."""
        def autoscale_task():
            try:
                self.log_message("Executing autoscale...", "INFO")
                self.update_status("Autoscaling...")
                self.root.after(0, lambda: self.autoscale_btn.config(state='disabled'))
                osc = self.oscilloscope
                if not (osc and osc.is_connected):
                    self.status_queue.put(('error', 'Not connected'))
                    self.root.after(0, lambda: self.autoscale_btn.config(state='normal'))
                    return
                with self.io_lock:
                    success = osc.autoscale()
                self.root.after(0, lambda: self.autoscale_btn.config(state='normal'))
                if success:
                    self.status_queue.put(('autoscale_complete', 'Autoscale completed successfully'))
                else:
                    self.status_queue.put(('error', 'Autoscale failed'))
            except Exception as e:
                self.log_message(f"Autoscale error: {e}", "ERROR")
                self.status_queue.put(('error', f"Autoscale error: {e}"))
                self.root.after(0, lambda: self.autoscale_btn.config(state='normal'))
        threading.Thread(target=autoscale_task, daemon=True).start()

    def run(self):
        """Start the application main event loop."""
        try:
            self.log_message("Keysight Oscilloscope Automation", "SUCCESS")
            self.log_message("New Features: Timebase controls + Individual trigger levels", "SUCCESS")
            self.log_message("All controls use methods from instrument_control module", "SUCCESS")
            self.log_message("Ready to connect to oscilloscope")
            self.root.mainloop()  # Start GUI event loop (blocks until window closed)
        except KeyboardInterrupt:  # User pressed Ctrl+C
            self.log_message("Application interrupted")
        except Exception as e:
            self.log_message(f"Application error: {e}", "ERROR")
        finally:
            if hasattr(self, 'oscilloscope') and self.oscilloscope:  # Clean up on exit
                try:
                    self.oscilloscope.disconnect()  # Close connection
                except:
                    pass  # Ignore cleanup errors

def main():
    """Application entry point."""
    print("Keysight Oscilloscope Automation")
    print("New Features: Horizontal timebase controls + Individual channel trigger levels")
    print("All methods called from instrument_control files")
    print("=" * 80)
    try:
        app = EnhancedResponsiveAutomationGUI()  # Create GUI application
        app.run()  # Start event loops
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()  # Execute applications