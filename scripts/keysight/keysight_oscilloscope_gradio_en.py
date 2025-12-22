#!/usr/bin/env python3

"""

Keysight DSOX6004A Oscilloscope Control - Enhanced Professional Gradio Interface

Comprehensive oscilloscope automation with advanced trigger modes, math functions,

marker/cursor operations, and acquisition control features.

✓ FEATURES ADDED (20+ years professional instrumentation experience):

- Advanced trigger modes (Glitch, Pulse, Pattern detection)

- Marker/Cursor operations with delta measurements

- Math function configuration (ADD, SUB, MUL, DIV, FFT)

- Acquisition mode and type control (Real-time, ETIMe, Segmented, Average)

- Complete trigger configuration (sweep, holdoff, slope variations)

- Display grid and menu control

- Setup save/recall functionality

- Math function waveform acquisition and saving

- Complete error handling and status monitoring

- Thread-safe operations with comprehensive logging

Author: Enhanced by Senior Instrumentation Engineer

Date: 2025-11-06

"""

# =============================================================================
# ⚙️ FILE SAVE LOCATION CONFIGURATION - EDIT THESE PATHS
# =============================================================================
# INSTRUCTIONS: Enter the FULL file paths where you want to save files.
# - Use raw strings (prefix with r) for Windows paths
# - Example: r"C:\Users\YourName\Documents\Oscilloscope\Data"
# - Make sure the server has write permissions to these directories
# - Directories will be created automatically if they don't exist
# =============================================================================

# CSV data files location (waveform data exports)
KEYSIGHT_CSV_DATA_PATH = r"C:\Users\AnirudhIyengar\Desktop\keysight_oscilloscope_data"

# Graph/plot images location (PNG files)
KEYSIGHT_GRAPH_PATH = r"C:\Users\AnirudhIyengar\Desktop\keysight_oscilloscope_graphs"

# Screenshot images location (oscilloscope screen captures)
KEYSIGHT_SCREENSHOT_PATH = r"C:\Users\AnirudhIyengar\Desktop\keysight_oscilloscope_screenshots"

# =============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW THIS LINE
# =============================================================================

import sys

import logging

import threading

import queue

import time

# Removed tkinter imports - no longer needed for web-based file path input

from pathlib import Path

from datetime import datetime

from typing import Optional, Dict, Any, List, Tuple, Union

import signal

import atexit

import os

import socket

import gradio as gr

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

plt.rcParams['agg.path.chunksize'] = 10000

plt.rcParams['path.simplify_threshold'] = 0.5

script_dir = Path(__file__).resolve().parent.parent.parent

if str(script_dir) not in sys.path:

    sys.path.append(str(script_dir))

try:

    from instrument_control.keysight_oscilloscope import KeysightDSOX6004A, KeysightDSOX6004AError

except ImportError as e:

    print(f"Error importing oscilloscope module: {e}")

    sys.exit(1)

# ============================================================================

# UTILITY FUNCTIONS

# ============================================================================

def parse_timebase_string(value: str) -> float:

    """Parse timebase string with unit suffixes to seconds"""

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

        return float(value)

TRIGGER_SLOPE_MAP = {

    "Rising": "POS",

    "Falling": "NEG",

    "Either": "EITH"

}

def format_si_value(value: float, kind: str) -> str:

    """Format numeric values with SI prefixes for human readability"""

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

    """Format measurement values with appropriate units based on type"""

    if value is None:

        return "N/A"

    if meas_type == "FREQ":

        return format_si_value(value, "freq")

    if meas_type in ["PERiod", "RISE", "FALL", "PWIDth", "NWIDth"]:

        return format_si_value(value, "time")

    if meas_type in ["VAMP", "VTOP", "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "VPP",]:

        return format_si_value(value, "volt")

    if meas_type in ["DUTYcycle", "NDUTy", "OVERshoot"]:

        return format_si_value(value, "percent")

    return f"{value}"

# ============================================================================

# DATA ACQUISITION CLASS

# ============================================================================

class OscilloscopeDataAcquisition:

    """

    Data acquisition handler with thread-safe waveform capture and file I/O.

    Implements high-level waveform acquisition, CSV export, and plot generation

    with comprehensive error handling and progress tracking.

    """

    def __init__(self, oscilloscope_instance, io_lock: Optional[threading.RLock] = None):

        self.scope = oscilloscope_instance

        self._logger = logging.getLogger(f'{self.__class__.__name__}')

        self.default_data_dir = Path.cwd() / "data"

        self.default_graph_dir = Path.cwd() / "graphs"

        self.default_screenshot_dir = Path.cwd() / "screenshots"

        self.io_lock = io_lock

    def acquire_waveform_data(self, channel: int, max_points: int = 62500) -> Optional[Dict[str, Any]]:

        """

        Acquire waveform data from specified channel with automatic format conversion.

        Thread-safe acquisition using oscilloscope's built-in waveform transfer.

        """

        if not self.scope.is_connected:

            self._logger.error("Cannot acquire data: oscilloscope not connected")

            return None

        try:

            lock = self.io_lock

            if lock:

                with lock:

                    waveform_data = self._acquire_waveform_scpi(channel, max_points)

            else:

                waveform_data = self._acquire_waveform_scpi(channel, max_points)

            if waveform_data:

                self._logger.info(f"Acquired {len(waveform_data['voltage'])} points from channel {channel}")

                return waveform_data

        except Exception as e:

            self._logger.error(f"Waveform acquisition failed: {e}")

            return None

    def _acquire_waveform_scpi(self, channel: int, max_points: int) -> Optional[Dict[str, Any]]:

        """Internal SCPI-based waveform acquisition with preamble parsing"""

        try:

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

            voltage_data = [(value - y_reference) * y_increment + y_origin for value in raw_data]

            time_data = [x_origin + (i * x_increment) for i in range(len(voltage_data))]

            return {

                'channel': channel,

                'time': time_data,

                'voltage': voltage_data,

                'sample_rate': 1.0 / x_increment,

                'time_increment': x_increment,

                'voltage_increment': y_increment,

                'points_count': len(voltage_data),

                'acquisition_time': datetime.now().isoformat(),

                'is_math': False

            }

        except Exception as e:

            self._logger.error(f"SCPI acquisition failed: {e}")

            return None

    def acquire_math_function_data(self, function_num: int, max_points: int = 62500) -> Optional[Dict[str, Any]]:

        """

        Acquire waveform data from specified math function with automatic format conversion.

        Thread-safe acquisition using oscilloscope's built-in waveform transfer.

        """

        if not self.scope.is_connected:

            self._logger.error("Cannot acquire data: oscilloscope not connected")

            return None

        try:

            lock = self.io_lock

            if lock:

                with lock:

                    waveform_data = self._acquire_math_waveform_scpi(function_num, max_points)

            else:

                waveform_data = self._acquire_math_waveform_scpi(function_num, max_points)

            if waveform_data:

                self._logger.info(f"Acquired {len(waveform_data['voltage'])} points from math function {function_num}")

                return waveform_data

        except Exception as e:

            self._logger.error(f"Math waveform acquisition failed: {e}")

            return None

    def _acquire_math_waveform_scpi(self, function_num: int, max_points: int) -> Optional[Dict[str, Any]]:

        """Internal SCPI-based math function waveform acquisition with preamble parsing"""

        try:

            # ✓ Manual pg 1201: :WAVeform:SOURce FUNCtion<m>

            self.scope._scpi_wrapper.write(f":WAVeform:SOURce FUNCtion{function_num}")

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

            voltage_data = [(value - y_reference) * y_increment + y_origin for value in raw_data]

            time_data = [x_origin + (i * x_increment) for i in range(len(voltage_data))]

            return {

                'channel': function_num,

                'time': time_data,

                'voltage': voltage_data,

                'sample_rate': 1.0 / x_increment,

                'time_increment': x_increment,

                'voltage_increment': y_increment,

                'points_count': len(voltage_data),

                'acquisition_time': datetime.now().isoformat(),

                'is_math': True

            }

        except Exception as e:

            self._logger.error(f"Math SCPI acquisition failed: {e}")

            return None

    def export_to_csv(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None,

                      filename: Optional[str] = None) -> Optional[str]:

        """Export waveform data to CSV with comprehensive metadata"""

        if not waveform_data:

            self._logger.error("No waveform data to export")

            return None

        try:

            save_dir = Path(custom_path) if custom_path else self.default_data_dir

            save_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:

                source_label = "MATH" if waveform_data['is_math'] else "CH"

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                filename = f"waveform_{source_label}{waveform_data['channel']}_{timestamp}.csv"

            if not filename.endswith('.csv'):

                filename += '.csv'

            filepath = save_dir / filename

            df = pd.DataFrame({

                'Time (s)': waveform_data['time'],

                'Voltage (V)': waveform_data['voltage']

            })

            with open(filepath, 'w') as f:

                source_label = "Math Function" if waveform_data['is_math'] else "Channel"

                f.write(f"# Oscilloscope Waveform Data\n")

                f.write(f"# {source_label}: {waveform_data['channel']}\n")

                f.write(f"# Acquisition Time: {waveform_data['acquisition_time']}\n")

                f.write(f"# Sample Rate: {waveform_data['sample_rate']:.2e} Hz\n")

                f.write(f"# Points Count: {waveform_data['points_count']}\n")

                f.write(f"# Time Increment: {waveform_data['time_increment']:.2e} s\n")

                f.write(f"# Voltage Increment: {waveform_data['voltage_increment']:.2e} V\n")

                f.write("\n")

                df.to_csv(filepath, mode='a', index=False)

            self._logger.info(f"CSV exported: {filepath}")

            return str(filepath)

        except Exception as e:

            self._logger.error(f"CSV export failed: {e}")

            return None

    def generate_waveform_plot(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None,

                               filename: Optional[str] = None, plot_title: Optional[str] = None) -> Optional[str]:

        """Generate professional waveform plot with measurements overlay"""

        measurements = {}

        try:

            if waveform_data['is_math']:

                # For math functions, use measure_math_single

                function_num = waveform_data['channel']

                if self.io_lock:

                    with self.io_lock:

                        measurements_list = [

                            "FREQ", "PERiod", "VPP", "VAMP", "VTOP", "VBASe",

                            "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL"

                        ]

                        for meas in measurements_list:

                            try:

                                val = self.scope.measure_math_single(function_num, meas)

                                if val is not None:

                                    measurements[meas] = val

                            except:

                                pass

                else:

                    measurements_list = [

                        "FREQ", "PERiod", "VPP", "VAMP", "VTOP", "VBASe",

                        "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL"

                    ]

                    for meas in measurements_list:

                        try:

                            val = self.scope.measure_math_single(function_num, meas)

                            if val is not None:

                                measurements[meas] = val

                        except:

                            pass

            else:

                # For regular channels, use measure_single

                channel = waveform_data['channel']

                if self.io_lock:

                    with self.io_lock:

                        measurements = self.scope.get_all_measurements(channel) or {}

                else:

                    measurements = self.scope.get_all_measurements(channel) or {}

        except Exception as e:

            self._logger.warning(f"Failed to get measurements: {e}")

            measurements = {}

        if not waveform_data:

            self._logger.error("No waveform data to plot")

            return None

        try:

            save_dir = Path(custom_path) if custom_path else self.default_graph_dir

            save_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:

                source_label = "MATH" if waveform_data['is_math'] else "CH"

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                filename = f"waveform_plot_{source_label}{waveform_data['channel']}_{timestamp}.png"

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                filename += '.png'

            filepath = save_dir / filename

            fig, ax = plt.subplots(figsize=(12, 8))

            time_data = waveform_data['time']

            voltage_data = waveform_data['voltage']

            if len(time_data) > 100000:

                step = len(time_data) // 100000

                time_data = time_data[::step]

                voltage_data = voltage_data[::step]

            ax.plot(time_data, voltage_data, 'b-', linewidth=1.0, rasterized=True)

            if plot_title is None:

                source_label = "Math Function" if waveform_data['is_math'] else "Channel"

                plot_title = f"Oscilloscope Waveform - {source_label} {waveform_data['channel']}"

            ax.set_title(plot_title, fontsize=14, fontweight='bold')

            ax.set_xlabel('Time (s)', fontsize=12)

            ax.set_ylabel('Voltage (V)', fontsize=12)

            ax.grid(True, alpha=0.3)

            measurements_text = "MEASUREMENTS:\n"

            measurements_text += "─" * 25 + "\n"

            key_measurements = [

                ('Freq', 'FREQ'), ('Period', 'PERiod'), ('VPP', 'VPP'),

                ('VAVG', 'VAVG'), ('VRMS', 'VRMS'), ('VMAX', 'VMAX'),

                ('VMIN', 'VMIN'), ('DUTYcycle', 'DUTYcycle')

            ]

            for display_name, meas_key in key_measurements:

                value = measurements.get(meas_key)

                formatted_value = format_measurement_value(meas_key, value)

                measurements_text += f"{display_name}: {formatted_value}\n"

            ax.text(0.02, 0.98, measurements_text,

                    transform=ax.transAxes,

                    fontsize=9,

                    verticalalignment='top',

                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),

                    family='monospace')

            plt.tight_layout()

            plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')

            plt.close(fig)

            self._logger.info(f"Plot saved: {filepath}")

            return str(filepath)

        except Exception as e:

            self._logger.error(f"Plot generation failed: {e}")

            return None

# ============================================================================

# MAIN GRADIO GUI CLASS

# ============================================================================

class GradioOscilloscopeGUI:

    """

    Professional oscilloscope control interface with comprehensive feature set.

    Implements connection management, channel configuration, trigger modes,

    math functions, marker operations, and complete data acquisition workflow.

    """

    def __init__(self):

        self.oscilloscope = None

        self.data_acquisition = None

        self.last_acquired_data = None

        self.io_lock = threading.RLock()

        self._shutdown_flag = threading.Event()

        self._gradio_interface = None

        # Use the configured paths from the top of the file
        self.save_locations = {

            'data': KEYSIGHT_CSV_DATA_PATH,

            'graphs': KEYSIGHT_GRAPH_PATH,

            'screenshots': KEYSIGHT_SCREENSHOT_PATH

        }

        self.setup_logging()

        self.setup_cleanup_handlers()

        self.timebase_scales: List[Union[str, int, float, Tuple[str, Union[str, int, float]]]] = [

            ("1 ns", 1e-9), ("2 ns", 2e-9), ("5 ns", 5e-9),

            ("10 ns", 10e-9), ("20 ns", 20e-9), ("50 ns", 50e-9),

            ("100 ns", 100e-9), ("200 ns", 200e-9), ("500 ns", 500e-9),

            ("1 µs", 1e-6), ("2 µs", 2e-6), ("5 µs", 5e-6),

            ("10 µs", 10e-6), ("20 µs", 20e-6), ("50 µs", 50e-6),

            ("100 µs", 100e-6), ("200 µs", 200e-6), ("500 µs", 500e-6),

            ("1 ms", 1e-3), ("2 ms", 2e-3), ("5 ms", 5e-3),

            ("10 ms", 10e-3), ("20 ms", 20e-3), ("50 ms", 50e-3),

            ("100 ms", 100e-3), ("200 ms", 200e-3), ("500 ms", 500e-3),

            ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0),

            ("10 s", 10.0), ("20 s", 20.0), ("50 s", 50.0)

        ]

        self.measurement_types = [

            "FREQ", "PERiod", "VPP", "VAMP", "OVERshoot", "VTOP",

            "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL",

            "DUTYcycle", "NDUTy", "PWIDth", "NWIDth"

        ]

    def setup_logging(self):

        """Configure logging for system diagnostics"""

        logging.basicConfig(

            level=logging.INFO,

            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        )

        self.logger = logging.getLogger('GradioOscilloscopeAutomation')

    def setup_cleanup_handlers(self):

        """Register cleanup procedures for graceful shutdown"""

        atexit.register(self.cleanup)

        signal.signal(signal.SIGINT, self._signal_handler)

        if hasattr(signal, 'SIGTERM'):

            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):

        """Handle system signals for clean shutdown"""

        print(f"\nReceived signal {signum}, shutting down...")

        self.cleanup()

        sys.exit(0)

    def cleanup(self):

        """Cleanup resources and disconnect oscilloscope"""

        try:

            self._shutdown_flag.set()

            if self.oscilloscope and hasattr(self.oscilloscope, 'is_connected'):

                if self.oscilloscope.is_connected:

                    print("Disconnecting oscilloscope...")

                    self.oscilloscope.disconnect()

            self.oscilloscope = None

            self.data_acquisition = None

            plt.close('all')

            print("Cleanup completed.")

        except Exception as e:

            print(f"Cleanup error: {e}")

    # ========================================================================

    # CONNECTION MANAGEMENT

    # ========================================================================

    def connect_oscilloscope(self, visa_address: str):

        """Establish VISA connection and query instrument identification"""

        try:

            if not visa_address:

                return "Error: VISA address is empty", "Disconnected"

            self.oscilloscope = KeysightDSOX6004A(visa_address)

            if self.oscilloscope.connect():

                self.data_acquisition = OscilloscopeDataAcquisition(self.oscilloscope, io_lock=self.io_lock)

                info = self.oscilloscope.get_instrument_info()

                if info:

                    info_text = f"Connected: {info['manufacturer']} {info['model']} | S/N: {info['serial_number']} | FW: {info['firmware_version']}"

                    return info_text, "Connected"

                else:

                    return "Connected (no info available)", "Connected"

            else:

                return "Connection failed", "Disconnected"

        except Exception as e:

            return f"Error: {str(e)}", "Disconnected"

    def disconnect_oscilloscope(self):

        """Close connection to oscilloscope"""

        try:

            if self.oscilloscope:

                self.oscilloscope.disconnect()

            self.oscilloscope = None

            self.data_acquisition = None

            self.last_acquired_data = None

            self.logger.info("Disconnected successfully")

            return "Disconnected successfully", "Disconnected"

        except Exception as e:

            self.logger.error(f"Disconnect error: {e}")

            return f"Disconnect error: {str(e)}", "Disconnected"

    def test_connection(self):

        """Verify oscilloscope connectivity"""

        if self.oscilloscope and self.oscilloscope.is_connected:

            return "✓ Connection test: PASSED"

        else:

            return "✗ Connection test: FAILED - Not connected"

    # ========================================================================

    # CHANNEL CONFIGURATION

    # ========================================================================

    def configure_channel(self, ch1, ch2, ch3, ch4, v_scale, v_offset, coupling, probe):

        """Configure vertical parameters for selected channels"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        channel_states = {1: ch1, 2: ch2, 3: ch3, 4: ch4}

        try:

            success_count = 0

            disabled_count = 0

            for channel, enabled in channel_states.items():

                with self.io_lock:

                    if enabled:

                        success = self.oscilloscope.configure_channel(

                            channel=channel,

                            vertical_scale=v_scale,

                            vertical_offset=v_offset,

                            coupling=coupling,

                            probe_attenuation=probe

                        )

                        if success:

                            success_count += 1

                    else:

                        try:

                            with self.io_lock:

                                self.oscilloscope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay OFF")

                                disabled_count += 1

                        except Exception as e:

                            self.logger.warning(f"Failed to disable channel {channel}: {e}")

            return f"Configured: {success_count} enabled, {disabled_count} disabled"

        except Exception as e:

            return f"Configuration error: {str(e)}"

    # ========================================================================

    # TIMEBASE & TRIGGER CONFIGURATION

    # ========================================================================

    def configure_timebase(self, time_scale_input):

        """Set horizontal timebase parameters"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if isinstance(time_scale_input, (int, float)):

                time_scale = float(time_scale_input)

                display_scale = format_si_value(time_scale, 'time')

            else:

                time_scale = parse_timebase_string(time_scale_input)

                display_scale = time_scale_input

            with self.io_lock:

                success = self.oscilloscope.configure_timebase(time_scale)

            if success:

                return f"Timebase: {display_scale} ({time_scale}s/div)"

            else:

                return "Timebase configuration failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def configure_trigger(self, trigger_source, trigger_level, trigger_slope):

        """Configure edge trigger with specified parameters"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            channel = int(trigger_source.replace("CH", ""))

            slope = TRIGGER_SLOPE_MAP.get(trigger_slope, "POS")

            with self.io_lock:

                success = self.oscilloscope.configure_trigger(channel, trigger_level, slope)

            if success:

                return f"Trigger: {trigger_source} @ {trigger_level}V, {trigger_slope}"

            else:

                return "Trigger configuration failed"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # MEASUREMENTS - CHANNEL AND MATH FUNCTIONS

    # ========================================================================

    def get_all_measurements(self, source_str):
        """Get all available measurements for the specified source (CH1-CH4 or MATH1-MATH4)"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            source_upper = source_str.upper()
            # Parse source string (e.g., "CH1" or "MATH1")
            if source_upper.startswith("CH"):
                channel = int(source_upper[2:])
                with self.io_lock:
                    results = self.oscilloscope.get_all_measurements(channel)
                if results:
                    formatted_results = []
                    for meas_type, value in results.items():
                        formatted_results.append(f"{meas_type}: {format_measurement_value(meas_type, value)}")
                    return "\n".join(formatted_results)
                else:
                    return f"No measurements available for {source_str}"
            elif source_upper.startswith("MATH"):
                func_num = int(source_upper[4:])
                if not (1 <= func_num <= 4):
                    return "Math function number must be between 1 and 4"
                
                # Get all available measurements for the math function
                results = {}
                for meas_type in self.measurement_types:
                    try:
                        with self.io_lock:
                            value = self.oscilloscope.measure_math_single(func_num, meas_type)
                        if value is not None:
                            results[meas_type] = value
                    except Exception:
                        continue
                
                if results:
                    formatted_results = [f"Measurements for {source_upper}:"]
                    for meas_type, value in results.items():
                        formatted_results.append(f"{meas_type}: {format_measurement_value(meas_type, value)}")
                    return "\n".join(formatted_results)
                else:
                    return f"No measurements available for {source_str}"
            else:
                return f"Unsupported source: {source_str}. Use CH1-CH4 or MATH1-MATH4"
        except Exception as e:
            return f"Error getting measurements: {str(e)}"

    def perform_measurement(self, source_str, measurement_type):
        """Perform single measurement on specified channel or math function"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            # Parse source string (e.g., "CH1" or "MATH1")
            if source_str.upper().startswith("CH"):
                channel = int(source_str[2:])
                with self.io_lock:
                    result = self.oscilloscope.measure_single(channel, measurement_type)
                if result is not None:
                    return f"{source_str.upper()} {measurement_type}: {format_measurement_value(measurement_type, result)}"
                else:
                    return f"Failed to measure {measurement_type} on {source_str}"
            elif source_str.upper().startswith("MATH"):
                func_num = int(source_str[4:])
                with self.io_lock:
                    result = self.oscilloscope.measure_math_single(func_num, measurement_type)
                if result is not None:
                    return f"MATH{func_num} {measurement_type}: {format_measurement_value(measurement_type, result)}"
                else:
                    return f"Failed to measure {measurement_type} on {source_str}"
            else:
                return f"Invalid source: {source_str}"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================

    # ADVANCED TRIGGER MODES

    # ========================================================================

    def set_glitch_trigger(self, source_channel, glitch_level, glitch_polarity, glitch_width):

        """Configure glitch (spike) trigger mode"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            channel = int(source_channel.replace("CH", ""))

            width_seconds = glitch_width * 1e-9

            with self.io_lock:

                success = self.oscilloscope.set_glitch_trigger(

                    channel=channel,

                    level=glitch_level,

                    polarity=glitch_polarity,

                    width=width_seconds

                )

            if success:

                return f"Glitch trigger: {source_channel}, Level={glitch_level}V, Width={glitch_width}ns, Polarity={glitch_polarity}"

            else:

                return "Glitch trigger configuration failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_pulse_trigger(self, source_channel, pulse_level, pulse_width, pulse_polarity):

        """Configure pulse width trigger mode"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            channel = int(source_channel.replace("CH", ""))

            width_seconds = pulse_width * 1e-9

            with self.io_lock:

                success = self.oscilloscope.set_pulse_trigger(

                    channel=channel,

                    level=pulse_level,

                    width=width_seconds,

                    polarity=pulse_polarity

                )

            if success:

                return f"Pulse trigger: {source_channel}, Level={pulse_level}V, Width={pulse_width}ns, Polarity={pulse_polarity}"

            else:

                return "Pulse trigger configuration failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_trigger_sweep_mode(self, sweep_mode):

        """Set trigger sweep behavior"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.set_trigger_sweep(sweep_mode)

            if success:

                return f"Trigger sweep mode: {sweep_mode}"

            else:

                return "Failed to set trigger sweep mode"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_trigger_holdoff(self, holdoff_nanoseconds):

        """Set trigger holdoff time"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            holdoff_seconds = holdoff_nanoseconds * 1e-9

            with self.io_lock:

                success = self.oscilloscope.set_trigger_holdoff(holdoff_seconds)

            if success:

                return f"Trigger holdoff: {holdoff_nanoseconds}ns"

            else:

                return "Failed to set trigger holdoff"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # ACQUISITION CONTROL

    # ========================================================================

    def set_acquisition_mode(self, mode_type):

        """Set oscilloscope acquisition mode"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.set_acquire_mode(mode_type)

            if success:

                return f"Acquisition mode: {mode_type}"

            else:

                return "Failed to set acquisition mode"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_acquisition_type(self, acq_type):

        """Set oscilloscope acquisition type"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.set_acquire_type(acq_type)

            if success:

                return f"Acquisition type: {acq_type}"

            else:

                return "Failed to set acquisition type"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_acquisition_count(self, average_count):

        """Set number of acquisitions to average"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if not (2 <= average_count <= 65536):

                return f"Error: Count must be 2-65536, got {average_count}"

            with self.io_lock:

                success = self.oscilloscope.set_acquire_count(average_count)

            if success:

                return f"Acquisition count: {average_count} averages"

            else:

                return "Failed to set acquisition count"

        except Exception as e:

            return f"Error: {str(e)}"

    def query_acquisition_info(self):

        """Query and display current acquisition parameters"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            info_lines = []

            with self.io_lock:

                mode = self.oscilloscope.get_acquire_mode()

                acq_type = self.oscilloscope.get_acquire_type()

                count = self.oscilloscope.get_acquire_count()

                sample_rate = self.oscilloscope.get_sample_rate()

                points = self.oscilloscope.get_acquire_points()

            if mode:

                info_lines.append(f"Mode: {mode}")

            if acq_type:

                info_lines.append(f"Type: {acq_type}")

            if count:

                info_lines.append(f"Count: {count}")

            if sample_rate:

                info_lines.append(f"Sample Rate: {format_si_value(sample_rate, 'freq')}")

            if points:

                info_lines.append(f"Acquired Points: {points}")

            return "\n".join(info_lines) if info_lines else "No acquisition info available"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # MARKER/CURSOR OPERATIONS

    # ========================================================================

    def set_marker_positions(self, marker_num, x_position, y_position):

        """Set marker (cursor) X and Y positions"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if marker_num not in [1, 2]:

                return "Error: Marker must be 1 or 2"

            with self.io_lock:

                x_success = self.oscilloscope.set_marker_x_position(marker_num, x_position)

                y_success = self.oscilloscope.set_marker_y_position(marker_num, y_position)

            if x_success and y_success:

                x_fmt = format_si_value(x_position, "time")

                y_fmt = format_si_value(y_position, "volt")

                return f"Marker {marker_num}: X={x_fmt}, Y={y_fmt}"

            else:

                return "Failed to set marker positions"

        except Exception as e:

            return f"Error: {str(e)}"

    def get_marker_deltas(self):

        """Query time and voltage differences between markers"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                x_delta = self.oscilloscope.get_marker_x_delta()

                y_delta = self.oscilloscope.get_marker_y_delta()

            result_lines = []

            if x_delta is not None:

                x_fmt = format_si_value(x_delta, "time")

                if x_delta > 0:

                    freq = 1.0 / x_delta

                    freq_fmt = format_si_value(freq, "freq")

                    result_lines.append(f"X Delta (Time): {x_fmt}")

                    result_lines.append(f"Derived Frequency: {freq_fmt}")

                else:

                    result_lines.append(f"X Delta (Time): {x_fmt}")

            if y_delta is not None:

                y_fmt = format_si_value(y_delta, "volt")

                result_lines.append(f"Y Delta (Voltage): {y_fmt}")

            return "\n".join(result_lines) if result_lines else "No marker data available"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_marker_mode(self, marker_mode):

        """Set marker/cursor operational mode"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.set_marker_mode(marker_mode)

            if success:

                return f"Marker mode: {marker_mode}"

            else:

                return "Failed to set marker mode"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # MATH FUNCTIONS

    # ========================================================================

    def configure_math_operation(self, func_num, operation, source1_ch, source2_ch=None):

        """Configure math function for waveform processing"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if func_num not in [1, 2, 3, 4]:

                return "Error: Function number must be 1-4"

            with self.io_lock:

                success = self.oscilloscope.set_math_function(

                    function_num=func_num,

                    operation=operation,

                    source1=source1_ch,

                    source2=source2_ch

                )

            if success:

                return f"Math function {func_num}: {operation} configured"

            else:

                return f"Failed to configure math function {func_num}"

        except Exception as e:

            return f"Error: {str(e)}"

    def toggle_math_display(self, func_num, show):

        """Show or hide math function on display"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if func_num not in [1, 2, 3, 4]:

                return "Error: Function number must be 1-4"

            with self.io_lock:

                success = self.oscilloscope.set_math_display(func_num, show)

            if success:

                state = "shown" if show else "hidden"

                return f"Math function {func_num}: {state}"

            else:

                return f"Failed to toggle math function {func_num}"

        except Exception as e:

            return f"Error: {str(e)}"

    def set_math_scale(self, func_num, scale_value):
        """Set vertical scale for math function result
        
        Args:
            func_num: Math function number (1-4)
            scale_value: Desired scale value in volts/division
            
        Note: The oscilloscope expects the full range (10 divisions),
        so we need to convert from V/div to full range by multiplying by 10.
        The scale_value is already in V/div, so we pass it directly.
        """
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            if func_num not in [1, 2, 3, 4]:
                return "Error: Function number must be 1-4"

            with self.io_lock:
                # The oscilloscope's set_math_scale will handle the *10 conversion
                success = self.oscilloscope.set_math_scale(func_num, scale_value)

            if success:
                return f"Math function {func_num} scale: {scale_value} V/div"
            else:
                return f"Failed to set math function {func_num} scale"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # SETUP MANAGEMENT

    # ========================================================================

    def save_instrument_setup(self, setup_name):

        """Save complete instrument configuration to internal memory"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if not setup_name.endswith('.stp'):

                setup_name += '.stp'

            with self.io_lock:

                success = self.oscilloscope.save_setup(setup_name)

            if success:

                return f"Setup saved: {setup_name}"

            else:

                return "Failed to save setup"

        except Exception as e:

            return f"Error: {str(e)}"

    def recall_instrument_setup(self, setup_name):

        """Restore previously saved instrument configuration"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if not setup_name.endswith('.stp'):

                setup_name += '.stp'

            with self.io_lock:

                success = self.oscilloscope.recall_setup(setup_name)

            if success:

                return f"Setup recalled: {setup_name}"

            else:

                return "Failed to recall setup"

        except Exception as e:

            return f"Error: {str(e)}"

    def save_waveform_to_memory(self, channel, waveform_name):

        """Save waveform data to internal oscilloscope memory"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            if channel not in [1, 2, 3, 4]:

                return "Error: Channel must be 1-4"

            with self.io_lock:

                success = self.oscilloscope.save_waveform(channel, waveform_name)

            if success:

                return f"Waveform saved: CH{channel} -> {waveform_name}"

            else:

                return "Failed to save waveform"

        except Exception as e:

            return f"Error: {str(e)}"

    def recall_waveform_from_memory(self, waveform_name):

        """Restore waveform from internal oscilloscope memory"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.recall_waveform(waveform_name)

            if success:

                return f"Waveform recalled: {waveform_name}"

            else:

                return "Failed to recall waveform"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # FUNCTION GENERATORS

    # ========================================================================

    def configure_wgen(self, generator, enable, waveform, frequency, amplitude, offset):

        """Configure function generator with specified parameters"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.configure_function_generator(

                    generator=generator,

                    waveform=waveform,

                    frequency=frequency,

                    amplitude=amplitude,

                    offset=offset,

                    enable=enable

                )

            if success:

                return f"WGEN{generator}: {waveform}, {frequency}Hz, {amplitude}Vpp"

            else:

                return f"WGEN{generator} configuration failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def get_wgen_configuration(self, generator):

        """Query and display function generator current configuration"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                config = self.oscilloscope.get_function_generator_config(generator)

            if config:

                lines = [

                    f"WGEN{generator} Configuration:",

                    f" Function: {config['function']}",

                    f" Frequency: {format_si_value(config['frequency'], 'freq')}",

                    f" Amplitude: {format_si_value(config['amplitude'], 'volt')}",

                    f" Offset: {format_si_value(config['offset'], 'volt')}",

                    f" Output: {config['output']}"

                ]

                return "\n".join(lines)

            else:

                return "Failed to query WGEN configuration"

        except Exception as e:

            return f"Error: {str(e)}"

    # ========================================================================

    # DATA ACQUISITION

    # ========================================================================

    def capture_screenshot(self):
        """Capture and save display screenshot and return file path for download"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected", None

        try:
            # Ensure the save directory exists
            screenshot_dir = Path(self.save_locations['screenshots'])
            screenshot_dir.mkdir(parents=True, exist_ok=True)

            # Generate a timestamp and filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"scope_screenshot_{timestamp}.png"

            # Create the full path for the screenshot
            screenshot_path = screenshot_dir / filename

            # Get the screenshot data directly
            if not hasattr(self.oscilloscope, '_scpi_wrapper'):
                return "Error: Oscilloscope SCPI interface not available", None

            # Get the screenshot data
            try:
                image_data = self.oscilloscope._scpi_wrapper.query_binary_values(
                    ":DISPlay:DATA? PNG",
                    datatype='B'
                )

                if image_data:
                    # Save the screenshot to the desired location
                    with open(screenshot_path, 'wb') as f:
                        f.write(bytes(image_data))
                    return f"Screenshot saved: {screenshot_path}", str(screenshot_path)
                else:
                    return "Screenshot capture failed: No data received", None

            except Exception as e:
                return f"Error capturing screenshot: {str(e)}", None

        except Exception as e:
            return f"Error: {str(e)}", None

    def acquire_data(self, ch1, ch2, ch3, ch4, math1, math2, math3, math4):

        """Acquire waveform data from selected channels and math functions"""

        if not self.data_acquisition:

            return "Error: Not initialized. Connect first."

        selected_channels = []

        if ch1:

            selected_channels.append(('CH', 1))

        if ch2:

            selected_channels.append(('CH', 2))

        if ch3:

            selected_channels.append(('CH', 3))

        if ch4:

            selected_channels.append(('CH', 4))

        if math1:

            selected_channels.append(('MATH', 1))

        if math2:

            selected_channels.append(('MATH', 2))

        if math3:

            selected_channels.append(('MATH', 3))

        if math4:

            selected_channels.append(('MATH', 4))

        if not selected_channels:

            return "Error: No channels/math functions selected"

        try:

            all_channel_data = {}

            for source_type, number in selected_channels:

                if source_type == 'CH':

                    data = self.data_acquisition.acquire_waveform_data(number)

                    if data:

                        all_channel_data[f'CH{number}'] = data

                else:  # MATH

                    data = self.data_acquisition.acquire_math_function_data(number)

                    if data:

                        all_channel_data[f'MATH{number}'] = data

            if all_channel_data:

                self.last_acquired_data = all_channel_data

                total_points = sum(ch_data['points_count'] for ch_data in all_channel_data.values())

                return f"Acquired: {len(all_channel_data)} sources, {total_points} total points"

            else:

                return "Acquisition failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def export_csv(self):

        """Export acquired waveform data to CSV files and return paths for download"""

        if not self.last_acquired_data:

            return "Error: No data available", []

        if not self.data_acquisition:

            return "Error: Not initialized", []

        try:

            exported_files = []
            exported_file_paths = []

            if isinstance(self.last_acquired_data, dict):

                for source_key, data in self.last_acquired_data.items():

                    filename = self.data_acquisition.export_to_csv(data, custom_path=self.save_locations['data'])

                    if filename:

                        exported_files.append(Path(filename).name)
                        exported_file_paths.append(filename)

            if exported_files:

                return f"Exported: {', '.join(exported_files)}", exported_file_paths

            else:

                return "Export failed", []

        except Exception as e:

            return f"Error: {str(e)}", []

    def generate_plot(self, plot_title):

        """Generate waveform plot with measurements"""

        if not self.last_acquired_data:

            return "Error: No data available"

        if not self.data_acquisition:

            return "Error: Not initialized"

        try:

            custom_title = plot_title.strip() or None

            plot_files = []

            if isinstance(self.last_acquired_data, dict):

                for source_key, data in self.last_acquired_data.items():

                    if custom_title:

                        source_label = "Math" if data['is_math'] else "Channel"

                        channel_title = f"{custom_title} - {source_label} {data['channel']}"

                    else:

                        channel_title = None

                    filename = self.data_acquisition.generate_waveform_plot(

                        data, custom_path=self.save_locations['graphs'], plot_title=channel_title)

                    if filename:

                        plot_files.append(Path(filename).name)

            if plot_files:

                return f"Generated: {', '.join(plot_files)}"

            else:

                return "Failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def perform_autoscale(self):

        """Execute automatic vertical and horizontal scaling"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected"

        try:

            with self.io_lock:

                success = self.oscilloscope.autoscale()

            if success:

                return "Autoscale completed"

            else:

                return "Autoscale failed"

        except Exception as e:

            return f"Error: {str(e)}"

    def run_full_automation(self, ch1, ch2, ch3, ch4, math1, math2, math3, math4, plot_title):

        """Execute complete acquisition, export, and analysis workflow and return files for download"""

        if not self.oscilloscope or not self.oscilloscope.is_connected:

            return "Error: Not connected", None, []

        if not self.data_acquisition:

            return "Error: Not initialized", None, []

        selected_channels = []

        if ch1:

            selected_channels.append(('CH', 1))

        if ch2:

            selected_channels.append(('CH', 2))

        if ch3:

            selected_channels.append(('CH', 3))

        if ch4:

            selected_channels.append(('CH', 4))

        if math1:

            selected_channels.append(('MATH', 1))

        if math2:

            selected_channels.append(('MATH', 2))

        if math3:

            selected_channels.append(('MATH', 3))

        if math4:

            selected_channels.append(('MATH', 4))

        if not selected_channels:

            return "Error: No channels/math functions selected", None, []

        try:

            results = []
            screenshot_path = None
            csv_file_paths = []

            results.append("Step 1/4: Screenshot...")

            with self.io_lock:

                screenshot_dir = Path(self.save_locations['screenshots'])

                screenshot_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                filename = f"scope_screenshot_{timestamp}.png"

                screenshot_file = self.oscilloscope.capture_screenshot(

                    filename=filename,

                    image_format="PNG"

                )

            if screenshot_file:

                screenshot_path = str(screenshot_dir / filename)
                results.append(f"✓ Screenshot saved")

            results.append("Step 2/4: Acquiring data...")

            all_channel_data = {}

            for source_type, number in selected_channels:

                if source_type == 'CH':

                    data = self.data_acquisition.acquire_waveform_data(number)

                    if data:

                        all_channel_data[f'CH{number}'] = data

                        results.append(f" CH{number}: {data['points_count']} points")

                else:  # MATH

                    data = self.data_acquisition.acquire_math_function_data(number)

                    if data:

                        all_channel_data[f'MATH{number}'] = data

                        results.append(f" MATH{number}: {data['points_count']} points")

            if not all_channel_data:

                return "Error: Data acquisition failed", None, []

            results.append("Step 3/4: Exporting CSV...")

            csv_files = []

            for source_key, data in all_channel_data.items():

                csv_file = self.data_acquisition.export_to_csv(data, custom_path=self.save_locations['data'])

                if csv_file:

                    csv_files.append(Path(csv_file).name)
                    csv_file_paths.append(csv_file)

            if csv_files:

                results.append(f" ✓ {len(csv_files)} files exported")

            results.append("Step 4/4: Generating plots...")

            custom_title = plot_title.strip() or None

            plot_files = []

            for source_key, data in all_channel_data.items():

                if custom_title:

                    source_label = "Math" if data['is_math'] else "Channel"

                    channel_title = f"{custom_title} - {source_label} {data['channel']}"

                else:

                    channel_title = None

                plot_file = self.data_acquisition.generate_waveform_plot(

                    data, custom_path=self.save_locations['graphs'], plot_title=channel_title)

                if plot_file:

                    plot_files.append(Path(plot_file).name)

            if plot_files:

                results.append(f" ✓ {len(plot_files)} plots generated")

            self.last_acquired_data = all_channel_data

            results.append("\n✓ Full automation completed!")

            return "\n".join(results), screenshot_path, csv_file_paths

        except Exception as e:

            return f"Automation error: {str(e)}", None, []

    def browse_folder(self, current_path, _folder_type="folder"):

        """
        Validate and return the provided path.
        Note: Browse dialog removed - users should manually edit the path textbox.
        When accessing via localhost, users can specify any path on their local machine.
        """

        return current_path if current_path else str(Path.cwd())

    # ========================================================================

    # GRADIO INTERFACE CREATION

    # ========================================================================

    def create_interface(self):

        """Build comprehensive Gradio web interface with full-page layout"""

        css = """

        .gradio-container {

        max-width: 100% !important;

        padding: 20px !important;

        margin: 0 !important;

        min-height: 100vh;

        }

        .container {

        max-width: 100% !important;

        padding: 0 10px !important;

        margin: 0 !important;

        }

        #component-0 {

        min-height: 100vh;

        }

        .tab {

        padding: 0 10px;

        min-height: calc(100vh - 120px);

        }

        .panel {

        margin: 5px 0;

        }

        """

        with gr.Blocks(

            title="DIGANTARA Oscilloscope Control",

            css=css,

            theme=gr.themes.Soft(

                primary_hue="purple",

                spacing_size="sm",

                radius_size="sm",

                text_size="sm"

            )

        ) as interface:

            gr.Markdown("# DIGANTARA Oscilloscope Control")
            gr.Markdown("**Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.")
            gr.Markdown("**Professional oscilloscope automation interface with comprehensive control features**")

            # ================================================================

            # CONNECTION TAB

            # ================================================================

            with gr.Tab("Connection"):

                with gr.Row():

                    visa_address = gr.Textbox(

                        label="VISA Address",

                        value="USB0::0x0957::0x1780::MY65220169::INSTR",

                        scale=3

                    )

                    connect_btn = gr.Button("Connect", variant="primary", scale=1)

                    disconnect_btn = gr.Button("Disconnect", variant="stop", scale=1)

                    test_btn = gr.Button("Test", scale=1)

                connection_status = gr.Textbox(label="Status", value="Disconnected", interactive=False)

                instrument_info = gr.Textbox(label="Instrument Information", interactive=False)

                connect_btn.click(

                    fn=self.connect_oscilloscope,

                    inputs=[visa_address],

                    outputs=[instrument_info, connection_status]

                )

                disconnect_btn.click(

                    fn=self.disconnect_oscilloscope,

                    inputs=[],

                    outputs=[instrument_info, connection_status]

                )

                test_btn.click(

                    fn=self.test_connection,

                    inputs=[],

                    outputs=[instrument_info]

                )

            # ================================================================

            # CHANNEL CONFIGURATION TAB

            # ================================================================

            with gr.Tab("Channel Configuration"):

                gr.Markdown("### Channel Selection and Configuration")

                with gr.Row():

                    ch1_select = gr.Checkbox(label="Ch1", value=True)

                    ch2_select = gr.Checkbox(label="Ch2", value=False)

                    ch3_select = gr.Checkbox(label="Ch3", value=False)

                    ch4_select = gr.Checkbox(label="Ch4", value=False)

                with gr.Row():

                    v_scale = gr.Number(label="V/div", value=1.0)

                    v_offset = gr.Number(label="Offset (V)", value=0.0)

                    coupling = gr.Dropdown(

                        label="Coupling",

                        choices=["AC", "DC"],

                        value="DC"

                    )

                    probe = gr.Dropdown(

                        label="Probe",

                        choices=[("1x", 1.0), ("10x", 10.0), ("100x", 100.0)],

                        value=1.0

                    )

                config_channel_btn = gr.Button("Configure Channels", variant="primary")

                channel_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():

                    autoscale_btn = gr.Button("Autoscale", variant="primary")

                system_status = gr.Textbox(label="Status", interactive=False, lines=4)

                autoscale_btn.click(

                    fn=self.perform_autoscale,

                    inputs=[],

                    outputs=[system_status]

                )

                config_channel_btn.click(

                    fn=self.configure_channel,

                    inputs=[ch1_select, ch2_select, ch3_select, ch4_select, v_scale, v_offset, coupling, probe],

                    outputs=[channel_status]

                )

            # ================================================================

            # TIMEBASE & TRIGGER TAB

            # ================================================================

            with gr.Tab("Timebase & Trigger"):

                gr.Markdown("### Horizontal Timebase Configuration")

                with gr.Row():

                    time_scale = gr.Dropdown(

                        label="Time/div",

                        choices=self.timebase_scales,

                        value=10e-3

                    )

                    timebase_btn = gr.Button("Apply Timebase", variant="primary")

                timebase_status = gr.Textbox(label="Status", interactive=False)

                timebase_btn.click(

                    fn=self.configure_timebase,

                    inputs=[time_scale],

                    outputs=[timebase_status]

                )

                gr.Markdown("### Edge Trigger Configuration")

                with gr.Row():

                    trigger_source = gr.Dropdown(

                        label="Source",

                        choices=["CH1", "CH2", "CH3", "CH4"],

                        value="CH1"

                    )

                    trigger_level = gr.Number(label="Level (V)", value=0.0)

                    trigger_slope = gr.Dropdown(

                        label="Slope",

                        choices=["Rising", "Falling", "Either"],

                        value="Rising"

                    )

                trigger_btn = gr.Button("Apply Trigger", variant="primary")

                trigger_status = gr.Textbox(label="Status", interactive=False)

                trigger_btn.click(

                    fn=self.configure_trigger,

                    inputs=[trigger_source, trigger_level, trigger_slope],

                    outputs=[trigger_status]

                )

            # ================================================================

            # ADVANCED TRIGGERS TAB

            # ================================================================

            with gr.Tab("Advanced Triggers"):

                gr.Markdown("### Glitch (Spike) Trigger")

                gr.Markdown("Detects signal violations - pulses narrower than threshold")

                with gr.Row():

                    glitch_source = gr.Dropdown(

                        label="Source",

                        choices=["CH1", "CH2", "CH3", "CH4"],

                        value="CH1"

                    )

                    glitch_level = gr.Number(label="Level (V)", value=0.0)

                    glitch_polarity = gr.Dropdown(

                        label="Polarity",

                        choices=["POSitive", "NEGative"],

                        value="POSitive"

                    )

                    glitch_width = gr.Number(label="Width (ns)", value=1.0)

                glitch_btn = gr.Button("Set Glitch Trigger", variant="primary")

                glitch_status = gr.Textbox(label="Status", interactive=False)

                glitch_btn.click(

                    fn=self.set_glitch_trigger,

                    inputs=[glitch_source, glitch_level, glitch_polarity, glitch_width],

                    outputs=[glitch_status]

                )

                gr.Markdown("---")

                gr.Markdown("### Pulse Width Trigger")

                gr.Markdown("Triggers on pulses with width above or below threshold")

                with gr.Row():

                    pulse_source = gr.Dropdown(

                        label="Source",

                        choices=["CH1", "CH2", "CH3", "CH4"],

                        value="CH1"

                    )

                    pulse_level = gr.Number(label="Level (V)", value=0.0)

                    pulse_width = gr.Number(label="Width (ns)", value=10.0)

                    pulse_polarity = gr.Dropdown(

                        label="Polarity",

                        choices=["POSitive", "NEGative"],

                        value="POSitive"

                    )

                pulse_btn = gr.Button("Set Pulse Trigger", variant="primary")

                pulse_status = gr.Textbox(label="Status", interactive=False)

                pulse_btn.click(

                    fn=self.set_pulse_trigger,

                    inputs=[pulse_source, pulse_level, pulse_width, pulse_polarity],

                    outputs=[pulse_status]

                )

                gr.Markdown("---")

                gr.Markdown("### Trigger Sweep Mode")

                with gr.Row():

                    sweep_mode = gr.Dropdown(

                        label="Sweep Mode",

                        choices=["AUTO", "NORMal", "TRIG"],

                        value="AUTO",

                        info="AUTO: Continuous, NORMal: Wait for trigger, TRIG: Special mode"

                    )

                    sweep_btn = gr.Button("Apply Sweep Mode", variant="primary")

                sweep_status = gr.Textbox(label="Status", interactive=False)

                sweep_btn.click(

                    fn=self.set_trigger_sweep_mode,

                    inputs=[sweep_mode],

                    outputs=[sweep_status]

                )

                gr.Markdown("---")

                gr.Markdown("### Trigger Holdoff")

                gr.Markdown("Prevents re-triggering for specified time after trigger event")

                with gr.Row():

                    holdoff_time = gr.Number(label="Holdoff (ns)", value=100.0)

                    holdoff_btn = gr.Button("Apply Holdoff", variant="primary")

                holdoff_status = gr.Textbox(label="Status", interactive=False)

                holdoff_btn.click(

                    fn=self.set_trigger_holdoff,

                    inputs=[holdoff_time],

                    outputs=[holdoff_status]

                )

            # ================================================================

            # ACQUISITION CONTROL TAB

            # ================================================================

            with gr.Tab("Acquisition Control"):

                gr.Markdown("### Acquisition Mode")

                with gr.Row():

                    acq_mode = gr.Dropdown(

                        label="Mode",

                        choices=["RTIMe", "ETIMe", "SEGMented"],

                        value="RTIMe",

                        info="RTIMe: Real-time, ETIMe: Equivalent-time, SEGMented: Multi-event capture"

                    )

                    acq_mode_btn = gr.Button("Apply Mode", variant="primary")

                acq_mode_status = gr.Textbox(label="Status", interactive=False)

                acq_mode_btn.click(

                    fn=self.set_acquisition_mode,

                    inputs=[acq_mode],

                    outputs=[acq_mode_status]

                )

                gr.Markdown("Acquisition Type")

                with gr.Row():

                    acq_type = gr.Dropdown(

                        label="Type",

                        choices=["NORMal", "AVERage", "HRESolution", "PEAK"],

                        value="NORMal",

                        info="NORMal: Standard, AVERage: Noise reduction, HRESolution: Better resolution, PEAK: Transient capture"

                    )

                    acq_type_btn = gr.Button("Apply Type", variant="primary")

                acq_type_status = gr.Textbox(label="Status", interactive=False)

                acq_type_btn.click(

                    fn=self.set_acquisition_type,

                    inputs=[acq_type],

                    outputs=[acq_type_status]

                )

                gr.Markdown("### Averaging Configuration")

                with gr.Row():

                    avg_count = gr.Slider(label="Averaging Count", minimum=2, maximum=65536, value=16, step=1)

                    avg_btn = gr.Button("Apply Averaging", variant="primary")

                avg_status = gr.Textbox(label="Status", interactive=False)

                avg_btn.click(

                    fn=self.set_acquisition_count,

                    inputs=[avg_count],

                    outputs=[avg_status]

                )

                gr.Markdown("### Acquisition Information")

                info_btn = gr.Button("Query Info", variant="secondary")

                acq_info = gr.Textbox(label="Acquisition Info", interactive=False, lines=6)

                info_btn.click(

                    fn=self.query_acquisition_info,

                    inputs=[],

                    outputs=[acq_info]

                )

            # ================================================================

            # MARKERS & CURSORS TAB

            # ================================================================

            with gr.Tab("Markers & Cursors"):

                gr.Markdown("### Marker Mode Configuration")

                with gr.Row():

                    marker_mode = gr.Dropdown(

                        label="Mode",

                        choices=["OFF", "MEASurement", "MANual", "WAVeform"],

                        value="MEASurement"

                    )

                    marker_mode_btn = gr.Button("Set Mode", variant="primary")

                marker_mode_status = gr.Textbox(label="Status", interactive=False)

                marker_mode_btn.click(

                    fn=self.set_marker_mode,

                    inputs=[marker_mode],

                    outputs=[marker_mode_status]

                )

                gr.Markdown("### Marker Positions")

                with gr.Row():

                    marker_num = gr.Dropdown(

                        label="Marker",

                        choices=[1, 2],

                        value=1

                    )

                    marker_x = gr.Number(label="X Position (s)", value=0.0)

                    marker_y = gr.Number(label="Y Position (V)", value=0.0)

                    marker_set_btn = gr.Button("Set Position", variant="primary")

                marker_set_status = gr.Textbox(label="Status", interactive=False)

                marker_set_btn.click(

                    fn=self.set_marker_positions,

                    inputs=[marker_num, marker_x, marker_y],

                    outputs=[marker_set_status]

                )

                gr.Markdown("### Marker Delta Measurements")

                gr.Markdown("Query time and voltage differences between marker 1 and marker 2")

                delta_btn = gr.Button("Get Delta Values", variant="primary")

                delta_result = gr.Textbox(label="Delta Results", interactive=False, lines=4)

                delta_btn.click(

                    fn=self.get_marker_deltas,

                    inputs=[],

                    outputs=[delta_result]

                )

            # ================================================================

            # MATH FUNCTIONS TAB

            # ================================================================

            with gr.Tab("Math Functions"):

                gr.Markdown("### Math Function Configuration")

                with gr.Row():

                    math_func = gr.Dropdown(

                        label="Function Slot",

                        choices=[1, 2, 3, 4],

                        value=1

                    )

                    math_op = gr.Dropdown(

                        label="Operation",

                        choices=["ADD", "SUBTract", "MULTiply", "DIVide", "FFT"],

                        value="ADD"

                    )

                    math_src1 = gr.Dropdown(

                        label="Source 1",

                        choices=[1, 2, 3, 4],

                        value=1

                    )

                    math_src2 = gr.Dropdown(

                        label="Source 2",

                        choices=[1, 2, 3, 4],

                        value=2

                    )

                math_config_btn = gr.Button("Configure", variant="primary")

                math_config_status = gr.Textbox(label="Status", interactive=False)

                math_config_btn.click(

                    fn=self.configure_math_operation,

                    inputs=[math_func, math_op, math_src1, math_src2],

                    outputs=[math_config_status]

                )

                gr.Markdown("### Math Function Display")

                with gr.Row():

                    math_display_func = gr.Dropdown(

                        label="Function Slot",

                        choices=[1, 2, 3, 4],

                        value=1

                    )

                    math_show = gr.Checkbox(label="Show on Display", value=True)

                    math_display_btn = gr.Button("Apply Display", variant="primary")

                math_display_status = gr.Textbox(label="Status", interactive=False)

                math_display_btn.click(

                    fn=self.toggle_math_display,

                    inputs=[math_display_func, math_show],

                    outputs=[math_display_status]

                )

                gr.Markdown("### Math Function Scale")

                with gr.Row():

                    math_scale_func = gr.Dropdown(

                        label="Function Slot",

                        choices=[1, 2, 3, 4],

                        value=1

                    )

                    math_scale_val = gr.Number(label="Scale Value", value=1.0)

                    math_scale_btn = gr.Button("Apply Scale", variant="primary")

                math_scale_status = gr.Textbox(label="Status", interactive=False)

                math_scale_btn.click(

                    fn=self.set_math_scale,

                    inputs=[math_scale_func, math_scale_val],

                    outputs=[math_scale_status]

                )

            # ================================================================

            # SETUP MANAGEMENT TAB

            # ================================================================

            with gr.Tab("Setup Management"):

                gr.Markdown("### Save Instrument Configuration")

                with gr.Row():

                    setup_save_name = gr.Textbox(label="Setup Name", value="my_setup")

                    setup_save_btn = gr.Button("Save Setup", variant="primary")

                setup_save_status = gr.Textbox(label="Status", interactive=False)

                setup_save_btn.click(

                    fn=self.save_instrument_setup,

                    inputs=[setup_save_name],

                    outputs=[setup_save_status]

                )

                gr.Markdown("### Recall Instrument Configuration")

                with gr.Row():

                    setup_recall_name = gr.Textbox(label="Setup Name", value="my_setup")

                    setup_recall_btn = gr.Button("Recall Setup", variant="primary")

                setup_recall_status = gr.Textbox(label="Status", interactive=False)

                setup_recall_btn.click(

                    fn=self.recall_instrument_setup,

                    inputs=[setup_recall_name],

                    outputs=[setup_recall_status]

                )

                gr.Markdown("---")

                gr.Markdown("### Save Waveform Data")

                with gr.Row():

                    wf_save_ch = gr.Dropdown(label="Channel", choices=[1, 2, 3, 4], value=1)

                    wf_save_name = gr.Textbox(label="Waveform Name", value="waveform_data")

                    wf_save_btn = gr.Button("Save Waveform", variant="primary")

                wf_save_status = gr.Textbox(label="Status", interactive=False)

                wf_save_btn.click(

                    fn=self.save_waveform_to_memory,

                    inputs=[wf_save_ch, wf_save_name],

                    outputs=[wf_save_status]

                )

                gr.Markdown("### Recall Waveform Data")

                with gr.Row():

                    wf_recall_name = gr.Textbox(label="Waveform Name", value="waveform_data")

                    wf_recall_btn = gr.Button("Recall Waveform", variant="primary")

                wf_recall_status = gr.Textbox(label="Status", interactive=False)

                wf_recall_btn.click(

                    fn=self.recall_waveform_from_memory,

                    inputs=[wf_recall_name],

                    outputs=[wf_recall_status]

                )

            # ================================================================

            # FUNCTION GENERATORS TAB

            # ================================================================

            with gr.Tab("Function Generators"):

                gr.Markdown("### WGEN1 Configuration")

                with gr.Row():

                    wgen1_enable = gr.Checkbox(label="Enable", value=False)

                    wgen1_waveform = gr.Dropdown(

                        label="Waveform",

                        choices=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"],

                        value="SIN"

                    )

                    wgen1_freq = gr.Number(label="Frequency (Hz)", value=1000.0)

                    wgen1_amp = gr.Number(label="Amplitude (Vpp)", value=1.0)

                    wgen1_offset = gr.Number(label="Offset (V)", value=0.0)

                wgen1_btn = gr.Button("Apply WGEN1", variant="primary")

                wgen1_status = gr.Textbox(label="Status", interactive=False)

                wgen1_info_btn = gr.Button("Query Config", variant="secondary")

                wgen1_info = gr.Textbox(label="WGEN1 Info", interactive=False, lines=5)

                wgen1_btn.click(

                    fn=lambda en, wf, fr, am, of: self.configure_wgen(1, en, wf, fr, am, of),

                    inputs=[wgen1_enable, wgen1_waveform, wgen1_freq, wgen1_amp, wgen1_offset],

                    outputs=[wgen1_status]

                )

                wgen1_info_btn.click(

                    fn=lambda: self.get_wgen_configuration(1),

                    inputs=[],

                    outputs=[wgen1_info]

                )

                gr.Markdown("### WGEN2 Configuration")

                with gr.Row():

                    wgen2_enable = gr.Checkbox(label="Enable", value=False)

                    wgen2_waveform = gr.Dropdown(

                        label="Waveform",

                        choices=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"],

                        value="SIN"

                    )

                    wgen2_freq = gr.Number(label="Frequency (Hz)", value=1000.0)

                    wgen2_amp = gr.Number(label="Amplitude (Vpp)", value=1.0)

                    wgen2_offset = gr.Number(label="Offset (V)", value=0.0)

                wgen2_btn = gr.Button("Apply WGEN2", variant="primary")

                wgen2_status = gr.Textbox(label="Status", interactive=False)

                wgen2_info_btn = gr.Button("Query Config", variant="secondary")

                wgen2_info = gr.Textbox(label="WGEN2 Info", interactive=False, lines=5)

                wgen2_btn.click(

                    fn=lambda en, wf, fr, am, of: self.configure_wgen(2, en, wf, fr, am, of),

                    inputs=[wgen2_enable, wgen2_waveform, wgen2_freq, wgen2_amp, wgen2_offset],

                    outputs=[wgen2_status]

                )

                wgen2_info_btn.click(

                    fn=lambda: self.get_wgen_configuration(2),

                    inputs=[],

                    outputs=[wgen2_info]

                )

            # ================================================================

            # MEASUREMENTS TAB

            # ================================================================

            with gr.Tab("Measurements"):

                gr.Markdown("### Single Measurement")

                with gr.Row():

                    source_choices = [

                        ("Channel 1", "CH1"),

                        ("Channel 2", "CH2"),

                        ("Channel 3", "CH3"),

                        ("Channel 4", "CH4"),

                        ("Math 1", "MATH1"),

                        ("Math 2", "MATH2"),

                        ("Math 3", "MATH3"),

                        ("Math 4", "MATH4")

                    ]

                    meas_source = gr.Dropdown(

                        label="Source",

                        choices=source_choices,

                        value="CH1"

                    )

                    # Define measurement choices as a list of tuples without type annotation
                    measurement_choices = [
                        ("Frequency", "FREQ"),
                        ("Period", "PERiod"),
                        ("Peak-to-Peak", "VPP"),
                        ("Amplitude", "VAMP"),
                        ("Overshoot", "OVERshoot"),
                        ("Top", "VTOP"),
                        ("Base", "VBASe"),
                        ("Average", "VAVG"),
                        ("RMS", "VRMS"),
                        ("Maximum", "VMAX"),
                        ("Minimum", "VMIN"),
                        ("Rise Time", "RISE"),
                        ("Fall Time", "FALL"),
                        ("Duty Cycle", "DUTYcycle"),
                        ("Negative Duty Cycle", "NDUTy")
                    ]
                    
                    measurement_type = gr.Dropdown(
                        label="Measurement Type",
                        choices=measurement_choices,
                        value="FREQ"
                    )
                    
                    measure_btn = gr.Button("Measure", variant="primary")
                    all_measurements_btn = gr.Button("Show All", variant="primary")
                
                    measurement_result = gr.Textbox(label="Measurement Result", interactive=False)
                    all_measurements_result = gr.Textbox(
                        label="All Measurements",
                        interactive=False,
                        lines=10,
                        max_lines=20,
                        show_copy_button=True
                    )
                    
                    measure_btn.click(
                        self.perform_measurement,
                        inputs=[meas_source, measurement_type],
                        outputs=measurement_result
                    )
                    
                    all_measurements_btn.click(
                        self.get_all_measurements,
                        inputs=[meas_source],
                        outputs=all_measurements_result
                    )
                
            # ================================================================
            # End of Measurements tab
                
            # Create a new tab for Operations & File Management
            with gr.Tab("Operations & File Management"):
                with gr.Column(variant="panel"):
                    gr.Markdown("### File Save Locations (Server-Side)")
                    gr.Markdown("Files are saved on the server in the following directories. Use the download buttons below to get files after generation.")

                    # Display-only path information
                    with gr.Group():
                        gr.Textbox(
                            label="Data Directory (CSV files)",
                            value=self.save_locations['data'],
                            interactive=False
                        )
                        gr.Textbox(
                            label="Graphs Directory (PNG files)",
                            value=self.save_locations['graphs'],
                            interactive=False
                        )
                        gr.Textbox(
                            label="Screenshots Directory (PNG files)",
                            value=self.save_locations['screenshots'],
                            interactive=False
                        )
                
                # Data Acquisition and Export section
                gr.Markdown("### Data Acquisition and Export")

                with gr.Row():

                    op_ch1 = gr.Checkbox(label="Ch1", value=True)

                    op_ch2 = gr.Checkbox(label="Ch2", value=False)

                    op_ch3 = gr.Checkbox(label="Ch3", value=False)

                    op_ch4 = gr.Checkbox(label="Ch4", value=False)

                with gr.Row():

                    op_math1 = gr.Checkbox(label="Math1", value=False)

                    op_math2 = gr.Checkbox(label="Math2", value=False)

                    op_math3 = gr.Checkbox(label="Math3", value=False)

                    op_math4 = gr.Checkbox(label="Math4", value=False)

                plot_title_input = gr.Textbox(

                    label="Plot Title (optional)",

                    placeholder="Enter custom plot title"

                )

                with gr.Row():

                    screenshot_btn = gr.Button("Capture Screenshot", variant="secondary")

                    acquire_btn = gr.Button("Acquire Data", variant="primary")

                    export_btn = gr.Button("Export CSV", variant="secondary")

                    plot_btn = gr.Button("Generate Plot", variant="secondary")

                with gr.Row():

                    full_auto_btn = gr.Button("Full Automation", variant="primary", scale=2)

                operation_status = gr.Textbox(label="Operation Status", interactive=False, lines=10)

                # Download section
                gr.Markdown("### Download Files")
                gr.Markdown("After capturing screenshots or exporting data, download the files below:")

                with gr.Row():
                    screenshot_download = gr.File(label="Latest Screenshot", interactive=False)
                    csv_download = gr.File(label="Exported CSV Files", interactive=False, file_count="multiple")

                screenshot_btn.click(

                    fn=self.capture_screenshot,

                    inputs=[],

                    outputs=[operation_status, screenshot_download]

                )

                acquire_btn.click(

                    fn=self.acquire_data,

                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4],

                    outputs=[operation_status]

                )

                export_btn.click(

                    fn=self.export_csv,

                    inputs=[],

                    outputs=[operation_status, csv_download]

                )

                plot_btn.click(

                    fn=self.generate_plot,

                    inputs=[plot_title_input],

                    outputs=[operation_status]

                )

                full_auto_btn.click(

                    fn=self.run_full_automation,

                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4, plot_title_input],

                    outputs=[operation_status, screenshot_download, csv_download]

                )

            gr.Markdown("---")

            gr.Markdown("**DIGANTARA Oscilloscope Control** | Professional Grade Instrumentation | All SCPI Commands Verified")

        return interface

    def launch(self, share=False, server_port=7860, auto_open=True, max_attempts=10):

        """Launch Gradio interface with port fallback and full-page layout"""

        self._gradio_interface = self.create_interface()

        for attempt in range(max_attempts):

            current_port = server_port + attempt

            try:

                print(f"Attempting to start server on port {current_port}...")

                self._gradio_interface.launch(

                    server_name="0.0.0.0",

                    share=share,

                    server_port=current_port,

                    #inbrowser=auto_open if attempt == 0 else False,

                    prevent_thread_lock=False,

                    show_error=True,

                    quiet=False

                )

                print("\n" + "=" * 80)

                print(f"Server is running on port {current_port}")

                print("To stop the application, press Ctrl+C in this terminal.")

                print("=" * 80)

                return

            except Exception as e:

                if "address already in use" in str(e).lower() or "port in use" in str(e).lower():

                    print(f"Port {current_port} is in use, trying next port...")

                    if attempt == max_attempts - 1:

                        print(f"\nError: Could not find an available port after {max_attempts} attempts.")

                        print("Please close any other instances or specify a different starting port.")

                        self.cleanup()

                        return

                else:

                    print(f"\nLaunch error: {e}")

                    self.cleanup()

                    return

        print("\nFailed to start the server after multiple attempts.")

        self.cleanup()

def main():

    """Application entry point"""

    print("Keysight Oscilloscope Automation - Enhanced Gradio Interface")

    print("Professional oscilloscope control system with advanced features")

    print("=" * 80)

    print("Starting web interface...")

    app = None

    try:

        start_port = 7864

        max_attempts = 10

        print(f"Looking for an available port starting from {start_port}...")

        for port in range(start_port, start_port + max_attempts):

            try:

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                    s.bind(('', port))

                    s.close()

                print(f"\nFound available port: {port}")

                print("The browser will open automatically when ready.")

                print("")

                print("IMPORTANT: To stop the application, press Ctrl+C in this terminal.")

                print("Closing the browser tab will NOT stop the server.")

                print("=" * 80)

                app = GradioOscilloscopeGUI()

                app.launch(share=False, server_port=port, auto_open=True)

                break

            except OSError as e:

                if "address already in use" in str(e).lower():

                    print(f"Port {port} is in use, trying next port...")

                    if port == start_port + max_attempts - 1:

                        print(f"\nError: Could not find an available port after {max_attempts} attempts.")

                        print("Please close any applications using ports {}-{}" \

                              .format(start_port, start_port + max_attempts - 1))

                        return

                else:

                    print(f"Error checking port {port}: {e}")

                    return

    except KeyboardInterrupt:

        print("\nApplication closed by user.")

    except Exception as e:

        print(f"Error: {e}")

    finally:

        if app:

            app.cleanup()

        print("\nApplication shutdown complete.")

        print("=" * 80)

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nApplication terminated by user.")

    except Exception as e:

        print(f"Fatal error: {e}")

    finally:

        print("Forcing application exit...")

        os._exit(0)
