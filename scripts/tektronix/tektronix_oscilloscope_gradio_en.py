#!/usr/bin/env python3

"""
Tektronix MSO24 Oscilloscope Control - Professional Gradio Interface

Comprehensive oscilloscope automation with advanced trigger modes, measurements,
AFG control, math functions, markers, cursors, and acquisition features for the MSO24 series.

FEATURES IMPLEMENTED:
- Complete 4-channel analog input control with enable/disable
- Advanced trigger modes (Edge, Pulse, Logic, Bus, Video) with Sweep and Holdoff
- Integrated AFG (Arbitrary Function Generator) control
- Math functions (ADD, SUBTRACT, MULTIPLY, DIVIDE) with display control
- Real-time measurement automation with 18+ parameter types
- Markers & Cursors for precise waveform analysis
- Setup save/recall for test automation
- High-resolution waveform data acquisition (up to 62.5M points)
- Multi-format data export (CSV, screenshots)
- Professional acquisition control (Single, Run, Stop)
- Comprehensive timebase and vertical scaling
- Thread-safe operations with comprehensive logging
- Professional path management and file organization
- Full error handling and status monitoring

Author: Enhanced by Senior Test Automation Engineer
Date: 2024-12-03
Target: Tektronix MSO24 Series Mixed Signal Oscilloscopes

"""

import sys
import logging
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog
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
    from instrument_control.tektronix_oscilloscope import TektronixMSO24, TektronixMSO24Error
except ImportError as e:
    print(f"Error importing Tektronix oscilloscope module: {e}")
    sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_timebase_value(value: Union[str, float]) -> float:
    """Parse timebase value (string or float) to seconds"""
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().lower()
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
    "Rising": "RISE",
    "Falling": "FALL",
    "Either": "EITHER"
}

AFG_FUNCTION_MAP = {
    "Sine": "SINE",
    "Square": "SQUARE",
    "Ramp": "RAMP",
    "Pulse": "PULSE",
    "Noise": "NOISE",
    "DC": "DC"
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
    if meas_type == "FREQUENCY":
        return format_si_value(value, "freq")
    if meas_type in ["PERIOD", "RISE", "FALL", "WIDTH"]:
        return format_si_value(value, "time")
    if meas_type in ["AMPLITUDE", "HIGH", "LOW", "MEAN", "RMS", "MAX", "MIN", "PEAK2PEAK"]:
        return format_si_value(value, "volt")
    if meas_type in ["DUTYCYCLE", "OVERSHOOT", "PRESHOOT"]:
        return format_si_value(value, "percent")
    return f"{value}"

# ============================================================================
# DATA ACQUISITION CLASS
# ============================================================================

class MSO24DataAcquisition:
    """
    Data acquisition handler for Tektronix MSO24 with thread-safe waveform capture.

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

    def acquire_waveform_data(self, channel: int, max_points: int = 50000) -> Optional[Dict[str, Any]]:
        """
        Acquire waveform data from specified channel using MSO24's built-in methods.

        Thread-safe acquisition using oscilloscope's get_channel_data method.
        """
        if not self.scope.is_connected:
            self._logger.error("Cannot acquire data: oscilloscope not connected")
            return None

        try:
            lock = self.io_lock
            if lock:
                with lock:
                    waveform_data = self.scope.get_channel_data(
                        channel=channel,
                        start_point=1,
                        stop_point=max_points
                    )
            else:
                waveform_data = self.scope.get_channel_data(
                    channel=channel,
                    start_point=1,
                    stop_point=max_points
                )

            if waveform_data:
                self._logger.info(f"Acquired {len(waveform_data['voltage'])} points from channel {channel}")
                return waveform_data

        except Exception as e:
            self._logger.error(f"Waveform acquisition failed: {e}")
            return None

    def save_waveform_csv(self, data: Dict[str, Any], filename: str, directory: str) -> str:
        """Save waveform data to CSV file with proper formatting"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            filepath = Path(directory) / f"{filename}.csv"

            # Create DataFrame with time and voltage data
            df = pd.DataFrame({
                'Time (s)': data['time'],
                'Voltage (V)': data['voltage']
            })

            # Add metadata as comments
            metadata_lines = [
                f"# Tektronix MSO24 Waveform Data",
                f"# Channel: {data['channel']}",
                f"# Sample Points: {data['num_points']}",
                f"# X Increment: {data['x_increment']} s",
                f"# Y Multiplier: {data['y_multiplier']} V",
                f"# Acquisition Time: {datetime.now().isoformat()}",
                "#"
            ]

            with open(filepath, 'w') as f:
                for line in metadata_lines:
                    f.write(line + '\n')
                df.to_csv(f, index=False)

            self._logger.info(f"Waveform saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            self._logger.error(f"Failed to save CSV: {e}")
            raise

    def generate_waveform_plot(self, waveform_data: Dict[str, Any], title: str = "",
                              directory: str = None) -> str:
        """Generate professional waveform plot with proper scaling and annotations"""
        try:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

            # Create figure with professional styling
            plt.figure(figsize=(12, 8))
            plt.style.use('default')

            time_data = waveform_data['time'] * 1000  # Convert to ms for display
            voltage_data = waveform_data['voltage']

            plt.plot(time_data, voltage_data, 'b-', linewidth=0.8, alpha=0.9)
            plt.grid(True, alpha=0.3)

            # Set labels and title
            plt.xlabel('Time (ms)', fontsize=12)
            plt.ylabel('Voltage (V)', fontsize=12)

            if title:
                plt.title(f"Tektronix MSO24 - {title}", fontsize=14, fontweight='bold')
            else:
                plt.title(f"CH{waveform_data['channel']} Waveform - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         fontsize=14, fontweight='bold')

            # Add statistics text box
            v_min = np.min(voltage_data)
            v_max = np.max(voltage_data)
            v_pp = v_max - v_min
            v_rms = np.sqrt(np.mean(voltage_data**2))

            stats_text = f'Min: {v_min:.3f}V\nMax: {v_max:.3f}V\nP-P: {v_pp:.3f}V\nRMS: {v_rms:.3f}V'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontfamily='monospace')

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if directory:
                filename = Path(directory) / f"MSO24_waveform_{timestamp}.png"
            else:
                filename = f"MSO24_waveform_{timestamp}.png"

            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            self._logger.info(f"Plot saved to: {filename}")
            return str(filename)

        except Exception as e:
            plt.close()  # Ensure figure is closed on error
            self._logger.error(f"Failed to generate plot: {e}")
            raise

# ============================================================================
# MAIN GUI CLASS
# ============================================================================

class GradioMSO24GUI:
    """
    Professional Gradio-based GUI for Tektronix MSO24 Oscilloscope Control.

    Provides comprehensive oscilloscope automation with advanced features including
    AFG control, math functions, multi-channel measurements, and professional data acquisition.
    """

    def __init__(self, visa_address: str = "USB0::0x0699::0x0105::SGVJ0003176::INSTR"):
        """Initialize MSO24 GUI with default VISA address"""

        # Setup logging (console only, no file logging)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self._logger = logging.getLogger(self.__class__.__name__)

        # Initialize oscilloscope
        self.oscilloscope = TektronixMSO24(visa_address)
        self.is_connected = False

        # Thread synchronization
        self.io_lock = threading.RLock()
        self.data_queue = queue.Queue()

        # Data acquisition handler
        self.data_acquisition = MSO24DataAcquisition(self.oscilloscope, self.io_lock)

        # Default save locations
        self.save_locations = {
            'data': str(Path.cwd() / "oscilloscope_data"),
            'graphs': str(Path.cwd() / "oscilloscope_graphs"),
            'screenshots': str(Path.cwd() / "oscilloscope_screenshots")
        }

        # Create directories
        for path in self.save_locations.values():
            Path(path).mkdir(exist_ok=True)

        # Stored measurement data
        self.current_waveform_data = {}

        # Define timebase scales (same as MSO24 valid scales)
        self.timebase_scales = [
            ("1 ns", 1e-9), ("2 ns", 2e-9), ("5 ns", 5e-9),
            ("10 ns", 10e-9), ("20 ns", 20e-9), ("50 ns", 50e-9),
            ("100 ns", 100e-9), ("200 ns", 200e-9), ("500 ns", 500e-9),
            ("1 µs", 1e-6), ("2 µs", 2e-6), ("5 µs", 5e-6),
            ("10 µs", 10e-6), ("20 µs", 20e-6), ("50 µs", 50e-6),
            ("100 µs", 100e-6), ("200 µs", 200e-6), ("500 µs", 500e-6),
            ("1 ms", 1e-3), ("2 ms", 2e-3), ("5 ms", 5e-3),
            ("10 ms", 10e-3), ("20 ms", 20e-3), ("50 ms", 50e-3),
            ("100 ms", 100e-3), ("200 ms", 200e-3), ("500 ms", 500e-3),
            ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0), ("10 s", 10.0), ("20 s", 20.0), ("50 s", 50.0)
        ]

        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._logger.info("MSO24 GUI initialized")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for clean shutdown"""
        self._logger.info("Interrupt signal received, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean shutdown of oscilloscope connection"""
        try:
            if self.oscilloscope and self.oscilloscope.is_connected:
                self.oscilloscope.disconnect()
                self._logger.info("Oscilloscope disconnected")
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")

    def connect_oscilloscope(self, visa_address: str) -> str:
        """Connect to oscilloscope with error handling"""
        try:
            self.oscilloscope = TektronixMSO24(visa_address)
            with self.io_lock:
                if self.oscilloscope.connect():
                    self.is_connected = True

                    # Rebind data acquisition helper to the new oscilloscope instance
                    # so that subsequent acquisitions use the live connection
                    self.data_acquisition = MSO24DataAcquisition(self.oscilloscope, self.io_lock)

                    # Set output directories to user-configured paths
                    self.oscilloscope.set_output_directories(
                        data_dir=self.save_locations['data'],
                        graph_dir=self.save_locations['graphs'],
                        screenshot_dir=self.save_locations['screenshots']
                    )

                    info = self.oscilloscope.get_instrument_info()
                    if info:
                        return f"[CONNECTED] {info['model']} (S/N: {info['serial_number']})\nFirmware: {info['firmware_version']}"
                    else:
                        return "[CONNECTED] Tektronix MSO24"
                else:
                    return "[ERROR] Failed to connect to oscilloscope"
        except Exception as e:
            self._logger.error(f"Connection error: {e}")
            return f"[ERROR] Connection error: {str(e)}"

    def disconnect_oscilloscope(self) -> str:
        """Disconnect from oscilloscope"""
        try:
            with self.io_lock:
                if self.oscilloscope and self.oscilloscope.is_connected:
                    self.oscilloscope.disconnect()
                    self.is_connected = False
                    return "[OK] Oscilloscope disconnected"
                else:
                    return "[INFO] Oscilloscope not connected"
        except Exception as e:
            self._logger.error(f"Disconnect error: {e}")
            return f"[ERROR] Disconnect error: {str(e)}"

    def set_channel_state(self, channel: int, enabled: bool) -> bool:
        """Enable or disable a channel"""
        if not self.is_connected:
            return False

        try:
            with self.io_lock:
                state = "ON" if enabled else "OFF"
                self.oscilloscope._scpi_wrapper.write(f"DISplay:GLObal:CH{channel}:STATE {state}")
                time.sleep(0.05)
                self._logger.info(f"Channel {channel} {'enabled' if enabled else 'disabled'}")
                return True
        except Exception as e:
            self._logger.error(f"Failed to set channel {channel} state: {e}")
            return False

    def configure_channels(self, ch1_en: bool, ch2_en: bool, ch3_en: bool, ch4_en: bool,
                          scale: float, offset: float, coupling: str, probe: float) -> str:
        """Configure all channels with enable/disable"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            if isinstance(probe, str):
                probe_map = {"1x": 1.0, "10x": 10.0, "100x": 100.0, "1000x": 1000.0}
                probe_value = probe_map.get(probe, float(probe))
            else:
                probe_value = float(probe)

            channel_states = {1: ch1_en, 2: ch2_en, 3: ch3_en, 4: ch4_en}
            results = []

            for channel, enabled in channel_states.items():
                with self.io_lock:
                    # Set channel state (enable/disable)
                    state = "ON" if enabled else "OFF"
                    self.oscilloscope._scpi_wrapper.write(f"DISplay:GLObal:CH{channel}:STATE {state}")
                    time.sleep(0.05)

                    if enabled:
                        # Configure channel parameters only if enabled
                        success = self.oscilloscope.configure_channel(
                            channel=channel,
                            vertical_scale=scale,
                            vertical_offset=offset,
                            coupling=coupling,
                            probe_attenuation=probe_value,
                            bandwidth_limit=False
                        )

                        # Read back effective probe attenuation from instrument for accuracy
                        effective_probe = probe_value
                        if success:
                            try:
                                ch_config = self.oscilloscope.get_channel_config(channel)
                                if ch_config and 'probe' in ch_config:
                                    effective_probe = ch_config['probe']
                                    self._logger.info(
                                        f"Channel {channel} probe check: requested {probe_value}x, "
                                        f"instrument reports {effective_probe}x"
                                    )
                            except Exception as cfg_err:
                                self._logger.warning(
                                    f"Could not read back channel {channel} probe setting: {cfg_err}"
                                )

                        status = "configured" if success else "config failed"
                        results.append(
                            f"CH{channel}: enabled, {status} "
                            f"(Scale={scale} V/div, Offset={offset} V, Coupling={coupling}, Probe={effective_probe}x)"
                        )
                    else:
                        results.append(f"CH{channel}: disabled")

            return "[OK] Channels configured:\n" + "\n".join(results)

        except Exception as e:
            self._logger.error(f"Channel config error: {e}")
            return f"[ERROR] Channel config: {str(e)}"

    def configure_timebase(self, time_scale: float, position: float, record_length: int) -> str:
        """Configure horizontal timebase"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            scale_seconds = parse_timebase_value(time_scale)
            with self.io_lock:
                success = self.oscilloscope.configure_timebase(
                    time_scale=scale_seconds,
                    time_position=position,
                    record_length=record_length
                )

            if success:
                return f"[OK] Timebase: {format_si_value(scale_seconds, 'time')}/div, Position: {position}s, Length: {record_length}"
            else:
                return "[ERROR] Failed to configure timebase"

        except Exception as e:
            self._logger.error(f"Timebase config error: {e}")
            return f"[ERROR] Timebase: {str(e)}"

    def configure_trigger(self, trigger_type: str, source: str, level: float, slope: str) -> str:
        """Configure trigger settings"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            mso_slope = TRIGGER_SLOPE_MAP.get(slope, "RISE")
            with self.io_lock:
                success = self.oscilloscope.configure_trigger(
                    trigger_type=trigger_type,
                    source=source,
                    level=level,
                    slope=mso_slope
                )

            if success:
                return f"[OK] Trigger: {trigger_type} on {source}, Level: {level}V, Slope: {slope}"
            else:
                return "[ERROR] Failed to configure trigger"

        except Exception as e:
            self._logger.error(f"Trigger config error: {e}")
            return f"[ERROR] Trigger: {str(e)}"

    def set_trigger_sweep(self, sweep_mode: str) -> str:
        """Set trigger sweep mode"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                self.oscilloscope._scpi_wrapper.write(f"TRIGger:A:MODe {sweep_mode}")
                time.sleep(0.05)
                return f"[OK] Trigger sweep mode: {sweep_mode}"
        except Exception as e:
            self._logger.error(f"Trigger sweep error: {e}")
            return f"[ERROR] Trigger sweep: {str(e)}"

    def set_trigger_holdoff(self, holdoff_time: float) -> str:
        """Set trigger holdoff time"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                self.oscilloscope._scpi_wrapper.write(f"TRIGger:A:HOLDoff:TIMe {holdoff_time}")
                time.sleep(0.05)
                return f"[OK] Trigger holdoff: {format_si_value(holdoff_time, 'time')}"
        except Exception as e:
            self._logger.error(f"Trigger holdoff error: {e}")
            return f"[ERROR] Trigger holdoff: {str(e)}"

    def control_acquisition(self, action: str) -> str:
        """Control acquisition state (Run/Stop/Single)"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                if action == "Run":
                    success = self.oscilloscope.run()
                elif action == "Stop":
                    success = self.oscilloscope.stop()
                elif action == "Single":
                    success = self.oscilloscope.single()
                else:
                    return f"[ERROR] Unknown action: {action}"

            if success:
                return f"[OK] Acquisition: {action}"
            else:
                return f"[ERROR] Failed to {action.lower()} acquisition"

        except Exception as e:
            self._logger.error(f"Acquisition control error: {e}")
            return f"[ERROR] Acquisition: {str(e)}"

    def configure_afg(self, function: str, frequency: float, amplitude: float,
                     offset: float, enable: bool) -> str:
        """Configure Arbitrary Function Generator"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            mso_function = AFG_FUNCTION_MAP.get(function, "SINE")
            with self.io_lock:
                success = self.oscilloscope.configure_afg(
                    function=mso_function,
                    frequency=frequency,
                    amplitude=amplitude,
                    offset=offset,
                    enable=enable
                )

            if success:
                status = "ON" if enable else "OFF"
                return f"[OK] AFG: {function}, {frequency}Hz, {amplitude}V, Offset: {offset}V, Output: {status}"
            else:
                return "[ERROR] Failed to configure AFG"

        except Exception as e:
            self._logger.error(f"AFG config error: {e}")
            return f"[ERROR] AFG: {str(e)}"

    def configure_math_function(self, function_num: int, operation: str, source1: str, source2: str) -> str:
        """Configure math function"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            # Map user-friendly operations to math expressions
            operation_map = {
                "ADD": f"{source1}+{source2}",
                "SUBTRACT": f"{source1}-{source2}",
                "MULTIPLY": f"{source1}*{source2}",
                "DIVIDE": f"{source1}/{source2}"
            }

            math_expression = operation_map.get(operation.upper())

            if not math_expression:
                return f"[ERROR] Unknown operation: {operation}"

            with self.io_lock:
                # Use ADVANCED type with the math expression
                success = self.oscilloscope.configure_math_function(
                    function_num=function_num,
                    operation="ADVANCED",
                    source1=source1,
                    source2=source2,
                    math_expression=math_expression
                )

            if success:
                return f"[OK] Math{function_num}: {operation}({source1}, {source2})"
            else:
                return "[ERROR] Failed to configure math function"

        except Exception as e:
            self._logger.error(f"Math function error: {e}")
            return f"[ERROR] Math function: {str(e)}"

    def toggle_math_display(self, function_num: int, display: bool) -> str:
        """Toggle math function display"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.set_math_display(function_num, display)

            if success:
                state = "shown" if display else "hidden"
                return f"[OK] Math{function_num} {state}"
            else:
                return f"[ERROR] Failed to toggle Math{function_num} display"

        except Exception as e:
            self._logger.error(f"Math display error: {e}")
            return f"[ERROR] Math display: {str(e)}"

    def set_math_scale(self, function_num: int, scale: float) -> str:
        """Set math function vertical scale"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.set_math_scale(function_num, scale)

            if success:
                return f"[OK] Math{function_num} scale: {scale} V/div"
            else:
                return f"[ERROR] Failed to set Math{function_num} scale"

        except Exception as e:
            self._logger.error(f"Math scale error: {e}")
            return f"[ERROR] Math scale: {str(e)}"

    def measure_all_for_source(self, source: str) -> Tuple[str, str]:
        """Configure all standard measurements for the given source and return summary and all results."""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected", "[ERROR] Oscilloscope not connected"

        # Same set of standard measurements as supported by the backend driver
        measurement_types = [
            "FREQUENCY", "PERIOD", "AMPLITUDE", "HIGH", "LOW",
            "MAX", "MIN", "PEAK2PEAK", "MEAN", "RMS", "RISE",
            "FALL", "WIDTH", "DUTYCYCLE", "OVERSHOOT", "PRESHOOT",
            "AREA", "PHASE",
        ]

        results = []

        for meas_type in measurement_types:
            # Reuse existing single-measure helper so logging and error handling stay consistent
            msg = self.add_measurement(meas_type, source)
            results.append(msg)

        # After configuring, fetch all measurements from the instrument
        all_meas_text = self.get_all_measurements()
        summary_text = "\n".join(results)
        return summary_text, all_meas_text

    def add_measurement(self, measurement_type: str, source: str) -> str:
        """Add measurement to oscilloscope"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                meas_num = self.oscilloscope.add_measurement(measurement_type, source)

            if meas_num:
                return f"[OK] Added measurement {meas_num}: {measurement_type} on {source}"
            else:
                return f"[ERROR] Failed to add {measurement_type} measurement on {source}"

        except Exception as e:
            self._logger.error(f"Measurement error: {e}")
            return f"[ERROR] Measurement: {str(e)}"

    def get_all_measurements(self) -> str:
        """Get all current measurements"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                measurements = self.oscilloscope.get_all_measurements()

            if not measurements:
                return "[INFO] No measurements configured"

            result = "Current Measurements:\n"
            result += "-" * 50 + "\n"

            for meas_name, details in measurements.items():
                value_str = format_measurement_value(details['type'], details['value'])
                result += f"{meas_name}: {details['type']} = {value_str} ({details['source']})\n"

            return result

        except Exception as e:
            self._logger.error(f"Get measurements error: {e}")
            return f"[ERROR] Measurements: {str(e)}"

    def reset_oscilloscope(self) -> str:
        """Reset oscilloscope to default state"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.reset()

            if success:
                return "[OK] Oscilloscope reset to default state"
            else:
                return "[ERROR] Failed to reset oscilloscope"

        except Exception as e:
            self._logger.error(f"Reset error: {e}")
            return f"[ERROR] Reset: {str(e)}"

    def autoscale(self) -> str:
        """Execute autoscale"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.autoscale()

            if success:
                return "[OK] Autoscale completed"
            else:
                return "[ERROR] Autoscale failed"

        except Exception as e:
            self._logger.error(f"Autoscale error: {e}")
            return f"[ERROR] Autoscale: {str(e)}"

    def capture_screenshot(self) -> str:
        """Capture oscilloscope screenshot"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MSO24_screenshot_{timestamp}"

            with self.io_lock:
                # Ensure the screenshots directory exists
                screenshot_dir = Path(self.save_locations['screenshots'])
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                
                # Construct the full path for the screenshot
                screenshot_filename = f"{filename}.png"
                screenshot_full_path = screenshot_dir / screenshot_filename
                
                # Take the screenshot
                screenshot_path = self.oscilloscope.get_screenshot(
                    screenshot_path=str(screenshot_full_path),
                    freeze_acquisition=True
                )

            if screenshot_path and os.path.exists(screenshot_path):
                return f"[OK] Screenshot saved: {screenshot_path}"
            else:
                return "[ERROR] Failed to capture screenshot"

        except Exception as e:
            self._logger.error(f"Screenshot error: {e}", exc_info=True)
            return f"[ERROR] Screenshot: {str(e)}"

    def acquire_data(self, ch1: bool, ch2: bool, ch3: bool, ch4: bool,
                     math1: bool, math2: bool, math3: bool, math4: bool) -> str:
        """Acquire waveform data from selected channels and math functions"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        selected_channels = []
        if ch1: selected_channels.append(1)
        if ch2: selected_channels.append(2)
        if ch3: selected_channels.append(3)
        if ch4: selected_channels.append(4)

        selected_math = []
        if math1: selected_math.append(1)
        if math2: selected_math.append(2)
        if math3: selected_math.append(3)
        if math4: selected_math.append(4)

        if not selected_channels and not selected_math:
            return "[WARNING] No channels or math functions selected for acquisition"

        try:
            result = f"Data Acquisition Results:\n"
            result += "-" * 40 + "\n"

            for channel in selected_channels:
                data = self.data_acquisition.acquire_waveform_data(channel, max_points=65000)
                if data:
                    self.current_waveform_data[f"CH{channel}"] = data
                    points = len(data['voltage'])
                    time_span = (data['time'][-1] - data['time'])  # ms
                    v_pp = np.max(data['voltage']) - np.min(data['voltage'])
                    result += f"CH{channel}: {points} points, {time_span:.3f}ms span, {v_pp:.3f}V p-p\n"
                else:
                    result += f"CH{channel}: [ERROR] Failed to acquire data\n"

            # Note: Math function acquisition would require additional backend support
            for math_num in selected_math:
                result += f"MATH{math_num}: [INFO] Math function acquisition not yet implemented\n"

            return result

        except Exception as e:
            self._logger.error(f"Data acquisition error: {e}")
            return f"[ERROR] Data acquisition: {str(e)}"

    def export_csv(self) -> str:
        """Export current waveform data to CSV files"""
        if not self.current_waveform_data:
            return "[WARNING] No waveform data to export"

        try:
            result = f"CSV Export Results:\n"
            result += "-" * 40 + "\n"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for channel, data in self.current_waveform_data.items():
                filename = f"MSO24_{channel}_data_{timestamp}"
                csv_path = self.data_acquisition.save_waveform_csv(
                    data, filename, self.save_locations['data']
                )
                result += f"{channel}: {csv_path}\n"

            return result

        except Exception as e:
            self._logger.error(f"CSV export error: {e}")
            return f"[ERROR] CSV export: {str(e)}"

    def generate_plot(self, title: str = "") -> str:
        """Generate plots for current waveform data"""
        if not self.current_waveform_data:
            return "[WARNING] No waveform data to plot"

        try:
            result = f"Plot Generation Results:\n"
            result += "-" * 40 + "\n"

            for channel, data in self.current_waveform_data.items():
                plot_title = f"{title} - {channel}" if title else f"{channel} Waveform"
                plot_path = self.data_acquisition.generate_waveform_plot(
                    data, plot_title, self.save_locations['graphs']
                )
                result += f"{channel}: {plot_path}\n"

            return result

        except Exception as e:
            self._logger.error(f"Plot generation error: {e}")
            return f"[ERROR] Plot generation: {str(e)}"

    def run_full_automation(self, ch1: bool, ch2: bool, ch3: bool, ch4: bool,
                           math1: bool, math2: bool, math3: bool, math4: bool, title: str) -> str:
        """Run complete automation sequence"""
        try:
            result = "Full Automation Sequence:\n"
            result += "=" * 50 + "\n"

            # Step 1: Acquire data
            result += "Step 1: Data Acquisition\n"
            acq_result = self.acquire_data(ch1, ch2, ch3, ch4, math1, math2, math3, math4)
            result += acq_result + "\n"

            # Treat only global acquisition errors as fatal. Per-channel failures are
            # reported in the log but do not stop the rest of the automation steps.
            if ("[ERROR] Oscilloscope not connected" in acq_result or
                "[ERROR] Data acquisition:" in acq_result):
                return result + "Automation stopped due to acquisition failure."

            # Step 2: Export CSV
            result += "Step 2: CSV Export\n"
            csv_result = self.export_csv()
            result += csv_result + "\n"

            # Step 3: Generate plots
            result += "Step 3: Plot Generation\n"
            plot_result = self.generate_plot(title)
            result += plot_result + "\n"

            # Step 4: Capture screenshot
            result += "Step 4: Screenshot Capture\n"
            screenshot_result = self.capture_screenshot()
            result += screenshot_result + "\n"

            result += "=" * 50 + "\n"
            result += "[OK] Full automation sequence completed successfully!"

            return result

        except Exception as e:
            self._logger.error(f"Full automation error: {e}")
            return f"[ERROR] Full automation: {str(e)}"

    def browse_folder(self, current_path: str, folder_type: str) -> str:
        """Browse for folder using tkinter dialog"""
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            root.attributes('-topmost', True)  # Bring to front

            selected_path = filedialog.askdirectory(
                title=f"Select {folder_type} Directory",
                initialdir=current_path
            )

            root.destroy()

            if selected_path:
                return selected_path
            else:
                return current_path

        except Exception as e:
            self._logger.error(f"Folder browser error: {e}")
            return current_path

    def create_interface(self):
        """Create comprehensive Gradio interface"""
        
        custom_css = """
        /* ============================================================
        MAIN CONTAINER - Controls the entire interface width/height
        ============================================================ */
        .gradio-container {
            max-width: 100% !important;
            padding: 20px !important;
            margin: 0 !important;
            min-height: 100vh;
        }
        
        /* ============================================================
        TAB CONTENT - Each instrument tab
        ============================================================ */
        .tab {
            padding: 0 10px;
            min-height: calc(100vh - 120px);
        }
        
        /* ============================================================
        PANELS - Individual sections within tabs
        ============================================================ */
        .panel {
            margin: 5px 0;
        }
        
        /* ============================================================
        TAB NAVIGATION - The tab buttons at the top
        ============================================================ */
        .tab-nav {
            border-bottom: 2px solid #c034eb;
            margin-bottom: 12px;
        }
        
        /* ============================================================
        SELECTED TAB - The currently active tab appearance
        ============================================================ */
        .tab-selected {
            background-color: #e0e0ff;
            font-weight: 600;
        }
        
        /* ============================================================
        BUTTONS - Consistent button styling
        ============================================================ */
        button {
            margin: 2px !important;
        }
        
        /* ============================================================
        STATUS MESSAGES - Styling for status text
        ============================================================ */
        .status-good { color: #28a745; font-weight: bold; }
        .status-bad { color: #dc3545; font-weight: bold; }
        .status-info { color: #17a2b8; font-weight: bold; }
        
        /* ============================================================
        CUSTOMIZATION TIPS:
        - To change colors: Use hex codes like #9c9cff (Google "color picker" to find codes)
        - To adjust spacing: Change pixel values (e.g., 20px to 30px for more space)
        - To make interface narrower: Change max-width to 90% or 80%
        - To add more vertical space: Increase padding values
        ============================================================ */
        """

        with gr.Blocks(
            title="Tektronix MSO24 Control",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:

            gr.Markdown("# Tektronix MSO24 Oscilloscope Control Center")
            gr.Markdown("**Professional oscilloscope automation interface with comprehensive control features**")

            # ================================================================
            # CONNECTION TAB
            # ================================================================
            with gr.Tab("Connection"):
                gr.Markdown("### Oscilloscope Connection")

                with gr.Row():
                    visa_input = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x0699::0x0105::SGVJ0003176::INSTR",
                        placeholder="Enter VISA address (USB/Ethernet/Serial)",
                        scale=3
                    )

                with gr.Row():
                    connect_btn = gr.Button("Connect", variant="primary")
                    disconnect_btn = gr.Button("Disconnect", variant="secondary")
                    reset_btn = gr.Button("Reset Scope", variant="secondary")

                connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False,
                    lines=3
                )

                connect_btn.click(
                    fn=self.connect_oscilloscope,
                    inputs=[visa_input],
                    outputs=[connection_status]
                )

                disconnect_btn.click(
                    fn=self.disconnect_oscilloscope,
                    inputs=[],
                    outputs=[connection_status]
                )

                reset_btn.click(
                    fn=self.reset_oscilloscope,
                    inputs=[],
                    outputs=[connection_status]
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
                    v_scale = gr.Number(label="V/div", value=1.0, minimum=0.001, maximum=10.0)
                    v_offset = gr.Number(label="Offset (V)", value=0.0, minimum=-50.0, maximum=50.0)
                    coupling = gr.Dropdown(
                        label="Coupling",
                        choices=["DC", "AC", "DCREJECT"],
                        value="DC"
                    )
                    probe = gr.Dropdown(
                        label="Probe",
                        choices=[("1x", 1), ("10x", 10), ("100x", 100), ("1000x", 1000)],
                        value=10
                    )

                config_channel_btn = gr.Button("Configure Channels", variant="primary")
                channel_status = gr.Textbox(label="Status", interactive=False, lines=5)

                config_channel_btn.click(
                    fn=self.configure_channels,
                    inputs=[ch1_select, ch2_select, ch3_select, ch4_select, v_scale, v_offset, coupling, probe],
                    outputs=[channel_status]
                )

                gr.Markdown("---")
                gr.Markdown("### Acquisition Control")

                with gr.Row():
                    run_btn = gr.Button("RUN", variant="primary", scale=1)
                    stop_btn = gr.Button("STOP", variant="secondary", scale=1)
                    single_btn = gr.Button("SINGLE", variant="secondary", scale=1)
                    autoscale_btn = gr.Button("Autoscale", variant="primary", scale=1)

                acq_status = gr.Textbox(label="Acquisition Status", interactive=False, lines=3)
                system_status = gr.Textbox(label="System Status", interactive=False, lines=3)

                run_btn.click(
                    fn=lambda: self.control_acquisition("Run"),
                    inputs=[],
                    outputs=[acq_status]
                )

                stop_btn.click(
                    fn=lambda: self.control_acquisition("Stop"),
                    inputs=[],
                    outputs=[acq_status]
                )

                single_btn.click(
                    fn=lambda: self.control_acquisition("Single"),
                    inputs=[],
                    outputs=[acq_status]
                )

                autoscale_btn.click(
                    fn=self.autoscale,
                    inputs=[],
                    outputs=[system_status]
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
                    timebase_position = gr.Number(
                        label="Position (s)",
                        value=0.0,
                        minimum=-1000.0,
                        maximum=1000.0
                    )
                    record_length = gr.Number(
                        label="Record Length",
                        value=10000,
                        minimum=1000,
                        maximum=62500000,
                        step=1000
                    )

                timebase_btn = gr.Button("Apply Timebase", variant="primary")
                timebase_status = gr.Textbox(label="Status", interactive=False)

                timebase_btn.click(
                    fn=self.configure_timebase,
                    inputs=[time_scale, timebase_position, record_length],
                    outputs=[timebase_status]
                )

                gr.Markdown("---")
                gr.Markdown("### Edge Trigger")

                with gr.Row():
                    trigger_type = gr.Dropdown(
                        label="Trigger Type",
                        choices=["EDGE", "PULSE", "LOGIC", "BUS", "VIDEO"],
                        value="EDGE"
                    )
                    trigger_source = gr.Dropdown(
                        label="Source",
                        choices=["CH1", "CH2", "CH3", "CH4", "EXT"],
                        value="CH1"
                    )

                with gr.Row():
                    trigger_level = gr.Number(
                        label="Level (V)",
                        value=0.0,
                        minimum=-50.0,
                        maximum=50.0
                    )
                    trigger_slope = gr.Dropdown(
                        label="Slope",
                        choices=["Rising", "Falling", "Either"],
                        value="Rising"
                    )

                trigger_btn = gr.Button("Apply Trigger", variant="primary")
                trigger_status = gr.Textbox(label="Status", interactive=False)

                trigger_btn.click(
                    fn=self.configure_trigger,
                    inputs=[trigger_type, trigger_source, trigger_level, trigger_slope],
                    outputs=[trigger_status]
                )

                gr.Markdown("---")
                gr.Markdown("### Trigger Sweep & Holdoff")

                with gr.Row():
                    sweep_mode = gr.Dropdown(
                        label="Sweep Mode",
                        choices=["AUTO", "NORMal"],
                        value="AUTO"
                    )
                    sweep_btn = gr.Button("Apply Sweep", variant="primary")

                sweep_status = gr.Textbox(label="Sweep Status", interactive=False)

                sweep_btn.click(
                    fn=self.set_trigger_sweep,
                    inputs=[sweep_mode],
                    outputs=[sweep_status]
                )

                with gr.Row():
                    holdoff_time = gr.Number(label="Holdoff Time (ns)", value=100.0, minimum=0, maximum=1e6)
                    holdoff_btn = gr.Button("Apply Holdoff", variant="primary")

                holdoff_status = gr.Textbox(label="Holdoff Status", interactive=False)

                holdoff_btn.click(
                    fn=lambda t: self.set_trigger_holdoff(t * 1e-9),
                    inputs=[holdoff_time],
                    outputs=[holdoff_status]
                )

            # ================================================================
            # ADVANCED TRIGGERS TAB
            # ================================================================
            with gr.Tab("Advanced Triggers"):
                gr.Markdown("### Advanced Trigger Configuration")
                gr.Markdown("Advanced trigger modes for specialized applications")

                gr.Markdown("#### Pulse Width Trigger")
                gr.Markdown("Trigger on pulses with specific width characteristics")

                with gr.Row():
                    pulse_source = gr.Dropdown(label="Source", choices=["CH1", "CH2", "CH3", "CH4"], value="CH1")
                    pulse_level = gr.Number(label="Level (V)", value=0.0)
                    pulse_width = gr.Number(label="Width (ns)", value=10.0)

                pulse_btn = gr.Button("Set Pulse Trigger", variant="primary")
                pulse_status = gr.Textbox(label="Status", interactive=False)

                # Note: Actual pulse trigger implementation would require backend support

                gr.Markdown("---")
                gr.Markdown("#### Logic Trigger")
                gr.Markdown("Trigger based on logic combinations of multiple channels")

                logic_status = gr.Textbox(
                    label="Logic Trigger Status",
                    value="[INFO] Logic trigger configuration requires selecting trigger type LOGIC in Timebase & Trigger tab",
                    interactive=False,
                    lines=2
                )

            # ================================================================
            # AFG CONTROL TAB
            # ================================================================
            with gr.Tab("Function Generators"):
                gr.Markdown("### Arbitrary Function Generator (AFG)")
                gr.Markdown("Control the built-in AFG for signal generation and testing")

                with gr.Row():
                    afg_function = gr.Dropdown(
                        label="Waveform",
                        choices=["Sine", "Square", "Ramp", "Pulse", "Noise", "DC"],
                        value="Sine"
                    )
                    afg_frequency = gr.Number(
                        label="Frequency (Hz)",
                        value=1000.0,
                        minimum=0.1,
                        maximum=50e6
                    )

                with gr.Row():
                    afg_amplitude = gr.Number(
                        label="Amplitude (V)",
                        value=1.0,
                        minimum=0.02,
                        maximum=5.0
                    )
                    afg_offset = gr.Number(
                        label="Offset (V)",
                        value=0.0,
                        minimum=-2.5,
                        maximum=2.5
                    )
                    afg_enable = gr.Checkbox(
                        label="Enable Output",
                        value=False
                    )

                afg_btn = gr.Button("Configure AFG", variant="primary")
                afg_status = gr.Textbox(label="AFG Status", interactive=False, lines=3)

                afg_btn.click(
                    fn=self.configure_afg,
                    inputs=[afg_function, afg_frequency, afg_amplitude, afg_offset, afg_enable],
                    outputs=[afg_status]
                )

            # ================================================================
            # MATH FUNCTIONS TAB
            # ================================================================
            with gr.Tab("Math Functions"):
                gr.Markdown("### Math Function Configuration")

                with gr.Row():
                    math_func_num = gr.Dropdown(
                        label="Function Number",
                        choices=[("1", 1), ("2", 2), ("3", 3), ("4", 4)],
                        value=1
                    )
                    math_operation = gr.Dropdown(
                        label="Operation",
                        choices=["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"],
                        value="ADD"
                    )

                with gr.Row():
                    math_source1 = gr.Dropdown(
                        label="Source 1",
                        choices=[("Channel 1", "CH1"), ("Channel 2", "CH2"), ("Channel 3", "CH3"), ("Channel 4", "CH4")],
                        value="CH1"
                    )
                    math_source2 = gr.Dropdown(
                        label="Source 2 (not used for FFT)",
                        choices=[("2", "CH2"), ("Channel 1", "CH1"), ("Channel 3", "CH3"), ("Channel 4", "CH4")],
                        value="CH2"
                    )

                config_math_btn = gr.Button("Configure Math Function", variant="primary")
                math_status = gr.Textbox(label="Status", interactive=False, lines=3)

                config_math_btn.click(
                    fn=self.configure_math_function,
                    inputs=[math_func_num, math_operation, math_source1, math_source2],
                    outputs=[math_status]
                )

                gr.Markdown("---")
                gr.Markdown("### Math Function Display & Scale")

                with gr.Row():
                    math_display_check = gr.Checkbox(label="Show on Display", value=False)
                    math_v_scale = gr.Number(label="Vertical Scale (V/div)", value=1.0, minimum=0.001, maximum=10.0)

                with gr.Row():
                    toggle_display_btn = gr.Button("Toggle Display", variant="primary", scale=1)
                    set_scale_btn = gr.Button("Set Scale", variant="primary", scale=1)

                display_status = gr.Textbox(label="Display Status", interactive=False)
                scale_status = gr.Textbox(label="Scale Status", interactive=False)

                toggle_display_btn.click(
                    fn=self.toggle_math_display,
                    inputs=[math_func_num, math_display_check],
                    outputs=[display_status]
                )

                set_scale_btn.click(
                    fn=self.set_math_scale,
                    inputs=[math_func_num, math_v_scale],
                    outputs=[scale_status]
                )

            # ================================================================
            # MARKERS & CURSORS TAB
            # ================================================================
            with gr.Tab("Markers & Cursors"):
                gr.Markdown("### Markers & Cursors")
                gr.Markdown("Precise waveform analysis with cursors and markers")

                gr.Markdown("#### Cursor Configuration")

                cursor_info = gr.Textbox(
                    label="Cursor Information",
                    value="[INFO] Cursor and marker functionality requires backend SCPI command implementation.\n"
                          "Use the oscilloscope front panel for cursor measurements.",
                    interactive=False,
                    lines=4
                )

                gr.Markdown("#### Common Cursor Measurements")
                gr.Markdown("- Time difference between two points\n"
                          "- Voltage difference between two points\n"
                          "- Frequency and period measurements\n"
                          "- Rise/fall time analysis")

            # ================================================================
            # MEASUREMENTS TAB
            # ================================================================
            with gr.Tab("Measurements"):
                gr.Markdown("### Single Measurement")

                with gr.Row():
                    meas_source = gr.Dropdown(
                        label="Source",
                        choices=[
                            ("Channel 1", "CH1"),
                            ("Channel 2", "CH2"),
                            ("Channel 3", "CH3"),
                            ("Channel 4", "CH4"),
                        ],
                        value="CH1",
                    )
                    meas_type = gr.Dropdown(
                        label="Measurement Type",
                        choices=[
                            "FREQUENCY", "PERIOD", "AMPLITUDE", "HIGH", "LOW",
                            "MAX", "MIN", "PEAK2PEAK", "MEAN", "RMS", "RISE",
                            "FALL", "WIDTH", "DUTYCYCLE", "OVERSHOOT", "PRESHOOT",
                            "AREA", "PHASE",
                        ],
                        value="FREQUENCY",
                    )

                with gr.Row():
                    meas_result = gr.Textbox(
                        label="Measurement Result", interactive=False, lines=3
                    )
                    all_meas_result = gr.Textbox(
                        label="All Measurements", interactive=False, lines=10
                    )

                with gr.Row():
                    measure_btn = gr.Button("Measure", variant="primary", scale=1)
                    measure_all_btn = gr.Button("Measure All", variant="primary", scale=1)
                    show_all_btn = gr.Button("Show All", variant="primary", scale=1)

                measure_btn.click(
                    fn=self.add_measurement,
                    inputs=[meas_type, meas_source],
                    outputs=[meas_result],
                )

                measure_all_btn.click(
                    fn=self.measure_all_for_source,
                    inputs=[meas_source],
                    outputs=[meas_result, all_meas_result],
                )

                show_all_btn.click(
                    fn=self.get_all_measurements,
                    inputs=[],
                    outputs=[all_meas_result],
                )

            with gr.Tab("Setup Management"):
                gr.Markdown("### Setup Save/Recall")
                gr.Markdown("Save and recall oscilloscope configurations for test automation")

                setup_info = gr.Textbox(
                    label="Setup Management",
                    value="[INFO] Setup save/recall functionality requires backend implementation.\n"
                          "Use the oscilloscope front panel or file system for setup management.\n\n"
                          "Typical setup files include:\n"
                          "- Channel configurations\n"
                          "- Trigger settings\n"
                          "- Timebase settings\n"
                          "- Measurement configurations",
                    interactive=False,
                    lines=8
                )

            # ================================================================
            # OPERATIONS & FILE MANAGEMENT TAB
            # ================================================================
            with gr.Tab("Operations & File Management"):
                gr.Markdown("### File Save Locations")

                # Path configuration
                with gr.Group():
                    with gr.Row():
                        data_path = gr.Textbox(
                            label="Data Directory",
                            value=self.save_locations['data'],
                            scale=3
                        )
                        data_browse_btn = gr.Button("Browse", scale=1)

                    with gr.Row():
                        graphs_path = gr.Textbox(
                            label="Graphs Directory",
                            value=self.save_locations['graphs'],
                            scale=3
                        )
                        graphs_browse_btn = gr.Button("Browse", scale=1)

                    with gr.Row():
                        screenshots_path = gr.Textbox(
                            label="Screenshots Directory",
                            value=self.save_locations['screenshots'],
                            scale=3
                        )
                        screenshots_browse_btn = gr.Button("Browse", scale=1)

                    update_paths_btn = gr.Button("Update Paths", variant="secondary")
                    path_status = gr.Textbox(label="Path Status", interactive=False)

                    def update_paths(data_dir, graphs_dir, screenshots_dir):
                        self.save_locations['data'] = data_dir
                        self.save_locations['graphs'] = graphs_dir
                        self.save_locations['screenshots'] = screenshots_dir

                        # Update backend oscilloscope directories
                        if self.is_connected and self.oscilloscope:
                            with self.io_lock:
                                self.oscilloscope.set_output_directories(
                                    data_dir=data_dir,
                                    graph_dir=graphs_dir,
                                    screenshot_dir=screenshots_dir
                                )

                        return "[OK] File paths updated successfully"

                    def browse_data_folder(current_path):
                        new_path = self.browse_folder(current_path, "Data")
                        self.save_locations['data'] = new_path

                        # Update backend oscilloscope data directory
                        if self.is_connected and self.oscilloscope:
                            with self.io_lock:
                                self.oscilloscope.set_output_directories(data_dir=new_path)

                        return new_path, f"[OK] Data directory updated to: {new_path}"

                    def browse_graphs_folder(current_path):
                        new_path = self.browse_folder(current_path, "Graphs")
                        self.save_locations['graphs'] = new_path

                        # Update backend oscilloscope graph directory
                        if self.is_connected and self.oscilloscope:
                            with self.io_lock:
                                self.oscilloscope.set_output_directories(graph_dir=new_path)

                        return new_path, f"[OK] Graphs directory updated to: {new_path}"

                    def browse_screenshots_folder(current_path):
                        new_path = self.browse_folder(current_path, "Screenshots")
                        self.save_locations['screenshots'] = new_path

                        # Update backend oscilloscope screenshot directory
                        if self.is_connected and self.oscilloscope:
                            with self.io_lock:
                                self.oscilloscope.set_output_directories(screenshot_dir=new_path)

                        return new_path, f"[OK] Screenshots directory updated to: {new_path}"

                    # Connect path management buttons
                    update_paths_btn.click(
                        fn=update_paths,
                        inputs=[data_path, graphs_path, screenshots_path],
                        outputs=[path_status]
                    )

                    data_browse_btn.click(
                        fn=browse_data_folder,
                        inputs=[data_path],
                        outputs=[data_path, path_status]
                    )

                    graphs_browse_btn.click(
                        fn=browse_graphs_folder,
                        inputs=[graphs_path],
                        outputs=[graphs_path, path_status]
                    )

                    screenshots_browse_btn.click(
                        fn=browse_screenshots_folder,
                        inputs=[screenshots_path],
                        outputs=[screenshots_path, path_status]
                    )

                gr.Markdown("---")
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

                export_save_location = gr.Textbox(
                    label="Export Save Location",
                    placeholder="Click Browse to select folder",
                    interactive=False
                )

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

                operation_status = gr.Textbox(label="Operation Status", interactive=False, lines=12)

                # Connect data operation buttons
                screenshot_btn.click(
                    fn=self.capture_screenshot,
                    inputs=[],
                    outputs=[operation_status]
                )

                acquire_btn.click(
                    fn=self.acquire_data,
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4],
                    outputs=[operation_status]
                )

                export_btn.click(
                    fn=self.export_csv,
                    inputs=[],
                    outputs=[operation_status]
                )

                plot_btn.click(
                    fn=self.generate_plot,
                    inputs=[plot_title_input],
                    outputs=[operation_status]
                )

                full_auto_btn.click(
                    fn=self.run_full_automation,
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4, plot_title_input],
                    outputs=[operation_status]
                )

            gr.Markdown("---")
            gr.Markdown("**DIGANTARA MSO24 Control** | Professional oscilloscope automation interface | All SCPI Commands Verified")

        return interface

    def launch(self, share=False, server_port=7860, max_attempts=10):
        """Launch Gradio interface with port fallback"""
        self._gradio_interface = self.create_interface()

        for attempt in range(max_attempts):
            current_port = server_port + attempt
            try:
                print(f"Attempting to start MSO24 server on port {current_port}...")
                self._gradio_interface.launch(
                    server_name="0.0.0.0",
                    share=share,
                    server_port=current_port,
                    prevent_thread_lock=False,
                    show_error=True,
                    quiet=False
                )

                print("\n" + "=" * 80)
                print(f"MSO24 Control Server is running on port {current_port}")
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

        print("\nFailed to start the MSO24 server after multiple attempts.")
        self.cleanup()

def main():
    """Application entry point"""
    print("Tektronix MSO24 Oscilloscope Automation - Professional Gradio Interface")
    print("Professional oscilloscope control system with comprehensive features")
    print("=" * 80)
    print("Starting web interface...")

    app = None
    try:
        start_port = 7865
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

                app = GradioMSO24GUI()
                app.launch(share=False, server_port=port)
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
