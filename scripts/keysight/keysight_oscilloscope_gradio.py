#!/usr/bin/env python3
"""
Keysight DSOX6004A Oscilloscope Control - Clean Gradio Interface

Professional-grade web-based GUI for oscilloscope automation.
This interface uses high-level methods from the instrument control module
and avoids direct SCPI commands for better maintainability.

Features:
- Clean separation of concerns
- High-level instrument control methods
- Comprehensive error handling
- Professional UI with file management
- Automated measurements and plotting

Author: Professional Instrumentation Control System
Date: 2025-11-03
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

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib to handle large plots
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify_threshold'] = 0.5
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
# Import instrument control modules
try:
    from instrument_control.keysight_oscilloscope import KeysightDSOX6004A, KeysightDSOX6004AError
except ImportError as e:
    print(f"Error importing instrument control modules: {e}")
    print("Please ensure the instrument_control module is properly installed.")
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
        return float(value)


TRIGGER_SLOPE_MAP = {
    "Rising": "POS",
    "Falling": "NEG"
}


def format_si_value(value: float, kind: str) -> str:
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
        return format_si_value(value, "freq")
    if meas_type in ["PERiod", "RISE", "FALL"]:
        return format_si_value(value, "time")
    if meas_type in ["VAMP", "VTOP", "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "VPP"]:
        return format_si_value(value, "volt")
    if meas_type in ["DUTYcycle", "NDUTy", "OVERshoot"]:
        return format_si_value(value, "percent")
    return f"{value}"


class OscilloscopeDataAcquisition:
    """Simplified data acquisition class using high-level oscilloscope methods"""
    
    def __init__(self, oscilloscope_instance, io_lock: Optional[threading.RLock] = None):
        self.scope = oscilloscope_instance
        self._logger = logging.getLogger(f'{self.__class__.__name__}')
        self.default_data_dir = Path.cwd() / "data"
        self.default_graph_dir = Path.cwd() / "graphs"
        self.default_screenshot_dir = Path.cwd() / "screenshots"
        self.io_lock = io_lock

    def acquire_waveform_data(self, channel: int, max_points: int = 62500) -> Optional[Dict[str, Any]]:
        """Acquire waveform data using high-level oscilloscope methods"""
        if not self.scope.is_connected:
            self._logger.error("Cannot acquire data: oscilloscope not connected")
            return None

        try:
            # Use high-level waveform acquisition (would need to be implemented in oscilloscope class)
            # For now, we'll use the existing SCPI method but in a cleaner way
            lock = self.io_lock
            if lock:
                with lock:
                    waveform_data = self._acquire_waveform_scpi(channel, max_points)
            else:
                waveform_data = self._acquire_waveform_scpi(channel, max_points)
            
            if waveform_data:
                self._logger.info(f"Successfully acquired {len(waveform_data['voltage'])} points from channel {channel}")
            
            return waveform_data
            
        except Exception as e:
            self._logger.error(f"Failed to acquire waveform data from channel {channel}: {e}")
            return None
    
    def _acquire_waveform_scpi(self, channel: int, max_points: int) -> Optional[Dict[str, Any]]:
        """Internal SCPI-based waveform acquisition"""
        try:
            # Configure waveform acquisition
            self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")
            self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")
            self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")
            self.scope._scpi_wrapper.write(f":WAVeform:POINts {max_points}")
            
            # Get preamble
            preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
            preamble_parts = preamble.split(',')
            y_increment = float(preamble_parts[7])
            y_origin = float(preamble_parts[8])
            y_reference = float(preamble_parts[9])
            x_increment = float(preamble_parts[4])
            x_origin = float(preamble_parts[5])
            
            # Get raw data
            raw_data = self.scope._scpi_wrapper.query_binary_values(":WAVeform:DATA?", datatype='B')
            
            # Convert to physical units
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
                'acquisition_time': datetime.now().isoformat()
            }
        except Exception as e:
            self._logger.error(f"SCPI waveform acquisition failed: {e}")
            return None

    def export_to_csv(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None, 
                     filename: Optional[str] = None) -> Optional[str]:
        if not waveform_data:
            self._logger.error("No waveform data to export")
            return None

        try:
            save_dir = Path(custom_path) if custom_path else self.default_data_dir
            self.scope.setup_output_directories()
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"waveform_ch{waveform_data['channel']}_{timestamp}.csv"
            
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            filepath = save_dir / filename
            df = pd.DataFrame({
                'Time (s)': waveform_data['time'],
                'Voltage (V)': waveform_data['voltage']
            })
            
            with open(filepath, 'w') as f:
                f.write(f"# Oscilloscope Waveform Data\n")
                f.write(f"# Channel: {waveform_data['channel']}\n")
                f.write(f"# Acquisition Time: {waveform_data['acquisition_time']}\n")
                f.write(f"# Sample Rate: {waveform_data['sample_rate']:.2e} Hz\n")
                f.write(f"# Points Count: {waveform_data['points_count']}\n")
                f.write(f"# Time Increment: {waveform_data['time_increment']:.2e} s\n")
                f.write(f"# Voltage Increment: {waveform_data['voltage_increment']:.2e} V\n")
                f.write("\n")
            
            df.to_csv(filepath, mode='a', index=False)
            self._logger.info(f"CSV exported successfully: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.error(f"Failed to export CSV: {e}")
            return None

    def generate_waveform_plot(self, waveform_data: Dict[str, Any], custom_path: Optional[str] = None,
                                filename: Optional[str] = None, plot_title: Optional[str] = None) -> Optional[str]:
        """Generate waveform plot with measurements using high-level oscilloscope methods"""
        # Get all measurements using the oscilloscope's built-in method
        measurements = {}
        try:
            if self.io_lock:
                with self.io_lock:
                    measurements = self.scope.get_all_measurements(waveform_data['channel']) or {}
            else:
                measurements = self.scope.get_all_measurements(waveform_data['channel']) or {}
        except Exception as e:
            self._logger.warning(f"Failed to get measurements: {e}")
            measurements = {}
        
        if not waveform_data:
            self._logger.error("No waveform data to plot")
            return None

        try:
            save_dir = Path(custom_path) if custom_path else self.default_graph_dir
            self.scope.setup_output_directories()
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"waveform_plot_ch{waveform_data['channel']}_{timestamp}.png"
            
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            filepath = save_dir / filename

            # Create figure with proper settings for large datasets
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Downsample data if too large to prevent Agg errors
            time_data = waveform_data['time']
            voltage_data = waveform_data['voltage']
            
            if len(time_data) > 10000:
                # Downsample to 10000 points max
                step = len(time_data) // 10000
                time_data = time_data[::step]
                voltage_data = voltage_data[::step]
            
            ax.plot(time_data, voltage_data, 'b-', linewidth=1, rasterized=True)
            
            if plot_title is None:
                plot_title = f"Oscilloscope Waveform - Channel {waveform_data['channel']}"
            
            ax.set_title(plot_title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Voltage (V)', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Format measurements for display
            measurements_text = "MEASUREMENTS:\n"
            measurements_text += "─" * 25 + "\n"
            
            # Display key measurements
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
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close specific figure
            self._logger.info(f"Plot saved successfully: {filepath}")
            return str(filepath)

        except Exception as e:
            self._logger.error(f"Failed to generate plot: {e}")
            return None


class GradioOscilloscopeGUI:
    def __init__(self):
        self.oscilloscope = None
        self.data_acquisition = None
        self.last_acquired_data = None
        self.io_lock = threading.RLock()
        self.live_feed_active = False
        self.live_feed_paused = False
        self.live_feed_worker_running = False
        self._shutdown_flag = threading.Event()
        self._gradio_interface = None
        
        self.save_locations = {
            'data': str(Path.cwd() / "data"),
            'graphs': str(Path.cwd() / "graphs"),
            'screenshots': str(Path.cwd() / "screenshots")
        }
        
        self.setup_logging()
        self.setup_cleanup_handlers()
        
        # Timebase scales as a list of mixed types for Gradio compatibility
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
        
        # Measurement types as a list of strings
        self.measurement_types = [
            "FREQ", "PERiod", "VPP", "VAMP", "OVERshoot", "VTOP",
            "VBASe", "VAVG", "VRMS", "VMAX", "VMIN", "RISE", "FALL",
            "DUTYcycle", "NDUTy"
        ]

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GradioOscilloscopeAutomation')
    
    def setup_cleanup_handlers(self):
        """Setup cleanup handlers for graceful shutdown"""
        # Register cleanup function to run on exit
        atexit.register(self.cleanup)
        
        # Handle Ctrl+C and other signals
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
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
                    print("Oscilloscope disconnected.")
            
            self.oscilloscope = None
            self.data_acquisition = None
            
            # Close any remaining matplotlib figures
            plt.close('all')
            
            print("Cleanup completed.")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def connect_oscilloscope(self, visa_address: str):
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
        try:
            if self.oscilloscope:
                self.oscilloscope.disconnect()
                self.oscilloscope = None
                self.data_acquisition = None
                self.last_acquired_data = None
                self.logger.info("Oscilloscope disconnected successfully")
            return "Disconnected successfully", "Disconnected"
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            return f"Disconnect error: {str(e)}", "Disconnected"

    def test_connection(self):
        if self.oscilloscope and self.oscilloscope.is_connected:
            return "Connection test: PASSED"
        else:
            return "Connection test: FAILED - Not connected"

    def configure_channel(self, ch1, ch2, ch3, ch4, v_scale, v_offset, coupling, probe):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        # Configure all channels (enable selected ones, disable others)
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
                        # Disable the channel using high-level method
                        try:
                            # Use the oscilloscope's built-in method to disable channel
                            with self.io_lock:
                                self.oscilloscope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay OFF")
                            disabled_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to disable channel {channel}: {e}")
            
            return f"Configured: {success_count} enabled, {disabled_count} disabled"
        except Exception as e:
            return f"Configuration error: {str(e)}"

    def configure_timebase(self, time_scale_input, time_offset):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            # If time_scale_input is already a number, use it directly
            if isinstance(time_scale_input, (int, float)):
                time_scale = float(time_scale_input)
                display_scale = format_si_value(time_scale, 's')
            else:
                # For string inputs, use the parse function
                time_scale = parse_timebase_string(time_scale_input)
                display_scale = time_scale_input
            
            with self.io_lock:
                success = self.oscilloscope.configure_timebase(time_scale, time_offset)
            
            if success:
                return f"Timebase configured: {display_scale} ({time_scale}s/div), offset {time_offset}s"
            else:
                return "Timebase configuration failed"
        except Exception as e:
            return f"Timebase error: {str(e)}"

    def configure_trigger(self, trigger_source, trigger_level, trigger_slope):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            channel = int(trigger_source.replace("CH", ""))
            slope = TRIGGER_SLOPE_MAP.get(trigger_slope, "POS")
            
            with self.io_lock:
                success = self.oscilloscope.configure_trigger(channel, trigger_level, slope)
            
            if success:
                return f"Trigger configured: {trigger_source} @ {trigger_level}V, {trigger_slope}"
            else:
                return "Trigger configuration failed"
        except Exception as e:
            return f"Trigger error: {str(e)}"

    def configure_wgen(self, generator, enable, waveform, frequency, amplitude, offset):
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
                return f"WGEN{generator} configured: {waveform}, {frequency}Hz, {amplitude}Vpp"
            else:
                return f"WGEN{generator} configuration failed"
        except Exception as e:
            return f"WGEN{generator} error: {str(e)}"

    def capture_screenshot(self):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            # Create custom screenshot directory if needed
            screenshot_dir = Path(self.save_locations['screenshots'])
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"scope_screenshot_{timestamp}.png"
            filepath = screenshot_dir / filename
            
            # Use the oscilloscope's built-in screenshot method
            screenshot_file = self.oscilloscope.capture_screenshot(
                filename=filename,
                image_format="PNG"
            )
            
            if screenshot_file:
                # Move to user-specified directory if different
                if str(screenshot_dir) != str(Path(screenshot_file).parent):
                    import shutil
                    final_path = screenshot_dir / Path(screenshot_file).name
                    shutil.move(screenshot_file, final_path)
                    return f"Screenshot saved: {final_path}"
                else:
                    return f"Screenshot saved: {screenshot_file}"
            else:
                return "Screenshot capture failed"
        except Exception as e:
            return f"Screenshot error: {str(e)}"

    def acquire_data(self, ch1, ch2, ch3, ch4):
        if not self.data_acquisition:
            return "Error: Data acquisition module not initialized. Connect to oscilloscope first."
        
        selected_channels = []
        if ch1:
            selected_channels.append(1)
        if ch2:
            selected_channels.append(2)
        if ch3:
            selected_channels.append(3)
        if ch4:
            selected_channels.append(4)
        
        if not selected_channels:
            return "Error: No channels selected"
        
        try:
            all_channel_data = {}
            for channel in selected_channels:
                data = self.data_acquisition.acquire_waveform_data(channel)
                if data:
                    all_channel_data[channel] = data
            
            if all_channel_data:
                self.last_acquired_data = all_channel_data
                total_points = sum(ch_data['points_count'] for ch_data in all_channel_data.values())
                return f"Data acquired: {len(all_channel_data)} channels, {total_points} total points"
            else:
                return "Data acquisition failed for all channels"
        except Exception as e:
            return f"Acquisition error: {str(e)}"

    def export_csv(self):
        if not self.last_acquired_data:
            return "Error: No data available. Acquire data first."
        
        if not self.data_acquisition:
            return "Error: Data acquisition module not initialized. Connect to oscilloscope first."
        
        try:
            exported_files = []
            if isinstance(self.last_acquired_data, dict) and 'channel' not in self.last_acquired_data:
                for channel, data in self.last_acquired_data.items():
                    filename = self.data_acquisition.export_to_csv(data, custom_path=self.save_locations['data'])
                    if filename:
                        exported_files.append(Path(filename).name)
            else:
                filename = self.data_acquisition.export_to_csv(self.last_acquired_data, custom_path=self.save_locations['data'])
                if filename:
                    exported_files.append(Path(filename).name)
            
            if exported_files:
                return f"CSV exported: {', '.join(exported_files)}"
            else:
                return "CSV export failed"
        except Exception as e:
            return f"Export error: {str(e)}"

    def generate_plot(self, plot_title):
        if not self.last_acquired_data:
            return "Error: No data available. Acquire data first."
        
        if not self.data_acquisition:
            return "Error: Data acquisition module not initialized. Connect to oscilloscope first."
        
        try:
            custom_title = plot_title.strip() or None
            plot_files = []
            
            if isinstance(self.last_acquired_data, dict) and 'channel' not in self.last_acquired_data:
                for channel, data in self.last_acquired_data.items():
                    if custom_title:
                        channel_title = f"{custom_title} - Channel {channel}"
                    else:
                        channel_title = None
                    
                    filename = self.data_acquisition.generate_waveform_plot(
                        data, custom_path=self.save_locations['graphs'], plot_title=channel_title)
                    if filename:
                        plot_files.append(Path(filename).name)
            else:
                filename = self.data_acquisition.generate_waveform_plot(
                    self.last_acquired_data, custom_path=self.save_locations['graphs'], plot_title=custom_title)
                if filename:
                    plot_files.append(Path(filename).name)
            
            if plot_files:
                return f"Plots generated: {', '.join(plot_files)}"
            else:
                return "Plot generation failed"
        except Exception as e:
            return f"Plot error: {str(e)}"

    def perform_measurement(self, channel_str, measurement_type):
        """Perform single measurement using high-level oscilloscope methods"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            channel = int(channel_str.replace("CH", ""))
            
            # Use high-level measurement method
            with self.io_lock:
                result = self.oscilloscope.measure_single(channel, measurement_type)
            
            if result is not None:
                formatted_result = format_measurement_value(measurement_type, result)
                return f"{channel_str} {measurement_type}: {formatted_result}"
            else:
                return f"Measurement failed: No data from {channel_str}"
        except Exception as e:
            return f"Measurement error: {str(e)}"

    def perform_autoscale(self):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        try:
            with self.io_lock:
                success = self.oscilloscope.autoscale()
            
            if success:
                return "Autoscale completed successfully"
            else:
                return "Autoscale failed"
        except Exception as e:
            return f"Autoscale error: {str(e)}"

    def run_full_automation(self, ch1, ch2, ch3, ch4, plot_title):
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        
        if not self.data_acquisition:
            return "Error: Data acquisition module not initialized. Connect to oscilloscope first."
        
        selected_channels = []
        if ch1:
            selected_channels.append(1)
        if ch2:
            selected_channels.append(2)
        if ch3:
            selected_channels.append(3)
        if ch4:
            selected_channels.append(4)
        
        if not selected_channels:
            return "Error: No channels selected"
        
        try:
            results = []
            
            results.append("Step 1/4: Screenshot...")
            with self.io_lock:
                screenshot_file = self.oscilloscope.capture_screenshot()
            if screenshot_file:
                results.append(f"Screenshot: {Path(screenshot_file).name}")
            
            results.append("Step 2/4: Acquiring data...")
            all_channel_data = {}
            for channel in selected_channels:
                data = self.data_acquisition.acquire_waveform_data(channel)
                if data:
                    all_channel_data[channel] = data
                    results.append(f"Ch{channel}: {data['points_count']} points")
            
            if not all_channel_data:
                return "Error: Data acquisition failed for all channels"
            
            results.append("Step 3/4: Exporting CSV...")
            csv_files = []
            for channel, data in all_channel_data.items():
                csv_file = self.data_acquisition.export_to_csv(data, custom_path=self.save_locations['data'])
                if csv_file:
                    csv_files.append(Path(csv_file).name)
                    results.append(f"Ch{channel} CSV: {Path(csv_file).name}")
            
            results.append("Step 4/4: Generating plots...")
            custom_title = plot_title.strip() or None
            plot_files = []
            for channel, data in all_channel_data.items():
                if custom_title:
                    channel_title = f"{custom_title} - Channel {channel}"
                else:
                    channel_title = None
                plot_file = self.data_acquisition.generate_waveform_plot(
                    data, custom_path=self.save_locations['graphs'], plot_title=channel_title)
                if plot_file:
                    plot_files.append(Path(plot_file).name)
                    results.append(f"Ch{channel} Plot: {Path(plot_file).name}")
            
            self.last_acquired_data = all_channel_data
            results.append("Full automation completed successfully!")
            return "\n".join(results)
            
        except Exception as e:
            return f"Automation error: {str(e)}"

    def browse_folder(self, current_path, folder_type="folder"):
        """Open file dialog to browse for folder"""
        try:
            # Create a temporary tkinter root window (hidden)
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            root.lift()  # Bring to front
            root.attributes('-topmost', True)  # Keep on top
            
            # Set initial directory
            initial_dir = current_path if Path(current_path).exists() else str(Path.cwd())
            
            # Open folder selection dialog
            selected_path = filedialog.askdirectory(
                title=f"Select {folder_type} Directory",
                initialdir=initial_dir
            )
            
            root.destroy()  # Clean up
            
            if selected_path:
                return selected_path
            else:
                return current_path  # Return original if cancelled
        except Exception as e:
            print(f"Browse error: {e}")
            return current_path
    
    def create_interface(self):
        with gr.Blocks(title="Keysight Oscilloscope Automation", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Keysight DSOX6004A Oscilloscope Control System")
            gr.Markdown("Professional oscilloscope automation interface with comprehensive control features")
            
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
                
                config_channel_btn.click(
                    fn=self.configure_channel,
                    inputs=[ch1_select, ch2_select, ch3_select, ch4_select, v_scale, v_offset, coupling, probe],
                    outputs=[channel_status]
                )
            
            with gr.Tab("Timebase & Trigger"):
                gr.Markdown("### Horizontal Timebase Configuration")
                with gr.Row():
                    time_scale = gr.Dropdown(
                        label="Time/div",
                        choices=self.timebase_scales,
                        value=10e-3  # 10 ms in seconds
                    )
                    time_offset = gr.Number(label="Offset (s)", value=0.0)
                    timebase_btn = gr.Button("Apply Timebase", variant="primary")
                
                timebase_status = gr.Textbox(label="Timebase Status", interactive=False)
                
                timebase_btn.click(
                    fn=self.configure_timebase,
                    inputs=[time_scale, time_offset],
                    outputs=[timebase_status]
                )
                
                gr.Markdown("### Trigger Configuration")
                with gr.Row():
                    trigger_source = gr.Dropdown(
                        label="Source",
                        choices=["CH1", "CH2", "CH3", "CH4"],
                        value="CH1"
                    )
                    trigger_level = gr.Number(label="Level (V)", value=0.0)
                    trigger_slope = gr.Dropdown(
                        label="Slope",
                        choices=["Rising", "Falling"],
                        value="Rising"
                    )
                    trigger_btn = gr.Button("Apply Trigger", variant="primary")
                
                trigger_status = gr.Textbox(label="Trigger Status", interactive=False)
                
                trigger_btn.click(
                    fn=self.configure_trigger,
                    inputs=[trigger_source, trigger_level, trigger_slope],
                    outputs=[trigger_status]
                )
                
                gr.Markdown("### Autoscale")
                autoscale_btn = gr.Button("Execute Autoscale", variant="primary")
                autoscale_status = gr.Textbox(label="Autoscale Status", interactive=False)
                
                autoscale_btn.click(
                    fn=self.perform_autoscale,
                    inputs=[],
                    outputs=[autoscale_status]
                )
            
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
                
                wgen1_status = gr.Textbox(label="WGEN1 Status", interactive=False)
                
                wgen1_btn.click(
                    fn=lambda en, wf, fr, am, of: self.configure_wgen(1, en, wf, fr, am, of),
                    inputs=[wgen1_enable, wgen1_waveform, wgen1_freq, wgen1_amp, wgen1_offset],
                    outputs=[wgen1_status]
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
                
                wgen2_status = gr.Textbox(label="WGEN2 Status", interactive=False)
                
                wgen2_btn.click(
                    fn=lambda en, wf, fr, am, of: self.configure_wgen(2, en, wf, fr, am, of),
                    inputs=[wgen2_enable, wgen2_waveform, wgen2_freq, wgen2_amp, wgen2_offset],
                    outputs=[wgen2_status]
                )
            
            with gr.Tab("Operations & File Management"):
                gr.Markdown("### File Save Locations")
                
                with gr.Row():
                    data_path = gr.Textbox(
                        label="Data Directory",
                        value=self.save_locations['data'],
                        scale=4
                    )
                    data_browse_btn = gr.Button("Browse", scale=1)
                
                with gr.Row():
                    graphs_path = gr.Textbox(
                        label="Graphs Directory",
                        value=self.save_locations['graphs'],
                        scale=4
                    )
                    graphs_browse_btn = gr.Button("Browse", scale=1)
                
                with gr.Row():
                    screenshots_path = gr.Textbox(
                        label="Screenshots Directory",
                        value=self.save_locations['screenshots'],
                        scale=4
                    )
                    screenshots_browse_btn = gr.Button("Browse", scale=1)
                
                def update_paths(data, graphs, screenshots):
                    self.save_locations['data'] = data
                    self.save_locations['graphs'] = graphs
                    self.save_locations['screenshots'] = screenshots
                    return "Paths updated successfully"
                
                update_paths_btn = gr.Button("Update Paths", variant="primary")
                path_status = gr.Textbox(label="Path Status", interactive=False)
                
                update_paths_btn.click(
                    fn=update_paths,
                    inputs=[data_path, graphs_path, screenshots_path],
                    outputs=[path_status]
                )
                
                # File browser functionality
                def browse_data_folder(current_path):
                    new_path = self.browse_folder(current_path, "Data")
                    self.save_locations['data'] = new_path
                    return new_path, f"Data directory updated to: {new_path}"
                
                def browse_graphs_folder(current_path):
                    new_path = self.browse_folder(current_path, "Graphs")
                    self.save_locations['graphs'] = new_path
                    return new_path, f"Graphs directory updated to: {new_path}"
                
                def browse_screenshots_folder(current_path):
                    new_path = self.browse_folder(current_path, "Screenshots")
                    self.save_locations['screenshots'] = new_path
                    return new_path, f"Screenshots directory updated to: {new_path}"
                
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
                
                gr.Markdown("### Data Acquisition and Export")
                
                with gr.Row():
                    op_ch1 = gr.Checkbox(label="Ch1", value=True)
                    op_ch2 = gr.Checkbox(label="Ch2", value=False)
                    op_ch3 = gr.Checkbox(label="Ch3", value=False)
                    op_ch4 = gr.Checkbox(label="Ch4", value=False)
                
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
                
                screenshot_btn.click(
                    fn=self.capture_screenshot,
                    inputs=[],
                    outputs=[operation_status]
                )
                
                acquire_btn.click(
                    fn=self.acquire_data,
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4],
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
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, plot_title_input],
                    outputs=[operation_status]
                )
            
            with gr.Tab("Measurements"):
                gr.Markdown("### Waveform Measurements")
                
                with gr.Row():
                    # Create channel choices with explicit typing
                    channel_choices: List[Union[str, int, float, Tuple[str, Union[str, int, float]]]] = [
                        ("Channel 1", "CH1"),
                        ("Channel 2", "CH2"),
                        ("Channel 3", "CH3"),
                        ("Channel 4", "CH4")
                    ]
                    meas_channel = gr.Dropdown(
                        label="Channel",
                        choices=channel_choices,
                        value="CH1"
                    )
                    # Measurement choices with explicit typing
                    measurement_choices: List[Union[str, int, float, Tuple[str, Union[str, int, float]]]] = [
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
                    meas_type = gr.Dropdown(
                        label="Measurement Type",
                        choices=measurement_choices,
                        value="FREQ"
                    )
                    measure_btn = gr.Button("Measure", variant="primary")
                
                measurement_result = gr.Textbox(label="Measurement Result", interactive=False)
            
                
                measure_btn.click(
                    fn=self.perform_measurement,
                    inputs=[meas_channel, meas_type],
                    outputs=[measurement_result]
                )
            
            gr.Markdown("---")
            gr.Markdown("Keysight DSOX6004A Oscilloscope Automation System | Professional Grade Control Interface")
        
        return interface

    def launch(self, share=False, server_port=7860, auto_open=True, max_attempts=10):
        self._gradio_interface = self.create_interface()
        
        for attempt in range(max_attempts):
            current_port = server_port + attempt
            try:
                print(f"Attempting to start server on port {current_port}...")
                
                # Launch with blocking=True to keep the process alive
                self._gradio_interface.launch(
                    share=share, 
                    server_port=current_port,
                    inbrowser=auto_open if attempt == 0 else False,  # Only try to open browser on first attempt
                    prevent_thread_lock=False,
                    show_error=True,
                    quiet=False
                )
                
                # If we get here, launch was successful
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


import socket

def find_available_port(start_port=7860, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")

def main():
    print("Keysight Oscilloscope Automation - Gradio Interface")
    print("Professional oscilloscope control system")
    print("=" * 80)
    print("Starting web interface...")
    
    app = None
    try:
        # Try a higher port range to avoid conflicts
        start_port = 9000
        max_attempts = 10
        
        print(f"Looking for an available port starting from {start_port}...")
        
        for port in range(start_port, start_port + max_attempts):
            try:
                # Try to create a socket to check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.close()
                    
                    # If we get here, the port is available
                    print(f"\nFound available port: {port}")
                    print("The browser will open automatically when ready.")
                    print("")
                    print("IMPORTANT: To stop the application, press Ctrl+C in this terminal.")
                    print("Closing the browser tab will NOT stop the server.")
                    print("=" * 80)
                    
                    # Create and launch the app
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
        # Force exit to ensure all threads are terminated
        print("Forcing application exit...")
        os._exit(0)  # Force immediate exit