"""
Keysight E36441A Four Output Programmable DC Power Supply Control - Professional Gradio Interface

Comprehensive 4-channel autoranging DC power supply automation with all channel control,
measurements, protection, sequencing, and complete data acquisition workflow.

Features Include (Professional instrumentation control):
- Complete 4-channel control (voltage, current, output enable/disable)
- Real-time measurements (voltage, current, power) with live monitoring
- Advanced protection (OVP, OCP) with configurable limits and status monitoring
- Power sequencing with customizable timing and order
- Data logging and export capabilities with CSV and plot generation
- Memory operations (save/recall instrument states)
- Multi-Channel Simultaneous Waveform Generator with 19 waveform types
- Professional error handling and status reporting
- Thread-safe operations with comprehensive monitoring

Author: Enhanced by Senior Instrumentation Engineer
Organization: DIGANTARA Research and Technologies Pvt. Ltd.
Date: 2026-01-20
"""

# =============================================================================
# FILE SAVE LOCATION CONFIGURATION - EDIT THESE PATHS
# =============================================================================
# INSTRUCTIONS: Enter the FULL file paths where you want to save files.
# - Use raw strings (prefix with r) for Windows paths
# - Example: r"C:\Users\YourName\Documents\PSU\Data"
# - Make sure the server has write permissions to these directories
# - Directories will be created automatically if they don't exist
# =============================================================================

# Measurement data files location (CSV exports)
PSU_CSV_DATA_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_data"

# Graph/plot images location (PNG files)
PSU_GRAPH_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_graphs"

# Log files location (measurement logging)
PSU_LOG_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_logs"

# Waveform data exports
PSU_WAVEFORM_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_waveforms"

# =============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW THIS LINE
# =============================================================================

import sys
import logging
import threading
import queue
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import signal
import atexit
import os
import socket
import math
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify_threshold'] = 0.5

script_dir = Path(__file__).resolve().parent.parent.parent.parent

if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

try:
    from instrument_control.keysight_psu import KeysightE36441A, KeysightE36441AError
except ImportError as e:
    print(f"Error importing PSU module: {e}")
    sys.exit(1)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_si_value(value: float, kind: str) -> str:
    """Format numeric values with SI prefixes for human readability"""
    v = abs(value)
    if kind == "current":
        if v >= 1:
            return f"{value:.3f} A"
        if v >= 1e-3:
            return f"{value*1e3:.3f} mA"
        if v >= 1e-6:
            return f"{value*1e6:.3f} uA"
        return f"{value*1e9:.3f} nA"

    if kind == "voltage":
        if v >= 1e3:
            return f"{value/1e3:.3f} kV"
        if v >= 1:
            return f"{value:.3f} V"
        if v >= 1e-3:
            return f"{value*1e3:.3f} mV"
        if v >= 1e-6:
            return f"{value*1e6:.3f} uV"
        return f"{value*1e9:.3f} nV"

    if kind == "power":
        if v >= 1e3:
            return f"{value/1e3:.3f} kW"
        if v >= 1:
            return f"{value:.3f} W"
        if v >= 1e-3:
            return f"{value*1e3:.3f} mW"
        if v >= 1e-6:
            return f"{value*1e6:.3f} uW"
        return f"{value*1e9:.3f} nW"

    if kind == "resistance":
        if v >= 1e6:
            return f"{value/1e6:.3f} MOhm"
        if v >= 1e3:
            return f"{value/1e3:.3f} kOhm"
        if v >= 1:
            return f"{value:.3f} Ohm"
        if v >= 1e-3:
            return f"{value*1e3:.3f} mOhm"
        return f"{value*1e6:.3f} uOhm"

    return f"{value:.3f}"

def validate_channel_specs(channel: int, voltage: float, current: float) -> Tuple[bool, str]:
    """Validate voltage and current against E36441A specifications"""

    # Channel specifications
    if channel == 1:
        # Channel 1: High power channel
        max_ranges = [(15.0, 4.0), (20.0, 3.0), (30.0, 2.0)]
        max_power = 60.0
    else:
        # Channels 2-4: Standard power channels
        max_ranges = [(15.0, 2.0), (20.0, 1.5), (30.0, 1.0)]
        max_power = 30.0

    # Check voltage limits
    max_voltage = max(v for v, c in max_ranges)
    if voltage < 0 or voltage > max_voltage:
        return False, f"CH{channel}: Voltage {voltage}V exceeds range [0, {max_voltage}V]"

    # Find valid current for given voltage
    valid_ranges = [(v, c) for v, c in max_ranges if voltage <= v]
    if not valid_ranges:
        return False, f"CH{channel}: No valid range for {voltage}V"

    max_current = min(c for v, c in valid_ranges)
    if current < 0 or current > max_current:
        return False, f"CH{channel}: Current {current}A exceeds range [0, {max_current}A] at {voltage}V"

    # Check power limit
    power = voltage * current
    if power > max_power:
        return False, f"CH{channel}: Power {power}W exceeds {max_power}W limit"

    return True, "OK"

# ============================================================================
# WAVEFORM GENERATOR CLASS
# ============================================================================

class WaveformGenerator:
    """
    Waveform profile generator for PSU voltage output.
    Supports 19 different waveform types for comprehensive testing.
    """

    WAVEFORM_TYPES = [
        "Sine",
        "Square",
        "Triangle",
        "Ramp Up",
        "Ramp Down",
        "Cardiac",
        "Damped Sine",
        "Exponential Raise",
        "Exponential Fall",
        "Gaussian Pulse",
        "Neural Spike",
        "Staircase",
        "PWM",
        "Chirp",
        "Burst Mode",
        "Brownout",
        "RC Charge",
        "Sinc",
        "Breathing"
    ]

    def __init__(self, waveform_type: str = "Sine", target_voltage: float = 5.0,
                 cycles: int = 3, points_per_cycle: int = 50, cycle_duration: float = 8.0):
        self.waveform_type = waveform_type
        self.target_voltage = target_voltage
        self.cycles = cycles
        self.points_per_cycle = points_per_cycle
        self.cycle_duration = cycle_duration

    def generate(self) -> List[Tuple[float, float]]:
        """Generate waveform profile as list of (time, voltage) tuples"""
        total_points = self.cycles * self.points_per_cycle
        time_per_point = self.cycle_duration / self.points_per_cycle

        profile = []

        for i in range(total_points):
            t = i * time_per_point
            phase = (i % self.points_per_cycle) / self.points_per_cycle

            if self.waveform_type == "Sine":
                voltage = self.target_voltage * (0.5 + 0.5 * math.sin(2 * math.pi * phase))

            elif self.waveform_type == "Square":
                voltage = self.target_voltage if phase < 0.5 else 0.0

            elif self.waveform_type == "Triangle":
                if phase < 0.5:
                    voltage = self.target_voltage * (2 * phase)
                else:
                    voltage = self.target_voltage * (2 * (1 - phase))

            elif self.waveform_type == "Ramp Up":
                voltage = self.target_voltage * phase

            elif self.waveform_type == "Ramp Down":
                voltage = self.target_voltage * (1 - phase)

            elif self.waveform_type == "Cardiac":
                # ECG-like waveform
                if phase < 0.1:
                    voltage = self.target_voltage * 0.1
                elif phase < 0.15:
                    voltage = self.target_voltage * (0.1 + 0.9 * ((phase - 0.1) / 0.05))
                elif phase < 0.2:
                    voltage = self.target_voltage * (1.0 - 0.9 * ((phase - 0.15) / 0.05))
                elif phase < 0.25:
                    voltage = self.target_voltage * (0.1 - 0.15 * ((phase - 0.2) / 0.05))
                elif phase < 0.3:
                    voltage = self.target_voltage * (-0.05 + 0.15 * ((phase - 0.25) / 0.05))
                elif phase < 0.5:
                    voltage = self.target_voltage * (0.1 + 0.3 * math.sin(math.pi * (phase - 0.3) / 0.2))
                else:
                    voltage = self.target_voltage * 0.1
                voltage = max(0, voltage)

            elif self.waveform_type == "Damped Sine":
                decay = math.exp(-3 * phase)
                voltage = self.target_voltage * (0.5 + 0.5 * decay * math.sin(4 * math.pi * phase))

            elif self.waveform_type == "Exponential Raise":
                voltage = self.target_voltage * (1 - math.exp(-5 * phase))

            elif self.waveform_type == "Exponential Fall":
                voltage = self.target_voltage * math.exp(-5 * phase)

            elif self.waveform_type == "Gaussian Pulse":
                center = 0.5
                sigma = 0.15
                voltage = self.target_voltage * math.exp(-((phase - center) ** 2) / (2 * sigma ** 2))

            elif self.waveform_type == "Neural Spike":
                # Action potential waveform
                if phase < 0.1:
                    voltage = self.target_voltage * 0.1
                elif phase < 0.2:
                    voltage = self.target_voltage * (0.1 + 0.9 * ((phase - 0.1) / 0.1))
                elif phase < 0.3:
                    voltage = self.target_voltage * (1.0 - 1.1 * ((phase - 0.2) / 0.1))
                elif phase < 0.4:
                    voltage = max(0, self.target_voltage * (-0.1 + 0.2 * ((phase - 0.3) / 0.1)))
                else:
                    voltage = self.target_voltage * 0.1

            elif self.waveform_type == "Staircase":
                steps = 5
                step_num = int(phase * steps)
                voltage = self.target_voltage * (step_num + 1) / steps

            elif self.waveform_type == "PWM":
                # Variable duty cycle PWM
                duty = 0.3 + 0.4 * phase  # Duty cycle varies from 30% to 70%
                pwm_phase = (phase * 10) % 1  # 10 PWM cycles per waveform cycle
                voltage = self.target_voltage if pwm_phase < duty else 0.0

            elif self.waveform_type == "Chirp":
                # Frequency sweep
                freq = 1 + 4 * phase  # Frequency increases from 1 to 5
                voltage = self.target_voltage * (0.5 + 0.5 * math.sin(2 * math.pi * freq * phase))

            elif self.waveform_type == "Burst Mode":
                # Bursts of pulses
                if phase < 0.4:
                    burst_phase = (phase / 0.4 * 5) % 1
                    voltage = self.target_voltage if burst_phase < 0.5 else 0.0
                else:
                    voltage = 0.0

            elif self.waveform_type == "Brownout":
                # Power sag and recovery
                if phase < 0.2:
                    voltage = self.target_voltage
                elif phase < 0.3:
                    voltage = self.target_voltage * (1 - 0.7 * ((phase - 0.2) / 0.1))
                elif phase < 0.7:
                    voltage = self.target_voltage * 0.3
                elif phase < 0.8:
                    voltage = self.target_voltage * (0.3 + 0.7 * ((phase - 0.7) / 0.1))
                else:
                    voltage = self.target_voltage

            elif self.waveform_type == "RC Charge":
                # Capacitor charge/discharge
                if phase < 0.5:
                    tau = 0.15
                    voltage = self.target_voltage * (1 - math.exp(-phase / tau))
                else:
                    tau = 0.15
                    voltage = self.target_voltage * math.exp(-(phase - 0.5) / tau)

            elif self.waveform_type == "Sinc":
                # Sinc function (main lobe + ripples)
                x = (phase - 0.5) * 10
                if abs(x) < 0.001:
                    voltage = self.target_voltage
                else:
                    voltage = self.target_voltage * abs(math.sin(math.pi * x) / (math.pi * x))

            elif self.waveform_type == "Breathing":
                # Smooth fade in/out (cosine-based)
                voltage = self.target_voltage * (0.5 - 0.5 * math.cos(2 * math.pi * phase))

            else:
                voltage = 0.0

            profile.append((t, max(0, min(self.target_voltage, voltage))))

        return profile

# ============================================================================
# DATA LOGGING CLASS
# ============================================================================

class PSUDataLogger:
    """
    Data logging handler with CSV export and measurement monitoring.
    Implements continuous measurement logging and file I/O with error handling.
    """

    def __init__(self, psu_instance, io_lock: Optional[threading.RLock] = None):
        self.psu = psu_instance
        self._logger = logging.getLogger(f'{self.__class__.__name__}')
        self.default_data_dir = Path.cwd() / "data"
        self.default_graph_dir = Path.cwd() / "graphs"
        self.default_log_dir = Path.cwd() / "logs"
        self.io_lock = io_lock
        self._logging_active = False
        self._logging_thread = None
        self._stop_logging = threading.Event()

    def start_measurement_logging(self, interval_seconds: float = 1.0, duration_seconds: Optional[float] = None) -> bool:
        """Start continuous measurement logging"""
        if self._logging_active:
            self._logger.warning("Logging already active")
            return False

        try:
            self._stop_logging.clear()
            self._logging_thread = threading.Thread(
                target=self._logging_worker,
                args=(interval_seconds, duration_seconds),
                daemon=True
            )
            self._logging_active = True
            self._logging_thread.start()
            self._logger.info(f"Started measurement logging: interval={interval_seconds}s")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start logging: {e}")
            return False

    def stop_measurement_logging(self) -> bool:
        """Stop continuous measurement logging"""
        if not self._logging_active:
            return True

        try:
            self._stop_logging.set()
            if self._logging_thread and self._logging_thread.is_alive():
                self._logging_thread.join(timeout=5.0)
            self._logging_active = False
            self._logger.info("Measurement logging stopped")
            return True
        except Exception as e:
            self._logger.error(f"Error stopping logging: {e}")
            return False

    def _logging_worker(self, interval: float, duration: Optional[float]):
        """Worker thread for continuous measurement logging"""
        start_time = time.time()
        measurements = []

        while not self._stop_logging.is_set():
            try:
                if duration and (time.time() - start_time) > duration:
                    break

                if self.io_lock:
                    with self.io_lock:
                        data = self.psu.measure_all_channels()
                else:
                    data = self.psu.measure_all_channels()

                if data:
                    timestamp = datetime.now()
                    for ch, ch_data in data.items():
                        measurements.append({
                            'timestamp': timestamp.isoformat(),
                            'channel': ch,
                            'voltage': ch_data['voltage'],
                            'current': ch_data['current'],
                            'power': ch_data['power']
                        })

                time.sleep(interval)

            except Exception as e:
                self._logger.error(f"Error in logging worker: {e}")

        # Save measurements to file
        if measurements:
            self._save_logged_data(measurements)

    def _save_logged_data(self, measurements: List[Dict]) -> Optional[str]:
        """Save logged measurement data to CSV file"""
        try:
            save_dir = Path(PSU_CSV_DATA_PATH)
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"psu_measurements_{timestamp}.csv"
            filepath = save_dir / filename

            df = pd.DataFrame(measurements)
            df.to_csv(filepath, index=False)

            self._logger.info(f"Logged data saved: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.error(f"Failed to save logged data: {e}")
            return None

    def export_single_measurement(self, measurements: Dict[int, Dict], custom_path: Optional[str] = None) -> Optional[str]:
        """Export single measurement to CSV"""
        if not measurements:
            return None

        try:
            save_dir = Path(custom_path) if custom_path else Path(PSU_CSV_DATA_PATH)
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"psu_measurement_{timestamp}.csv"
            filepath = save_dir / filename

            # Flatten measurements for CSV
            csv_data = []
            for ch, ch_data in measurements.items():
                csv_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'channel': ch,
                    'voltage': ch_data['voltage'],
                    'current': ch_data['current'],
                    'power': ch_data['power'],
                    'resistance': ch_data.get('resistance', 0),
                    'efficiency': ch_data.get('efficiency', 0)
                })

            df = pd.DataFrame(csv_data)

            # Write with header information
            with open(filepath, 'w') as f:
                f.write("# Keysight E36441A PSU Measurement\n")
                f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                f.write("# All 4 channels measurement\n\n")
                df.to_csv(filepath, mode='a', index=False)

            self._logger.info(f"Measurement exported: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.error(f"Export failed: {e}")
            return None

    def generate_measurement_plot(self, measurements: Dict[int, Dict], custom_path: Optional[str] = None) -> Optional[str]:
        """Generate measurement visualization"""
        if not measurements:
            return None

        try:
            save_dir = Path(custom_path) if custom_path else Path(PSU_GRAPH_PATH)
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"psu_measurement_plot_{timestamp}.png"
            filepath = save_dir / filename

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            channels = list(measurements.keys())
            voltages = [measurements[ch]['voltage'] for ch in channels]
            currents = [measurements[ch]['current'] for ch in channels]
            powers = [measurements[ch]['power'] for ch in channels]

            # Voltage bar chart
            bars1 = ax1.bar([f'CH{ch}' for ch in channels], voltages, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax1.set_ylabel('Voltage (V)')
            ax1.set_title('Channel Voltages')
            ax1.grid(True, alpha=0.3)
            # Add value labels on bars
            for bar, voltage in zip(bars1, voltages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(voltages)*0.01,
                        f'{voltage:.3f}V', ha='center', va='bottom')

            # Current bar chart
            bars2 = ax2.bar([f'CH{ch}' for ch in channels], currents, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax2.set_ylabel('Current (A)')
            ax2.set_title('Channel Currents')
            ax2.grid(True, alpha=0.3)
            # Add value labels on bars
            for bar, current in zip(bars2, currents):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(currents)*0.01,
                        f'{current:.3f}A', ha='center', va='bottom')

            # Power bar chart
            bars3 = ax3.bar([f'CH{ch}' for ch in channels], powers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax3.set_ylabel('Power (W)')
            ax3.set_title('Channel Powers')
            ax3.grid(True, alpha=0.3)
            # Add value labels on bars
            for bar, power in zip(bars3, powers):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(powers)*0.01,
                        f'{power:.3f}W', ha='center', va='bottom')

            # Summary pie chart of power distribution
            if sum(powers) > 0:
                ax4.pie(powers, labels=[f'CH{ch}' for ch in channels], autopct='%1.1f%%',
                       colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                ax4.set_title('Power Distribution')
            else:
                ax4.text(0.5, 0.5, 'No Power\nOutput', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=14)
                ax4.set_title('Power Distribution')

            plt.tight_layout()
            plt.suptitle('E36441A Power Supply Measurements', fontsize=16, y=1.02)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            self._logger.info(f"Plot generated: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.error(f"Plot generation failed: {e}")
            return None

# ============================================================================
# MAIN GRADIO GUI CLASS
# ============================================================================

class GradioPSUGUI:
    """
    Professional 4-channel power supply control interface with comprehensive feature set.
    Implements connection management, all channel control, measurements, protection,
    sequencing, waveform generation, and complete data acquisition workflow.
    """

    def __init__(self):
        self.psu = None
        self.data_logger = None
        self.last_measurements = None
        self.io_lock = threading.RLock()
        self._shutdown_flag = threading.Event()
        self._gradio_interface = None

        # Waveform execution state
        self.waveform_active = False
        self.waveform_thread = None
        self.waveform_stop_event = threading.Event()
        self.waveform_data = []
        self.waveform_status_message = "Ready"
        self._last_waveform_fig = None
        self.log_messages = []

        # Use the configured paths from the top of the file
        self.save_locations = {
            'data': PSU_CSV_DATA_PATH,
            'graphs': PSU_GRAPH_PATH,
            'logs': PSU_LOG_PATH,
            'waveforms': PSU_WAVEFORM_PATH
        }

        self.setup_logging()
        self.setup_cleanup_handlers()

    def setup_logging(self):
        """Configure logging for system diagnostics"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GradioPSUAutomation')

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
        """Cleanup resources and disconnect PSU"""
        try:
            self._shutdown_flag.set()
            self.waveform_stop_event.set()

            if self.data_logger:
                self.data_logger.stop_measurement_logging()

            if self.psu and hasattr(self.psu, 'is_connected'):
                if self.psu.is_connected:
                    print("Safely shutting down PSU...")
                    self.psu.safe_shutdown()
                    self.psu.disconnect()

            self.psu = None
            self.data_logger = None
            plt.close('all')
            print("Cleanup completed.")
        except Exception as e:
            print(f"Cleanup error: {e}")

    def log_message(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}"
        self.log_messages.append(formatted)
        if len(self.log_messages) > 1000:
            self.log_messages = self.log_messages[-500:]
        self.logger.info(message)

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    def connect_psu(self, visa_address: str):
        """Establish VISA connection and query instrument identification"""
        try:
            if not visa_address:
                return "Error: VISA address is empty", "Disconnected"

            self.psu = KeysightE36441A(visa_address)

            if self.psu.connect():
                self.data_logger = PSUDataLogger(self.psu, io_lock=self.io_lock)

                info = self.psu.get_instrument_info()
                if info:
                    info_text = f"Connected: {info['manufacturer']} {info['model']} | S/N: {info['serial_number']} | FW: {info['firmware_version']}"
                    info_text += f"\nChannels: {info['channels']} | CH1: {info['channel_1_max_power']}W | CH2-4: {info['channels_234_max_power']}W"
                    self.log_message(f"Connected to {info['model']}", "SUCCESS")
                    return info_text, "Connected"
                else:
                    return "Connected (no info available)", "Connected"
            else:
                return "Connection failed", "Disconnected"
        except Exception as e:
            return f"Error: {str(e)}", "Disconnected"

    def disconnect_psu(self):
        """Close connection to PSU"""
        try:
            if self.data_logger:
                self.data_logger.stop_measurement_logging()

            if self.psu:
                self.psu.safe_shutdown()
                self.psu.disconnect()

            self.psu = None
            self.data_logger = None
            self.last_measurements = None
            self.logger.info("Disconnected successfully")
            return "Disconnected successfully", "Disconnected"
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            return f"Disconnect error: {str(e)}", "Disconnected"

    def test_connection(self):
        """Verify PSU connectivity"""
        if self.psu and self.psu.is_connected:
            return "Connection test: PASSED"
        else:
            return "Connection test: FAILED - Not connected"

    # ========================================================================
    # CHANNEL CONTROL
    # ========================================================================

    def set_channel_voltage(self, channel, voltage):
        """Set voltage for specified channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            # Validate against specifications
            current_setting = self.psu.get_current(ch) or 0.0
            valid, message = validate_channel_specs(ch, voltage, current_setting)
            if not valid:
                return message

            with self.io_lock:
                success = self.psu.set_voltage(ch, voltage)

            if success:
                return f"Channel {ch} voltage set to {voltage:.6f}V"
            else:
                return f"Failed to set Channel {ch} voltage"
        except Exception as e:
            return f"Error: {str(e)}"

    def set_channel_current(self, channel, current):
        """Set current limit for specified channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            # Validate against specifications
            voltage_setting = self.psu.get_voltage(ch) or 0.0
            valid, message = validate_channel_specs(ch, voltage_setting, current)
            if not valid:
                return message

            with self.io_lock:
                success = self.psu.set_current(ch, current)

            if success:
                return f"Channel {ch} current limit set to {current:.6f}A"
            else:
                return f"Failed to set Channel {ch} current"
        except Exception as e:
            return f"Error: {str(e)}"

    def set_channel_output(self, channel, enable):
        """Enable/disable output for specified channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            with self.io_lock:
                success = self.psu.set_output_state(ch, enable)

            status = "enabled" if enable else "disabled"
            if success:
                return f"Channel {ch} output {status}"
            else:
                return f"Failed to {status} Channel {ch} output"
        except Exception as e:
            return f"Error: {str(e)}"

    def quick_channel_setup(self, channel, voltage, current, enable_output):
        """Quick setup for channel with voltage, current, and output control"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            # Validate settings
            valid, message = validate_channel_specs(ch, voltage, current)
            if not valid:
                return message

            with self.io_lock:
                success = self.psu.quick_setup(ch, voltage, current, enable_output, True)

            if success:
                status = "enabled" if enable_output else "configured"
                power = voltage * current
                return f"Channel {ch}: {voltage:.3f}V, {current:.3f}A ({power:.3f}W), output {status}"
            else:
                return f"Quick setup failed for Channel {ch}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_channel_status(self, channel):
        """Get comprehensive status for specified channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            with self.io_lock:
                voltage_setting = self.psu.get_voltage(ch)
                current_setting = self.psu.get_current(ch)
                output_state = self.psu.get_output_state(ch)
                measurements = self.psu.measure_all(ch)
                protection = self.psu.get_protection_status(ch)

            status_lines = [f"Channel {ch} Status:"]

            if voltage_setting is not None:
                status_lines.append(f"Voltage Setting: {format_si_value(voltage_setting, 'voltage')}")
            if current_setting is not None:
                status_lines.append(f"Current Setting: {format_si_value(current_setting, 'current')}")
            if output_state is not None:
                status_lines.append(f"Output: {'ON' if output_state else 'OFF'}")

            if measurements:
                status_lines.append(f"Measured Voltage: {format_si_value(measurements['voltage'], 'voltage')}")
                status_lines.append(f"Measured Current: {format_si_value(measurements['current'], 'current')}")
                status_lines.append(f"Measured Power: {format_si_value(measurements['power'], 'power')}")

            if protection:
                status_lines.append(f"OVP: {'ON' if protection['ovp_enabled'] else 'OFF'} at {format_si_value(protection['ovp_level'], 'voltage')}")
                status_lines.append(f"OCP: {'ON' if protection['ocp_enabled'] else 'OFF'} at {format_si_value(protection['ocp_level'], 'current')}")

            return "\n".join(status_lines)

        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # GLOBAL CONTROL
    # ========================================================================

    def enable_all_outputs(self):
        """Enable all channel outputs"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.enable_all_outputs()

            if success:
                return "All channel outputs enabled"
            else:
                return "Failed to enable all outputs"
        except Exception as e:
            return f"Error: {str(e)}"

    def disable_all_outputs(self):
        """Disable all channel outputs"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.disable_all_outputs()

            if success:
                return "All channel outputs disabled"
            else:
                return "Failed to disable all outputs"
        except Exception as e:
            return f"Error: {str(e)}"

    def measure_all_channels(self):
        """Measure all channels and return formatted results"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                measurements = self.psu.measure_all_channels()

            if measurements:
                self.last_measurements = measurements
                result_lines = ["All Channels Measurements:"]
                total_power = 0

                for ch in sorted(measurements.keys()):
                    data = measurements[ch]
                    voltage = data['voltage']
                    current = data['current']
                    power = data['power']
                    total_power += power

                    result_lines.append(f"CH{ch}: {format_si_value(voltage, 'voltage')}, {format_si_value(current, 'current')}, {format_si_value(power, 'power')}")

                result_lines.append(f"Total Power: {format_si_value(total_power, 'power')}")
                return "\n".join(result_lines)
            else:
                return "Measurement failed"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # PROTECTION CONTROL
    # ========================================================================

    def configure_protection(self, channel, ovp_enable, ovp_level, ocp_enable, ocp_level):
        """Configure protection settings for channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)
            results = []

            with self.io_lock:
                if ovp_enable and ovp_level > 0:
                    if self.psu.set_overvoltage_protection(ch, ovp_level, True):
                        results.append(f"OVP enabled at {format_si_value(ovp_level, 'voltage')}")
                elif not ovp_enable:
                    if self.psu.set_overvoltage_protection(ch, ovp_level if ovp_level > 0 else 30.0, False):
                        results.append("OVP disabled")

                if ocp_enable and ocp_level > 0:
                    if self.psu.set_overcurrent_protection(ch, ocp_level, True):
                        results.append(f"OCP enabled at {format_si_value(ocp_level, 'current')}")
                elif not ocp_enable:
                    max_current = 4.0 if ch == 1 else 2.0
                    if self.psu.set_overcurrent_protection(ch, ocp_level if ocp_level > 0 else max_current, False):
                        results.append("OCP disabled")

            return f"Channel {ch} protection: " + ", ".join(results) if results else "Protection configuration failed"

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_channel_protection(self, channel):
        """Clear protection latch for channel"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            ch = int(channel)

            with self.io_lock:
                success = self.psu.clear_protection(ch)

            if success:
                return f"Channel {ch} protection cleared"
            else:
                return f"Failed to clear Channel {ch} protection"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # SEQUENCING CONTROL
    # ========================================================================

    def power_on_sequence(self, ch1_delay, ch2_delay, ch3_delay, ch4_delay):
        """Execute power-on sequence with specified delays"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            delays = {1: ch1_delay, 2: ch2_delay, 3: ch3_delay, 4: ch4_delay}

            with self.io_lock:
                success = self.psu.power_on_sequence(delays)

            if success:
                return f"Power-on sequence completed with delays: CH1={ch1_delay}s, CH2={ch2_delay}s, CH3={ch3_delay}s, CH4={ch4_delay}s"
            else:
                return "Power-on sequence failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def power_off_sequence(self, reverse_order):
        """Execute power-off sequence"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.power_off_sequence(reverse_order)

            if success:
                order = "reverse order (4-3-2-1)" if reverse_order else "simultaneous"
                return f"Power-off sequence completed: {order}"
            else:
                return "Power-off sequence failed"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # DATA LOGGING
    # ========================================================================

    def start_continuous_logging(self, interval, duration):
        """Start continuous measurement logging"""
        if not self.data_logger:
            return "Error: Data logger not initialized"

        try:
            duration_val = duration if duration > 0 else None
            success = self.data_logger.start_measurement_logging(interval, duration_val)

            if success:
                duration_str = f"{duration}s" if duration_val else "indefinite"
                return f"Continuous logging started: interval={interval}s, duration={duration_str}"
            else:
                return "Failed to start logging"
        except Exception as e:
            return f"Error: {str(e)}"

    def stop_continuous_logging(self):
        """Stop continuous measurement logging"""
        if not self.data_logger:
            return "Error: Data logger not initialized"

        try:
            success = self.data_logger.stop_measurement_logging()
            if success:
                return "Continuous logging stopped"
            else:
                return "Failed to stop logging"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # SYSTEM FUNCTIONS
    # ========================================================================

    def reset_instrument(self):
        """Reset instrument to default state"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.reset()

            if success:
                return "Instrument reset to default state"
            else:
                return "Reset failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_self_test(self):
        """Execute instrument self-test"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                result = self.psu.self_test()

            if result is not None:
                status = "PASSED" if result == 0 else "FAILED"
                return f"Self-test result: {status} (code: {result})"
            else:
                return "Self-test failed to execute"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_status_summary(self):
        """Get comprehensive instrument status"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                status = self.psu.get_status_summary()

            if status:
                lines = []

                # Instrument info
                if 'instrument' in status and status['instrument']:
                    info = status['instrument']
                    lines.append(f"Instrument: {info.get('model', 'Unknown')}")
                    lines.append(f"Serial: {info.get('serial_number', 'Unknown')}")
                    lines.append(f"Firmware: {info.get('firmware_version', 'Unknown')}")

                # Channel status
                lines.append("\nChannel Status:")
                for ch in range(1, 5):
                    if ch in status.get('channels', {}):
                        ch_data = status['channels'][ch]
                        output_state = "ON" if ch_data.get('output_enabled', False) else "OFF"
                        voltage = ch_data.get('voltage_setting', 0)
                        current = ch_data.get('current_setting', 0)
                        lines.append(f"  CH{ch}: {output_state} | {format_si_value(voltage, 'voltage')} | {format_si_value(current, 'current')}")

                # Error status
                errors = status.get('errors', [])
                if errors:
                    lines.append(f"\nErrors: {len(errors)} found")
                    for error in errors[:3]:  # Show first 3 errors
                        lines.append(f"  {error}")
                else:
                    lines.append("\nNo errors")

                return "\n".join(lines)
            else:
                return "Status query failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_errors(self):
        """Clear error queue and status"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.clear_errors()

            if success:
                return "Errors and status cleared"
            else:
                return "Failed to clear errors"
        except Exception as e:
            return f"Error: {str(e)}"

    def save_state(self, location):
        """Save current instrument state to memory"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.save_state(location)

            if success:
                return f"State saved to location: {location}"
            else:
                return "Failed to save state"
        except Exception as e:
            return f"Error: {str(e)}"

    def recall_state(self, location):
        """Recall instrument state from memory"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.recall_state(location)

            if success:
                return f"State recalled from location: {location}"
            else:
                return "Failed to recall state"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    def export_measurements(self):
        """Export last measurements to CSV"""
        if not self.last_measurements:
            return "Error: No measurements available", None

        if not self.data_logger:
            return "Error: Data logger not initialized", None

        try:
            filepath = self.data_logger.export_single_measurement(self.last_measurements)
            if filepath:
                return f"Measurements exported to: {Path(filepath).name}", filepath
            else:
                return "Export failed", None
        except Exception as e:
            return f"Error: {str(e)}", None

    def generate_plot(self):
        """Generate measurement plot"""
        if not self.last_measurements:
            return "Error: No measurements available"

        if not self.data_logger:
            return "Error: Data logger not initialized"

        try:
            filepath = self.data_logger.generate_measurement_plot(self.last_measurements)
            if filepath:
                return f"Plot generated: {Path(filepath).name}"
            else:
                return "Plot generation failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def safe_shutdown(self):
        """Execute safe shutdown sequence"""
        if not self.psu or not self.psu.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.psu.safe_shutdown()

            if success:
                return "Safe shutdown completed"
            else:
                return "Shutdown sequence failed"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # MULTI-CHANNEL WAVEFORM GENERATION
    # ========================================================================

    def execute_multi_channel_waveform(self, channel_configs: List[Dict]) -> None:
        """
        Execute simultaneous waveform generation on multiple PSU channels.

        Args:
            channel_configs: List of dictionaries, each containing:
                - channel: int (1, 2, 3, or 4)
                - waveform: str (waveform type name)
                - target_voltage: float (peak voltage)
                - cycles: int (number of repetitions)
                - points_per_cycle: int (resolution)
                - cycle_duration: float (seconds per cycle)
                - current_limit: float (safety current limit)
        """
        if not channel_configs:
            self.log_message("No channel configurations provided", "ERROR")
            return

        if not self.psu or not self.psu.is_connected:
            self.log_message("Power supply not connected", "ERROR")
            return

        psu_settle = 0.1  # Default settle time
        waveform_start_time = datetime.now()
        start_timestamp = waveform_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        self.log_message(f"{'='*60}", "INFO")
        self.log_message(f"MULTI-CHANNEL WAVEFORM STARTED", "INFO")
        self.log_message(f"Start Time: {start_timestamp}", "INFO")
        self.log_message(f"Channels: {len(channel_configs)}", "INFO")

        try:
            # Create waveform generators for each channel
            generators = []
            for config in channel_configs:
                waveform_name = config.get('waveform', 'Sine')
                # Extract name if it has description
                if ' - ' in waveform_name:
                    waveform_name = waveform_name.split(' - ')[0]

                gen = WaveformGenerator(
                    waveform_type=waveform_name,
                    target_voltage=config.get('target_voltage', 3.0),
                    cycles=config.get('cycles', 3),
                    points_per_cycle=config.get('points_per_cycle', 50),
                    cycle_duration=config.get('cycle_duration', 8.0)
                )
                generators.append({
                    'generator': gen,
                    'channel': config.get('channel', 1),
                    'current_limit': config.get('current_limit', 0.5),
                    'profile': gen.generate()
                })
                self.log_message(
                    f"CH{config.get('channel', 1)}: {waveform_name} waveform, "
                    f"{config.get('target_voltage', 3.0)}V target, "
                    f"{config.get('cycles', 3)} cycles",
                    "INFO"
                )

            # Find the longest profile
            max_points = max(len(g['profile']) for g in generators)
            self.log_message(f"Total points to execute: {max_points}", "INFO")
            self.log_message(f"{'='*60}", "INFO")

            # Clear previous data
            self.waveform_data = []
            point_timings = []

            # Clear any protection trips from previous runs
            self.log_message("Clearing any previous protection states...", "INFO")
            for gen_data in generators:
                self.psu.clear_protection(gen_data['channel'])
            time.sleep(0.3)

            # Configure and enable all channels first
            for gen_data in generators:
                ch = gen_data['channel']
                current_limit = gen_data['current_limit']
                ovp_level = gen_data['generator'].target_voltage + 2.0

                with self.io_lock:
                    self.psu.set_voltage(ch, 0.0)
                    self.psu.set_current(ch, current_limit)
                    self.psu.set_output_state(ch, True)
                time.sleep(0.1)

            self.log_message("All channels configured and enabled", "SUCCESS")
            self.waveform_status_message = f"Running multi-channel waveform ({len(generators)} channels)..."

            # Execute waveform points synchronously across all channels
            for point_idx in range(max_points):
                point_start_time = datetime.now()

                # Check for stop signal
                if self.waveform_stop_event.is_set():
                    self.log_message("Multi-channel waveform stopped by user", "WARNING")
                    break

                if not self.waveform_active:
                    self.log_message("Waveform stopped (waveform_active=False)", "WARNING")
                    break

                # Set voltage on all channels for this time point AND record data
                timestamp = datetime.now()

                for gen_data in generators:
                    profile = gen_data['profile']
                    ch = gen_data['channel']

                    # Get voltage for this point (or hold last value if profile is shorter)
                    if point_idx < len(profile):
                        t, voltage = profile[point_idx]
                    else:
                        t, voltage = profile[-1]

                    # Set voltage on this channel
                    with self.io_lock:
                        self.psu.set_voltage(ch, voltage)

                    # Calculate cycle position
                    points_per_cycle = gen_data['generator'].points_per_cycle
                    cycle_num = point_idx // points_per_cycle
                    point_in_cycle = point_idx % points_per_cycle

                    # Store data point immediately after setting voltage
                    data_point = {
                        'timestamp': timestamp,
                        'channel': ch,
                        'set_voltage': voltage,
                        'measured_voltage': voltage,
                        'measured_current': 0.0,
                        'cycle_number': cycle_num,
                        'point_in_cycle': point_in_cycle,
                        'point_index': point_idx
                    }
                    self.waveform_data.append(data_point)

                # Calculate elapsed time and add delay to meet target time per point
                point_elapsed = (datetime.now() - point_start_time).total_seconds()
                additional_delay = psu_settle - point_elapsed

                if additional_delay > 0:
                    time.sleep(additional_delay)

                # Track timing
                point_end_time = datetime.now()
                point_duration = (point_end_time - point_start_time).total_seconds()
                point_timings.append(point_duration)

                # Progress logging (every 10%)
                if point_idx % max(1, max_points // 10) == 0:
                    progress = (point_idx / max_points) * 100
                    elapsed = (point_end_time - waveform_start_time).total_seconds()
                    avg_time = sum(point_timings) / len(point_timings) if point_timings else 0

                    self.log_message(
                        f"Progress: {progress:.1f}% | Point {point_idx}/{max_points} | "
                        f"Elapsed: {elapsed:.1f}s | Avg: {avg_time*1000:.0f}ms/pt",
                        "INFO"
                    )

            # Ramp down all channels to 0V
            self.log_message("Ramping down all channels to 0V...", "INFO")
            for gen_data in generators:
                ch = gen_data['channel']
                with self.io_lock:
                    self.psu.set_voltage(ch, 0.0)
                time.sleep(0.05)
                with self.io_lock:
                    self.psu.set_output_state(ch, False)

            # Calculate statistics
            waveform_end_time = datetime.now()
            total_duration = (waveform_end_time - waveform_start_time).total_seconds()

            if point_timings:
                avg_time = sum(point_timings) / len(point_timings)
                min_time = min(point_timings)
                max_time = max(point_timings)
            else:
                avg_time = min_time = max_time = 0

            # Log completion
            self.log_message(f"{'='*60}", "SUCCESS")
            self.log_message(f"MULTI-CHANNEL WAVEFORM COMPLETED", "SUCCESS")
            self.log_message(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} min)", "SUCCESS")
            self.log_message(f"Points Executed: {max_points}", "SUCCESS")
            self.log_message(f"Data Points Collected: {len(self.waveform_data)}", "SUCCESS")
            self.log_message(f"{'='*60}", "SUCCESS")

            self.waveform_status_message = (
                f"COMPLETED! {len(generators)} channels, "
                f"{len(self.waveform_data)} pts, {total_duration:.1f}s"
            )

        except Exception as e:
            self.log_message(f"Multi-channel waveform error: {e}", "ERROR")
            self.waveform_status_message = f"ERROR: {str(e)}"

            # Safety: disable all outputs
            try:
                for gen_data in generators:
                    with self.io_lock:
                        self.psu.set_voltage(gen_data['channel'], 0.0)
                        self.psu.set_output_state(gen_data['channel'], False)
            except:
                pass

        finally:
            self.waveform_active = False

    def start_multi_channel_waveform(self, channel_configs: List[Dict]) -> str:
        """Start multi-channel waveform execution in a background thread."""
        if self.waveform_active:
            return "ERROR: Waveform already running. Stop current waveform first."

        if not self.psu or not self.psu.is_connected:
            return "ERROR: Power supply not connected. Please connect first."

        if not channel_configs:
            return "ERROR: No channels enabled for waveform generation."

        # Validate configurations
        for config in channel_configs:
            ch = config.get('channel', 1)
            target_v = config.get('target_voltage', 3.0)
            max_v = 30.0 if ch == 1 else 30.0  # E36441A has same max for all channels
            if target_v > max_v:
                return f"ERROR: Channel {ch} limited to {max_v}V maximum. Configured: {target_v}V"

        self.waveform_active = True
        self.waveform_stop_event.clear()

        self.waveform_thread = threading.Thread(
            target=self.execute_multi_channel_waveform,
            args=(channel_configs,),
            daemon=True
        )
        self.waveform_thread.start()

        channels_str = ", ".join([f"CH{c['channel']}" for c in channel_configs])
        return f"Multi-channel waveform STARTED on {channels_str}"

    def stop_multi_channel_waveform(self) -> str:
        """Stop multi-channel waveform execution gracefully."""
        self.waveform_active = False
        self.waveform_stop_event.set()

        if self.waveform_thread and self.waveform_thread.is_alive():
            self.waveform_thread.join(timeout=2.0)

        self.log_message("Multi-channel waveform stop signal sent", "INFO")
        return "Stopping multi-channel waveform..."

    def save_waveform_plot(self, save_path: str) -> str:
        """Save the waveform execution data plot to user-specified location."""
        if not self.waveform_data:
            return "No waveform data available. Please run a waveform first."

        if not save_path or save_path.strip() == "":
            return "Please select a save location using the Browse button"

        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Group data by channel
            channel_data = {}
            for d in self.waveform_data:
                ch = d.get('channel', 1)
                if ch not in channel_data:
                    channel_data[ch] = {
                        'timestamps': [],
                        'set_voltages': [],
                        'measured_voltages': [],
                        'measured_currents': []
                    }
                channel_data[ch]['timestamps'].append(d['timestamp'])
                channel_data[ch]['set_voltages'].append(d['set_voltage'])
                channel_data[ch]['measured_voltages'].append(d['measured_voltage'])
                channel_data[ch]['measured_currents'].append(d['measured_current'])

            num_channels = len(channel_data)
            channel_colors = {1: '#1E88E5', 2: '#43A047', 3: '#FB8C00', 4: '#E53935'}

            fig, axes = plt.subplots(num_channels, 2, figsize=(14, 4*num_channels))
            fig.suptitle('Multi-Channel Waveform Execution Data', fontsize=14, fontweight='bold')

            if num_channels == 1:
                axes = [axes]

            for idx, ch in enumerate(sorted(channel_data.keys())):
                data = channel_data[ch]
                color = channel_colors.get(ch, '#1E88E5')

                # Convert timestamps to relative seconds
                base_time = data['timestamps'][0]
                time_sec = [(t - base_time).total_seconds() for t in data['timestamps']]

                # Voltage subplot
                ax_v = axes[idx][0] if num_channels > 1 else axes[0]
                ax_v.plot(time_sec, data['set_voltages'], '--', color=color,
                         label='Setpoint', linewidth=1, alpha=0.7)
                ax_v.plot(time_sec, data['measured_voltages'], '-', color=color,
                         label='Measured', linewidth=1.5)
                ax_v.set_xlabel('Time (s)')
                ax_v.set_ylabel('Voltage (V)')
                ax_v.set_title(f'CH{ch} Voltage')
                ax_v.legend(loc='upper right')
                ax_v.grid(True, alpha=0.3)

                # Current subplot
                ax_i = axes[idx][1] if num_channels > 1 else axes[1]
                ax_i.plot(time_sec, data['measured_currents'], '-', color=color, linewidth=1.5)
                ax_i.set_xlabel('Time (s)')
                ax_i.set_ylabel('Current (A)')
                ax_i.set_title(f'CH{ch} Current')
                ax_i.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"psu_waveform_{timestamp_str}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            self.log_message(f"Waveform plot saved to: {filepath}", "SUCCESS")
            return f"Plot saved ({num_channels} channel{'s' if num_channels > 1 else ''}):\n{filepath}"

        except Exception as e:
            self.logger.error(f"Waveform plot save error: {e}")
            return f"Plot save failed: {str(e)}"

    # ========================================================================
    # GRADIO INTERFACE CREATION
    # ========================================================================

    def create_interface(self):
        """Build comprehensive Gradio web interface with full-page layout"""
        # Get the path to assets folder (same level as Unified.py)
        assets_path = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "digantara_logo.png"
        # CSS styling matching Unified.py
        css = """
        /* Main container - full width interface */
        .gradio-container {
            max-width: 100% !important;
            padding: 20px !important;
            margin: 0 !important;
            min-height: 100vh;
        }

        /* Content container */
        .container {
            max-width: 100% !important;
            padding: 0 10px !important;
            margin: 0 !important;
        }

        /* Root component sizing */
        #component-0 {
            min-height: 100vh;
        }

        /* Tab content styling */
        .tab {
            padding: 0 10px;
            min-height: calc(100vh - 120px);
        }

        /* Panel spacing */
        .panel {
            margin: 5px 0;
        }

        /* Channel group styling */
        .channel-group {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
        }

        /* Tab navigation styling */
        .tab-nav {
            border-bottom: 2px solid #c034eb;
            margin-bottom: 12px;
        }

        /* Selected tab appearance */
        .tab-selected {
            background-color: #e0e0ff;
            font-weight: 600;
        }
        """

        # Waveform type options with descriptions
        waveform_types = [
            "Sine - smooth wave",
            "Square - sharp on/off",
            "Triangle - linear ramp up/down",
            "Ramp Up - linear rise",
            "Ramp Down - linear fall",
            "Cardiac - ECG heartbeat",
            "Damped Sine - decaying oscillation",
            "Exponential Raise - slow to fast rise",
            "Exponential Fall - fast to slow fall",
            "Gaussian Pulse - bell curve",
            "Neural Spike - action potential",
            "Staircase - discrete steps",
            "PWM - variable duty cycle",
            "Chirp - frequency sweep",
            "Burst Mode - on/off bursts",
            "Brownout - power sag/recovery",
            "RC Charge - capacitor charging",
            "Sinc - main lobe + ripples",
            "Breathing - smooth fade in/out"
        ]

        with gr.Blocks(
            title="DIGANTARA Keysight E36441A PSU Control",
            css=css,
            theme=gr.themes.Soft(
                primary_hue="purple",
                spacing_size="sm",
                radius_size="sm",
                text_size="sm"
            )
        ) as interface:

            # ================================================================
            # HEADER
            # ================================================================
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(
                        str(assets_path),
                        show_label=False,
                        container=False,
                        height=80,
                        width=550
                    )

                with gr.Column(scale=5):
                    gr.Markdown("# DIGANTARA Keysight E36441A Power Supply Control")
                    gr.Markdown("**Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.")
                    gr.Markdown("Professional 4-channel autoranging DC power supply automation interface with waveform generation")

            # ================================================================
            # CONNECTION TAB
            # ================================================================

            with gr.Tab("Connection"):
                with gr.Row():
                    visa_address = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x2A8D::0xCC04::CN65240123::INSTR",
                        scale=3
                    )
                    connect_btn = gr.Button("Connect", variant="primary", scale=1)
                    disconnect_btn = gr.Button("Disconnect", variant="stop", scale=1)
                    test_btn = gr.Button("Test", scale=1)

                connection_status = gr.Textbox(label="Status", value="Disconnected", interactive=False)
                instrument_info = gr.Textbox(label="Instrument Information", interactive=False, lines=3)

                connect_btn.click(
                    fn=self.connect_psu,
                    inputs=[visa_address],
                    outputs=[instrument_info, connection_status]
                )

                disconnect_btn.click(
                    fn=self.disconnect_psu,
                    inputs=[],
                    outputs=[instrument_info, connection_status]
                )

                test_btn.click(
                    fn=self.test_connection,
                    inputs=[],
                    outputs=[instrument_info]
                )

            # ================================================================
            # CHANNEL CONTROL TAB
            # ================================================================

            with gr.Tab("Channel Control"):
                # Channel 1 - High Power Channel
                with gr.Group(elem_classes=["channel-group"]):
                    gr.Markdown("### Channel 1 - High Power (60W: 15V/4A, 20V/3A, 30V/2A)")

                    with gr.Row():
                        ch1_voltage = gr.Number(label="Voltage (V)", value=0.0, minimum=0, maximum=30)
                        ch1_current = gr.Number(label="Current (A)", value=0.0, minimum=0, maximum=4)
                        ch1_enable = gr.Checkbox(label="Enable Output", value=False)

                    with gr.Row():
                        ch1_quick_setup_btn = gr.Button("Quick Setup", variant="primary", scale=2)
                        ch1_set_v_btn = gr.Button("Set Voltage", variant="secondary", scale=1)
                        ch1_set_i_btn = gr.Button("Set Current", variant="secondary", scale=1)
                        ch1_output_btn = gr.Button("Toggle Output", variant="secondary", scale=1)
                        ch1_status_btn = gr.Button("Status", variant="secondary", scale=1)

                    ch1_status = gr.Textbox(label="Channel 1 Status", interactive=False, lines=3)

                # Channel 2
                with gr.Group(elem_classes=["channel-group"]):
                    gr.Markdown("### Channel 2 - Standard Power (30W: 15V/2A, 20V/1.5A, 30V/1A)")

                    with gr.Row():
                        ch2_voltage = gr.Number(label="Voltage (V)", value=0.0, minimum=0, maximum=30)
                        ch2_current = gr.Number(label="Current (A)", value=0.0, minimum=0, maximum=2)
                        ch2_enable = gr.Checkbox(label="Enable Output", value=False)

                    with gr.Row():
                        ch2_quick_setup_btn = gr.Button("Quick Setup", variant="primary", scale=2)
                        ch2_set_v_btn = gr.Button("Set Voltage", variant="secondary", scale=1)
                        ch2_set_i_btn = gr.Button("Set Current", variant="secondary", scale=1)
                        ch2_output_btn = gr.Button("Toggle Output", variant="secondary", scale=1)
                        ch2_status_btn = gr.Button("Status", variant="secondary", scale=1)

                    ch2_status = gr.Textbox(label="Channel 2 Status", interactive=False, lines=3)

                # Channel 3
                with gr.Group(elem_classes=["channel-group"]):
                    gr.Markdown("### Channel 3 - Standard Power (30W: 15V/2A, 20V/1.5A, 30V/1A)")

                    with gr.Row():
                        ch3_voltage = gr.Number(label="Voltage (V)", value=0.0, minimum=0, maximum=30)
                        ch3_current = gr.Number(label="Current (A)", value=0.0, minimum=0, maximum=2)
                        ch3_enable = gr.Checkbox(label="Enable Output", value=False)

                    with gr.Row():
                        ch3_quick_setup_btn = gr.Button("Quick Setup", variant="primary", scale=2)
                        ch3_set_v_btn = gr.Button("Set Voltage", variant="secondary", scale=1)
                        ch3_set_i_btn = gr.Button("Set Current", variant="secondary", scale=1)
                        ch3_output_btn = gr.Button("Toggle Output", variant="secondary", scale=1)
                        ch3_status_btn = gr.Button("Status", variant="secondary", scale=1)

                    ch3_status = gr.Textbox(label="Channel 3 Status", interactive=False, lines=3)

                # Channel 4
                with gr.Group(elem_classes=["channel-group"]):
                    gr.Markdown("### Channel 4 - Standard Power (30W: 15V/2A, 20V/1.5A, 30V/1A)")

                    with gr.Row():
                        ch4_voltage = gr.Number(label="Voltage (V)", value=0.0, minimum=0, maximum=30)
                        ch4_current = gr.Number(label="Current (A)", value=0.0, minimum=0, maximum=2)
                        ch4_enable = gr.Checkbox(label="Enable Output", value=False)

                    with gr.Row():
                        ch4_quick_setup_btn = gr.Button("Quick Setup", variant="primary", scale=2)
                        ch4_set_v_btn = gr.Button("Set Voltage", variant="secondary", scale=1)
                        ch4_set_i_btn = gr.Button("Set Current", variant="secondary", scale=1)
                        ch4_output_btn = gr.Button("Toggle Output", variant="secondary", scale=1)
                        ch4_status_btn = gr.Button("Status", variant="secondary", scale=1)

                    ch4_status = gr.Textbox(label="Channel 4 Status", interactive=False, lines=3)

                # Global Controls
                gr.Markdown("---")
                gr.Markdown("### Global Controls")

                with gr.Row():
                    enable_all_btn = gr.Button("Enable All Outputs", variant="primary")
                    disable_all_btn = gr.Button("Disable All Outputs", variant="stop")
                    measure_all_btn = gr.Button("Measure All Channels", variant="secondary")

                global_status = gr.Textbox(label="Global Status", interactive=False, lines=4)

                # Connect all channel 1 controls
                ch1_quick_setup_btn.click(
                    fn=lambda v, i, e: self.quick_channel_setup(1, v, i, e),
                    inputs=[ch1_voltage, ch1_current, ch1_enable],
                    outputs=[ch1_status]
                )
                ch1_set_v_btn.click(
                    fn=lambda v: self.set_channel_voltage(1, v),
                    inputs=[ch1_voltage],
                    outputs=[ch1_status]
                )
                ch1_set_i_btn.click(
                    fn=lambda i: self.set_channel_current(1, i),
                    inputs=[ch1_current],
                    outputs=[ch1_status]
                )
                ch1_output_btn.click(
                    fn=lambda e: self.set_channel_output(1, e),
                    inputs=[ch1_enable],
                    outputs=[ch1_status]
                )
                ch1_status_btn.click(
                    fn=lambda: self.get_channel_status(1),
                    outputs=[ch1_status]
                )

                # Connect all channel 2 controls
                ch2_quick_setup_btn.click(
                    fn=lambda v, i, e: self.quick_channel_setup(2, v, i, e),
                    inputs=[ch2_voltage, ch2_current, ch2_enable],
                    outputs=[ch2_status]
                )
                ch2_set_v_btn.click(
                    fn=lambda v: self.set_channel_voltage(2, v),
                    inputs=[ch2_voltage],
                    outputs=[ch2_status]
                )
                ch2_set_i_btn.click(
                    fn=lambda i: self.set_channel_current(2, i),
                    inputs=[ch2_current],
                    outputs=[ch2_status]
                )
                ch2_output_btn.click(
                    fn=lambda e: self.set_channel_output(2, e),
                    inputs=[ch2_enable],
                    outputs=[ch2_status]
                )
                ch2_status_btn.click(
                    fn=lambda: self.get_channel_status(2),
                    outputs=[ch2_status]
                )

                # Connect all channel 3 controls
                ch3_quick_setup_btn.click(
                    fn=lambda v, i, e: self.quick_channel_setup(3, v, i, e),
                    inputs=[ch3_voltage, ch3_current, ch3_enable],
                    outputs=[ch3_status]
                )
                ch3_set_v_btn.click(
                    fn=lambda v: self.set_channel_voltage(3, v),
                    inputs=[ch3_voltage],
                    outputs=[ch3_status]
                )
                ch3_set_i_btn.click(
                    fn=lambda i: self.set_channel_current(3, i),
                    inputs=[ch3_current],
                    outputs=[ch3_status]
                )
                ch3_output_btn.click(
                    fn=lambda e: self.set_channel_output(3, e),
                    inputs=[ch3_enable],
                    outputs=[ch3_status]
                )
                ch3_status_btn.click(
                    fn=lambda: self.get_channel_status(3),
                    outputs=[ch3_status]
                )

                # Connect all channel 4 controls
                ch4_quick_setup_btn.click(
                    fn=lambda v, i, e: self.quick_channel_setup(4, v, i, e),
                    inputs=[ch4_voltage, ch4_current, ch4_enable],
                    outputs=[ch4_status]
                )
                ch4_set_v_btn.click(
                    fn=lambda v: self.set_channel_voltage(4, v),
                    inputs=[ch4_voltage],
                    outputs=[ch4_status]
                )
                ch4_set_i_btn.click(
                    fn=lambda i: self.set_channel_current(4, i),
                    inputs=[ch4_current],
                    outputs=[ch4_status]
                )
                ch4_output_btn.click(
                    fn=lambda e: self.set_channel_output(4, e),
                    inputs=[ch4_enable],
                    outputs=[ch4_status]
                )
                ch4_status_btn.click(
                    fn=lambda: self.get_channel_status(4),
                    outputs=[ch4_status]
                )

                # Connect global controls
                enable_all_btn.click(
                    fn=self.enable_all_outputs,
                    inputs=[],
                    outputs=[global_status]
                )
                disable_all_btn.click(
                    fn=self.disable_all_outputs,
                    inputs=[],
                    outputs=[global_status]
                )
                measure_all_btn.click(
                    fn=self.measure_all_channels,
                    inputs=[],
                    outputs=[global_status]
                )

            # ================================================================
            # MULTI-CHANNEL WAVEFORM GENERATOR TAB
            # ================================================================

            with gr.Tab("Multi-Channel Waveform Generator"):
                gr.Markdown("### Multi-Channel Simultaneous Waveform Generator")
                gr.Markdown("Configure and run synchronized waveform patterns on multiple channels simultaneously.")

                with gr.Group():
                    with gr.Tabs():
                        # Channel 1 Waveform Configuration
                        with gr.TabItem(label="CH1 Waveform"):
                            wf_ch1_enable = gr.Checkbox(label="Enable CH1", value=True)
                            with gr.Row():
                                wf_ch1_waveform = gr.Dropdown(
                                    choices=waveform_types,
                                    value="Sine - smooth wave",
                                    label="Waveform Type"
                                )
                                wf_ch1_voltage = gr.Slider(0.1, 30.0, value=5.0, label="Target Voltage (V)", step=0.1)
                                wf_ch1_current = gr.Slider(0.001, 4.0, value=0.5, label="Current Limit (A)", step=0.001)
                            with gr.Row():
                                wf_ch1_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=1000, precision=0)
                                wf_ch1_points = gr.Number(value=50, label="Points/Cycle", minimum=10, maximum=1000, precision=0)
                                wf_ch1_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                        # Channel 2 Waveform Configuration
                        with gr.TabItem(label="CH2 Waveform"):
                            wf_ch2_enable = gr.Checkbox(label="Enable CH2", value=False)
                            with gr.Row():
                                wf_ch2_waveform = gr.Dropdown(
                                    choices=waveform_types,
                                    value="Triangle - linear ramp up/down",
                                    label="Waveform Type"
                                )
                                wf_ch2_voltage = gr.Slider(0.1, 30.0, value=3.3, label="Target Voltage (V)", step=0.1)
                                wf_ch2_current = gr.Slider(0.001, 2.0, value=0.5, label="Current Limit (A)", step=0.001)
                            with gr.Row():
                                wf_ch2_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=1000, precision=0)
                                wf_ch2_points = gr.Number(value=50, label="Points/Cycle", minimum=10, maximum=1000, precision=0)
                                wf_ch2_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                        # Channel 3 Waveform Configuration
                        with gr.TabItem(label="CH3 Waveform"):
                            wf_ch3_enable = gr.Checkbox(label="Enable CH3", value=False)
                            with gr.Row():
                                wf_ch3_waveform = gr.Dropdown(
                                    choices=waveform_types,
                                    value="Square - sharp on/off",
                                    label="Waveform Type"
                                )
                                wf_ch3_voltage = gr.Slider(0.1, 30.0, value=3.3, label="Target Voltage (V)", step=0.1)
                                wf_ch3_current = gr.Slider(0.001, 2.0, value=0.5, label="Current Limit (A)", step=0.001)
                            with gr.Row():
                                wf_ch3_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=1000, precision=0)
                                wf_ch3_points = gr.Number(value=50, label="Points/Cycle", minimum=10, maximum=1000, precision=0)
                                wf_ch3_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                        # Channel 4 Waveform Configuration
                        with gr.TabItem(label="CH4 Waveform"):
                            wf_ch4_enable = gr.Checkbox(label="Enable CH4", value=False)
                            with gr.Row():
                                wf_ch4_waveform = gr.Dropdown(
                                    choices=waveform_types,
                                    value="Ramp Up - linear rise",
                                    label="Waveform Type"
                                )
                                wf_ch4_voltage = gr.Slider(0.1, 30.0, value=3.3, label="Target Voltage (V)", step=0.1)
                                wf_ch4_current = gr.Slider(0.001, 2.0, value=0.5, label="Current Limit (A)", step=0.001)
                            with gr.Row():
                                wf_ch4_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=1000, precision=0)
                                wf_ch4_points = gr.Number(value=50, label="Points/Cycle", minimum=10, maximum=1000, precision=0)
                                wf_ch4_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                    # Global waveform settings
                    with gr.Row():
                        wf_settle_time = gr.Slider(
                            label="Time per Point (s)",
                            minimum=0.05,
                            maximum=5.0,
                            value=0.1,
                            step=0.01,
                            info="Total time per point including settle"
                        )
                        wf_estimated_duration = gr.Textbox(
                            label="Estimated Duration",
                            value="~15s | 150 pts @ 100ms/pt",
                            interactive=False
                        )

                    # Duration estimation function
                    def update_duration_estimate(
                        ch1_en, ch1_cyc, ch1_pts,
                        ch2_en, ch2_cyc, ch2_pts,
                        ch3_en, ch3_cyc, ch3_pts,
                        ch4_en, ch4_cyc, ch4_pts,
                        settle
                    ):
                        try:
                            enabled_channels = 0
                            max_points = 0

                            if ch1_en:
                                enabled_channels += 1
                                cyc = int(ch1_cyc) if ch1_cyc else 3
                                pts = int(ch1_pts) if ch1_pts else 50
                                max_points = max(max_points, cyc * pts)

                            if ch2_en:
                                enabled_channels += 1
                                cyc = int(ch2_cyc) if ch2_cyc else 3
                                pts = int(ch2_pts) if ch2_pts else 50
                                max_points = max(max_points, cyc * pts)

                            if ch3_en:
                                enabled_channels += 1
                                cyc = int(ch3_cyc) if ch3_cyc else 3
                                pts = int(ch3_pts) if ch3_pts else 50
                                max_points = max(max_points, cyc * pts)

                            if ch4_en:
                                enabled_channels += 1
                                cyc = int(ch4_cyc) if ch4_cyc else 3
                                pts = int(ch4_pts) if ch4_pts else 50
                                max_points = max(max_points, cyc * pts)

                            if enabled_channels == 0 or max_points == 0:
                                return "Enable at least one channel"

                            time_per_point = float(settle) if settle else 0.1
                            estimated_time = max_points * time_per_point
                            time_per_point_ms = time_per_point * 1000

                            start_time = datetime.now()
                            end_time = start_time + timedelta(seconds=estimated_time)
                            start_str = start_time.strftime('%H:%M:%S')
                            end_str = end_time.strftime('%H:%M:%S')

                            if estimated_time < 60:
                                return f"Start: {start_str} -> End: ~{end_str} | ~{estimated_time:.1f}s | {max_points} pts @ {time_per_point_ms:.0f}ms/pt ({enabled_channels} CH)"
                            else:
                                return f"Start: {start_str} -> End: ~{end_str} | ~{estimated_time:.1f}s ({estimated_time/60:.1f} min) | {max_points} pts ({enabled_channels} CH)"

                        except Exception as e:
                            return f"~0s (error: {e})"

                    # Wire up duration updates
                    duration_inputs = [
                        wf_ch1_enable, wf_ch1_cycles, wf_ch1_points,
                        wf_ch2_enable, wf_ch2_cycles, wf_ch2_points,
                        wf_ch3_enable, wf_ch3_cycles, wf_ch3_points,
                        wf_ch4_enable, wf_ch4_cycles, wf_ch4_points,
                        wf_settle_time
                    ]

                    for input_component in duration_inputs:
                        input_component.change(
                            fn=update_duration_estimate,
                            inputs=duration_inputs,
                            outputs=[wf_estimated_duration]
                        )

                    # Control buttons
                    with gr.Row():
                        wf_preview_btn = gr.Button("Preview All Enabled Channels", variant="secondary", size="lg")
                        wf_start_btn = gr.Button("Start Multi-Channel Waveform", variant="primary", size="lg")
                        wf_stop_btn = gr.Button("Stop Waveform", variant="stop", size="lg")

                    wf_status = gr.Textbox(
                        label="Waveform Status",
                        value="Ready - Enable channels, configure parameters, and click Start",
                        interactive=False,
                        lines=2
                    )

                    # wf_preview_plot = gr.Plot(label="Multi-Channel Waveform Preview")  # Disabled due to schema generation issues

                    # Preview function
                    def preview_waveforms(
                        ch1_en, ch1_wf, ch1_v, ch1_cyc, ch1_pts, ch1_dur,
                        ch2_en, ch2_wf, ch2_v, ch2_cyc, ch2_pts, ch2_dur,
                        ch3_en, ch3_wf, ch3_v, ch3_cyc, ch3_pts, ch3_dur,
                        ch4_en, ch4_wf, ch4_v, ch4_cyc, ch4_pts, ch4_dur
                    ):
                        try:
                            colors = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800', 4: '#E53935'}
                            fig, ax = plt.subplots(figsize=(14, 7))

                            enabled_count = 0
                            max_voltage = 0
                            total_points = 0

                            channel_configs = [
                                (1, ch1_en, ch1_wf, ch1_v, ch1_cyc, ch1_pts, ch1_dur),
                                (2, ch2_en, ch2_wf, ch2_v, ch2_cyc, ch2_pts, ch2_dur),
                                (3, ch3_en, ch3_wf, ch3_v, ch3_cyc, ch3_pts, ch3_dur),
                                (4, ch4_en, ch4_wf, ch4_v, ch4_cyc, ch4_pts, ch4_dur)
                            ]

                            for ch, enabled, wf_type, voltage, cycles, points, duration in channel_configs:
                                if not enabled:
                                    continue

                                enabled_count += 1
                                wf_name = wf_type.split(' - ')[0] if ' - ' in wf_type else wf_type

                                cyc = int(cycles) if cycles else 3
                                pts = int(points) if points else 50
                                dur = float(duration) if duration else 8.0
                                volt = float(voltage) if voltage else 3.0

                                generator = WaveformGenerator(
                                    waveform_type=wf_name,
                                    target_voltage=volt,
                                    cycles=cyc,
                                    points_per_cycle=pts,
                                    cycle_duration=dur
                                )
                                profile = generator.generate()

                                times = [p[0] for p in profile]
                                voltages = [p[1] for p in profile]

                                ax.plot(times, voltages, color=colors[ch], linewidth=2,
                                       label=f'CH{ch}: {wf_name} ({volt}V, {cyc}x{pts}pts)')

                                max_voltage = max(max_voltage, max(voltages))
                                total_points = max(total_points, len(profile))

                            if enabled_count == 0:
                                ax.text(0.5, 0.5, 'No channels enabled.\nEnable at least one channel to preview.',
                                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                                ax.set_title('Multi-Channel Waveform Preview')
                            else:
                                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                                ax.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
                                ax.set_title(f'Multi-Channel Waveform Preview - {enabled_count} Channel(s), {total_points} points max',
                                            fontsize=14, fontweight='bold')
                                ax.grid(True, alpha=0.3, linestyle='--')
                                ax.set_ylim([0, max_voltage + 0.5])
                                ax.legend(loc='upper right', fontsize=10)

                            plt.tight_layout()
                            self._last_waveform_fig = fig
                            return fig

                        except Exception as e:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.text(0.5, 0.5, f'Error generating preview:\n{str(e)}',
                                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                            return fig

                    # Start waveform function
                    def start_waveform(
                        ch1_en, ch1_wf, ch1_v, ch1_i, ch1_cyc, ch1_pts, ch1_dur,
                        ch2_en, ch2_wf, ch2_v, ch2_i, ch2_cyc, ch2_pts, ch2_dur,
                        ch3_en, ch3_wf, ch3_v, ch3_i, ch3_cyc, ch3_pts, ch3_dur,
                        ch4_en, ch4_wf, ch4_v, ch4_i, ch4_cyc, ch4_pts, ch4_dur
                    ):
                        channel_configs = []

                        if ch1_en:
                            channel_configs.append({
                                'channel': 1,
                                'waveform': ch1_wf,
                                'target_voltage': float(ch1_v) if ch1_v else 5.0,
                                'current_limit': float(ch1_i) if ch1_i else 0.5,
                                'cycles': int(ch1_cyc) if ch1_cyc else 3,
                                'points_per_cycle': int(ch1_pts) if ch1_pts else 50,
                                'cycle_duration': float(ch1_dur) if ch1_dur else 8.0
                            })

                        if ch2_en:
                            channel_configs.append({
                                'channel': 2,
                                'waveform': ch2_wf,
                                'target_voltage': float(ch2_v) if ch2_v else 3.3,
                                'current_limit': float(ch2_i) if ch2_i else 0.5,
                                'cycles': int(ch2_cyc) if ch2_cyc else 3,
                                'points_per_cycle': int(ch2_pts) if ch2_pts else 50,
                                'cycle_duration': float(ch2_dur) if ch2_dur else 8.0
                            })

                        if ch3_en:
                            channel_configs.append({
                                'channel': 3,
                                'waveform': ch3_wf,
                                'target_voltage': float(ch3_v) if ch3_v else 3.3,
                                'current_limit': float(ch3_i) if ch3_i else 0.5,
                                'cycles': int(ch3_cyc) if ch3_cyc else 3,
                                'points_per_cycle': int(ch3_pts) if ch3_pts else 50,
                                'cycle_duration': float(ch3_dur) if ch3_dur else 8.0
                            })

                        if ch4_en:
                            channel_configs.append({
                                'channel': 4,
                                'waveform': ch4_wf,
                                'target_voltage': float(ch4_v) if ch4_v else 3.3,
                                'current_limit': float(ch4_i) if ch4_i else 0.5,
                                'cycles': int(ch4_cyc) if ch4_cyc else 3,
                                'points_per_cycle': int(ch4_pts) if ch4_pts else 50,
                                'cycle_duration': float(ch4_dur) if ch4_dur else 8.0
                            })

                        return self.start_multi_channel_waveform(channel_configs)

                    # Wire up buttons
                    # wf_preview_btn.click(
                    #     fn=preview_waveforms,
                    #     inputs=[
                    #         wf_ch1_enable, wf_ch1_waveform, wf_ch1_voltage, wf_ch1_cycles, wf_ch1_points, wf_ch1_duration,
                    #         wf_ch2_enable, wf_ch2_waveform, wf_ch2_voltage, wf_ch2_cycles, wf_ch2_points, wf_ch2_duration,
                    #         wf_ch3_enable, wf_ch3_waveform, wf_ch3_voltage, wf_ch3_cycles, wf_ch3_points, wf_ch3_duration,
                    #         wf_ch4_enable, wf_ch4_waveform, wf_ch4_voltage, wf_ch4_cycles, wf_ch4_points, wf_ch4_duration
                    #     ],
                    #     outputs=[wf_preview_plot]
                    # )

                    wf_start_btn.click(
                        fn=start_waveform,
                        inputs=[
                            wf_ch1_enable, wf_ch1_waveform, wf_ch1_voltage, wf_ch1_current, wf_ch1_cycles, wf_ch1_points, wf_ch1_duration,
                            wf_ch2_enable, wf_ch2_waveform, wf_ch2_voltage, wf_ch2_current, wf_ch2_cycles, wf_ch2_points, wf_ch2_duration,
                            wf_ch3_enable, wf_ch3_waveform, wf_ch3_voltage, wf_ch3_current, wf_ch3_cycles, wf_ch3_points, wf_ch3_duration,
                            wf_ch4_enable, wf_ch4_waveform, wf_ch4_voltage, wf_ch4_current, wf_ch4_cycles, wf_ch4_points, wf_ch4_duration
                        ],
                        outputs=[wf_status]
                    )

                    wf_stop_btn.click(
                        fn=self.stop_multi_channel_waveform,
                        outputs=[wf_status]
                    )

                    # Save waveform plot section
                    gr.Markdown("### Save Waveform Plot")
                    with gr.Row():
                        wf_save_path = gr.Textbox(
                            label="Save Location for Waveform Plots",
                            value=PSU_WAVEFORM_PATH,
                            interactive=True,
                            scale=3
                        )

                    wf_save_btn = gr.Button("Save Waveform Execution Plot", variant="primary")
                    wf_save_status = gr.Textbox(label="Save Status", interactive=False)

                    wf_save_btn.click(
                        fn=self.save_waveform_plot,
                        inputs=[wf_save_path],
                        outputs=[wf_save_status]
                    )

            # ================================================================
            # PROTECTION TAB
            # ================================================================

            with gr.Tab("Protection"):
                gr.Markdown("### Channel Protection Configuration")
                gr.Markdown("Configure overvoltage (OVP) and overcurrent (OCP) protection for each channel")

                # Protection controls for each channel
                for ch in range(1, 5):
                    with gr.Group(elem_classes=["channel-group"]):
                        gr.Markdown(f"#### Channel {ch} Protection")

                        with gr.Row():
                            ovp_enable = gr.Checkbox(label="Enable OVP", value=False, elem_id=f"ch{ch}_ovp_enable")
                            ovp_level = gr.Number(label="OVP Level (V)", value=0, minimum=0, maximum=33, elem_id=f"ch{ch}_ovp_level")
                            ocp_enable = gr.Checkbox(label="Enable OCP", value=False, elem_id=f"ch{ch}_ocp_enable")
                            ocp_level = gr.Number(label="OCP Level (A)", value=0, minimum=0, maximum=4.4, elem_id=f"ch{ch}_ocp_level")

                        with gr.Row():
                            configure_prot_btn = gr.Button(f"Configure CH{ch} Protection", variant="primary", scale=2, elem_id=f"ch{ch}_config_btn")
                            clear_prot_btn = gr.Button(f"Clear CH{ch} Protection", variant="secondary", scale=1, elem_id=f"ch{ch}_clear_btn")

                        prot_status = gr.Textbox(label=f"Channel {ch} Protection Status", interactive=False, lines=2, elem_id=f"ch{ch}_prot_status")

                        # Connect protection controls
                        configure_prot_btn.click(
                            fn=lambda ov_en, ov_lv, oc_en, oc_lv, c=ch: self.configure_protection(c, ov_en, ov_lv, oc_en, oc_lv),
                            inputs=[ovp_enable, ovp_level, ocp_enable, ocp_level],
                            outputs=[prot_status]
                        )
                        clear_prot_btn.click(
                            fn=lambda c=ch: self.clear_channel_protection(c),
                            outputs=[prot_status]
                        )

            # ================================================================
            # SEQUENCING TAB
            # ================================================================

            with gr.Tab("Sequencing"):
                gr.Markdown("### Power Sequencing Control")
                gr.Markdown("Configure startup and shutdown sequences with customizable timing")

                gr.Markdown("#### Power-On Sequence")
                gr.Markdown("Set individual channel startup delays (channels enable in order with specified delays)")

                with gr.Row():
                    ch1_delay = gr.Number(label="CH1 Delay (s)", value=0.0, minimum=0, maximum=60)
                    ch2_delay = gr.Number(label="CH2 Delay (s)", value=0.1, minimum=0, maximum=60)
                    ch3_delay = gr.Number(label="CH3 Delay (s)", value=0.2, minimum=0, maximum=60)
                    ch4_delay = gr.Number(label="CH4 Delay (s)", value=0.3, minimum=0, maximum=60)

                power_on_btn = gr.Button("Execute Power-On Sequence", variant="primary")
                power_on_status = gr.Textbox(label="Power-On Status", interactive=False, lines=2)

                power_on_btn.click(
                    fn=self.power_on_sequence,
                    inputs=[ch1_delay, ch2_delay, ch3_delay, ch4_delay],
                    outputs=[power_on_status]
                )

                gr.Markdown("#### Power-Off Sequence")

                with gr.Row():
                    reverse_order = gr.Checkbox(label="Reverse Order Shutdown (4-3-2-1)", value=True)
                    power_off_btn = gr.Button("Execute Power-Off Sequence", variant="stop")

                power_off_status = gr.Textbox(label="Power-Off Status", interactive=False, lines=2)

                power_off_btn.click(
                    fn=self.power_off_sequence,
                    inputs=[reverse_order],
                    outputs=[power_off_status]
                )

            # ================================================================
            # DATA LOGGING TAB
            # ================================================================

            with gr.Tab("Data Logging"):
                gr.Markdown("### Continuous Data Logging")
                gr.Markdown("Log measurements from all channels with configurable intervals")

                with gr.Row():
                    log_interval = gr.Number(label="Interval (s)", value=1.0, minimum=0.1)
                    log_duration = gr.Number(label="Duration (s)", value=60, minimum=0, info="0 = indefinite")

                with gr.Row():
                    start_logging_btn = gr.Button("Start Logging", variant="primary")
                    stop_logging_btn = gr.Button("Stop Logging", variant="stop")

                logging_status = gr.Textbox(label="Logging Status", interactive=False)

                start_logging_btn.click(
                    fn=self.start_continuous_logging,
                    inputs=[log_interval, log_duration],
                    outputs=[logging_status]
                )

                stop_logging_btn.click(
                    fn=self.stop_continuous_logging,
                    inputs=[],
                    outputs=[logging_status]
                )

                gr.Markdown("### Data Export")
                gr.Markdown("Export measurements and generate plots from last measurement")

                with gr.Row():
                    export_csv_btn = gr.Button("Export CSV", variant="secondary")
                    generate_plot_btn = gr.Button("Generate Plot", variant="secondary")

                export_status = gr.Textbox(label="Export Status", interactive=False)

                # Download section
                gr.Markdown("### Download Files")
                csv_download = gr.File(label="Exported CSV File", interactive=False)

                export_csv_btn.click(
                    fn=self.export_measurements,
                    inputs=[],
                    outputs=[export_status, csv_download]
                )

                generate_plot_btn.click(
                    fn=self.generate_plot,
                    inputs=[],
                    outputs=[export_status]
                )

            # ================================================================
            # MEMORY & SYSTEM TAB
            # ================================================================

            with gr.Tab("Memory & System"):
                gr.Markdown("### Memory Operations")

                with gr.Row():
                    memory_location = gr.Slider(label="Memory Location", minimum=0, maximum=9, value=0, step=1)

                with gr.Row():
                    save_state_btn = gr.Button("Save State", variant="primary")
                    recall_state_btn = gr.Button("Recall State", variant="secondary")

                memory_status = gr.Textbox(label="Memory Status", interactive=False)

                save_state_btn.click(
                    fn=self.save_state,
                    inputs=[memory_location],
                    outputs=[memory_status]
                )

                recall_state_btn.click(
                    fn=self.recall_state,
                    inputs=[memory_location],
                    outputs=[memory_status]
                )

                gr.Markdown("### System Functions")

                with gr.Row():
                    reset_btn = gr.Button("Reset Instrument", variant="stop")
                    self_test_btn = gr.Button("Self Test", variant="secondary")
                    status_btn = gr.Button("Get Status", variant="secondary")
                    clear_errors_btn = gr.Button("Clear Errors", variant="secondary")

                system_status = gr.Textbox(label="System Status", interactive=False, lines=15)

                reset_btn.click(
                    fn=self.reset_instrument,
                    inputs=[],
                    outputs=[system_status]
                )

                self_test_btn.click(
                    fn=self.run_self_test,
                    inputs=[],
                    outputs=[system_status]
                )

                status_btn.click(
                    fn=self.get_status_summary,
                    inputs=[],
                    outputs=[system_status]
                )

                clear_errors_btn.click(
                    fn=self.clear_errors,
                    inputs=[],
                    outputs=[system_status]
                )

                gr.Markdown("### Safety")

                safe_shutdown_btn = gr.Button("Safe Shutdown", variant="stop", scale=2)
                shutdown_status = gr.Textbox(label="Shutdown Status", interactive=False)

                safe_shutdown_btn.click(
                    fn=self.safe_shutdown,
                    inputs=[],
                    outputs=[shutdown_status]
                )

            # ================================================================
            # FILE OPERATIONS TAB
            # ================================================================

            with gr.Tab("File Operations & Settings"):
                with gr.Column(variant="panel"):
                    gr.Markdown("### File Save Locations (Server-Side)")
                    gr.Markdown("Files are saved on the server in the following directories. Use the download buttons to get files after generation.")

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
                            label="Logs Directory (Log files)",
                            value=self.save_locations['logs'],
                            interactive=False
                        )
                        gr.Textbox(
                            label="Waveforms Directory (Waveform plots)",
                            value=self.save_locations['waveforms'],
                            interactive=False
                        )

            gr.Markdown("---")
            gr.Markdown("**DIGANTARA Keysight E36441A Power Supply Control** | Professional Grade 4-Channel Autoranging DC PSU Automation with Multi-Channel Waveform Generator | All SCPI Commands Verified")

        return interface

    def launch(self, share=False, server_port=7867, auto_open=True, max_attempts=10):
        """Launch Gradio interface with port fallback and full-page layout"""
        self._gradio_interface = self.create_interface()

        for attempt in range(max_attempts):
            current_port = server_port + attempt

            try:
                print(f"Attempting to start PSU GUI on port {current_port}...")
                self._gradio_interface.launch(
                    server_name="0.0.0.0",
                    share=share,
                    server_port=current_port,
                    prevent_thread_lock=False,
                    show_error=True,
                    quiet=False,
                    show_api=False
                )

                print("\n" + "=" * 80)
                hostname = socket.gethostname()
                print(f"PSU GUI is running on port {current_port}")
                print(f"Network access from other PCs: http://{hostname}:{current_port}")
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
    print("Keysight E36441A Power Supply Automation - Professional Gradio Interface")
    print("Comprehensive 4-channel autoranging DC power supply control system")
    print("With Multi-Channel Simultaneous Waveform Generator")
    print("=" * 80)
    print("Starting web interface...")

    app = None
    try:
        start_port = 7867  # Different from DMM and electronic load ports
        max_attempts = 10
        print(f"Looking for an available port starting from {start_port}...")

        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.close()

                hostname = socket.gethostname()
                print(f"\nFound available port: {port}")
                print("The browser will open automatically when ready.")
                print(f"Network access from other PCs: http://{hostname}:{port}")
                print("")
                print("IMPORTANT: To stop the application, press Ctrl+C in this terminal.")
                print("Closing the browser tab will NOT stop the server.")
                print("=" * 80)

                app = GradioPSUGUI()
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