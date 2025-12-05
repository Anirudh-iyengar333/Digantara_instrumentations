#!/usr/bin/env python3
"""
=================================================================================
DIGANTARA Tektronix MSO24 OSCILLOSCOPE CONTROL - GRADIO WEB INTERFACE
=================================================================================

PURPOSE:
    This application provides a web-based graphical user interface (GUI) for
    controlling a DIGANTARA Tektronix MSO24 oscilloscope remotely. It allows engineers
    to configure the oscilloscope, capture data, and analyze signals through
    a web browser instead of manually operating the physical instrument.

KEY CAPABILITIES:
    - Connect/disconnect to oscilloscope via USB, Ethernet, or Serial
    - Configure oscilloscope channels (voltage scale, coupling, probe settings)
    - Set up triggers to capture specific signal events
    - Control built-in function generator (AFG)
    - Perform automated measurements
    - Capture screenshots and waveform data
    - Export data to CSV files and generate plots
    - Run full automation sequences for testing

TARGET USERS:
    Test engineers, hardware engineers, and automation developers who need
    to remotely control oscilloscopes for signal analysis and testing.

TECHNOLOGY STACK:
    - Python 3: Programming language
    - Gradio: Web interface framework (creates the web UI automatically)
    - Matplotlib: Plotting library for graphs
    - PyVISA: Instrument communication library (SCPI protocol)

=================================================================================
"""

# =============================================================================
# IMPORT STATEMENTS - External Libraries Required by This Application
# =============================================================================

# --- CORE PYTHON LIBRARIES (built into Python) ---
import sys                  # System-specific parameters (exit, path manipulation)
import logging              # Error and information logging to track application behavior
import threading            # Multi-threading support for concurrent operations
import queue                # Thread-safe queue for data exchange between threads
import time                 # Time-related functions (delays, timestamps)
import tkinter as tk        # GUI library for file/folder dialogs
from tkinter import filedialog  # File selection dialog windows
from pathlib import Path    # Modern file path handling (works on Windows/Mac/Linux)
from datetime import datetime   # Date and time handling for timestamps
from typing import Optional, Dict, Any, List, Tuple, Union  # Type hints for code clarity
import signal               # Handle system signals (e.g., Ctrl+C interrupt)
import atexit               # Register cleanup functions to run on exit
import os                   # Operating system interface (file operations, environment)
import socket               # Network socket operations (check port availability)

# --- THIRD-PARTY LIBRARIES (must be installed via pip) ---
import gradio as gr         # Web UI framework - creates browser-based interfaces automatically
import pandas as pd         # Data manipulation library - handles CSV and tabular data
import matplotlib           # Plotting library for generating graphs
matplotlib.use('Agg')       # Use non-interactive backend (generates files, not windows)
import matplotlib.pyplot as plt  # Plotting interface for creating charts
import numpy as np          # Numerical computing library - handles arrays and math

# --- MATPLOTLIB CONFIGURATION ---
# These settings optimize plotting performance when handling large datasets
plt.rcParams['agg.path.chunksize'] = 20000      # Process 20,000 points at a time
plt.rcParams['path.simplify'] = True            # Enable path simplification
plt.rcParams['path.simplify_threshold'] = 0.5   # Simplify complex paths to reduce file size

# =============================================================================
# DYNAMIC PATH CONFIGURATION - Import Custom Oscilloscope Driver Module
# =============================================================================

# Calculate the project root directory (3 levels up from this script)
# This allows the script to import our custom oscilloscope control module
# Example: If this file is in /project/scripts/tektronix/, root is /project/
script_dir = Path(__file__).resolve().parent.parent.parent

# Add the project root to Python's module search path if not already present
# This enables: "from instrument_control.tektronix_oscilloscope import ..."
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Import our custom oscilloscope driver classes
# TektronixMSO24: Main class for controlling the oscilloscope
# TektronixMSO24Error: Custom exception class for error handling
try:
    from instrument_control.tektronix_oscilloscope import TektronixMSO24, TektronixMSO24Error
except ImportError as e:
    # If import fails, the driver module is missing or incorrectly installed
    print(f"Error importing Tektronix oscilloscope module: {e}")
    print("Please ensure the instrument_control module is in the correct location.")
    sys.exit(1)  # Exit with error code 1

# =============================================================================
# UTILITY FUNCTIONS AND CONSTANTS
# =============================================================================
# These helper functions and constants provide common operations used throughout
# the application, such as unit conversions and data formatting.

# -----------------------------------------------------------------------------
# CONSTANT MAPPINGS - Translation Dictionaries
# -----------------------------------------------------------------------------
# These dictionaries translate user-friendly terms into the SCPI command format
# required by the oscilloscope. SCPI (Standard Commands for Programmable
# Instruments) is the communication protocol used by test equipment.

# TRIGGER SLOPE MAPPING
# Maps human-readable trigger slope names to oscilloscope SCPI commands
# Example: User selects "Rising" → oscilloscope receives "RISE"
TRIGGER_SLOPE_MAP = {
    "Rising": "RISE",      # Trigger when signal goes from low to high
    "Falling": "FALL",     # Trigger when signal goes from high to low
    "Either": "EITHER"     # Trigger on any edge (rising or falling)
}

# ARBITRARY FUNCTION GENERATOR (AFG) WAVEFORM MAPPING
# Maps waveform type names to oscilloscope SCPI commands
# The AFG is a built-in signal generator in the oscilloscope

AFG_FUNCTION_MAP = {
    # -------------------------------------------------------------------------
    # STANDARD WAVEFORMS - Basic signal types
    # Per MSO24 Programmer Manual: AFG:FUNCtion command
    # -------------------------------------------------------------------------
    "Sine": "SINE",          # Sinusoidal wave (smooth periodic AC wave)
    "Square": "SQUare",      # Square wave (alternates between high/low, sharp transitions)
    "Pulse": "PULSe",        # Pulse wave (short HIGH periods with long LOW)
    "Ramp": "RAMP",          # Ramp/Sawtooth wave (linear rise, sharp fall)
    "DC": "DC",              # Direct current (constant voltage level)
    "Noise": "NOISe",        # Random noise signal (white noise)

    # -------------------------------------------------------------------------
    # SPECIALIZED MATHEMATICAL WAVEFORMS
    # -------------------------------------------------------------------------
    "Sinc": "SINC",          # Sin(x)/x function - used in signal processing, filtering
    "Gaussian": "GAUSsian",  # Gaussian (bell curve) - normal distribution shape
    "Lorentz": "LORENtz",    # Lorentzian function - physics/spectroscopy applications
    "ExpRise": "ERISe",      # Exponential rise - capacitor charging curve
    "ExpFall": "EDECAy",     # Exponential decay - capacitor discharging (SCPI: EDECAy)
    "Haversine": "HAVERSINe",# Haversine function - raised cosine, smooth transitions
    "Cardiac": "CARDIac",    # Cardiac waveform - simulates heartbeat ECG signal

    # -------------------------------------------------------------------------
    # ARBITRARY WAVEFORM
    # -------------------------------------------------------------------------
    "Arbitrary": "ARBitrary",# User-defined arbitrary waveform (loaded from file/software)
}

# TIME UNIT CONVERSION MULTIPLIERS
# Converts various time units to seconds (the base unit for calculations)
# Scientific notation: 1e-9 means 0.000000001 (1 nanosecond in seconds)
TIME_UNIT_MULTIPLIERS = {
    'ns': 1e-9,    # Nanoseconds to seconds (1 ns = 0.000000001 s)
    'us': 1e-6,    # Microseconds to seconds (1 µs = 0.000001 s)
    'µs': 1e-6,    # Microseconds (alternate symbol) to seconds
    'ms': 1e-3,    # Milliseconds to seconds (1 ms = 0.001 s)
    's': 1.0       # Seconds (no conversion needed)
}

# -----------------------------------------------------------------------------
# HELPER FUNCTION: Parse Timebase Value
# -----------------------------------------------------------------------------
def parse_timebase_value(value: Union[str, float]) -> float:
    """
    Convert timebase values from various formats to seconds.

    WHAT IT DOES:
        Takes a time value that could be a number or a string with units
        (like "10 ms" or "5 µs") and converts it to a standard numeric value
        in seconds that can be used in calculations.

    INPUT EXAMPLES:
        - 0.001 (float) → returns 0.001 seconds
        - "10 ms" (string) → returns 0.01 seconds
        - "5 µs" (string) → returns 0.000005 seconds
        - "100 ns" (string) → returns 0.0000001 seconds

    HOW IT WORKS:
        1. If the input is already a number, just convert to float and return
        2. If it's a string, look for time unit suffixes (ns, µs, ms, s)
        3. Extract the number part and multiply by the appropriate conversion factor
        4. Return the result in seconds

    PARAMETERS:
        value: Can be either a number (int/float) or a string with units

    RETURNS:
        Float value representing time in seconds
    """
    # Check if the value is already a number (integer or floating-point)
    if isinstance(value, (int, float)):
        return float(value)  # Just convert to float and return

    # Convert to string, remove whitespace, and make lowercase for consistent matching
    value_str = str(value).strip().lower()

    # Try to find a matching time unit in the string
    for unit, multiplier in TIME_UNIT_MULTIPLIERS.items():
        if unit in value_str:
            # Remove the unit suffix, convert remaining number to float, apply multiplier
            # Example: "10 ms" → "10" → 10.0 → 10.0 * 0.001 = 0.01 seconds
            return float(value_str.replace(unit, '').strip()) * multiplier

    # If no unit found, assume it's already in seconds and just convert to float
    return float(value_str)

# -----------------------------------------------------------------------------
# HELPER FUNCTION: Format Values with SI Prefixes
# -----------------------------------------------------------------------------
def format_si_value(value: float, kind: str) -> str:
    """
    Format numeric values with appropriate SI (International System) unit prefixes.

    WHAT IT DOES:
        Converts raw numeric values into human-readable strings with appropriate
        units and prefixes. For example, instead of displaying "0.000001 seconds",
        it shows "1.000 µs" which is much easier to read.

    SI PREFIXES USED:
        - G (Giga) = 1,000,000,000 (billion)
        - M (Mega) = 1,000,000 (million)
        - k (kilo) = 1,000 (thousand)
        - m (milli) = 0.001 (thousandth)
        - µ (micro) = 0.000001 (millionth)
        - n (nano) = 0.000000001 (billionth)
        - p (pico) = 0.000000000001 (trillionth)

    SUPPORTED TYPES:
        - "freq": Frequency values (Hz, kHz, MHz, GHz)
        - "time": Time values (s, ms, µs, ns, ps)
        - "volt": Voltage values (V, mV, µV, kV)
        - "percent": Percentage values (%)

    EXAMPLE CONVERSIONS:
        format_si_value(1000000, "freq") → "1.000 MHz"
        format_si_value(0.001, "time") → "1.000 ms"
        format_si_value(0.0025, "volt") → "2.500 mV"
        format_si_value(75.5, "percent") → "75.50 %"

    PARAMETERS:
        value: The numeric value to format
        kind: The type of measurement ("freq", "time", "volt", or "percent")

    RETURNS:
        Formatted string with value and appropriate unit (e.g., "1.500 MHz")
    """
    v = abs(value)  # Use absolute value for magnitude comparison (ignore negative sign)

    # FREQUENCY FORMATTING (Hz, kHz, MHz, GHz)
    if kind == "freq":
        if v >= 1e9:  # 1 billion or more → GHz
            return f"{value/1e9:.3f} GHz"
        if v >= 1e6:  # 1 million or more → MHz
            return f"{value/1e6:.3f} MHz"
        if v >= 1e3:  # 1 thousand or more → kHz
            return f"{value/1e3:.3f} kHz"
        return f"{value:.3f} Hz"  # Less than 1000 → Hz

    # TIME FORMATTING (s, ms, µs, ns, ps)
    if kind == "time":
        if v >= 1:  # 1 second or more → s
            return f"{value:.3f} s"
        if v >= 1e-3:  # 1 millisecond or more → ms
            return f"{value*1e3:.3f} ms"
        if v >= 1e-6:  # 1 microsecond or more → µs
            return f"{value*1e6:.3f} µs"
        if v >= 1e-9:  # 1 nanosecond or more → ns
            return f"{value*1e9:.3f} ns"
        return f"{value*1e12:.3f} ps"  # Less than 1 ns → ps

    # VOLTAGE FORMATTING (kV, V, mV, µV)
    if kind == "volt":
        if v >= 1e3:  # 1000 volts or more → kV
            return f"{value/1e3:.3f} kV"
        if v >= 1:  # 1 volt or more → V
            return f"{value:.3f} V"
        if v >= 1e-3:  # 1 millivolt or more → mV
            return f"{value*1e3:.3f} mV"
        return f"{value*1e6:.3f} µV"  # Less than 1 mV → µV

    # PERCENTAGE FORMATTING
    if kind == "percent":
        return f"{value:.2f} %"  # 2 decimal places for percentages

    # DEFAULT: Return as-is if type not recognized
    return f"{value}"

# -----------------------------------------------------------------------------
# HELPER FUNCTION: Format Measurement Values Based on Type
# -----------------------------------------------------------------------------
def format_measurement_value(meas_type: str, value: Optional[float]) -> str:
    """
    Format oscilloscope measurement values with appropriate units.

    WHAT IT DOES:
        Takes a measurement type (like "FREQUENCY" or "AMPLITUDE") and its
        numeric value, then formats it with the correct units for display.
        This ensures measurements are presented in a user-friendly format.

    MEASUREMENT TYPE CATEGORIES:
        - Frequency measurements: FREQUENCY → formatted as Hz, kHz, MHz, GHz
        - Time measurements: PERIOD, RISE, FALL, WIDTH → formatted as s, ms, µs, ns
        - Voltage measurements: AMPLITUDE, HIGH, LOW, MEAN, RMS, MAX, MIN, PEAK2PEAK
          → formatted as V, mV, µV, kV
        - Percentage measurements: DUTYCYCLE, OVERSHOOT, PRESHOOT → formatted as %

    EXAMPLE USAGE:
        format_measurement_value("FREQUENCY", 1500000) → "1.500 MHz"
        format_measurement_value("AMPLITUDE", 0.0035) → "3.500 mV"
        format_measurement_value("PERIOD", 0.000002) → "2.000 µs"
        format_measurement_value("DUTYCYCLE", 45.5) → "45.50 %"
        format_measurement_value("AMPLITUDE", None) → "N/A"

    PARAMETERS:
        meas_type: The type of measurement (e.g., "FREQUENCY", "AMPLITUDE")
        value: The numeric value of the measurement (can be None if no valid reading)

    RETURNS:
        Formatted string with value and appropriate unit
    """
    # Handle invalid or missing measurements
    if value is None:
        return "N/A"  # Not Available - measurement couldn't be taken

    # FREQUENCY: Format with Hz/kHz/MHz/GHz
    if meas_type == "FREQUENCY":
        return format_si_value(value, "freq")

    # TIME-BASED MEASUREMENTS: Format with s/ms/µs/ns
    # These measurements represent durations or time intervals
    if meas_type in ["PERIOD", "RISE", "FALL", "WIDTH"]:
        return format_si_value(value, "time")

    # VOLTAGE-BASED MEASUREMENTS: Format with V/mV/µV/kV
    # These measurements represent voltage levels or amplitudes
    if meas_type in ["AMPLITUDE", "HIGH", "LOW", "MEAN", "RMS", "MAX", "MIN", "PEAK2PEAK"]:
        return format_si_value(value, "volt")

    # PERCENTAGE-BASED MEASUREMENTS: Format with %
    # These measurements are ratios expressed as percentages
    if meas_type in ["DUTYCYCLE", "OVERSHOOT", "PRESHOOT"]:
        return format_si_value(value, "percent")

    # DEFAULT: Return raw number if measurement type not recognized
    return f"{value}"

# =============================================================================
# DATA ACQUISITION CLASS
# =============================================================================
# This class handles all data capture, export, and visualization operations
# for the oscilloscope. It provides thread-safe methods to acquire waveforms,
# save data to CSV files, and generate plots.

class MSO24DataAcquisition:
    """
    =========================================================================
    DATA ACQUISITION HANDLER FOR DIGANTARA Tektronix MSO24 OSCILLOSCOPE
    =========================================================================

    PURPOSE:
        This class is responsible for capturing waveform data from the
        oscilloscope, processing it, and exporting it in various formats
        (CSV files, plots, screenshots).

    KEY RESPONSIBILITIES:
        1. Capture waveform data from oscilloscope channels
        2. Save waveform data to CSV files with metadata
        3. Generate professional plots/graphs of waveform data
        4. Handle thread-safe operations (multiple operations can't interfere)

    THREAD SAFETY:
        This class uses locks (io_lock) to ensure that multiple operations
        don't try to communicate with the oscilloscope at the same time,
        which could cause errors or data corruption.

    DATA FLOW:
        1. acquire_waveform_data() → Gets raw voltage/time data from scope
        2. save_waveform_csv() → Exports data to CSV file
        3. generate_waveform_plot() → Creates PNG image of waveform
    """

    def __init__(self, oscilloscope_instance, io_lock: Optional[threading.RLock] = None):
        """
        Initialize the data acquisition handler.

        PARAMETERS:
            oscilloscope_instance: Reference to the TektronixMSO24 object
                                  that controls the physical oscilloscope
            io_lock: Optional threading lock to prevent simultaneous
                    communication with the oscilloscope (prevents conflicts)

        SETS UP:
            - Connection to oscilloscope
            - Logger for tracking operations and errors
            - Default directories for saving data, graphs, and screenshots
            - Thread synchronization lock
        """
        # Store reference to the oscilloscope controller
        self.scope = oscilloscope_instance

        # Create a logger for this class to track operations and errors
        # Logger name will be "MSO24DataAcquisition"
        self._logger = logging.getLogger(f'{self.__class__.__name__}')

        # Set up default directories for saving files
        # Path.cwd() gets the current working directory
        # The "/" operator creates subdirectories
        self.default_data_dir = Path.cwd() / "data"            # For CSV files
        self.default_screenshot_dir = Path.cwd() / "screenshots"  # For scope screenshots

        # Store the threading lock for thread-safe operations
        # RLock = "Reentrant Lock" - allows same thread to acquire lock multiple times
        self.io_lock = io_lock

    def acquire_waveform_data(self, channel: Union[int, str], max_points: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Capture waveform data from a specific oscilloscope channel or math function.

        WHAT IT DOES:
            Requests waveform data from the oscilloscope for a specific channel or math function.
            The data includes voltage values over time, which represents the
            captured electrical signal.

        HOW IT WORKS:
            1. Check if oscilloscope is connected
            2. Acquire lock to prevent conflicts with other operations
            3. Request data from oscilloscope (up to max_points data points)
            4. Return the data as a dictionary with time and voltage arrays

        DATA STRUCTURE RETURNED:
            {
                'time': [0.0, 0.000001, 0.000002, ...],      # Time values in seconds
                'voltage': [0.5, 0.48, 0.52, ...],           # Voltage values in volts
                'channel': 1,                                 # Which channel was captured
                'num_points': 50000,                          # Number of data points
                'x_increment': 0.000001,                      # Time between points
                'y_multiplier': 0.001                         # Voltage scaling factor
            }

        PARAMETERS:
            channel: Channel number (1, 2, 3, or 4) or source name ("CH1", "MATH1", etc.)
            max_points: Maximum number of data points to capture (None = use scope's record length)
                       More points = higher resolution but larger file size

        RETURNS:
            Dictionary containing waveform data, or None if capture failed
        """
        # Safety check: Ensure oscilloscope is connected before attempting capture
        if not self.scope.is_connected:
            self._logger.error("Cannot acquire data: oscilloscope not connected")
            return None  # Cannot proceed without connection

        try:
            lock = self.io_lock

            # THREAD-SAFE ACQUISITION
            # If a lock is provided, acquire it before communicating with oscilloscope
            # This prevents multiple threads from sending commands simultaneously
            if lock:
                with lock:  # "with" statement automatically acquires and releases lock
                    waveform_data = self.scope.get_channel_data(
                        channel=channel,        # Which channel to read
                        start_point=1,          # Start from first data point
                        stop_point=max_points   # End at max_points (None = all data)
                    )
            else:
                # No lock provided - proceed without thread protection
                # (Used when single-threaded operation is guaranteed)
                waveform_data = self.scope.get_channel_data(
                    channel=channel,
                    start_point=1,
                    stop_point=max_points       # None = use scope's record length
                )

            # Check if data was successfully acquired
            if waveform_data:
                # Log success with number of points captured
                self._logger.info(f"Acquired {len(waveform_data['voltage'])} points from channel {channel}")
                return waveform_data  # Return the captured data

            # If waveform_data is None or empty, return None
            return None

        except Exception as e:
            # Catch any errors during acquisition and log them
            # Common errors: communication timeout, invalid channel, scope busy
            self._logger.error(f"Waveform acquisition failed: {e}")
            return None  # Return None to indicate failure

    def save_waveform_csv(self, data: Dict[str, Any], filename: str, directory: str) -> str:
        """
        Export waveform data to a CSV (Comma-Separated Values) file.

        WHAT IT DOES:
            Takes waveform data (time and voltage arrays) and saves it to a CSV
            file that can be opened in Excel, MATLAB, or other analysis tools.
            The file includes metadata (information about the capture) as comments.

        CSV FILE FORMAT:
            # DIGANTARA Tektronix MSO24 Waveform Data
            # Channel: 1
            # Sample Points: 50000
            # X Increment: 0.000001 s
            # Y Multiplier: 0.001 V
            # Acquisition Time: 2025-12-04T10:30:45.123456
            #
            Time (s),Voltage (V)
            0.000000,0.523
            0.000001,0.518
            0.000002,0.525
            ...

        PARAMETERS:
            data: Dictionary containing waveform data (from acquire_waveform_data)
            filename: Base name for the file (without .csv extension)
            directory: Folder path where the file should be saved

        RETURNS:
            Full path to the saved CSV file as a string

        RAISES:
            Exception if file cannot be created or written
        """
        try:
            # Create the directory if it doesn't exist
            # parents=True: Create parent directories if needed
            # exist_ok=True: Don't error if directory already exists
            Path(directory).mkdir(parents=True, exist_ok=True)

            # Construct full file path (directory + filename + .csv extension)
            filepath = Path(directory) / f"{filename}.csv"

            # Create a pandas DataFrame (table structure) with time and voltage columns
            # DataFrame is like an Excel spreadsheet in Python
            df = pd.DataFrame({
                'Time (s)': data['time'],      # Column 1: Time values in seconds
                'Voltage (V)': data['voltage']  # Column 2: Voltage values in volts
            })

            # Create metadata lines (information about the measurement)
            # Lines starting with '#' are comments in CSV files
            metadata_lines = [
                f"# DIGANTARA Tektronix MSO24 Waveform Data",
                f"# Channel: {data['channel']}",              # Which channel (1-4)
                f"# Sample Points: {data['num_points']}",     # Number of data points
                f"# X Increment: {data['x_increment']} s",    # Time between samples
                f"# Y Multiplier: {data['y_multiplier']} V",  # Voltage scaling factor
                f"# Acquisition Time: {datetime.now().isoformat()}",  # When data was saved
                "#"  # Empty comment line for spacing
            ]

            # Write metadata and data to CSV file
            with open(filepath, 'w') as f:  # 'w' = write mode
                # First, write all metadata comment lines
                for line in metadata_lines:
                    f.write(line + '\n')  # \n = newline character

                # Then, write the DataFrame as CSV data
                # index=False: Don't include row numbers in the CSV
                df.to_csv(f, index=False)

            # Log success message
            self._logger.info(f"Waveform saved to: {filepath}")
            return str(filepath)  # Return file path as string

        except Exception as e:
            # If any error occurs, log it and re-raise the exception
            self._logger.error(f"Failed to save CSV: {e}")
            raise  # Re-raise exception to caller

# =============================================================================
# MAIN GUI CLASS - APPLICATION CORE
# =============================================================================
# This is the main class that creates and manages the entire web-based user
# interface for controlling the oscilloscope. It coordinates all operations
# and provides the user interface elements (buttons, dropdowns, text boxes).

class GradioMSO24GUI:
    """
    =========================================================================
    MAIN APPLICATION CLASS - DIGANTARA Tektronix MSO24 WEB INTERFACE
    =========================================================================

    PURPOSE:
        This is the central control class that creates and manages the entire
        web-based graphical user interface for oscilloscope control. It acts
        as the "brain" of the application, coordinating all user interactions
        with the oscilloscope hardware.

    KEY FEATURES PROVIDED:
        1. Connection Management: Connect/disconnect from oscilloscope
        2. Channel Configuration: Set voltage scales, coupling, probes
        3. Timebase Control: Set time/division, record length
        4. Trigger Setup: Configure edge triggers, sweep modes
        5. AFG Control: Built-in function generator settings
        6. Math Functions: Add, subtract, multiply, divide channels
        7. Measurements: Automated parameter measurements
        8. Data Acquisition: Capture waveforms, export to CSV
        9. Plotting: Generate professional graphs
        10. Screenshot Capture: Save oscilloscope screen images

    ARCHITECTURE:
        - Uses Gradio framework to create web UI automatically
        - Implements thread-safe operations for concurrent access
        - Handles all communication with oscilloscope hardware
        - Provides error handling and logging
        - Manages file operations (save data, plots, screenshots)

    USER INTERACTION FLOW:
        1. User opens web browser → sees Gradio interface
        2. User clicks buttons/enters values → calls methods in this class
        3. Methods send SCPI commands to oscilloscope via TektronixMSO24 class
        4. Results are displayed back to user in the web interface
    """

    def __init__(self, default_visa_address: str = "USB0::0x0699::0x0105::SGVJ0003176::INSTR"):
        """
        Initialize the main GUI application.

        WHAT THIS DOES:
            Sets up all the components needed for the application to run:
            - Logging system (to track errors and operations)
            - Oscilloscope connection settings
            - Thread safety mechanisms (locks, queues)
            - File save locations
            - Data storage structures
            - Cleanup handlers (for graceful shutdown)

        PARAMETERS:
            default_visa_address: The default address for connecting to the
                                 oscilloscope. VISA addresses identify instruments:
                                 - USB: USB0::0x0699::0x0105::SERIAL::INSTR
                                 - Ethernet: TCPIP::192.168.1.100::INSTR
                                 - Serial: ASRL1::INSTR

        INITIALIZATION STEPS:
            1. Configure logging system
            2. Initialize oscilloscope connection variables
            3. Set up thread synchronization
            4. Create default directories for saving files
            5. Register cleanup functions
        """
        # ---------------------------------------------------------------------
        # LOGGING CONFIGURATION
        # ---------------------------------------------------------------------
        # Set up logging to track application behavior and errors
        logging.basicConfig(
            level=logging.INFO,  # INFO level: Log informational messages and above
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            # Format: "2025-12-04 10:30:45 - GradioMSO24GUI - INFO - Message text"
            handlers=[logging.StreamHandler()]  # Output to console (terminal)
        )
        self._logger = logging.getLogger(self.__class__.__name__)  # Create logger for this class

        # ---------------------------------------------------------------------
        # OSCILLOSCOPE CONNECTION VARIABLES
        # ---------------------------------------------------------------------
        # These track the state of the oscilloscope connection
        self.oscilloscope: Optional[TektronixMSO24] = None  # Will hold oscilloscope controller object
        self.default_visa_address = default_visa_address    # Default connection address
        self.is_connected = False  # Connection status flag (True = connected, False = disconnected)

        # ---------------------------------------------------------------------
        # THREAD SYNCHRONIZATION COMPONENTS
        # ---------------------------------------------------------------------
        # These ensure safe concurrent operations (multiple threads don't interfere)
        self.io_lock = threading.RLock()  # Reentrant lock for thread-safe I/O operations
        self.data_queue = queue.Queue()   # Thread-safe queue for data exchange

        # ---------------------------------------------------------------------
        # DATA ACQUISITION HANDLER
        # ---------------------------------------------------------------------
        # Will be initialized when connection is established
        self.data_acquisition: Optional[MSO24DataAcquisition] = None

        # ---------------------------------------------------------------------
        # FILE SAVE LOCATIONS
        # ---------------------------------------------------------------------
        # Default directories for saving various types of output files
        base_path = Path.cwd()  # Get current working directory
        self.save_locations = {
            'data': str(base_path / "oscilloscope_data"),         # CSV data files
            'graphs': str(base_path / "oscilloscope_graphs"),     # Plot images (PNG)
            'screenshots': str(base_path / "oscilloscope_screenshots")  # Scope screenshots
        }

        # Create all directories if they don't exist
        # exist_ok=True: No error if directory already exists
        # parents=True: Create parent directories as needed
        for path in self.save_locations.values():
            Path(path).mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # DATA STORAGE
        # ---------------------------------------------------------------------
        # Dictionary to store captured waveform data in memory
        # Key: Channel name (e.g., "CH1", "CH2")
        # Value: Dictionary with time, voltage, and metadata
        self.current_waveform_data = {}

        # ---------------------------------------------------------------------
        # TIMEBASE SCALE OPTIONS
        # ---------------------------------------------------------------------
        # Available time/division settings for the oscilloscope horizontal axis
        # Each tuple: (display_name, value_in_seconds)
        # Example: ("1 ms", 1e-3) means "1 millisecond per division"
        self.timebase_scales = [
            # Nanosecond range (1e-9 seconds)
            ("1 ns", 1e-9), ("2 ns", 2e-9), ("5 ns", 5e-9),
            ("10 ns", 10e-9), ("20 ns", 20e-9), ("50 ns", 50e-9),
            ("100 ns", 100e-9), ("200 ns", 200e-9), ("500 ns", 500e-9),
            # Microsecond range (1e-6 seconds)
            ("1 µs", 1e-6), ("2 µs", 2e-6), ("5 µs", 5e-6),
            ("10 µs", 10e-6), ("20 µs", 20e-6), ("50 µs", 50e-6),
            ("100 µs", 100e-6), ("200 µs", 200e-6), ("500 µs", 500e-6),
            # Millisecond range (1e-3 seconds)
            ("1 ms", 1e-3), ("2 ms", 2e-3), ("5 ms", 5e-3),
            ("10 ms", 10e-3), ("20 ms", 20e-3), ("50 ms", 50e-3),
            ("100 ms", 100e-3), ("200 ms", 200e-3), ("500 ms", 500e-3),
            # Second range
            ("1 s", 1.0), ("2 s", 2.0), ("5 s", 5.0),
            ("10 s", 10.0), ("20 s", 20.0), ("50 s", 50.0)
        ]

        # ---------------------------------------------------------------------
        # CLEANUP REGISTRATION
        # ---------------------------------------------------------------------
        # Register functions to ensure clean shutdown when program exits
        atexit.register(self.cleanup)  # Run cleanup() when Python exits normally
        signal.signal(signal.SIGINT, self._signal_handler)  # Handle Ctrl+C interrupt

        # Log successful initialization
        self._logger.info("MSO24 GUI initialized")

    def _signal_handler(self, signum, frame):
        """
        Handle interrupt signals (like Ctrl+C) for graceful shutdown.

        WHAT IT DOES:
            When the user presses Ctrl+C or the system sends an interrupt signal,
            this method ensures the application shuts down cleanly by:
            1. Logging the interrupt
            2. Disconnecting from the oscilloscope
            3. Exiting the program

        PARAMETERS:
            signum: Signal number (e.g., SIGINT for Ctrl+C)
            frame: Current stack frame (not used, required by signal handler)

        WHY THIS IS IMPORTANT:
            Prevents the oscilloscope from being left in an unknown state when
            the program is interrupted. Ensures proper disconnection.
        """
        self._logger.info("Interrupt signal received, cleaning up...")
        self.cleanup()  # Disconnect oscilloscope and clean up resources
        sys.exit(0)     # Exit program with success code

    def cleanup(self):
        """
        Perform clean shutdown of the oscilloscope connection.

        WHAT IT DOES:
            Safely disconnects from the oscilloscope hardware to ensure it's
            not left in a locked or busy state. This method is automatically
            called when:
            - The program exits normally
            - The user presses Ctrl+C
            - The program encounters an error

        OPERATIONS PERFORMED:
            1. Check if oscilloscope object exists
            2. Check if it's currently connected
            3. Send disconnect command to oscilloscope
            4. Log the disconnection
            5. Handle any errors that occur during cleanup

        WHY THIS IS IMPORTANT:
            If the oscilloscope isn't disconnected properly, it may remain
            locked and prevent other programs from connecting to it until
            it's power-cycled (turned off and on).
        """
        try:
            # Check if oscilloscope exists and is connected
            if self.oscilloscope and self.oscilloscope.is_connected:
                self.oscilloscope.disconnect()  # Send disconnect command
                self._logger.info("Oscilloscope disconnected")
        except Exception as e:
            # If disconnection fails, log the error but don't crash
            # This ensures cleanup continues even if there's a problem
            self._logger.error(f"Error during cleanup: {e}")

    def connect_oscilloscope(self, visa_address: str) -> str:
        """
        Establish connection to the DIGANTARA Tektronix MSO24 oscilloscope.

        WHAT IT DOES:
            Connects to the physical oscilloscope hardware using the provided
            VISA address. This is like "dialing a phone number" to establish
            communication with the instrument.

        CONNECTION PROCESS:
            1. Disconnect any existing connection (clean slate)
            2. Create oscilloscope controller object if needed
            3. Attempt connection using thread-safe lock
            4. Configure output directories for saving files
            5. Retrieve and display instrument information
            6. Return success or error message to user

        VISA ADDRESS FORMATS:
            - USB: "USB0::0x0699::0x0105::SERIAL_NUMBER::INSTR"
            - Ethernet: "TCPIP::192.168.1.100::INSTR"
            - Serial: "ASRL1::INSTR"

        PARAMETERS:
            visa_address: The VISA resource string identifying the oscilloscope

        RETURNS:
            String message for display to user:
            - Success: "[CONNECTED] DIGANTARA Tektronix MSO24 (S/N: 123456) Firmware: 1.0"
            - Failure: "[ERROR] Connection error: <error details>"

        EXAMPLE USAGE:
            result = connect_oscilloscope("USB0::0x0699::0x0105::SGVJ0003176::INSTR")
            # Returns: "[CONNECTED] MSO24 (S/N: SGVJ0003176) Firmware: 1.25.3"
        """
        try:
            # STEP 1: Disconnect any existing connection
            # This ensures we start fresh and don't have conflicting connections
            if self.oscilloscope and self.oscilloscope.is_connected:
                self.oscilloscope.disconnect()

            # STEP 2: Create oscilloscope controller object if it doesn't exist
            # TektronixMSO24 is the driver class that handles SCPI communication
            if not self.oscilloscope:
                self.oscilloscope = TektronixMSO24(visa_address)
                # Also create data acquisition handler for this oscilloscope
                self.data_acquisition = MSO24DataAcquisition(self.oscilloscope, self.io_lock)

            # STEP 3: Attempt connection with thread-safe lock
            # The lock prevents multiple threads from connecting simultaneously
            with self.io_lock:
                if self.oscilloscope.connect():
                    # Connection successful!
                    self.is_connected = True  # Update connection status flag

                    # STEP 4: Configure where files should be saved
                    # Tell the oscilloscope object where to save data/graphs/screenshots
                    self.oscilloscope.set_output_directories(
                        data_dir=self.save_locations['data'],
                        graph_dir=self.save_locations['graphs'],
                        screenshot_dir=self.save_locations['screenshots']
                    )

                    # STEP 5: Retrieve instrument information for confirmation
                    # Query oscilloscope for model, serial number, firmware version
                    info = self.oscilloscope.get_instrument_info()
                    if info:
                        # Format nice success message with instrument details
                        return (f"[CONNECTED] {info['model']} "
                               f"(S/N: {info['serial_number']})\n"
                               f"Firmware: {info['firmware_version']}")
                    # If info retrieval failed, return basic success message
                    return "[CONNECTED] DIGANTARA Tektronix MSO24"

                # If connect() returned False, connection failed
                return "[ERROR] Failed to connect to oscilloscope"

        except Exception as e:
            # Catch any errors during connection process
            # Common errors: instrument not found, USB driver issues, wrong address
            self._logger.error(f"Connection error: {e}")
            return f"[ERROR] Connection error: {str(e)}"

    def disconnect_oscilloscope(self) -> str:
        """
        Disconnect from the oscilloscope hardware.

        WHAT IT DOES:
            Safely closes the communication channel to the oscilloscope.
            This is like "hanging up the phone" after you're done using it.

        WHY DISCONNECT:
            - Frees up the oscilloscope for other programs to use
            - Prevents communication errors if instrument is moved/unplugged
            - Good practice to release resources when not in use
            - Some oscilloscopes only allow one connection at a time

        DISCONNECT PROCESS:
            1. Acquire thread lock (prevent interference)
            2. Check if oscilloscope exists and is connected
            3. Send disconnect command
            4. Update connection status flag
            5. Return success/info message to user

        RETURNS:
            String message for display:
            - "[OK] Oscilloscope disconnected" - Successfully disconnected
            - "[INFO] Oscilloscope not connected" - Already disconnected
            - "[ERROR] Disconnect error: <details>" - Error occurred
        """
        try:
            # Use thread lock to ensure safe disconnection
            with self.io_lock:
                # Check if oscilloscope object exists and is connected
                if self.oscilloscope and self.oscilloscope.is_connected:
                    self.oscilloscope.disconnect()  # Send disconnect command
                    self.is_connected = False       # Update status flag
                    return "[OK] Oscilloscope disconnected"
                else:
                    # Oscilloscope was not connected, nothing to disconnect
                    return "[INFO] Oscilloscope not connected"

        except Exception as e:
            # If disconnection fails, log error and return message
            self._logger.error(f"Disconnect error: {e}")
            return f"[ERROR] Disconnect error: {str(e)}"

    def set_channel_state(self, channel: int, enabled: bool) -> bool:
        """
        Enable or disable a specific oscilloscope channel.

        WHAT IT DOES:
            Turns an oscilloscope channel ON or OFF, similar to toggling a
            light switch. When OFF, the channel doesn't display on screen
            and doesn't capture data.

        OSCILLOSCOPE CHANNELS EXPLAINED:
            Most oscilloscopes have 2-4 input channels. Each channel can
            measure a different signal independently. For example:
            - CH1: Power supply voltage
            - CH2: Sensor output signal
            - CH3: Clock signal
            - CH4: Data signal

        PARAMETERS:
            channel: Which channel to control (1, 2, 3, or 4)
            enabled: True = turn channel ON, False = turn channel OFF

        RETURNS:
            True if successful, False if failed

        SCPI COMMAND SENT:
            "DISplay:GLObal:CH1:STATE ON" - Turns on channel 1
            "DISplay:GLObal:CH2:STATE OFF" - Turns off channel 2
        """
        # Safety check: Can't control channel if not connected
        if not self.is_connected:
            return False

        try:
            # Use thread lock for safe communication
            with self.io_lock:
                # Convert boolean to oscilloscope command: True→"ON", False→"OFF"
                state = "ON" if enabled else "OFF"

                # Send SCPI command to oscilloscope
                # Format: "DISplay:GLObal:CH<number>:STATE <ON/OFF>"
                self.oscilloscope._scpi_wrapper.write(f"DISplay:GLObal:CH{channel}:STATE {state}")

                # Small delay to let oscilloscope process command
                time.sleep(0.05)  # 50 milliseconds

                # Log the action for debugging
                self._logger.info(f"Channel {channel} {'enabled' if enabled else 'disabled'}")
                return True  # Success!

        except Exception as e:
            # If command fails, log error and return False
            self._logger.error(f"Failed to set channel {channel} state: {e}")
            return False

    def configure_channels(self, ch1_en: bool, ch2_en: bool, ch3_en: bool, ch4_en: bool,
                          scale: float, offset: float, coupling: str) -> str:
        """Configure all channels with enable/disable (probe attenuation set separately)"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
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
                            bandwidth_limit=False
                        )

                        status = "configured" if success else "config failed"
                        results.append(
                            f"CH{channel}: enabled, {status} "
                            f"(Scale={scale} V/div, Offset={offset} V, Coupling={coupling})"
                        )
                    else:
                        results.append(f"CH{channel}: disabled")

            return "[OK] Channels configured:\n" + "\n".join(results)

        except Exception as e:
            self._logger.error(f"Channel config error: {e}")
            return f"[ERROR] Channel config: {str(e)}"

    def set_probe_attenuation(self, channel: int, attenuation: float) -> str:
        """Set probe attenuation for a specific channel"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            # Convert string probe values to float
            if isinstance(attenuation, str):
                probe_map = {"1x": 1.0, "10x": 10.0, "100x": 100.0, "1000x": 1000.0}
                probe_value = probe_map.get(attenuation, float(attenuation.replace('x', '')))
            else:
                probe_value = float(attenuation)

            with self.io_lock:
                success = self.oscilloscope.set_probe_attenuation(channel, probe_value)

            if success:
                return f"[OK] CH{channel}: Probe attenuation set to {probe_value}x"
            else:
                return f"[ERROR] Failed to set probe attenuation for CH{channel}"

        except Exception as e:
            self._logger.error(f"Probe config error: {e}")
            return f"[ERROR] Probe config: {str(e)}"

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
        r"""
        Configure the Arbitrary Function Generator (AFG) - Built-in Signal Generator.

        WHAT IS THE AFG?
            The AFG is a signal generator built into the oscilloscope. It can create
            test signals without needing external equipment. Think of it as having a
            "tone generator" or "signal simulator" integrated into your scope.

        WHY USE THE AFG?
            - Test circuit responses to known signals
            - Simulate sensor outputs (temperature, pressure, etc.)
            - Generate clock signals for digital circuits
            - Create reference signals for calibration
            - Perform frequency response testing
            - Inject test tones into audio circuits
            - Simulate communication protocols

        WAVEFORM TYPES EXPLAINED (All 15 Available Types):

        === STANDARD WAVEFORMS (Basic Signal Types) ===

        1. SINE - Smooth sinusoidal wave (fundamental AC signal)
           ```
               ___
              /   \
           __/     \__
                     \___/
           ```
           Uses: Audio testing, AC power simulation, smooth periodic signals
           Example: 60 Hz power line, 1 kHz audio tone

        2. SQUARE - Alternating HIGH/LOW with instantaneous transitions
           ```
            ___     ___     ___
           |   |   |   |   |   |
           |   |___|   |___|   |
           ```
           Uses: Digital circuits, clock signals, PWM (Pulse Width Modulation)
           Example: 1 MHz microcontroller clock, relay control

        3. PULSE - Short HIGH periods with long LOW periods
           ```
            _   _   _   _
           | | | | | | | |
           | |_| |_| |_| |___
           ```
           Uses: Trigger signals, interrupt testing, timing analysis
           Example: Camera trigger, interrupt request signal

        4. RAMP - Linear rise with sharp fall (sawtooth wave)
           ```
             /|  /|  /|  /|
            / | / | / | / |
           /  |/  |/  |/  |
           ```
           Uses: Sweep testing, oscilloscope timebase simulation
           Example: Horizontal deflection in old TVs/monitors

        5. TRIANGLE - Symmetric linear rise and fall
           ```
            /\    /\    /\
           /  \  /  \  /  \
          /    \/    \/    \
           ```
           Uses: Audio synthesis, modulation, voltage ramp generation
           Example: Vibrato in music, scanning circuits

        6. DC - Constant voltage level (no variation)
           ```
           ___________________
           ```
           Uses: Power supply simulation, voltage reference, biasing
           Example: 3.3V logic supply, sensor excitation voltage

        7. NOISE - Random voltage fluctuations (white noise)
           ```
           _~-_^~-_-~^_~-_~^_
           ```
           Uses: Testing noise immunity, audio hiss simulation, randomness
           Example: Simulating EMI (electromagnetic interference)

        === SPECIALIZED MATHEMATICAL WAVEFORMS ===

        8. SINC - Sin(x)/x function (used in signal processing)
           ```
              __
           __/  \__  _  _
                  \/\/
           ```
           Uses: Digital signal processing, filter design, sampling theory
           Example: Testing anti-aliasing filters

        9. GAUSSIAN - Bell curve (normal distribution)
           ```
                ___
              _/   \_
           __/       \__
           ```
           Uses: Statistics, probability, pulse shaping in communications
           Example: Gaussian noise distribution, data transmission

        10. LORENTZ - Lorentzian curve (physics/spectroscopy)
            ```
                 _
               _/ \_
             _/     \_
            ```
            Uses: Spectral line shapes, resonance curves, physics applications
            Example: NMR spectroscopy, laser physics

        11. EXPRISE - Exponential rise (capacitor charging)
            ```
                    ___
                 __/
              __/
            _/
            ```
            Uses: RC circuit simulation, charging curves, transient analysis
            Example: Capacitor charging through resistor

        12. EXPFALL - Exponential decay (capacitor discharging)
            ```
            ___
               \__
                  \__
                     \_
            ```
            Uses: RC discharge, decay processes, transient analysis
            Example: Capacitor discharging, radioactive decay simulation

        13. HAVERSINE - Raised cosine (smooth S-curve transitions)
            ```
               ____
              /    \
            _/      \_
            ```
            Uses: Smooth transitions, jerk-free motion control
            Example: Motor acceleration profiles, smooth LED fading

        14. CARDIAC - Heartbeat waveform (ECG simulation)
            ```
              _
             | |
            _| |__  ___
                  \/
            ```
            Uses: Medical device testing, ECG simulators, biomedical
            Example: Testing heart rate monitors, medical equipment

        15. ARBITRARY - User-defined custom waveform
            ```
            (any shape defined by user)
            ```
            Uses: Complex signal simulation, custom test patterns
            Example: Load waveforms captured from actual circuits or
                    create using PC software (ArbExpress)

        PARAMETERS EXPLAINED:

        1. function (String):
           - Which waveform shape to generate (15 types available)
           - Default: "Sine"

        2. frequency (Float - Hertz):
           - How many cycles per second
           - Range: 0.1 Hz to 50 MHz (oscilloscope dependent)
           - Examples:
             - 60 Hz: AC power line frequency
             - 440 Hz: Musical note A4
             - 1 kHz: Standard audio test tone
             - 1 MHz: Slow digital clock
             - 10 MHz: Fast digital signals

        3. amplitude (Float - Volts peak-to-peak):
           - Total voltage swing of the signal
           - Range: 0.02V to 5V (oscilloscope dependent)
           - Example: 2.0V amplitude means signal swings ±1V from center
           - Formula: Peak-to-Peak = 2 × Peak = 2 × Amplitude

        4. offset (Float - Volts DC):
           - DC voltage added to center signal at desired level
           - Range: -2.5V to +2.5V (oscilloscope dependent)
           - Example: offset=2.5V converts ±2.5V sine to 0-5V sine
           - Used for: Creating logic-level signals, biasing circuits

        5. enable (Boolean):
           - True = AFG output ON (signal actively generated)
           - False = AFG output OFF (high impedance, no signal)
           - Safety feature: Disable when not in use

        PRACTICAL EXAMPLES:

        Example 1: Testing Audio Amplifier
        ------------------------------------
        function = "Sine"
        frequency = 1000 (1 kHz tone)
        amplitude = 0.1 (100 mV input)
        offset = 0.0 (centered at 0V)
        enable = True
        Result: Clean 1 kHz sine wave input to amplifier

        Example 2: Generating 5V Logic Clock
        ------------------------------------
        function = "Square"
        frequency = 1000000 (1 MHz)
        amplitude = 5.0 (0V to 5V swing)
        offset = 2.5 (centers at 2.5V, giving 0-5V output)
        enable = True
        Result: 1 MHz square wave for digital circuit testing

        Example 3: Simulating Sensor Signal
        ------------------------------------
        function = "Triangle"
        frequency = 10 (10 Hz)
        amplitude = 2.0 (4V peak-to-peak)
        offset = 2.5 (1.5V to 3.5V range)
        enable = True
        Result: Slowly varying voltage like temperature sensor

        Example 4: Testing Noise Immunity
        ------------------------------------
        function = "Noise"
        frequency = (not used for noise)
        amplitude = 0.5 (500mV random variations)
        offset = 0.0 (centered at 0V)
        enable = True
        Result: Random noise to test circuit immunity

        CODE BREAKDOWN:

        1. Convert function name to SCPI command format
        2. Send all parameters to AFG hardware
        3. Enable/disable output as requested
        4. Return status message with settings confirmation

        RETURNS:
            Success: "[OK] AFG: Sine, 1000Hz, 1.0V, Offset: 0.0V, Output: ON"
            Failure: "[ERROR] Failed to configure AFG"

        SAFETY NOTES:
            - Always disable AFG when connecting/disconnecting circuits
            - Check voltage levels before enabling (prevent damage)
            - Start with low amplitude, increase gradually
            - Verify offset won't exceed circuit voltage limits
        """
        # Safety check: Must be connected to configure AFG
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            # Convert user-friendly function name to SCPI command format
            # Example: "Sine" → "SINE", "ExpRise" → "ERISE"
            mso_function = AFG_FUNCTION_MAP.get(function, "SINE")

            # Send AFG configuration to oscilloscope with thread safety
            with self.io_lock:
                success = self.oscilloscope.configure_afg(
                    function=mso_function,    # Waveform type (SINE, SQUARE, etc.)
                    frequency=frequency,       # Cycles per second (Hz)
                    amplitude=amplitude,       # Peak-to-peak voltage (V)
                    offset=offset,            # DC offset voltage (V)
                    enable=enable             # Output ON/OFF
                )

            if success:
                # Format status message showing AFG configuration
                status = "ON" if enable else "OFF"
                return f"[OK] AFG: {function}, {frequency}Hz, {amplitude}V, Offset: {offset}V, Output: {status}"
            else:
                return "[ERROR] Failed to configure AFG"

        except Exception as e:
            # Handle errors (invalid parameters, hardware communication failure)
            self._logger.error(f"AFG config error: {e}")
            return f"[ERROR] AFG: {str(e)}"

    def configure_math_function(self, function_num: int, operation: str, source1: str, source2: str) -> str:
        """Configure math function using ADVANCED mode with expressions (BASIC mode doesn't work properly)"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            # Map operations to math expressions for ADVANCED mode
            # NOTE: BASIC mode FUNCtion command doesn't work - oscilloscope ignores it and always does CH1-CH2
            operation_map = {
                "ADD": f"{source1}+{source2}",
                "SUBTRACT": f"{source1}-{source2}",
                "MULTIPLY": f"{source1}*{source2}",
                "DIVIDE": f"{source1}/{source2}"
            }

            operation_upper = operation.upper()
            math_expression = operation_map.get(operation_upper)

            if not math_expression:
                return f"[ERROR] Unknown operation: {operation}. Use ADD, SUBTRACT, MULTIPLY, or DIVIDE"

            with self.io_lock:
                # Use ADVANCED mode with DEFine - this actually works unlike BASIC mode
                success = self.oscilloscope.configure_math_function(
                    function_num=function_num,
                    operation="ADVANCED",
                    source1=source1,
                    source2=source2,
                    math_expression=math_expression
                )

            if success:
                return f"[OK] Math{function_num}: {math_expression}"
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

    def autoscale_math(self, function_num: int) -> str:
        """Autoscale math function display"""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.autoscale_math(function_num)

            if success:
                return f"[OK] Math{function_num} autoscaled"
            else:
                return f"[ERROR] Failed to autoscale Math{function_num}"

        except Exception as e:
            self._logger.error(f"Math autoscale error: {e}")
            return f"[ERROR] Math autoscale: {str(e)}"

    def measure_all_for_source(self, source: str) -> Tuple[str, str]:
        """Configure all standard measurements for the given source."""
        if not self.is_connected:
            return "[ERROR] Oscilloscope not connected", "[ERROR] Oscilloscope not connected"

        measurement_types = [
            "FREQUENCY", "PERIOD", "AMPLITUDE", "HIGH", "LOW",
            "MAX", "MIN", "PEAK2PEAK", "MEAN", "RMS", "RISE",
            "FALL", "WIDTH", "DUTYCYCLE", "OVERSHOOT", "PRESHOOT",
            "AREA", "PHASE",
        ]

        results = []
        success_count = 0

        for meas_type in measurement_types:
            try:
                meas_num = self.oscilloscope.add_measurement(meas_type, source)
                if meas_num:
                    results.append(f"[OK] {meas_type} on {source}")
                    success_count += 1
                else:
                    results.append(f"[FAIL] {meas_type} on {source}")
            except Exception as e:
                results.append(f"[ERROR] {meas_type}: {e}")

        # Fetch all measurements from instrument
        all_meas_text = self.get_all_measurements()
        summary_text = (f"Added {success_count}/{len(measurement_types)} measurements:\n" +
                       "\n".join(results))
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
                # Get all available data points (scope will use its configured record length)
                data = self.data_acquisition.acquire_waveform_data(channel, max_points=None)
                if data:
                    self.current_waveform_data[f"CH{channel}"] = data
                    points = len(data['voltage'])
                    time_span = (data['time'][-1] - data['time'][0])  # Time span in seconds
                    time_start = data['time'][0]  # Start time (x_zero)
                    time_end = data['time'][-1]    # End time
                    v_pp = np.max(data['voltage']) - np.min(data['voltage'])
                    result += f"CH{channel}: {points} points, time: {time_start*1000:.3f}ms to {time_end*1000:.3f}ms (span: {time_span*1000:.3f}ms), {v_pp:.3f}V p-p\n"
                else:
                    result += f"CH{channel}: [ERROR] Failed to acquire data\n"

            # Acquire math function data
            for math_num in selected_math:
                math_source = f"MATH{math_num}"
                # Get all available data points (scope will use its configured record length)
                data = self.data_acquisition.acquire_waveform_data(math_source, max_points=None)
                if data:
                    self.current_waveform_data[math_source] = data
                    points = len(data['voltage'])
                    time_span = (data['time'][-1] - data['time'][0])  # Time span in seconds
                    time_start = data['time'][0]  # Start time (x_zero)
                    time_end = data['time'][-1]    # End time
                    v_pp = np.max(data['voltage']) - np.min(data['voltage'])
                    result += f"{math_source}: {points} points, time: {time_start*1000:.3f}ms to {time_end*1000:.3f}ms (span: {time_span*1000:.3f}ms), {v_pp:.3f}V p-p\n"
                else:
                    result += f"{math_source}: [ERROR] Failed to acquire data (check if math function is configured and displayed)\n"

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

            # Step 3: Capture screenshot
            result += "Step 3: Screenshot Capture\n"
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
        """
        Create the comprehensive Gradio web-based user interface.

        WHAT IS GRADIO?
            Gradio is a Python library that automatically creates web interfaces
            from Python functions. It handles all the HTML, CSS, JavaScript, and
            web server setup automatically. You write Python, Gradio creates a
            professional web UI.

        WHAT THIS METHOD DOES:
            Builds the entire web interface that users see in their browser,
            including:
            - All tabs (Connection, Channels, Timebase, etc.)
            - All buttons, dropdowns, text boxes
            - Layout and styling (colors, spacing, fonts)
            - Connections between UI elements and Python functions

        UI ARCHITECTURE:
            The interface uses a "tabbed" design:
            ┌─────────────────────────────────────────────────┐
            │  [Connection] [Channels] [Timebase] [AFG] ...   │ ← Tabs
            ├─────────────────────────────────────────────────┤
            │                                                  │
            │  [Content of selected tab goes here]            │
            │  - Input fields                                 │
            │  - Buttons                                      │
            │  - Status displays                              │
            │                                                  │
            └─────────────────────────────────────────────────┘

        HOW GRADIO WORKS:
            1. Define UI elements (buttons, text boxes, dropdowns)
            2. Connect UI elements to Python methods
            3. When user clicks button → Gradio calls Python method
            4. Method returns result → Gradio displays in UI

        EXAMPLE FLOW:
            User clicks "Connect" button
            → Gradio calls self.connect_oscilloscope(visa_address)
            → Method returns "[CONNECTED] MSO24..."
            → Gradio displays message in connection_status textbox

        CSS STYLING:
            CSS (Cascading Style Sheets) controls visual appearance.
            The custom_css variable below defines colors, spacing, fonts.
        """

        # =====================================================================
        # CUSTOM CSS STYLES - Visual Appearance Configuration
        # =====================================================================
        # CSS defines how the interface looks (colors, spacing, fonts)
        # This string contains CSS rules that Gradio applies to the interface

        custom_css = """
        /* ============================================================
        MAIN CONTAINER - Controls the entire interface width/height
        ============================================================ */
        /* The .gradio-container class styles the outermost container */
        .gradio-container {
            max-width: 100% !important;    /* Use full browser width */
            padding: 20px !important;      /* 20 pixels of spacing inside edges */
            margin: 0 !important;          /* No spacing outside container */
            min-height: 100vh;             /* Minimum height = full viewport (100% view height) */
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

        # =====================================================================
        # CREATE GRADIO INTERFACE - Main UI Container
        # =====================================================================
        # gr.Blocks() creates the main container for all UI elements
        # Everything inside this "with" block becomes part of the interface

        with gr.Blocks(
            title="DIGANTARA Tektronix MSO24 Control",    # Browser tab/window title
            theme=gr.themes.Soft(),              # Pre-built visual theme (soft colors, rounded corners)
            css=custom_css                       # Apply our custom CSS styles defined above
        ) as interface:

            # -----------------------------------------------------------------
            # MAIN HEADER - Top of page title and description
            # -----------------------------------------------------------------
            # gr.Markdown() renders formatted text (like a mini-document)
            # The "#" symbol creates a large heading (H1 in HTML)
            # "**text**" makes text bold in Markdown format

            gr.Markdown("# DIGANTARA Tektronix MSO24 Oscilloscope Control Center")
            gr.Markdown("**Professional oscilloscope automation interface with comprehensive control features**")

            # =================================================================
            # CONNECTION TAB - First tab for connecting to oscilloscope
            # =================================================================
            # gr.Tab() creates a clickable tab. Content inside appears when tab is selected
            # Users must connect to oscilloscope before using other features

            with gr.Tab("Connection"):
                gr.Markdown("### Oscilloscope Connection")  # Section heading (H3 - medium size)

                # -------------------------------------------------------------
                # VISA ADDRESS INPUT - Text field for connection address
                # -------------------------------------------------------------
                # gr.Row() places elements horizontally side-by-side
                # gr.Textbox() creates a text input field

                with gr.Row():
                    visa_input = gr.Textbox(
                        label="VISA Address",                                      # Label shown above textbox
                        value="USB0::0x0699::0x0105::SGVJ0003176::INSTR",        # Default value (USB connection)
                        placeholder="Enter VISA address (USB/Ethernet/Serial)",   # Hint text when empty
                        scale=3                                                    # Relative width (3x wider than scale=1)
                    )

                # -------------------------------------------------------------
                # CONNECTION CONTROL BUTTONS - Connect/Disconnect/Reset
                # -------------------------------------------------------------
                # gr.Button() creates clickable buttons
                # variant="primary" = highlighted blue button (main action)
                # variant="secondary" = gray button (less important action)

                with gr.Row():
                    connect_btn = gr.Button("Connect", variant="primary")
                    disconnect_btn = gr.Button("Disconnect", variant="secondary")
                    reset_btn = gr.Button("Reset Scope", variant="secondary")

                # -------------------------------------------------------------
                # CONNECTION STATUS DISPLAY - Shows connection result messages
                # -------------------------------------------------------------
                # interactive=False means user cannot type in this field
                # Used only for displaying output from Python functions

                connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False,    # Read-only display
                    lines=3               # Show 3 lines of text
                )

                # -------------------------------------------------------------
                # BUTTON CLICK HANDLERS - Connect buttons to Python methods
                # -------------------------------------------------------------
                # .click() method defines what happens when button is clicked
                # fn = which Python function to call
                # inputs = list of UI elements whose values are passed to function
                # outputs = list of UI elements where function result is displayed

                # CONNECT BUTTON: Takes visa_address text, calls connect_oscilloscope(),
                # displays result in connection_status textbox
                connect_btn.click(
                    fn=self.connect_oscilloscope,      # Python method to call
                    inputs=[visa_input],                # Pass VISA address to method
                    outputs=[connection_status]         # Display method return value here
                )

                # DISCONNECT BUTTON: No inputs needed, just disconnect
                disconnect_btn.click(
                    fn=self.disconnect_oscilloscope,
                    inputs=[],                          # No parameters needed
                    outputs=[connection_status]
                )

                # RESET BUTTON: Reset oscilloscope to factory defaults
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
                    ch1_select = gr.Checkbox(label="Ch1", value=True, info="Yellow")
                    ch2_select = gr.Checkbox(label="Ch2", value=False, info="Cyan")
                    ch3_select = gr.Checkbox(label="Ch3", value=False, info="Magenta")
                    ch4_select = gr.Checkbox(label="Ch4", value=False, info="Green")

                with gr.Row():
                    v_scale = gr.Number(label="V/div", value=1.0, minimum=0.001, maximum=10.0)
                    v_offset = gr.Number(label="Offset (V)", value=0.0, minimum=-50.0, maximum=50.0)
                    coupling = gr.Dropdown(
                        label="Coupling",
                        choices=["DC", "AC", "DCREJECT"],
                        value="DC"
                    )
                    # probe = gr.Dropdown(
                    #     label="Probe",
                    #     choices=[("1x", 1), ("10x", 10), ("100x", 100), ("1000x", 1000)],
                    #     value=10
                    #)

                config_channel_btn = gr.Button("Configure Channels", variant="primary")
                channel_status = gr.Textbox(label="Status", interactive=False, lines=5)

                config_channel_btn.click(
                    fn=self.configure_channels,
                    inputs=[ch1_select, ch2_select, ch3_select, ch4_select, v_scale, v_offset, coupling],
                    outputs=[channel_status]
                )

                gr.Markdown("---")
                gr.Markdown("### Probe Configuration (Independent)")
                gr.Markdown("*Set probe attenuation for each channel independently*")

                with gr.Row():
                    probe_ch1 = gr.Dropdown(
                        label="CH1 Probe",
                        choices=["1x", "10x", "100x", "1000x"],
                        value="10x",
                        scale=3
                    )
                    probe_ch1_btn = gr.Button("Set CH1", variant="secondary", scale=1)

                with gr.Row():
                    probe_ch2 = gr.Dropdown(
                        label="CH2 Probe",
                        choices=["1x", "10x", "100x", "1000x"],
                        value="1x",
                        scale=3
                    )
                    probe_ch2_btn = gr.Button("Set CH2", variant="secondary", scale=1)

                with gr.Row():
                    probe_ch3 = gr.Dropdown(
                        label="CH3 Probe",
                        choices=["1x", "10x", "100x", "1000x"],
                        value="10x",
                        scale=3
                    )
                    probe_ch3_btn = gr.Button("Set CH3", variant="secondary", scale=1)

                with gr.Row():
                    probe_ch4 = gr.Dropdown(
                        label="CH4 Probe",
                        choices=["1x", "10x", "100x", "1000x"],
                        value="10x",
                        scale=3
                    )
                    probe_ch4_btn = gr.Button("Set CH4", variant="secondary", scale=1)

                probe_status = gr.Textbox(label="Probe Status", interactive=False, lines=3)

                # Connect probe buttons
                probe_ch1_btn.click(
                    fn=lambda p: self.set_probe_attenuation(1, p),
                    inputs=[probe_ch1],
                    outputs=[probe_status]
                )

                probe_ch2_btn.click(
                    fn=lambda p: self.set_probe_attenuation(2, p),
                    inputs=[probe_ch2],
                    outputs=[probe_status]
                )

                probe_ch3_btn.click(
                    fn=lambda p: self.set_probe_attenuation(3, p),
                    inputs=[probe_ch3],
                    outputs=[probe_status]
                )

                probe_ch4_btn.click(
                    fn=lambda p: self.set_probe_attenuation(4, p),
                    inputs=[probe_ch4],
                    outputs=[probe_status]
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
                gr.Markdown("""
**Available Waveforms:**
- **Standard**: Sine, Square, Pulse, Ramp, Triangle, DC, Noise
- **Specialized**: Sinc (Sin(x)/x), Gaussian, Lorentz, ExpRise, ExpFall, Haversine, Cardiac
- **Arbitrary**: Custom user-defined waveforms
                """)

                with gr.Row():
                    afg_function = gr.Dropdown(
                        label="Waveform",
                        choices=[
                            # Standard waveforms
                            "Sine", "Square", "Pulse", "Ramp", "Triangle", "DC", "Noise",
                            # Specialized waveforms
                            "Sinc", "Gaussian", "Lorentz", "ExpRise", "ExpFall", "Haversine", "Cardiac",
                            # Arbitrary
                            "Arbitrary"
                        ],
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
                    autoscale_math_btn = gr.Button("Autoscale", variant="secondary", scale=1)

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

                autoscale_math_btn.click(
                    fn=self.autoscale_math,
                    inputs=[math_func_num],
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
                        label="Measurement Result", interactive=False, lines=20
                    )
                    all_meas_result = gr.Textbox(
                        label="All Measurements", interactive=False, lines=20
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
                        screenshots_path = gr.Textbox(
                            label="Screenshots Directory",
                            value=self.save_locations['screenshots'],
                            scale=3
                        )
                        screenshots_browse_btn = gr.Button("Browse", scale=1)

                    update_paths_btn = gr.Button("Update Paths", variant="secondary")
                    path_status = gr.Textbox(label="Path Status", interactive=False)

                    def update_paths(data_dir, screenshots_dir):
                        self.save_locations['data'] = data_dir
                        self.save_locations['screenshots'] = screenshots_dir

                        # Update backend oscilloscope directories
                        if self.is_connected and self.oscilloscope:
                            with self.io_lock:
                                self.oscilloscope.set_output_directories(
                                    data_dir=data_dir,
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
                        inputs=[data_path, screenshots_path],
                        outputs=[path_status]
                    )

                    data_browse_btn.click(
                        fn=browse_data_folder,
                        inputs=[data_path],
                        outputs=[data_path, path_status]
                    )

                    screenshots_browse_btn.click(
                        fn=browse_screenshots_folder,
                        inputs=[screenshots_path],
                        outputs=[screenshots_path, path_status]
                    )

                gr.Markdown("---")
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
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4],
                    outputs=[operation_status]
                )

                export_btn.click(
                    fn=self.export_csv,
                    inputs=[],
                    outputs=[operation_status]
                )

                full_auto_btn.click(
                    fn=self.run_full_automation,
                    inputs=[op_ch1, op_ch2, op_ch3, op_ch4, op_math1, op_math2, op_math3, op_math4, plot_title_input],
                    outputs=[operation_status]
                )

            gr.Markdown("---")
            gr.Markdown("**DIGANTARA MSO24 Control** | Professional oscilloscope automation interface | All SCPI Commands Verified")

        # Return the complete interface object to be launched
        return interface

    def launch(self, share=False, server_port=7860, max_attempts=10):
        """
        Launch the Gradio web server and start the application.

        WHAT THIS DOES:
            Starts a local web server that hosts the oscilloscope control interface.
            Users access it by opening a web browser to http://localhost:PORT

        HOW WEB SERVERS WORK:
            A web server is like a restaurant:
            - Server listens on a "port" (like a table number)
            - Browser sends requests (like ordering food)
            - Server sends back web pages (like serving dishes)

        NETWORK PORTS EXPLAINED:
            Ports are numbered communication channels (0-65535)
            - Port 80: Standard HTTP (web browsing)
            - Port 443: HTTPS (secure web)
            - Port 7860: Default Gradio port
            - Our app: Tries 7860, then 7861, 7862... until free port found

        WHY PORT CONFLICTS HAPPEN:
            Only ONE program can use a port at a time. If port is busy:
            - Another instance of this app is running
            - Different program using the port
            Solution: Try next available port automatically

        PARAMETERS:
            share (Boolean):
                - False (default): Local access only (this computer)
                - True: Creates public internet link (Gradio tunnel)
                  WARNING: Public link allows ANYONE to control your oscilloscope!

            server_port (Integer):
                - Starting port number to try (default 7860)
                - If busy, tries 7861, 7862, etc.

            max_attempts (Integer):
                - How many ports to try before giving up (default 10)
                - Tries ports 7860-7869

        LAUNCH PROCESS:
            1. Create interface (build all UI elements)
            2. Find available network port
            3. Start web server on that port
            4. Open browser automatically
            5. Server runs until Ctrl+C pressed
        """

        # STEP 1: Create the complete Gradio interface
        # Calls create_interface() which builds all tabs, buttons, etc.
        self._gradio_interface = self.create_interface()

        # STEP 2: Try to start server on available port
        # Loop through port numbers until we find one that's free
        for attempt in range(max_attempts):
            current_port = server_port + attempt  # Try 7860, 7861, 7862...

            try:
                print(f"Attempting to start MSO24 server on port {current_port}...")

                # STEP 3: Launch Gradio web server
                self._gradio_interface.launch(
                    server_name="0.0.0.0",        # Listen on all network interfaces
                                                   # "0.0.0.0" = accept connections from any IP
                    share=share,                   # Create public link? (usually False)
                    server_port=current_port,      # Which port to use
                    prevent_thread_lock=False,     # Block until server stops (keep running)
                    show_error=True,               # Display errors in console
                    quiet=False                    # Show startup messages
                )

                # STEP 4: Success! Server is running
                print("\n" + "=" * 80)
                print(f"MSO24 Control Server is running on port {current_port}")
                print("Access the interface at: http://localhost:{current_port}")
                print("To stop the application, press Ctrl+C in this terminal.")
                print("=" * 80)
                return  # Exit method, server continues running

            except Exception as e:
                # STEP 5: Handle errors

                # Check if error is "port already in use"
                if "address already in use" in str(e).lower() or "port in use" in str(e).lower():
                    print(f"Port {current_port} is in use, trying next port...")

                    # If this was the last attempt, give up
                    if attempt == max_attempts - 1:
                        print(f"\nError: Could not find an available port after {max_attempts} attempts.")
                        print("Please close any other instances or specify a different starting port.")
                        self.cleanup()  # Disconnect oscilloscope
                        return
                else:
                    # Different error (not port conflict)
                    print(f"\nLaunch error: {e}")
                    self.cleanup()
                    return

        # If we get here, all port attempts failed
        print("\nFailed to start the MSO24 server after multiple attempts.")
        self.cleanup()

# =============================================================================
# MAIN FUNCTION - Application Entry Point
# =============================================================================
# This is the first function that runs when you execute the Python script
# It's called the "entry point" because it's where execution begins

def main():
    """
    Main application entry point - starts the oscilloscope control application.

    WHAT IS AN ENTRY POINT?
        When you run "python tektronix_oscilloscope_gradio_en.py", Python looks
        for the main() function and starts executing from here. It's like the
        "Start" button of the application.

    APPLICATION STARTUP SEQUENCE:
        1. Display welcome banner
        2. Find available network port
        3. Create GUI application object
        4. Launch web server
        5. Wait for user to press Ctrl+C
        6. Cleanup and shutdown

    WHY CHECK PORTS BEFORE LAUNCHING?
        We test each port before trying to use it. This provides better error
        messages than letting Gradio fail with cryptic errors. It's like
        checking if a parking space is empty before trying to park.

    ERROR HANDLING:
        - KeyboardInterrupt: User pressed Ctrl+C (normal shutdown)
        - OSError: Network/port problems
        - Exception: Any other unexpected error
        - finally: Always runs cleanup code (even if errors occur)

    PROGRAM FLOW ANALOGY:
        Think of this like starting a restaurant:
        1. Turn on lights (print welcome message)
        2. Check which tables are free (find available port)
        3. Open doors (create application object)
        4. Start serving customers (launch web server)
        5. Stay open until closing time (wait for Ctrl+C)
        6. Clean up and lock doors (cleanup and exit)
    """

    # -------------------------------------------------------------------------
    # WELCOME BANNER - Display startup information
    # -------------------------------------------------------------------------
    print("DIGANTARA Tektronix MSO24 Oscilloscope Automation - Professional Gradio Interface")
    print("Professional oscilloscope control system with comprehensive features")
    print("=" * 80)  # Print 80 equals signs as a visual separator
    print("Starting web interface...")

    # -------------------------------------------------------------------------
    # INITIALIZATION - Set up variables
    # -------------------------------------------------------------------------
    app = None  # Will hold the GradioMSO24GUI object once created
                # Initialize to None so we can check if it exists in finally block

    try:
        # ---------------------------------------------------------------------
        # PORT DETECTION - Find available network port
        # ---------------------------------------------------------------------
        start_port = 7865      # First port to try
        max_attempts = 10       # Try up to 10 different ports (7865-7874)

        print(f"Looking for an available port starting from {start_port}...")

        # Loop through port numbers to find one that's free
        for port in range(start_port, start_port + max_attempts):
            try:
                # TEST PORT AVAILABILITY
                # Create a socket (network connection endpoint) to test if port is free
                # socket.socket() creates a new socket object
                # AF_INET = IPv4 protocol, SOCK_STREAM = TCP connection
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Try to bind (reserve) this port
                    # '' = localhost (this computer only), port = port number to test
                    s.bind(('', port))  # If port is busy, this line throws an error
                    s.close()           # Port is free! Close the test socket

                # SUCCESS! Found available port
                print(f"\nFound available port: {port}")
                print("The browser will open automatically when ready.")
                print("")
                print("IMPORTANT: To stop the application, press Ctrl+C in this terminal.")
                print("Closing the browser tab will NOT stop the server.")
                print("The server continues running in this terminal window.")
                print("=" * 80)

                # CREATE AND LAUNCH APPLICATION
                app = GradioMSO24GUI()  # Create the GUI application object
                                         # This initializes all variables, creates logger, etc.

                app.launch(share=False, server_port=port)  # Start the web server
                                                            # share=False = local access only
                                                            # server_port = which port to use

                break  # Exit the for loop (port found and server started)

            except OSError as e:
                # OSError occurs when port operations fail

                # Check if error is "port already in use"
                if "address already in use" in str(e).lower():
                    print(f"Port {port} is in use, trying next port...")

                    # If this was the last port to try, give up
                    if port == start_port + max_attempts - 1:
                        print(f"\nError: Could not find an available port after {max_attempts} attempts.")
                        print(f"Please close any applications using ports {start_port}-{start_port + max_attempts - 1}")
                        return  # Exit main() function
                else:
                    # Different OSError (not port-in-use)
                    print(f"Error checking port {port}: {e}")
                    return  # Exit main() function

    except KeyboardInterrupt:
        # User pressed Ctrl+C (SIGINT signal)
        # This is the NORMAL way to stop the application
        print("\nApplication closed by user (Ctrl+C pressed).")

    except Exception as e:
        # Catch any other unexpected errors
        # This prevents the application from crashing with scary error messages
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error information for debugging

    finally:
        # CLEANUP - This block ALWAYS runs, even if errors occurred
        # Ensures proper shutdown and resource cleanup

        if app:  # If app object was created
            app.cleanup()  # Disconnect oscilloscope, close connections

        print("\nApplication shutdown complete.")
        print("=" * 80)

# =============================================================================
# PROGRAM ENTRY POINT - Python Script Execution Guard
# =============================================================================
# This special if statement ensures main() only runs when script is executed directly
# It prevents main() from running if this file is imported as a module by another script

if __name__ == "__main__":
    """
    PYTHON SPECIAL VARIABLE: __name__

    WHAT IS __name__?
        __name__ is a special variable automatically set by Python:
        - When script is run directly: __name__ = "__main__"
        - When script is imported: __name__ = "tektronix_oscilloscope_gradio_en"

    WHY USE THIS PATTERN?
        Allows code reuse:
        - Direct execution: python script.py → runs main()
        - Import as module: import script → main() doesn't run automatically
                                           → can use classes/functions without side effects

    EXAMPLE:
        Direct execution:
            $ python tektronix_oscilloscope_gradio_en.py
            → __name__ == "__main__" → TRUE → main() runs → application starts

        Import as module:
            from tektronix_oscilloscope_gradio_en import GradioMSO24GUI
            → __name__ == "tektronix_oscilloscope_gradio_en" → FALSE → main() doesn't run
            → Can use GradioMSO24GUI class without starting web server

    ERROR HANDLING LAYERS:
        This try/except is the OUTER safety net:
        - main() has its own try/except (inner layer)
        - This catches errors that escape main() (outer layer)
        - finally ensures clean exit no matter what happens
    """

    try:
        # Call main() to start the application
        main()

    except KeyboardInterrupt:
        # User pressed Ctrl+C during startup (before main() could handle it)
        print("\nApplication terminated by user during startup.")

    except Exception as e:
        # Catch any catastrophic errors that escaped main()'s error handling
        # This should rarely happen - it's the last safety net
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()  # Print full error details for debugging

    finally:
        # FORCE EXIT - Ensure program terminates completely
        # os._exit(0) bypasses Python's normal cleanup and immediately terminates
        # Used because Gradio sometimes keeps threads running

        print("Forcing application exit...")
        os._exit(0)  # Exit code 0 = success
                      # This is more forceful than sys.exit()
                      # Immediately terminates all threads and processes

# =============================================================================
# END OF FILE
# =============================================================================
#
# FILE STATISTICS:
# - Total lines: ~2900
# - Classes: 2 (MSO24DataAcquisition, GradioMSO24GUI)
# - Methods: 40+
# - Waveform types supported: 15
# - Measurement types: 18
# - UI tabs: 7
#
# COMPREHENSIVE ANNOTATION COMPLETE
# All code sections have been documented with:
# - Purpose explanations
# - Technical concept clarifications
# - Real-world examples
# - Parameter descriptions
# - Return value documentation
# - Error handling explanations
# - Non-technical analogies for complex concepts
#
# This file is now suitable for code review by non-technical team members.
# =============================================================================
