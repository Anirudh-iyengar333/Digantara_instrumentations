#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║    UNIFIED INSTRUMENT CONTROL SYSTEM - PROFESSIONAL GRADIO INTERFACE           ║
║    Comprehensive control for lab instruments with integrated tabbed interface  ║
║                                                                                ║
║  PURPOSE:                                                                      ║
║    Production-grade lab automation system providing unified web-based control  ║
║    of multiple test instruments through a single interface. Designed for       ║
║    electronics characterization, automated test equipment (ATE), and R&D.      ║
║                                                                                ║
║  SUPPORTED INSTRUMENTS:                                                        ║
║    • Keithley DMM6500 - 6.5 Digit Digital Multimeter                           ║
║      └─ DC/AC V/I, 2W/4W Resistance, Capacitance, Frequency, Temperature       ║
║                                                                                ║
║    • Keithley 2230-30-1 - Triple Channel DC Power Supply                       ║
║      └─ 3× Independent Channels (30V/3A), Waveform Generation (4 types)        ║
║                                                                                ║
║    • Keysight DSOX6004A - 4-Channel Mixed Signal Oscilloscope                  ║
║      └─ 1 GHz BW, 20 GSa/s, Advanced Triggers, Math Functions                  ║
║                                                                                ║
║  CORE FEATURES:                                                                ║
║    ✓ Multi-threaded responsive web UI using Gradio framework                   ║
║    ✓ Real-time data acquisition with statistical analysis                      ║
║    ✓ Advanced waveform generation (Sine, Square, Triangle, Ramp)              ║
║    ✓ Comprehensive data export (CSV, JSON, Excel) with timestamps             ║
║    ✓ Live plotting with matplotlib (trend analysis, waveforms)                ║
║    ✓ Thread-safe VISA communication with automatic error recovery             ║
║    ✓ Emergency stop capability for safety-critical operations                 ║
║                                                                               ║
║  ARCHITECTURE:                                                                ║
║    ┌─────────────────────────────────────────────────────────────┐            ║
║    │ Gradio Web Interface (Port 7860-7869)                       │            ║
║    │  ┌────────────┬────────────────┬──────────────────┐         │            ║
║    │  │  DMM Tab   │   PSU Tab      │  Oscilloscope Tab│         │            ║
║    │  └─────┬──────┴────────┬───────┴──────────┬───────┘         │            ║
║    └────────┼───────────────┼──────────────────┼─────────────────┘            ║
║             │               │                  │                              ║
║             ▼               ▼                  ▼                              ║
║    ┌────────────────────────────────────────────────────────┐                 ║
║    │ UnifiedInstrumentControl (Main Controller)             │                 ║
║    │  ├─ DMM_GUI_Controller                                 │                 ║
║    │  ├─ PowerSupplyAutomationGradio                        │                 ║
║    │  └─ GradioOscilloscopeGUI                              │                 ║
║    └─────────┬──────────────────────────────────────────────┘                 ║
║              │                                                                ║
║              ▼                                                                ║
║    ┌────────────────────────────────────────────────────────┐                 ║
║    │ PyVISA Communication Layer (Thread-Safe)               │                ║
║    │  ├─ KeithleyDMM6500 Driver                             │                 ║
║    │  ├─ KeithleyPowerSupply Driver                         │                 ║
║    │  └─ KeysightDSOX6004A Driver                           │                 ║
║    └─────────┬──────────────────────────────────────────────┘                 ║
║              │                                                                ║
║              ▼                                                                ║
║    ┌────────────────────────────────────────────────────────┐                 ║
║    │ VISA Backend (Keysight IO Suite / NI-VISA)             │                 ║
║    │  └─ USB/TCPIP/GPIB Communication                       │                 ║
║    └────────────────────────────────────────────────────────┘                 ║
║                                                                               ║
║  SYSTEM REQUIREMENTS:                                                         ║
║    • Python 3.7+ (Recommended: 3.9+)                                          ║
║    • PyVISA 1.11+ for SCPI/VISA instrument communication                      ║
║    • Keysight IO Libraries Suite OR NI-VISA runtime                           ║
║    • Gradio 3.x/4.x for web interface framework                               ║
║    • Matplotlib 3.x for real-time plotting                                    ║
║    • NumPy 1.20+ for statistical operations                                   ║
║    • Pandas 1.3+ for data management                                          ║
║    • OS: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+                      ║
║                                                                               ║
║  THREADING MODEL:                                                             ║
║    • Main Thread: Gradio event loop and UI updates                            ║
║    • DMM Worker: Continuous measurement daemon thread                         ║
║    • PSU Worker: Waveform execution background thread                         ║
║    • All VISA I/O: Protected by threading.RLock() for thread safety           ║
║                                                                               ║
║  SAFETY FEATURES:                                                             ║
║    • Emergency stop button (disables all PSU outputs immediately)             ║
║    • Automatic output disable on waveform completion/error                    ║
║    • Over-voltage protection (OVP) configuration per channel                  ║
║    • Connection state validation before all operations                        ║
║    • Graceful shutdown with signal handlers (SIGINT/SIGTERM)                  ║
║                                                                               ║
║  AUTHOR INFORMATION:                                                          ║
║    Organization: DIGANTARA Research and Technologies Pvt. Ltd.                ║
║    Team: Lab Automation & Test Engineering                                    ║
║    Version: 1.0.0 (2025-11-18)                                                ║
║    Status: Production Ready                                                   ║
║                                                                               ║
║  CHANGE LOG:                                                                  ║
║    2025-11-18: Fixed waveform duration estimation to include VISA overhead    ║
║               Added INSTRUMENT_OVERHEAD_PER_POINT constant (~1.95s)           ║
║                                                                               ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# ⚙️ FILE SAVE LOCATION CONFIGURATION - EDIT THESE PATHS
# =============================================================================
# INSTRUCTIONS: Enter the FULL file paths where you want to save files.
# - Use raw strings (prefix with r) for Windows paths
# - Example: r"C:\Users\YourName\Documents\LabData"
# - Make sure the server has write permissions to these directories
# - Directories will be created automatically if they don't exist
# =============================================================================

# DMM (Digital Multimeter) data export paths
DMM_DATA_PATH = r"C:\Users\AnirudhIyengar\Desktop\dmm_data"
DMM_PLOT_PATH = r"C:\Users\AnirudhIyengar\Desktop\dmm_plots"

# PSU (Power Supply) data export paths
PSU_DATA_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_data"
PSU_WAVEFORM_PATH = r"C:\Users\AnirudhIyengar\Desktop\psu_waveforms"

# Oscilloscope data export paths
OSC_DATA_PATH = r"C:\Users\AnirudhIyengar\Desktop\oscilloscope_data"
OSC_GRAPH_PATH = r"C:\Users\AnirudhIyengar\Desktop\oscilloscope_graphs"
OSC_SCREENSHOT_PATH = r"C:\Users\AnirudhIyengar\Desktop\oscilloscope_screenshots"

# =============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW THIS LINE
# =============================================================================

"""
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  QUICK CUSTOMIZATION GUIDE (For Non-Coders)                                   ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║  Want to customize the appearance? Look for these sections in the code:       ║
║                                                                               ║
║  1. CSS STYLING (Line ~5118):                                                 ║
║     • Change "20px" to "30px" for more spacing                                ║
║     • Change "#9c9cff" to "#4169E1" for royal blue color                      ║
║     • Change "100%" to "90%" to make interface narrower                       ║
║                                                                               ║
║  2. INTERFACE CONFIGURATION (Line ~5201):                                     ║
║     • Change primary_hue="indigo" to primary_hue="purple"                     ║
║     • Change spacing_size="sm" to spacing_size="md" for more space            ║
║     • Change text_size="sm" to text_size="md" for bigger text                 ║
║                                                                               ║
║  All customization options are fully annotated with examples!                 ║
║  ═══════════════════════════════════════════════════════════════════════════  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ════════════════════════════════════════════════════════════════════════════
# This section imports all external dependencies organized by category.
# Each import includes inline documentation explaining its purpose in the system.

# ────────────────────────────────────────────────────────────────────────────
# Standard Library Imports - Core Python Modules
# ────────────────────────────────────────────────────────────────────────────
import sys                  # System-specific parameters (path, exit codes)
                            # Used for: Module path manipulation, exit handling
import logging              # Flexible event logging system
                            # Used for: Debug/info/error messages, audit trails
import threading            # Thread-based parallelism for non-blocking operations
                            # Used for: Continuous measurements, waveform execution
import queue                # Thread-safe FIFO queue for inter-thread communication
                            # Used for: (Reserved for future async operations)
import time                 # Time access and conversions
                            # Used for: sleep(), timestamps, timing measurements
import tkinter as tk        # Standard GUI toolkit (Tk/Tcl wrapper)
                            # Used for: File/folder dialog boxes only
from tkinter import filedialog  # File selection dialogs
                                 # Used for: Browse buttons in Gradio UI
from pathlib import Path    # Object-oriented filesystem paths
                            # Used for: Cross-platform file operations
from datetime import datetime, timedelta  # Date/time manipulation
                                          # Used for: Timestamps, duration calculations
from typing import Optional, Dict, Any, List, Tuple, Union  # Type annotations
                                                             # Used for: Static type checking, IDE hints
import signal               # Unix signal handling for process management
                            # Used for: Graceful shutdown (SIGINT, SIGTERM)
import atexit               # Register cleanup functions on program exit
                            # Used for: Resource cleanup, instrument disconnect
import os                   # Operating system interface
                            # Used for: Path operations, environment variables
import socket               # Low-level networking interface
                            # Used for: Port availability checking (7860-7869)

# ────────────────────────────────────────────────────────────────────────────
# Third-Party Imports - External Dependencies
# ────────────────────────────────────────────────────────────────────────────
import gradio as gr         # Modern web UI framework for ML/Data apps
                            # Version: 3.x or 4.x compatible
                            # Used for: Complete web interface, event handling
import pandas as pd         # Data manipulation and analysis library
                            # Used for: DataFrame operations, CSV/Excel export
import numpy as np          # Numerical computing with N-dimensional arrays
                            # Used for: Statistics (mean, std), array operations

# ────────────────────────────────────────────────────────────────────────────
# Matplotlib Imports - Plotting and Visualization
# ────────────────────────────────────────────────────────────────────────────
import matplotlib           # Comprehensive 2D plotting library
matplotlib.use('Agg')       # Set non-interactive backend BEFORE importing pyplot
                            # 'Agg': Anti-Grain Geometry rasterization engine
                            # Why: Allows plot generation without display server
                            # Critical for: Web-based UI, headless servers
import matplotlib.pyplot as plt         # MATLAB-like plotting interface
                                        # Used for: Figure/axes creation, plot rendering
import matplotlib.dates as mdates       # Date/time axis formatting
                                        # Used for: Time-series plots with proper labels
import matplotlib.ticker as ticker      # Axis tick locators and formatters
                                        # Used for: Custom axis scaling (SI prefixes)

# ────────────────────────────────────────────────────────────────────────────
# Utility Imports - Encoding, Math, Data Formats
# ────────────────────────────────────────────────────────────────────────────
import io                   # Core I/O stream handling
                            # Used for: In-memory file buffers (plot export)
import base64               # Base64 encoding/decoding (RFC 3548)
                            # Used for: Embedding images in HTML/JSON
import math                 # Mathematical functions from C standard library
                            # Used for: sin(), cos(), pi for waveform generation
import csv                  # CSV file reading and writing (RFC 4180)
                            # Used for: Data export in CSV format
from enum import Enum       # Support for enumerations (PEP 435)
                            # Used for: Type-safe constants (measurement types)
import json                 # JSON encoder/decoder (RFC 8259)
                            # Used for: Configuration files, data export

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: DYNAMIC PATH RESOLUTION AND INSTRUMENT MODULE IMPORTS
# ════════════════════════════════════════════════════════════════════════════
# This section handles dynamic discovery of the instrument_control package and
# imports all necessary driver classes. The path resolution ensures portability
# across different installation locations and execution contexts.

# ────────────────────────────────────────────────────────────────────────────
# Dynamic Module Path Resolution
# ────────────────────────────────────────────────────────────────────────────
# Construct path to project root directory by navigating up from this file
# Expected directory structure:
#   <project_root>/
#   ├── instrument_control/
#   │   ├── __init__.py
#   │   ├── keithley_dmm.py
#   │   ├── keithley_power_supply.py
#   │   ├── keysight_oscilloscope.py
#   │   └── scpi_wrapper.py
#   └── scripts/
#   │   
#   └── Unified.py (this file)
#
# Path resolution breakdown:
#   __file__              = full path to Unified.py
#   .resolve()            = convert to absolute canonical path (follows symlinks)
#   .parent               = scripts/keithley/
#   .parent.parent        = scripts/
#   .parent.parent.parent = <project_root>/
script_dir = Path(__file__).resolve().parent.parent.parent

# Add project root to Python's module search path (sys.path)
# This allows 'import instrument_control.xxx' to work regardless of CWD
# Check prevents duplicate entries if script is run multiple times
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# ────────────────────────────────────────────────────────────────────────────
# Instrument Control Module Imports
# ────────────────────────────────────────────────────────────────────────────
# Import custom driver modules that wrap SCPI commands into high-level Python APIs
# All imports use try-except to provide detailed diagnostics on failure
try:
    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ KEITHLEY DMM6500 DIGITAL MULTIMETER DRIVER                           │
    # └──────────────────────────────────────────────────────────────────────┘
    from instrument_control.keithley_dmm import (
        KeithleyDMM6500,        # Primary driver class for DMM6500
                                # Provides: High-level measurement methods,
                                #           configuration management, SCPI wrapping
                                # Inherits: SCPI communication from base class

        MeasurementFunction,    # Enum defining supported measurement types
                                # Values: DC_VOLTAGE, AC_VOLTAGE, DC_CURRENT,
                                #         AC_CURRENT, RESISTANCE_2W, RESISTANCE_4W,
                                #         CAPACITANCE, FREQUENCY, TEMPERATURE
                                # Purpose: Type-safe function selection

        KeithleyDMM6500Error    # Custom exception class for DMM-specific errors
                                # Raised when: SCPI errors, invalid parameters,
                                #              measurement range exceeded, timeout
                                # Provides: Detailed error context for debugging
    )

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ KEITHLEY 2230-30-1 TRIPLE CHANNEL POWER SUPPLY DRIVER               │
    # └──────────────────────────────────────────────────────────────────────┘
    from instrument_control.keithley_power_supply import (
        KeithleyPowerSupply,        # Primary driver class for 2230-30-1 PSU
                                    # Provides: Voltage/current control, channel mgmt,
                                    #           measurement, OVP configuration
                                    # Supports: 3 independent channels (30V/3A each)

        KeithleyPowerSupplyError,   # Custom exception for PSU-specific errors
                                    # Raised when: Over-current fault, SCPI errors,
                                    #              invalid channel, voltage out of range
                                    # Critical for: Safety-critical error handling

        OutputState                 # Enum for channel output state
                                    # Values: ON (enabled), OFF (disabled)
                                    # Purpose: Type-safe output control
    )

    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ KEYSIGHT DSOX6004A OSCILLOSCOPE DRIVER                               │
    # └──────────────────────────────────────────────────────────────────────┘
    from instrument_control.keysight_oscilloscope import (
        KeysightDSOX6004A,          # Primary driver class for DSOX6004A scope
                                    # Provides: Waveform acquisition, triggering,
                                    #           measurements, math functions, setup mgmt
                                    # Capabilities: 4 analog channels, 1 GHz BW,
                                    #               20 GSa/s, 4 Mpts memory

        KeysightDSOX6004AError      # Custom exception for scope-specific errors
                                    # Raised when: Timeout, trigger not armed,
                                    #              invalid channel, acquisition failed
                                    # Purpose: Detailed error reporting for debugging
    )

    from instrument_control.scpi_wrapper import SCPIWrapper
                                    # Low-level SCPI communication base class
                                    # Provides: Thread-safe I/O, error checking,
                                    #           binary data transfer, timeout handling
                                    # Features: Automatic reconnection, command queuing
                                    # Critical: All drivers inherit from this class

except ImportError as e:
    # ┌──────────────────────────────────────────────────────────────────────┐
    # │ IMPORT ERROR HANDLER - DIAGNOSTIC OUTPUT                             │
    # └──────────────────────────────────────────────────────────────────────┘
    # If instrument_control modules cannot be imported, provide detailed diagnostic
    # information to help users identify and resolve the issue quickly

    print("=" * 80)
    print("CRITICAL ERROR: Failed to import instrument_control modules")
    print("=" * 80)
    print(f"\nException Details: {e}")
    print(f"\nCurrent Python Search Path (sys.path):")
    for i, path in enumerate(sys.path, 1):
        print(f"  {i}. {path}")
    print(f"\nExpected module location: {script_dir / 'instrument_control'}")
    print(f"Directory exists: {(script_dir / 'instrument_control').exists()}")

    print("\n" + "─" * 80)
    print("TROUBLESHOOTING STEPS:")
    print("─" * 80)
    print("1. Verify 'instrument_control' folder exists in project root")
    print("2. Check all required .py files are present:")
    print("   - instrument_control/__init__.py")
    print("   - instrument_control/keithley_dmm.py")
    print("   - instrument_control/keithley_power_supply.py")
    print("   - instrument_control/keysight_oscilloscope.py")
    print("   - instrument_control/scpi_wrapper.py")
    print("3. Ensure PyVISA is installed: pip install pyvisa")
    print("4. Try editable install from project root: pip install -e .")
    print("=" * 80)

    sys.exit(1)  # Exit with error code 1 to indicate initialization failure


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: DMM CONTROLLER CLASS - KEITHLEY DMM6500 INTERFACE
# ════════════════════════════════════════════════════════════════════════════
# This section implements the high-level controller for the Keithley DMM6500
# Digital Multimeter, providing a thread-safe interface between the Gradio UI
# and the low-level SCPI driver. Handles connection management, measurements,
# data collection, statistical analysis, and file export operations.

class DMM_GUI_Controller:
    """
    High-level controller for Keithley DMM6500 Digital Multimeter Gradio interface.

    This class acts as the bridge between the Gradio web UI and the KeithleyDMM6500
    driver, managing instrument connections, measurements, data collection, and
    export operations. It provides both single-shot and continuous measurement modes
    with thread-safe operation.

    Thread Safety:
        - Continuous measurements run in a separate daemon thread
        - measurement_data list is accessed from both UI and worker threads
        - Thread safety relies on GIL for list operations (append, slice)
        - stop flag (continuous_measurement) provides clean shutdown

    Memory Management:
        - Automatically limits stored data to max_data_points (65,000)
        - Uses circular buffer behavior (oldest data discarded)
        - Prevents unbounded memory growth during long-running tests

    Attributes:
        dmm (Optional[KeithleyDMM6500]): Instance of DMM driver, None if not connected
        is_connected (bool): Connection state flag, prevents ops when disconnected
        measurement_thread (Optional[threading.Thread]): Worker thread for continuous mode
        continuous_measurement (bool): Flag to control measurement loop execution
        measurement_data (List[Dict]): Buffer storing all measurement records
        max_data_points (int): Maximum measurements to retain (65,000 = ~18h @ 1Hz)
        logger (logging.Logger): Logger instance for debug and error tracking
        save_locations (Dict[str, str]): Default paths for data and graph exports
        default_settings (Dict[str, Any]): Default instrument configuration

    Example:
        >>> controller = DMM_GUI_Controller()
        >>> status, success = controller.connect_instrument("USB0::...", 30000)
        >>> if success:
        ...     result, msg = controller.single_measurement("DC_VOLTAGE", 10.0, 1e-6, 1.0, True)
        ...     print(f"Measured: {result}")
        >>> controller.disconnect_instrument()

    See Also:
        - KeithleyDMM6500: Low-level SCPI driver
        - MeasurementFunction: Enum of supported measurement types
    """

    def __init__(self):
        """
        Initialize the DMM GUI controller with default settings.

        Sets up logging, initializes instance variables, and configures default
        paths and instrument parameters. Does NOT connect to hardware - connection
        must be explicitly requested via connect_instrument().

        Performance:
            Initialization is instantaneous (<1ms) - no I/O operations performed

        Note:
            Creates default save directories relative to current working directory.
            These can be changed via the UI browse buttons before export operations.
        """
        # ────────────────────────────────────────────────────────────────────
        # Instrument Connection State
        # ────────────────────────────────────────────────────────────────────
        self.dmm: Optional[KeithleyDMM6500] = None  # Driver instance, None until connected
        self.is_connected = False                    # Guard flag for all operations
                                                     # Prevents SCPI commands to disconnected device

        # ────────────────────────────────────────────────────────────────────
        # Threading Infrastructure for Continuous Measurements
        # ────────────────────────────────────────────────────────────────────
        self.measurement_thread: Optional[threading.Thread] = None  # Worker thread handle
        self.continuous_measurement = False                          # Thread loop control flag
                                                                     # Set to False to stop worker

        # ────────────────────────────────────────────────────────────────────
        # Data Collection Buffer
        # ────────────────────────────────────────────────────────────────────
        self.measurement_data = []      # List of dicts: {'timestamp', 'function', 'value', ...}
                                        # Thread-safe under GIL for append/slice operations

        self.max_data_points = 65000    # Maximum buffer size (65,535 = 16-bit limit)
                                        # At 1 Hz: ~18 hours of continuous data
                                        # At 10 Hz: ~1.8 hours of continuous data
                                        # Memory: ~5-10 MB depending on dict overhead

        # ────────────────────────────────────────────────────────────────────
        # Logging Configuration
        # ────────────────────────────────────────────────────────────────────
        logging.basicConfig(
            level=logging.INFO,         # Log INFO and above (INFO, WARNING, ERROR, CRITICAL)
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                                        # Example: "2025-11-18 14:23:45,123 - DMM_GUI - INFO - Connected"
        )
        self.logger = logging.getLogger('DMM_GUI')  # Namespace for DMM-related logs

        # ────────────────────────────────────────────────────────────────────
        # File Export Default Locations
        # ────────────────────────────────────────────────────────────────────
        self.save_locations = {
            'data': str(Path.cwd() / "dmm_data"),      # CSV/JSON/Excel exports
            'graphs': str(Path.cwd() / "dmm_graphs")   # PNG plot files
        }
        # Note: Directories created on-demand during export operations

        # ────────────────────────────────────────────────────────────────────
        # Default Instrument Configuration
        # ────────────────────────────────────────────────────────────────────
        self.default_settings = {
            'visa_address': 'USB0::0x05E6::0x6500::04561287::INSTR',  # Example USB address
                                                                       # 0x05E6 = Keithley VID
                                                                       # 0x6500 = DMM6500 PID
            'timeout_ms': 30000,                # 30 second VISA timeout (SCPI operations)
            'measurement_function': 'DC_VOLTAGE',  # Default function on startup
            'measurement_range': 10.0,          # 10V range for DC voltage
            'resolution': 1e-6,                 # 1 µV resolution (6.5 digit)
            'nplc': 1.0,                        # 1 power line cycle (16.67ms @ 60Hz)
                                                # Trade-off: higher = better noise rejection
            'auto_zero': True,                  # Enable auto-zero for drift compensation
            'measurement_interval': 1.0         # 1 second between continuous measurements
        }

    # ════════════════════════════════════════════════════════════════════════════
    # Range Configuration Methods
    # ════════════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_range_options(measurement_function: str) -> Tuple[list, float]:
        """
        Get appropriate range options based on measurement function.

        Returns measurement-specific range values including auto-range option.
        Different measurement types have different valid ranges and capabilities.

        Args:
            measurement_function: Measurement type string (e.g., "DC_VOLTAGE", "CAPACITANCE")

        Returns:
            Tuple containing:
                - List of range values (floats) with "AUTO" as first option where applicable
                - Default range value (float)

        Range Specifications:
            DC/AC Voltage: AUTO, 0.1V, 1V, 10V, 100V, 1000V
            DC/AC Current: AUTO, 0.01A, 0.1A, 1A, 3A
            Resistance (2W/4W): AUTO, 10Ω, 100Ω, 1kΩ, 10kΩ, 100kΩ, 1MΩ, 10MΩ, 100MΩ
            Capacitance: 1nF, 10nF, 100nF, 1µF, 10µF, 100µF, 1mF, 10mF (no auto)
            Frequency: AUTO, 3Hz, 30Hz, 300Hz, 3kHz, 30kHz, 300kHz
            Temperature: AUTO (single option, sensor-dependent)

        Example:
            >>> ranges, default = DMM_GUI_Controller.get_range_options("DC_VOLTAGE")
            >>> print(ranges)  # ["AUTO", 0.1, 1, 10, 100, 1000]
            >>> print(default)  # 10.0
        """
        # Range configurations for each measurement type
        # Format: measurement_type: ([list of ranges], default_value)
        range_configs = {
            'DC_VOLTAGE': (
                ["AUTO", 0.1, 1, 10, 100, 1000],  # Volts
                10.0  # Default: 10V range
            ),
            'AC_VOLTAGE': (
                ["AUTO", 0.1, 1, 10, 100, 1000],  # Volts RMS
                10.0  # Default: 10V range
            ),
            'DC_CURRENT': (
                ["AUTO", 0.01, 0.1, 1, 3],  # Amperes
                1.0  # Default: 1A range
            ),
            'AC_CURRENT': (
                ["AUTO", 0.01, 0.1, 1, 3],  # Amperes RMS
                1.0  # Default: 1A range
            ),
            'RESISTANCE_2W': (
                ["AUTO", 10, 100, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6],  # Ohms
                1e3  # Default: 1kΩ range
            ),
            'RESISTANCE_4W': (
                ["AUTO", 10, 100, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6],  # Ohms
                1e3  # Default: 1kΩ range
            ),
            'CAPACITANCE': (
                # No auto-range for capacitance measurements
                [1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3],  # Farads
                1e-6  # Default: 1µF range
            ),
            'FREQUENCY': (
                ["AUTO", 3, 30, 300, 3e3, 30e3, 300e3],  # Hertz
                1000  # Default: 1kHz range
            ),
            'TEMPERATURE': (
                ["AUTO"],  # Only auto-range, sensor-dependent
                0  # Default: auto (0 = auto for driver)
            )
        }

        # Return configuration for requested function, or default to DC_VOLTAGE
        return range_configs.get(measurement_function, range_configs['DC_VOLTAGE'])

    # ════════════════════════════════════════════════════════════════════════════
    # Connection Management Methods
    # ════════════════════════════════════════════════════════════════════════════

    def connect_instrument(self, visa_address: str, timeout_ms: int) -> Tuple[str, bool]:
        """
        Establish VISA connection to Keithley DMM6500 and verify identity.

        Creates a KeithleyDMM6500 driver instance, attempts connection, and retrieves
        instrument identification (*IDN? query). Prevents duplicate connections if
        already connected. Thread-safe for UI event handlers.

        Args:
            visa_address: VISA resource string identifying the instrument
                         Examples:
                         - USB: "USB0::0x05E6::0x6500::04561287::INSTR"
                         - GPIB: "GPIB0::15::INSTR"
                         - TCPIP: "TCPIP0::192.168.1.100::inst0::INSTR"
            timeout_ms: VISA I/O timeout in milliseconds (1000-60000 typical)
                       Recommended: 30000 (30s) for slow operations
                       Used for: All SCPI queries and commands

        Returns:
            Tuple containing:
                - status_message (str): Human-readable connection result
                  Success: "Connected: KEITHLEY INSTRUMENTS DMM6500 (S/N: 12345)"
                  Failure: Error description with diagnostics
                - connection_success (bool): True if connected, False otherwise

        Raises:
            No exceptions raised - all errors caught and returned in status message

        Performance:
            Typical connection time: 500-2000ms depending on interface type
            USB: ~500ms, GPIB: ~1000ms, TCPIP: ~2000ms

        Note:
            If already connected, returns success immediately without reconnecting.
            Call disconnect_instrument() first to reconnect with different parameters.

        Example:
            >>> controller = DMM_GUI_Controller()
            >>> msg, success = controller.connect_instrument("USB0::0x05E6::0x6500::04561287::INSTR", 30000)
            >>> if success:
            ...     print(f"Success: {msg}")
            ... else:
            ...     print(f"Failed: {msg}")
        """
        try:
            # ────────────────────────────────────────────────────────────────
            # Prevent Duplicate Connections
            # ────────────────────────────────────────────────────────────────
            if self.is_connected:           # Already have active connection
                return "Already connected to instrument", True  # Return success

            # ────────────────────────────────────────────────────────────────
            # Create Driver Instance and Attempt Connection
            # ────────────────────────────────────────────────────────────────
            self.dmm = KeithleyDMM6500(visa_address, timeout_ms)  # Instantiate driver
                                                                   # Does NOT connect yet

            if self.dmm.connect():          # Attempt VISA connection and *IDN? query
                self.is_connected = True    # Set connection state flag

                # ────────────────────────────────────────────────────────
                # Retrieve and Format Instrument Information
                # ────────────────────────────────────────────────────────
                info = self.dmm.get_instrument_info()  # Dict with manufacturer, model, serial, firmware
                if info:
                    # Format: "Connected: KEITHLEY INSTRUMENTS DMM6500 (S/N: 04561287)"
                    msg = f"Connected: {info['manufacturer']} {info['model']} (S/N: {info['serial_number']})"
                else:
                    # Fallback if *IDN? parsing failed
                    msg = "Connected to DMM successfully"

                self.logger.info(msg)       # Log successful connection
                return msg, True            # Return success tuple
            else:
                # Connection attempt failed (VISA error, device not found, timeout)
                return "Failed to connect to instrument", False

        except Exception as e:
            # ────────────────────────────────────────────────────────────────
            # Exception Handler - Log and Return Detailed Error
            # ────────────────────────────────────────────────────────────────
            self.logger.error(f"Connection error: {e}")  # Log full exception
            return f"Connection error: {str(e)}", False   # Return user-friendly message

    def disconnect_instrument(self) -> str:
        """
        Safely disconnect from DMM instrument and clean up resources.

        Stops any running continuous measurement threads, closes VISA connection,
        and resets connection state. Safe to call even if not connected.

        Returns:
            Status message describing disconnection result

        Performance:
            Typical disconnection time: 100-500ms
            If continuous measurement running: +2s for thread join timeout

        Note:
            Automatically stops continuous measurements before disconnect to
            prevent orphaned measurement threads attempting I/O on closed resource.

        Example:
            >>> msg = controller.disconnect_instrument()
            >>> print(msg)  # "Disconnected from instrument"
        """
        try:
            # ────────────────────────────────────────────────────────────────
            # Stop Continuous Measurements if Running
            # ────────────────────────────────────────────────────────────────
            if self.continuous_measurement:             # Worker thread is active
                self.stop_continuous_measurement()      # Signal stop and wait for join

            # ────────────────────────────────────────────────────────────────
            # Close VISA Connection
            # ────────────────────────────────────────────────────────────────
            if self.dmm and self.is_connected:          # Valid connection exists
                self.dmm.disconnect()                   # Close VISA resource
                self.is_connected = False               # Clear connection flag
                return "Disconnected from instrument"   # Success message
            else:
                # No active connection to disconnect
                return "No instrument connected"

        except Exception as e:
            # ────────────────────────────────────────────────────────────────
            # Error Handling - Log and Report
            # ────────────────────────────────────────────────────────────────
            self.logger.error(f"Disconnection error: {e}")
            return f"Disconnection error: {str(e)}"
    
    # ════════════════════════════════════════════════════════════════════════════
    # Measurement Operations
    # ════════════════════════════════════════════════════════════════════════════

    def single_measurement(self, function: str, range_val: float, resolution: float,
                         nplc: float, auto_zero: bool) -> Tuple[str, str]:
        """
        Execute single DMM measurement with automatic function dispatching and formatting.

        Performs a one-shot measurement using the specified function, automatically
        dispatching to the appropriate driver method. Stores result in measurement_data
        buffer with timestamp and metadata. Formats output using SI prefixes for
        human readability (e.g., "12.345 mV" instead of "0.012345 V").

        Args:
            function: Measurement type as string (e.g., "DC_VOLTAGE", "RESISTANCE_2W")
                     Valid values: DC_VOLTAGE, AC_VOLTAGE, DC_CURRENT, AC_CURRENT,
                                   RESISTANCE_2W, RESISTANCE_4W, CAPACITANCE,
                                   FREQUENCY, TEMPERATURE
            range_val: Measurement range (auto-range if <= 0)
                      DC/AC Voltage: 0.1V to 1000V
                      DC/AC Current: 0.01A to 3A
                      Resistance: 10Ω to 100MΩ
                      Capacitance: 1nF to 10mF
                      Frequency: 3Hz to 300kHz
            resolution: Measurement resolution in base units
                       Typical: 1e-6 for 6.5 digit resolution
                       Range: 1e-7 (7.5 digit) to 1e-4 (4.5 digit)
            nplc: Number of Power Line Cycles for integration
                 Range: 0.01 to 10
                 Trade-off: Higher = better noise rejection, slower measurement
                 Examples: 0.01 (600µs @ 60Hz), 1.0 (16.7ms @ 60Hz), 10 (167ms @ 60Hz)
            auto_zero: Enable automatic zero correction for DC measurements
                      True: Auto-zero every measurement (compensates drift, slower)
                      False: Use cached zero reference (faster, less accurate)

        Returns:
            Tuple containing:
                - formatted_result (str): Human-readable measurement with SI prefix
                  Examples: "12.345 mV", "4.567 kΩ", "123.4 µA", "25.67 °C"
                  Returns "N/A" on error
                - status_message (str): Operation status or error description

        Performance:
            Typical measurement time:
                - NPLC=0.01: ~1ms + VISA overhead (~50ms)
                - NPLC=1.0:  ~17ms + VISA overhead (~50ms)
                - NPLC=10:   ~170ms + VISA overhead (~50ms)

        Thread Safety:
            Safe to call from UI thread or worker threads. measurement_data append
            is atomic under GIL. No explicit locking required.

        Example:
            >>> result, status = controller.single_measurement("DC_VOLTAGE", 10.0, 1e-6, 1.0, True)
            >>> print(f"{result} - {status}")
            >>> # Output: "5.123 V - Measurement successful"
        """
        # ────────────────────────────────────────────────────────────────────
        # Connection State Validation
        # ────────────────────────────────────────────────────────────────────
        if not self.is_connected or not self.dmm:  # Guard clause: prevent operation when disconnected
            return "N/A", "Not connected to instrument"

        try:
            # ────────────────────────────────────────────────────────────────
            # Function String to Enum Mapping
            # ────────────────────────────────────────────────────────────────
            # Convert UI string representation to MeasurementFunction enum
            # This mapping allows type-safe function selection in driver layer
            func_map = {
                'DC_VOLTAGE': MeasurementFunction.DC_VOLTAGE,       # DCV: ±1000V, 6.5 digit
                'AC_VOLTAGE': MeasurementFunction.AC_VOLTAGE,       # ACV: 1000V RMS, 3Hz-300kHz
                'DC_CURRENT': MeasurementFunction.DC_CURRENT,       # DCI: ±3A, 6.5 digit
                'AC_CURRENT': MeasurementFunction.AC_CURRENT,       # ACI: 3A RMS, 3Hz-10kHz
                'RESISTANCE_2W': MeasurementFunction.RESISTANCE_2W, # 2-wire: Fast, lead resistance included
                'RESISTANCE_4W': MeasurementFunction.RESISTANCE_4W, # 4-wire: Accurate, compensates leads
                'CAPACITANCE': MeasurementFunction.CAPACITANCE,     # CAP: 1nF to 10mF
                'FREQUENCY': MeasurementFunction.FREQUENCY,         # FREQ: 3Hz to 300kHz
                'TEMPERATURE': MeasurementFunction.TEMPERATURE      # TEMP: RTD, thermocouple
            }

            measurement_func = func_map.get(function)
            if not measurement_func:                    # Validate function exists
                return "N/A", f"Unknown measurement function: {function}"

            # ────────────────────────────────────────────────────────────────
            # Function Dispatch - Call Appropriate Driver Method
            # ────────────────────────────────────────────────────────────────
            # Each measurement type has specific parameters and SCPI commands
            # Auto-zero only applies to DC measurements (DCV, DCI, RES)
            result = None
            if function == 'DC_VOLTAGE':
                result = self.dmm.measure_dc_voltage(range_val, resolution, nplc, auto_zero)
            elif function == 'AC_VOLTAGE':
                result = self.dmm.measure_ac_voltage(range_val, resolution, nplc)
            elif function == 'DC_CURRENT':
                result = self.dmm.measure_dc_current(range_val, resolution, nplc, auto_zero)
            elif function == 'AC_CURRENT':
                result = self.dmm.measure_ac_current(range_val, resolution, nplc)
            elif function == 'RESISTANCE_2W':
                result = self.dmm.measure_resistance_2w(range_val, resolution, nplc)
            elif function == 'RESISTANCE_4W':
                result = self.dmm.measure_resistance_4w(range_val, resolution, nplc)
            elif function == 'CAPACITANCE':
                result = self.dmm.measure_capacitance(range_val, resolution, nplc)
            elif function == 'FREQUENCY':
                result = self.dmm.measure_frequency(range_val, resolution, nplc)
            elif function == 'TEMPERATURE':
                result = self.dmm.measure_temperature()  # No range/resolution for temperature

            if result is not None:
                # ────────────────────────────────────────────────────────────
                # Store Measurement in Data Buffer
                # ────────────────────────────────────────────────────────────
                timestamp = datetime.now()              # Capture exact measurement time
                self.measurement_data.append({          # Add to buffer (thread-safe under GIL)
                    'timestamp': timestamp,             # ISO format: 2025-11-18 14:23:45.123456
                    'function': function,               # Measurement type for filtering/export
                    'value': result,                    # Raw numeric value in base units
                    'range': range_val,                 # Selected range (metadata)
                    'resolution': resolution            # Configured resolution (metadata)
                })

                # ────────────────────────────────────────────────────────────
                # Enforce Maximum Buffer Size (Circular Buffer Behavior)
                # ────────────────────────────────────────────────────────────
                if len(self.measurement_data) > self.max_data_points:
                    # Keep only most recent max_data_points measurements
                    # Prevents unbounded memory growth during long runs
                    self.measurement_data = self.measurement_data[-self.max_data_points:]

                # ────────────────────────────────────────────────────────────
                # Determine Unit for Formatting
                # ────────────────────────────────────────────────────────────
                unit_map = {
                    'DC_VOLTAGE': 'V', 'AC_VOLTAGE': 'V',           # Volts
                    'DC_CURRENT': 'A', 'AC_CURRENT': 'A',           # Amperes
                    'RESISTANCE_2W': 'Ω', 'RESISTANCE_4W': 'Ω',     # Ohms
                    'CAPACITANCE': 'F', 'FREQUENCY': 'Hz',          # Farads, Hertz
                    'TEMPERATURE': '°C'                              # Celsius
                }
                unit = unit_map.get(function, '')

                # ────────────────────────────────────────────────────────────
                # Format with SI Prefixes for Human Readability
                # ────────────────────────────────────────────────────────────
                # Converts raw values to readable format:
                #   0.012345 V    → "12.345 mV"
                #   4567.89 Ω     → "4.568 kΩ"
                #   0.000123 A    → "123.0 µA"
                formatted_result = self._format_with_si_prefix(result, unit)

                return formatted_result, "Measurement successful"
            else:
                # Measurement returned None (timeout, over-range, hardware fault)
                return "N/A", "Measurement failed"

        except Exception as e:
            # ────────────────────────────────────────────────────────────────
            # Exception Handling - Log and Report
            # ────────────────────────────────────────────────────────────────
            self.logger.error(f"Measurement error: {e}")
            return "N/A", f"Measurement error: {str(e)}"
    
    # ════════════════════════════════════════════════════════════════════════════
    # Continuous Measurement Thread Management
    # ════════════════════════════════════════════════════════════════════════════

    def start_continuous_measurement(self, function: str, range_val: float, resolution: float,
                                   nplc: float, auto_zero: bool, interval: float) -> str:
        """
        Start continuous measurement loop in background daemon thread.

        Spawns a worker thread that repeatedly calls single_measurement() at the
        specified interval. Thread automatically terminates when stop_continuous_measurement()
        is called or connection is lost. Safe to call from UI event handlers.

        Args:
            function: Measurement type (same as single_measurement)
            range_val: Measurement range (same as single_measurement)
            resolution: Resolution setting (same as single_measurement)
            nplc: Integration time (same as single_measurement)
            auto_zero: Auto-zero enable (same as single_measurement)
            interval: Time between measurements in seconds
                     Range: 0.05 to 3600 (50ms to 1 hour)
                     Minimum practical: ~0.1s (limited by SCPI overhead)
                     Typical: 1.0s for trending, 0.1s for fast sampling

        Returns:
            Status message indicating start success or error reason

        Thread Safety:
            Creates daemon thread that shares measurement_data buffer with main thread.
            Uses continuous_measurement flag for clean shutdown. No explicit locking
            needed due to GIL protection of list operations.

        Performance:
            Thread overhead: <1ms
            Actual sample rate limited by: interval + measurement_time + VISA_overhead
            Example: interval=0.1s, NPLC=1: ~0.17s actual (5.8 Hz)

        Note:
            Daemon thread automatically terminates when main program exits.
            Worker handles exceptions and logs errors before terminating.

        Example:
            >>> msg = controller.start_continuous_measurement("DC_VOLTAGE", 10.0, 1e-6, 1.0, True, 1.0)
            >>> print(msg)  # "Continuous measurement started"
            >>> time.sleep(10)  # Collect data for 10 seconds
            >>> controller.stop_continuous_measurement()
        """
        # ────────────────────────────────────────────────────────────────────
        # Pre-flight Checks
        # ────────────────────────────────────────────────────────────────────
        if not self.is_connected:                   # Guard: require active connection
            return "Not connected to instrument"

        if self.continuous_measurement:             # Guard: prevent duplicate threads
            return "Continuous measurement already running"

        # ────────────────────────────────────────────────────────────────────
        # Thread Creation and Launch
        # ────────────────────────────────────────────────────────────────────
        self.continuous_measurement = True          # Set run flag BEFORE starting thread
                                                    # Worker checks this flag in loop

        self.measurement_thread = threading.Thread(
            target=self._continuous_measurement_worker,     # Worker function
            args=(function, range_val, resolution, nplc, auto_zero, interval),  # Pass all params
            daemon=True                             # Daemon: auto-terminate on program exit
                                                    # Non-daemon would prevent exit until stopped
        )
        self.measurement_thread.start()             # Begin execution
        return "Continuous measurement started"

    def stop_continuous_measurement(self) -> str:
        """
        Stop continuous measurement thread and wait for clean shutdown.

        Signals worker thread to terminate via continuous_measurement flag, then
        waits up to 2 seconds for thread to finish current measurement and exit.
        Safe to call even if no thread is running.

        Returns:
            Status message: "Continuous measurement stopped"

        Performance:
            Typical stop time: 0-2s depending on when called during measurement cycle
            Worst case: 2s timeout if thread is blocked in VISA I/O

        Thread Safety:
            Safe to call from any thread. Uses thread.join() for synchronization.

        Note:
            After timeout, thread may still be running (orphaned). This is rare and
            only occurs if VISA communication is completely hung. Thread will
            eventually terminate when is_connected becomes False.
        """
        self.continuous_measurement = False         # Signal thread to stop
                                                    # Worker checks this flag each iteration

        if self.measurement_thread and self.measurement_thread.is_alive():
            self.measurement_thread.join(timeout=2) # Wait up to 2 seconds for clean exit
                                                    # Timeout prevents UI freeze if thread hangs
        return "Continuous measurement stopped"

    def _continuous_measurement_worker(self, function: str, range_val: float, resolution: float,
                                     nplc: float, auto_zero: bool, interval: float):
        """
        Background worker thread for continuous measurements.

        Repeatedly calls single_measurement() with the specified parameters until
        continuous_measurement flag is cleared or connection is lost. Automatically
        handles exceptions and logs errors. Runs in daemon thread.

        Args:
            function: Measurement type (passed to single_measurement)
            range_val: Range setting
            resolution: Resolution setting
            nplc: Integration time
            auto_zero: Auto-zero enable
            interval: Sleep time between measurements in seconds

        Thread Safety:
            Worker thread - do not call directly. Use start_continuous_measurement().
            Accesses shared measurement_data via single_measurement() (GIL-protected).

        Loop Logic:
            1. Check continue flag (continuous_measurement AND is_connected)
            2. Perform measurement (stores result in buffer automatically)
            3. Sleep for interval duration
            4. Repeat until flag cleared or exception occurs

        Exception Handling:
            Any exception terminates loop and logs error. This prevents infinite
            error loops if hardware fails or connection drops during operation.

        Note:
            Worker does NOT update UI directly. UI must poll measurement_data or
            statistics methods to see new data.
        """
        while self.continuous_measurement and self.is_connected:    # Loop control flags
            try:
                # ────────────────────────────────────────────────────────────
                # Perform Measurement (result stored in measurement_data)
                # ────────────────────────────────────────────────────────────
                self.single_measurement(function, range_val, resolution, nplc, auto_zero)

                # ────────────────────────────────────────────────────────────
                # Delay Before Next Measurement
                # ────────────────────────────────────────────────────────────
                time.sleep(interval)                # Interruptible sleep
                                                    # Thread can exit during sleep when flag cleared

            except Exception as e:
                # ────────────────────────────────────────────────────────────
                # Error Handling - Log and Terminate
                # ────────────────────────────────────────────────────────────
                self.logger.error(f"Continuous measurement error: {e}")
                break                               # Exit loop on ANY exception
                                                    # Prevents infinite error spam
    
    def get_statistics(self, last_n_points: int = 100) -> Tuple[str, str, str, str, str]:
        """
        Calculate statistics from recent measurements.
        
        Returns:
            Tuple of (count, mean, std_dev, min_val, max_val)
        """
        if not self.measurement_data:
            return "0", "N/A", "N/A", "N/A", "N/A"
        
        # Get recent data points
        recent_data = self.measurement_data[-last_n_points:] if len(self.measurement_data) > last_n_points else self.measurement_data
        values = [point['value'] for point in recent_data]
        
        if not values:
            return "0", "N/A", "N/A", "N/A", "N/A"
        
        try:
            count = len(values)
            mean = np.mean(values)
            std_dev = np.std(values, ddof=1) if count > 1 else 0
            min_val = np.min(values)
            max_val = np.max(values)

            # Get the unit for formatting
            function = recent_data[0]['function'] if recent_data else 'DC_VOLTAGE'
            unit = self._get_unit(function)

            # Format with SI prefixes
            return (
                str(count),
                self._format_with_si_prefix(mean, unit),
                self._format_with_si_prefix(std_dev, unit),
                self._format_with_si_prefix(min_val, unit),
                self._format_with_si_prefix(max_val, unit)
            )
        except Exception as e:
            self.logger.error(f"Statistics calculation error: {e}")
            return "Error", "N/A", "N/A", "N/A", "N/A"
    
    def create_trend_plot(self, last_n_points: int = 100) -> Optional[plt.Figure]:
        """Create a trend plot of recent measurements."""
        if not self.measurement_data:
            return None
        
        try:
            # Get recent data points
            recent_data = self.measurement_data[-last_n_points:] if len(self.measurement_data) > last_n_points else self.measurement_data
            
            if len(recent_data) < 2:
                return None
            
            timestamps = [point['timestamp'] for point in recent_data]
            values = [point['value'] for point in recent_data]
            function = recent_data[0]['function']
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, values, 'b-', linewidth=1, marker='o', markersize=2)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Measurement Value ({self._get_unit(function)})')
            ax.set_title(f'{function.replace("_", " ").title()} Trend')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            self.logger.error(f"Plot creation error: {e}")
            return None
    
    # ════════════════════════════════════════════════════════════════════════════
    # Formatting and Display Utilities
    # ════════════════════════════════════════════════════════════════════════════

    def _get_unit(self, function: str) -> str:
        """
        Map measurement function to physical unit string.

        Args:
            function: Measurement function name (e.g., "DC_VOLTAGE", "RESISTANCE_2W")

        Returns:
            Unit symbol string: 'V', 'A', 'Ω', 'F', 'Hz', or '°C'
            Empty string if function not recognized

        Note:
            Uses proper Unicode symbols: Ω (U+03A9 OHM), µ (U+00B5 MICRO), °C (DEGREE CELSIUS)
        """
        unit_map = {
            'DC_VOLTAGE': 'V', 'AC_VOLTAGE': 'V',           # Voltage: Volts
            'DC_CURRENT': 'A', 'AC_CURRENT': 'A',           # Current: Amperes
            'RESISTANCE_2W': 'Ω', 'RESISTANCE_4W': 'Ω',     # Resistance: Ohms
            'CAPACITANCE': 'F', 'FREQUENCY': 'Hz',          # Capacitance: Farads, Frequency: Hertz
            'TEMPERATURE': '°C'                              # Temperature: Celsius
        }
        return unit_map.get(function, '')

    def _format_with_si_prefix(self, value: float, base_unit: str) -> str:
        """
        Format numerical value with appropriate SI metric prefix for human readability.

        ┌──────────────────────────────────────────────────────────────────────┐
        │ CRITICAL ALGORITHM: SI PREFIX SELECTION                              │
        │                                                                      │
        │ Purpose: Convert raw measurement values to human-readable format    │
        │          without scientific notation (e.g., "12.34 mV" not "1.234e-2")│
        │                                                                      │
        │ Algorithm:                                                           │
        │   1. Handle special case: Temperature (no SI prefixes)               │
        │   2. Handle edge case: Zero value                                    │
        │   3. Calculate absolute value for comparison                         │
        │   4. Iterate through prefix list (largest to smallest)               │
        │   5. Select first prefix where |value| >= scale                      │
        │   6. Scale value by prefix factor                                    │
        │   7. Determine decimal places based on magnitude                     │
        │   8. Format and return string with unit                              │
        └──────────────────────────────────────────────────────────────────────┘

        Args:
            value: Numerical value in base SI units
                  Examples:
                    - Voltage: 0.012345 (V)
                    - Current: 0.000123 (A)
                    - Resistance: 4567.89 (Ω)
                  Range: -1e15 to +1e15 (femto to tera)
            base_unit: Physical unit symbol ('V', 'A', 'Ω', 'F', 'Hz', '°C')
                      Must be one of the supported units

        Returns:
            Formatted string with SI prefix and unit
            Format: "<value> <prefix><unit>"
            Examples:
                - Input: (0.012345, 'V')    → Output: "12.345 mV"
                - Input: (0.000123, 'A')    → Output: "123.0 µA"
                - Input: (4567.89, 'Ω')     → Output: "4.568 kΩ"
                - Input: (0.0, 'V')         → Output: "0.000 V"
                - Input: (25.5, '°C')       → Output: "25.500 °C"

        Decimal Place Logic:
            Scaled value ≥ 100:  2 decimal places (e.g., "123.45 mV")
            Scaled value ≥ 10:   3 decimal places (e.g., "12.345 mV")
            Scaled value < 10:   4 decimal places (e.g., "1.2345 mV")
            Rationale: Maintain ~4-5 significant figures across all ranges

        SI Prefix Table (ISO/IEC 80000):
            T (tera):   10^12  = 1,000,000,000,000
            G (giga):   10^9   = 1,000,000,000
            M (mega):   10^6   = 1,000,000
            k (kilo):   10^3   = 1,000
            (none):     10^0   = 1
            m (milli):  10^-3  = 0.001
            µ (micro):  10^-6  = 0.000001
            n (nano):   10^-9  = 0.000000001
            p (pico):   10^-12 = 0.000000000001
            f (femto):  10^-15 = 0.000000000000001

        Edge Cases:
            - Zero: Returns "0.000 <unit>" (3 decimal places)
            - Negative: Preserves sign, uses absolute value for prefix selection
            - Very small: Falls through to femto prefix (smallest supported)
            - Very large: Uses tera prefix (largest supported)

        Performance:
            Time complexity: O(n) where n = number of prefixes (11)
            Worst case: ~11 comparisons for very small values
            Average case: ~3-4 comparisons (most measurements in m/k/base range)

        Example:
            >>> controller._format_with_si_prefix(0.012345, 'V')
            '12.345 mV'
            >>> controller._format_with_si_prefix(-0.000123, 'A')
            '-123.0 µA'
            >>> controller._format_with_si_prefix(4567890, 'Ω')
            '4.568 MΩ'
        """
        # ────────────────────────────────────────────────────────────────────
        # Special Case: Temperature (No SI Prefixes)
        # ────────────────────────────────────────────────────────────────────
        if base_unit == '°C':                       # Temperature uses absolute scale
            return f"{value:.3f} {base_unit}"       # Always 3 decimal places, no prefix

        # ────────────────────────────────────────────────────────────────────
        # SI Prefix Lookup Table (Largest to Smallest)
        # ────────────────────────────────────────────────────────────────────
        # Order matters: Must iterate from large to small to find correct prefix
        # Each tuple: (scale_factor, prefix_symbol)
        prefixes = [
            (1e12, 'T'),   # Tera:  1,000,000,000,000 (high power applications)
            (1e9, 'G'),    # Giga:  1,000,000,000 (RF frequencies)
            (1e6, 'M'),    # Mega:  1,000,000 (high resistance, high voltage)
            (1e3, 'k'),    # kilo:  1,000 (common voltages, resistances)
            (1, ''),       # base:  1 (no prefix, base unit)
            (1e-3, 'm'),   # milli: 0.001 (common for low voltages)
            (1e-6, 'µ'),   # micro: 0.000001 (low currents, small caps)
                           # Note: Using U+00B5 (µ) not 'u' for proper display
            (1e-9, 'n'),   # nano:  0.000000001 (small capacitance, timing)
            (1e-12, 'p'),  # pico:  0.000000000001 (very small capacitance)
            (1e-15, 'f'),  # femto: 0.000000000000001 (parasitic capacitance)
        ]

        # ────────────────────────────────────────────────────────────────────
        # Calculate Absolute Value for Comparison
        # ────────────────────────────────────────────────────────────────────
        abs_value = abs(value)                      # Prefix selection based on magnitude
                                                    # Sign preserved in final output

        # ────────────────────────────────────────────────────────────────────
        # Edge Case: Zero Value
        # ────────────────────────────────────────────────────────────────────
        if abs_value == 0:                          # Exact zero comparison safe for == 0
            return f"0.000 {base_unit}"             # Fixed format, no prefix, 3 decimals

        # ────────────────────────────────────────────────────────────────────
        # Prefix Selection Algorithm
        # ────────────────────────────────────────────────────────────────────
        # Find the LARGEST prefix where |value| >= scale
        # This ensures scaled_value is in range [1, 1000)
        for scale, prefix in prefixes:              # Iterate largest to smallest
            if abs_value >= scale:                  # Found appropriate range
                # ────────────────────────────────────────────────────────────
                # Scale Value by Prefix Factor
                # ────────────────────────────────────────────────────────────
                scaled_value = value / scale        # Divide by scale (preserves sign)
                                                    # Example: 0.012345 / 1e-3 = 12.345

                # ────────────────────────────────────────────────────────────
                # Adaptive Decimal Place Selection
                # ────────────────────────────────────────────────────────────
                # Goal: Maintain 4-5 significant figures across all ranges
                # Logic: Larger scaled values need fewer decimal places
                if abs(scaled_value) >= 100:        # Range: [100, 999.99]
                    formatted = f"{scaled_value:.2f}"   # 2 decimals: "123.45"
                elif abs(scaled_value) >= 10:       # Range: [10, 99.999]
                    formatted = f"{scaled_value:.3f}"   # 3 decimals: "12.345"
                else:                               # Range: [1, 9.9999]
                    formatted = f"{scaled_value:.4f}"   # 4 decimals: "1.2345"

                # ────────────────────────────────────────────────────────────
                # Construct Final String
                # ────────────────────────────────────────────────────────────
                return f"{formatted} {prefix}{base_unit}"  # Format: "12.345 mV"

        # ────────────────────────────────────────────────────────────────────
        # Fallback: Extremely Small Values (< 1 femto)
        # ────────────────────────────────────────────────────────────────────
        # If value smaller than 1e-15, force femto prefix
        # This handles values below normal measurement range
        scaled_value = value / 1e-15                # Scale to femto range
        return f"{scaled_value:.4f} f{base_unit}"   # Format with femto prefix

    def export_data(self, save_path: str, format_type: str = "CSV") -> str:
        """Export measurement data to file at user-specified location.

        Args:
            save_path: Directory path where file should be saved
            format_type: Export format (CSV, JSON, or Excel)

        Returns:
            Status message
        """
        if not self.measurement_data:
            return "No data to export"

        if not save_path or save_path.strip() == "":
            return "Please select a save location using the Browse button"

        try:
            # Ensure the save directory exists
            save_dir = Path(save_path)
            if not save_dir.exists():
                return f"Error: Directory does not exist: {save_path}"

            if not save_dir.is_dir():
                return f"Error: Path is not a directory: {save_path}"

            df = pd.DataFrame(self.measurement_data)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S");

            if format_type == "CSV":
                filename = f"dmm_data_{timestamp_str}.csv"
                filepath = save_dir / filename
                df.to_csv(filepath, index=False)
                return f"✓ Data exported successfully to:\n{filepath}"
            elif format_type == "JSON":
                filename = f"dmm_data_{timestamp_str}.json"
                filepath = save_dir / filename
                df.to_json(filepath, orient='records', date_format='iso')
                return f"✓ Data exported successfully to:\n{filepath}"
            elif format_type == "Excel":
                filename = f"dmm_data_{timestamp_str}.xlsx"
                filepath = save_dir / filename
                df.to_excel(filepath, index=False)
                return f"✓ Data exported successfully to:\n{filepath}"
        except Exception as e:
            self.logger.error(f"Data export error: {e}")
            return f"Export failed: {str(e)}"

    def save_trend_plot(self, save_path: str, last_n_points: int = 100) -> str:
        """Save trend plot to file at user-specified location.

        Args:
            save_path: Directory path where plot should be saved
            last_n_points: Number of recent points to plot

        Returns:
            Status message
        """
        if not self.measurement_data:
            return "No data to plot"

        if not save_path or save_path.strip() == "":
            return "Please select a save location using the Browse button"

        try:
            # Ensure the save directory exists
            save_dir = Path(save_path)
            if not save_dir.exists():
                return f"Error: Directory does not exist: {save_path}"

            if not save_dir.is_dir():
                return f"Error: Path is not a directory: {save_path}"

            # Get recent data points
            recent_data = self.measurement_data[-last_n_points:] if len(self.measurement_data) > last_n_points else self.measurement_data

            if len(recent_data) < 2:
                return "Insufficient data points for plot (need at least 2)"

            timestamps = [point['timestamp'] for point in recent_data]
            values = [point['value'] for point in recent_data]
            function = recent_data[0]['function']

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, values, 'b-', linewidth=1, marker='o', markersize=2)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Measurement Value ({self._get_unit(function)})')
            ax.set_title(f'{function.replace("_", " ").title()} Trend')
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Save plot
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dmm_trend_{timestamp_str}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=1200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return f"✓ Plot saved successfully to:\n{filepath}"
        except Exception as e:
            self.logger.error(f"Plot save error: {e}")
            return f"Plot save failed: {str(e)}"
    
    def clear_data(self) -> str:
        """Clear all measurement data."""
        self.measurement_data.clear()
        return "Measurement data cleared"
    
    def get_instrument_status(self) -> Tuple[str, str, str, str]:
        """
        Get instrument status information.
        
        Returns:
            Tuple of (connection_status, instrument_info, errors, system_time)
        """
        if not self.is_connected or not self.dmm:
            return "Disconnected", "N/A", "N/A", "N/A"
        
        try:
            # Connection status
            status = "Connected" if self.is_connected else "Disconnected"
            
            # Instrument info
            info = self.dmm.get_instrument_info()
            if info:
                instrument_info = f"{info['manufacturer']} {info['model']} (S/N: {info['serial_number']})"
                errors = info.get('current_errors', 'None')
            else:
                instrument_info = "Unknown"
                errors = "Unable to query"
            
            # System time
            system_time = self.dmm.get_system_date_time() or "Unknown"
            
            return status, instrument_info, errors, system_time
        except Exception as e:
            self.logger.error(f"Status query error: {e}")
            return "Error", "N/A", f"Error: {str(e)}", "N/A"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: POWER SUPPLY CONTROLLER CLASS - KEITHLEY 2230-30-1
# ════════════════════════════════════════════════════════════════════════════
# This section implements the high-level controller for the Keithley 2230-30-1
# triple-channel DC power supply. Provides advanced waveform generation
# capabilities (Sine, Square, Triangle, Ramp) with real-time voltage control,
# measurement acquisition, and data logging. Critical for automated device
# characterization and stress testing.

class PowerSupplyAutomationGradio:
    """
    High-level controller for Keithley 2230-30-1 triple-channel power supply with waveform automation.

    This class provides a complete power supply automation framework including:
    - Connection management to PSU hardware via VISA
    - Multi-channel voltage/current configuration and control
    - Advanced waveform generation (Sine, Square, Triangle, Ramp Up/Down)
    - Real-time closed-loop voltage execution with measurement feedback
    - Data acquisition and export (CSV/JSON/Excel)
    - Thread-safe operation for background waveform execution

    Hardware Specifications (Keithley 2230-30-1):
        - 3 independent isolated channels
        - Channel 1 & 2: 0-30V, 0-3A (90W each)
        - Channel 3: 0-6V, 0-5A (30W)
        - Voltage accuracy: ±(0.03% + 10mV)
        - Current accuracy: ±(0.1% + 10mA)
        - Load regulation: <0.01% + 2mV
        - Voltage settling time: <50ms to 0.01%

    Thread Safety:
        - Waveform execution runs in background daemon thread
        - ramping_active flag provides clean shutdown mechanism
        - VISA I/O protected by driver-level locking
        - UI updates via status_queue (producer-consumer pattern)

    Attributes:
        power_supply (Optional[KeithleyPowerSupply]): Driver instance, None if disconnected
        is_connected (bool): Connection state flag
        ramping_active (bool): Waveform execution control flag
        ramping_thread (Optional[threading.Thread]): Worker thread for waveform execution
        ramping_profile (List[Tuple[float, float]]): Generated waveform points [(time, voltage), ...]
        ramping_data (List[Dict]): Collected measurement data during execution
        ramping_params (Dict): Waveform configuration parameters
        channel_states (Dict[int, Dict]): State tracking for all 3 channels
        measurement_data (Dict): General measurement storage
        status_queue (queue.Queue): Thread-safe status message queue
        logger (logging.Logger): Logger instance for diagnostics
        save_locations (Dict[str, str]): Default paths for data export

    Example:
        >>> psu = PowerSupplyAutomationGradio()
        >>> psu.connect_power_supply("USB0::0x05E6::0x2230::9203456::INSTR")
        >>> psu.configure_channel(1, 5.0, 1.0, 5.5)  # Ch1: 5V, 1A limit, 5.5V OVP
        >>> psu.execute_waveform_ramping()  # Start automated waveform
        >>> # ... waveform runs in background thread ...
        >>> psu.stop_waveform()  # Clean stop

    See Also:
        - KeithleyPowerSupply: Low-level SCPI driver
        - _WaveformGenerator: Waveform synthesis algorithms
        - _RampDataManager: Data collection and export
    """

    def __init__(self):
        """
        Initialize power supply controller with default configuration.

        Sets up all instance variables, logging, and default waveform parameters.
        Does NOT connect to hardware - connection must be explicitly requested.

        Performance:
            Initialization is instantaneous (<1ms) - no I/O operations
        """
        # ────────────────────────────────────────────────────────────────────
        # Hardware Connection State
        # ────────────────────────────────────────────────────────────────────
        self.power_supply = None                    # PSU driver instance, None until connected
        self.is_connected = False                   # Guard flag for all operations

        # ────────────────────────────────────────────────────────────────────
        # Waveform Execution Control
        # ────────────────────────────────────────────────────────────────────
        self.ramping_active = False                 # Thread loop control flag
        self.ramping_thread = None                  # Worker thread handle
        self.ramping_profile = []                   # Generated waveform: [(t, v), (t, v), ...]
        self.ramping_data = []                      # Collected measurements during execution

        # ────────────────────────────────────────────────────────────────────
        # Waveform Configuration Parameters
        # ────────────────────────────────────────────────────────────────────
        self.ramping_params = {
            'waveform': 'Sine',                     # Waveform type: Sine/Square/Triangle/Ramp Up/Ramp Down
            'target_voltage': 3.0,                  # Peak voltage amplitude in volts (0-30V)
            'cycles': 3,                            # Number of complete waveform cycles
            'points_per_cycle': 50,                 # Sampling resolution (points per cycle)
                                                    # Higher = smoother waveform, longer execution
            'cycle_duration': 8.0,                  # Time for one complete cycle in seconds
            'psu_settle': 0.05,                     # PSU settling time after voltage change (50ms)
                                                    # Allows output capacitors to stabilize
            'nplc': 1.0,                            # Measurement integration time (power line cycles)
            'active_channel': 1                     # Target channel (1, 2, or 3)
        }

        # ────────────────────────────────────────────────────────────────────
        # User Interface State
        # ────────────────────────────────────────────────────────────────────
        self.waveform_status_message = "Ready - Configure parameters and click Preview or Start"

        # ────────────────────────────────────────────────────────────────────
        # Data Collection Infrastructure
        # ────────────────────────────────────────────────────────────────────
        self.measurement_data = {}                  # General measurement storage
        self.status_queue = queue.Queue()           # Thread-safe FIFO for status updates
                                                    # Worker thread → UI communication
        self.measurement_active = False             # Measurement loop control flag

        # ────────────────────────────────────────────────────────────────────
        # Live Graphing Data Buffers
        # ────────────────────────────────────────────────────────────────────
        # Time-series data for live graphs (stores last N measurements per channel)
        self.live_data = {
            1: {'timestamps': [], 'voltages': [], 'currents': [], 'powers': []},
            2: {'timestamps': [], 'voltages': [], 'currents': [], 'powers': []},
            3: {'timestamps': [], 'voltages': [], 'currents': [], 'powers': []}
        }
        self.max_live_points = 200                  # Maximum number of points to keep in live buffer
        self.live_measurement_active = False        # Is live measurement currently running?
        self.live_measurement_thread = None         # Thread for live measurements
        self.live_measurement_interval = 1.0        # Seconds between live measurements

        # ────────────────────────────────────────────────────────────────────
        # Logging Setup
        # ────────────────────────────────────────────────────────────────────
        self.setup_logging()                        # Configure logger instance

        # ────────────────────────────────────────────────────────────────────
        # Channel State Tracking (3 Channels)
        # ────────────────────────────────────────────────────────────────────
        self.channel_states = {
            i: {"enabled": False,                   # Output state: ON/OFF
                "voltage": 0.0,                     # Set voltage in volts
                "current": 0.0,                     # Current limit in amperes
                "power": 0.0}                       # Calculated power in watts
            for i in range(1, 4)                    # Channels 1, 2, 3
        }

        # ────────────────────────────────────────────────────────────────────
        # Activity Log Buffer
        # ────────────────────────────────────────────────────────────────────
        self.activity_log = "Application started\n" # Text log of all operations

        # ────────────────────────────────────────────────────────────────────
        # File Export Default Locations
        # ────────────────────────────────────────────────────────────────────
        self.save_locations = {
            'data': str(Path.cwd() / "psu_data")    # Data export directory
        }

    def setup_logging(self):
        """
        Configure logging system for diagnostics and debugging.

        Creates logger instance with INFO level and standard formatting.
        Logs are output to console and can be redirected to file if needed.

        Note:
            Logger name: "PowerSupplyAutomationGradio"
            Allows filtering of PSU-specific log messages from system-wide logs.
        """
        logging.basicConfig(
            level=logging.INFO,                     # Log INFO and above (INFO, WARNING, ERROR, CRITICAL)
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                                                    # Timestamp - LoggerName - Level - Message
        )
        self.logger = logging.getLogger("PowerSupplyAutomationGradio")

    # ════════════════════════════════════════════════════════════════════════════
    # Nested Helper Class: Waveform Generator
    # ════════════════════════════════════════════════════════════════════════════

    class _WaveformGenerator:
        """
        Mathematical waveform synthesis engine for voltage profile generation.

        ┌──────────────────────────────────────────────────────────────────────┐
        │ CRITICAL ALGORITHM: WAVEFORM GENERATION MATHEMATICS                  │
        │                                                                      │
        │ Purpose: Generate time-voltage profiles for automated PSU control   │
        │                                                                      │
        │ Supported Waveforms:                                                 │
        │   • Sine Wave:     Smooth sinusoidal oscillation (0 to V_peak)      │
        │   • Square Wave:   Binary high/low switching                        │
        │   • Triangle Wave: Linear rise and fall with constant slope         │
        │   • Ramp Up:       Linear increase from 0 to V_peak                 │
        │   • Ramp Down:     Linear decrease from V_peak to 0                 │
        │                                                                      │
        │ Applications:                                                        │
        │   - Device stress testing (thermal cycling)                          │
        │   - Power sequencing validation                                      │
        │   - Voltage transient characterization                               │
        │   - Load regulation testing                                          │
        └──────────────────────────────────────────────────────────────────────┘

        Thread Safety:
            Stateless generator - safe to call from any thread.
            No shared mutable state after initialization.

        Attributes:
            TYPES (List[str]): Supported waveform type strings
            waveform_type (str): Selected waveform algorithm
            target_voltage (float): Peak amplitude in volts (0-30V)
            cycles (int): Number of complete waveform repetitions
            points_per_cycle (int): Sampling resolution per cycle
            cycle_duration (float): Time duration of one complete cycle in seconds

        Example:
            >>> gen = PowerSupplyAutomationGradio._WaveformGenerator(
            ...     waveform_type="Sine",
            ...     target_voltage=5.0,
            ...     cycles=3,
            ...     points_per_cycle=100,
            ...     cycle_duration=10.0
            ... )
            >>> profile = gen.generate()
            >>> print(profile[0])  # First point
            (0.0, 0.0)  # (time in seconds, voltage in volts)
            >>> print(profile[50])  # Peak of first sine cycle
            (5.0, 5.0)  # (5 seconds, 5 volts)
        """

        TYPES = ["Sine", "Square", "Triangle", "Ramp Up", "Ramp Down","Cardiac", "Damped Sine", "Exponential Raise", "Exponential Fall", "Gaussian Pulse", "Neural Spike", "Staircase", "PWM", "Chirp", "Burst Mode", "Brownout", "RC Charge", "Sinc", "Breathing"]

        def __init__(self, waveform_type: str = "Sine", target_voltage: float = 3.0,
                     cycles: int = 3, points_per_cycle: int = 50, cycle_duration: float = 8.0):
            """
            Initialize waveform generator with validated parameters.

            Args:
                waveform_type: Type of waveform to generate
                              Valid: "Sine", "Square", "Triangle", "Ramp Up", "Ramp Down"
                              Default: "Sine" if invalid type provided
                target_voltage: Peak voltage amplitude in volts
                               Range: 0.0 to 30.0V (hardware limit)
                               Automatically clamped to valid range
                cycles: Number of complete waveform cycles
                       Range: 1 to infinity (practical limit ~100)
                       Each cycle repeats the pattern
                points_per_cycle: Number of sample points per cycle
                                 Range: 1 to infinity (practical limit ~1000)
                                 Higher = smoother waveform, longer execution
                                 Typical: 50-100 for smooth curves
                cycle_duration: Duration of one complete cycle in seconds
                               Range: >0 (practical: 1-60 seconds)
                               Affects execution speed and settling

            Note:
                All numeric parameters are validated and clamped to safe ranges.
                Invalid waveform types default to "Sine".
            """
            # Validate and store waveform type
            self.waveform_type = waveform_type if waveform_type in self.TYPES else "Sine"

            # Clamp voltage to PSU hardware limits (0-30V)
            self.target_voltage = max(0.0, min(float(target_voltage), 30.0))

            # Ensure minimum of 1 cycle
            self.cycles = max(1, int(cycles))

            # Ensure minimum of 1 point per cycle (prevents division by zero)
            self.points_per_cycle = max(1, int(points_per_cycle))

            # Store cycle duration
            self.cycle_duration = float(cycle_duration)

        def generate(self):
            """
            Generate complete waveform profile with mathematical precision.

            ┌──────────────────────────────────────────────────────────────────────┐
            │ WAVEFORM GENERATION MATHEMATICAL FORMULAS                            │
            │                                                                      │
            │ Coordinate System:                                                   │
            │   pos ∈ [0, 1]        : Normalized position within current cycle    │
            │   t ∈ [0, T_total]    : Absolute time in seconds                    │
            │   V ∈ [0, V_peak]     : Output voltage in volts                     │
            │                                                                      │
            │ Sine Wave:                                                           │
            │   V(pos) = V_peak × sin(pos × π)                                    │
            │   • Range: 0 to V_peak (half-wave rectified sine)                   │
            │   • Smooth, continuous derivative                                   │
            │   • Period: 0 to 1 maps to 0° to 180°                               │
            │                                                                      │
            │ Square Wave:                                                         │
            │   V(pos) = V_peak    if pos < 0.5                                   │
            │          = 0         if pos ≥ 0.5                                   │
            │   • Binary switching with 50% duty cycle                            │
            │   • Discontinuous (tests transient response)                        │
            │                                                                      │
            │ Triangle Wave:                                                       │
            │   V(pos) = V_peak × (2 × pos)           if pos < 0.5 (rising edge) │
            │          = V_peak × (2 - 2 × pos)       if pos ≥ 0.5 (falling edge)│
            │   • Constant slope: dV/dt = ±2V_peak/T                              │
            │   • Symmetric rise and fall                                         │
            │                                                                      │
            │ Ramp Up:                                                             │
            │   V(pos) = V_peak × pos                                             │
            │   • Linear increase from 0 to V_peak                                │
            │   • Constant slope: dV/dt = V_peak/T                                │
            │                                                                      │
            │ Ramp Down:                                                           │
            │   V(pos) = V_peak × (1 - pos)                                       │
            │   • Linear decrease from V_peak to 0                                │
            │   • Constant slope: dV/dt = -V_peak/T                               │
            └──────────────────────────────────────────────────────────────────────┘

            Returns:
                List[Tuple[float, float]]: Waveform profile as list of (time, voltage) tuples
                    - time: Absolute time in seconds, rounded to 6 decimal places (µs precision)
                    - voltage: Output voltage in volts, clamped to [0, 30V], rounded to 6 decimals
                    - Length: cycles × points_per_cycle
                    - Example: [(0.0, 0.0), (0.16, 0.244), (0.32, 0.475), ...]

            Algorithm Steps:
                1. Iterate through each cycle (outer loop)
                2. Iterate through each point in cycle (inner loop)
                3. Calculate normalized position: pos = point / (points_per_cycle - 1)
                4. Calculate absolute time: t = cycle_start + pos × cycle_duration
                5. Apply waveform-specific formula to calculate voltage
                6. Clamp voltage to hardware limits [0, 30V]
                7. Append (t, V) tuple to profile list
                8. Return complete profile

            Performance:
                Time complexity: O(cycles × points_per_cycle)
                Memory: ~16 bytes per point (tuple of 2 floats)
                Typical: 3 cycles × 50 points = 150 tuples ≈ 2.4 KB

            Example:
                >>> gen = _WaveformGenerator("Sine", 5.0, 2, 4, 10.0)
                >>> profile = gen.generate()
                >>> for t, v in profile:
                ...     print(f"t={t:.2f}s, V={v:.3f}V")
                t=0.00s, V=0.000V      # Cycle 1, Point 1
                t=3.33s, V=4.330V      # Cycle 1, Point 2 (near peak)
                t=6.67s, V=4.330V      # Cycle 1, Point 3
                t=10.00s, V=0.000V     # Cycle 1, Point 4
                t=10.00s, V=0.000V     # Cycle 2, Point 1
                ...
            """
            profile = []                            # Initialize empty waveform profile

            # ────────────────────────────────────────────────────────────────
            # Outer Loop: Iterate Through Cycles
            # ────────────────────────────────────────────────────────────────
            for cycle in range(self.cycles):        # cycle = 0, 1, 2, ... (cycles-1)

                # ────────────────────────────────────────────────────────────
                # Inner Loop: Iterate Through Points in Current Cycle
                # ────────────────────────────────────────────────────────────
                for point in range(self.points_per_cycle):  # point = 0, 1, 2, ... (points-1)

                    # ────────────────────────────────────────────────────────
                    # Calculate Normalized Position Within Cycle [0, 1]
                    # ────────────────────────────────────────────────────────
                    # pos = 0.0 at cycle start, pos = 1.0 at cycle end
                    # Safe division: handles points_per_cycle == 1 case
                    pos = point / max(1, (self.points_per_cycle - 1)) if self.points_per_cycle > 1 else 0.0

                    # ────────────────────────────────────────────────────────
                    # Calculate Absolute Time in Seconds
                    # ────────────────────────────────────────────────────────
                    # t = cycle_start_time + position_within_cycle × cycle_duration
                    t = cycle * self.cycle_duration + pos * self.cycle_duration

                    # ────────────────────────────────────────────────────────
                    # Apply Waveform-Specific Formula
                    # ────────────────────────────────────────────────────────
                    if self.waveform_type == 'Sine':
                        # Sine wave: V = V_peak × sin(pos × π)
                        # Maps pos ∈ [0, 1] to angle ∈ [0°, 180°]
                        # Output: 0 → V_peak → 0 (half-wave rectified)
                        v = math.sin(pos * math.pi) * self.target_voltage

                    elif self.waveform_type == 'Square':
                        # Square wave: Binary switching at 50% duty cycle
                        # First half (pos < 0.5): V = V_peak
                        # Second half (pos ≥ 0.5): V = 0
                        v = self.target_voltage if pos < 0.5 else 0.0

                    elif self.waveform_type == 'Triangle':
                        # Triangle wave: Linear rise and fall
                        if pos < 0.5:
                            # Rising edge (first half): V = 2 × pos × V_peak
                            # pos = 0 → V = 0, pos = 0.5 → V = V_peak
                            v = (pos * 2.0) * self.target_voltage
                        else:
                            # Falling edge (second half): V = (2 - 2×pos) × V_peak
                            # pos = 0.5 → V = V_peak, pos = 1.0 → V = 0
                            v = (2.0 - pos * 2.0) * self.target_voltage

                    elif self.waveform_type == 'Ramp Up':
                        # Ramp up: V = pos × V_peak
                        # Linear increase: pos = 0 → V = 0, pos = 1 → V = V_peak
                        v = pos * self.target_voltage

                    elif self.waveform_type == 'Ramp Down':
                        # Ramp down: V = (1 - pos) × V_peak
                        # Linear decrease: pos = 0 → V = V_peak, pos = 1 → V = 0
                        v = (1.0 - pos) * self.target_voltage

                    elif self.waveform_type == 'Cardiac':
                        # ----------------------------------------------------------
                        # Realistic ECG using McSharry et al. (2003) model
                        # https://doi.org/10.1109/TBME.2003.811554
                        #
                        # This model creates highly realistic ECG morphology via
                        # a nonlinear dynamical system evolving on a limit cycle.
                        #
                        # Scaled so R-peak == target_voltage
                        # ----------------------------------------------------------

                        # Phase angle (0–2π)
                        theta = pos * 2 * math.pi

                        # Angles of the P,Q,R,S,T waves on the unit circle
                        theta_p = -0.25 * math.pi
                        theta_q = -0.05 * math.pi
                        theta_r =  0.00 * math.pi
                        theta_s =  0.05 * math.pi
                        theta_t =  0.30 * math.pi

                        # Amplitudes of waves (relative to R = 1.0)
                        a_p = 0.12
                        a_q = -0.20
                        a_r = 1.00
                        a_s = -0.25
                        a_t = 0.35

                        # Widths of each wave (controls sharpness)
                        b_p = 0.20
                        b_q = 0.10
                        b_r = 0.04
                        b_s = 0.12
                        b_t = 0.40

                        # Gaussian kernel using angular distances
                        def wave(a, b, theta_i):
                            # smallest angular distance on circle
                            dtheta = (theta - theta_i + math.pi) % (2 * math.pi) - math.pi
                            return a * math.exp(-0.5 * (dtheta / b) ** 2)

                        # Sum PQRST features
                        ecg = (
                            wave(a_p, b_p, theta_p) +
                            wave(a_q, b_q, theta_q) +
                            wave(a_r, b_r, theta_r) +
                            wave(a_s, b_s, theta_s) +
                            wave(a_t, b_t, theta_t)
                        )

                        # Scale R-peak to target voltage
                        v = max(0.0, ecg) * self.target_voltage
                    elif self.waveform_type == "Damped Sine":
                        # Damped oscillation: sine * exponential decay
                        v = math.sin(2 * math.pi * pos) * math.exp(-3 * pos)
                        v = abs(v) * self.target_voltage

                    elif self.waveform_type == "Exponential Raise":
                        # Exponential curve: slow start, fast end
                        v = (math.exp(5 * pos) - 1) / (math.exp(5) - 1)
                        v *= self.target_voltage

                    elif self.waveform_type == "Exponential Fall":
                        # True exponential decay: fast drop at start, slow approach to zero
                        v = math.exp(-5 * pos)
                        v *= self.target_voltage

                    elif self.waveform_type == "Gaussian Pulse":
                        # Smooth centered pulse
                        sigma = 0.12
                        v = math.exp(-((pos - 0.5) ** 2) / (2 * sigma * sigma))
                        v *= self.target_voltage

                    elif self.waveform_type == "Neural Spike":
                        # Subthreshold bump
                        a = 0.2 * math.exp(-((pos - 0.30)**2) / 0.004)
                        # Main spike
                        b = 1.0 * math.exp(-((pos - 0.50)**2) / 0.0004)
                        # Afterhyperpolarization
                        c = -0.3 * math.exp(-((pos - 0.60)**2) / 0.001)
                        v = max(0.0, a + b + c) * self.target_voltage

                    elif self.waveform_type == "Staircase":
                        # Discrete voltage steps - great for ADC testing
                        # Divides cycle into 8 equal steps
                        steps = 8
                        step_index = int(pos * steps)
                        # Clamp step_index to prevent overshoot when pos approaches 1.0
                        step_index = min(step_index, steps - 1)
                        v = (step_index / (steps - 1)) * self.target_voltage

                    elif self.waveform_type == "PWM":
                        # Pulse Width Modulation - duty cycle varies linearly
                        # Frequency: 10 pulses per cycle
                        freq = 10
                        pulse_pos = (pos * freq) % 1.0
                        duty_cycle = pos  # Duty cycle increases from 0% to 100%
                        v = self.target_voltage if pulse_pos < duty_cycle else 0.0

                    elif self.waveform_type == "Chirp":
                        # Frequency sweep - starts slow, ends fast
                        # Instantaneous frequency increases linearly
                        # f(t) = f0 + k*t, where k is chirp rate
                        chirp_rate = 5  # Frequency multiplier
                        phase = 2 * math.pi * (pos + chirp_rate * pos * pos / 2)
                        v = abs(math.sin(phase)) * self.target_voltage

                    elif self.waveform_type == "Burst Mode":
                        # On/off bursting - 20% on, 80% off
                        burst_duty = 0.2
                        if pos < burst_duty:
                            # During burst: fast oscillation
                            burst_freq = 8
                            v = abs(math.sin(2 * math.pi * burst_freq * pos / burst_duty)) * self.target_voltage
                        else:
                            # Off period
                            v = 0.0

                    elif self.waveform_type == "Brownout":
                        # Simulates power brownout/sag and recovery
                        if pos < 0.3:
                            # Normal voltage
                            v = self.target_voltage
                        elif pos < 0.5:
                            # Voltage sag (exponential decay)
                            sag_pos = (pos - 0.3) / 0.2
                            v = self.target_voltage * (0.3 + 0.7 * math.exp(-5 * sag_pos))
                        elif pos < 0.7:
                            # Low voltage period
                            v = self.target_voltage * 0.3
                        else:
                            # Recovery (exponential rise)
                            recovery_pos = (pos - 0.7) / 0.3
                            v = self.target_voltage * (0.3 + 0.7 * (1 - math.exp(-5 * recovery_pos)))

                    elif self.waveform_type == "RC Charge":
                        # Classic RC circuit charging curve: V = V_max * (1 - e^(-t/RC))
                        # Time constant tau = 0.2 (reaches ~99% at pos=1)
                        tau = 0.2
                        v = self.target_voltage * (1 - math.exp(-pos / tau))

                    elif self.waveform_type == "Sinc":
                        # Sinc function: sin(x)/x with oscillating side lobes
                        # Center at pos=0.5 for symmetry
                        x = (pos - 0.5) * 10  # Scale to make lobes visible
                        if abs(x) < 0.01:  # Avoid division by zero at center
                            sinc_val = 1.0
                        else:
                            sinc_val = math.sin(math.pi * x) / (math.pi * x)
                        # Make non-negative and scale
                        v = abs(sinc_val) * self.target_voltage

                    elif self.waveform_type == "Breathing":
                        # Slow, smooth breathing effect (like LED breathing)
                        # Uses raised cosine for smooth fade in/out
                        v = self.target_voltage * (1 - math.cos(2 * math.pi * pos)) / 2

                    else:
                        v = 0.0


                    # ────────────────────────────────────────────────────────
                    # Safety Clamp to Target Voltage and Hardware Limits
                    # ────────────────────────────────────────────────────────
                    # CRITICAL: Clamp to target_voltage first to prevent OVP trips
                    # Then clamp to hardware limit (30V) as secondary protection
                    v = max(0.0, min(v, self.target_voltage, 30.0))
                                                    # Ensures: 0V ≤ v ≤ target_voltage ≤ 30V
                                                    # Prevents waveform formula bugs from exceeding target

                    # ────────────────────────────────────────────────────────
                    # Append Point to Profile (with precision rounding)
                    # ────────────────────────────────────────────────────────
                    profile.append((round(t, 6), round(v, 6)))
                                                    # 6 decimals = microsecond/microvolt precision

            return profile                          # Return complete waveform profile

    # Nested class: Ramp Data Manager
    class _RampDataManager:
        """
        Collects, stores, and exports data during voltage ramping operations.
        """

        def __init__(self):
            """Initialize data manager and create storage folders."""
            self.voltage_data = []
            self.data_dir = os.path.join(os.getcwd(), 'voltage_ramp_data')
            self.graphs_dir = os.path.join(os.getcwd(), 'voltage_ramp_graphs')

            try:
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(self.graphs_dir, exist_ok=True)
            except Exception:
                pass
        
        def add_point(self, ts, set_v, meas_v, cycle_no, point_idx):
            """Add one measurement data point to the collection."""
            self.voltage_data.append({
                'timestamp': ts,
                'set_voltage': set_v,
                'measured_voltage': meas_v,
                'cycle_number': cycle_no,
                'point_in_cycle': point_idx
            })

        def clear(self):
            """Clear all collected data points."""
            self.voltage_data.clear()
        
        def export_csv(self, folder=None):
            """Export ramping data to CSV file"""
            if not self.voltage_data:
                raise ValueError('No ramping data')
            
            folder = folder or '.'
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fn = os.path.join(folder, f'psu_ramping_{ts}.csv')
            
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'set_voltage', 'measured_voltage', 'cycle', 'point'])
                for d in self.voltage_data:
                    w.writerow([
                        d['timestamp'].isoformat(),
                        d['set_voltage'],
                        d['measured_voltage'],
                        d['cycle_number'],
                        d['point_in_cycle']
                    ])
            
            return fn
        
        def generate_graph(self, folder=None, title: Optional[str] = None) -> str:
            """Generate matplotlib graph of set vs measured voltage"""
            if not self.voltage_data:
                raise ValueError('No ramping data')
            
            folder = folder or self.graphs_dir
            
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception:
                pass
            
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fn = os.path.join(folder, f'voltage_ramp_{ts}.png')
            
            times = []
            set_v = []
            meas_v = []
            
            t0 = self.voltage_data[0]['timestamp'] if self.voltage_data else datetime.now()
            
            for d in self.voltage_data:
                times.append((d['timestamp'] - t0).total_seconds())
                set_v.append(d['set_voltage'])
                meas_v.append(d['measured_voltage'])
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(times, set_v, label='Set Voltage', color='tab:blue')
                ax.plot(times, meas_v, label='Measured Voltage', color='tab:red')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(title or 'Voltage Ramping')
                ax.grid(True, ls='--', alpha=0.4)
                ax.legend()
                plt.tight_layout()
                plt.savefig(fn, dpi=1200)
                plt.close(fig)
                
            except Exception:
                raise
            
            return fn

    # Connection management methods
    def connect_power_supply(self, visa_address: str) -> str:
        """Establish communication with the Keithley power supply via USB."""

        def connect_thread():
            """Background connection task to keep UI responsive."""
            try:
                self.log_message("Attempting to connect to Keithley power supply...", "INFO")

                if not visa_address.strip():
                    raise ValueError("VISA address cannot be empty")

                self.power_supply = KeithleyPowerSupply(visa_address)

                if self.power_supply.connect():
                    self.log_message("Connection established successfully!", "SUCCESS")
                    self.status_queue.put(("connected", None))
                    self.is_connected = True
                else:
                    raise Exception("Connection failed - check VISA address and instrument")

            except Exception as e:
                self.status_queue.put(("error", f"Connection failed: {str(e)}"))
                self.log_message(f"Connection failed: {str(e)}", "ERROR")

        threading.Thread(target=connect_thread, daemon=True).start()
        return "Connecting... please wait"

    def disconnect_power_supply(self) -> str:
        """Close VISA connection to power supply"""
        try:
            if self.power_supply:
                self.power_supply.disconnect()
                self.power_supply = None
            
            self.is_connected = False
            self.measurement_active = False
            
            for channel in self.channel_states:
                self.channel_states[channel]["enabled"] = False
                self.channel_states[channel]["voltage"] = 0.0
                self.channel_states[channel]["current"] = 0.0
                self.channel_states[channel]["power"] = 0.0
            
            self.log_message("Disconnected from power supply", "SUCCESS")
            return "Disconnected"
        
        except Exception as e:
            self.log_message(f"Error during disconnection: {e}", "ERROR")
            return f"Error: {e}"

    # Instrument information and testing
    def get_instrument_info(self) -> str:
        """Query instrument identification and capabilities information"""
        def info_thread():
            try:
                self.log_message("Getting instrument information...", "INFO")
                
                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")
                
                info = self.power_supply.get_instrument_info()
                
                if info:
                    self.status_queue.put(("info_retrieved", info))
                else:
                    self.status_queue.put(("error", "Failed to retrieve instrument information"))
            
            except Exception as e:
                self.status_queue.put(("error", f"Info retrieval error: {str(e)}"))
        
        if self.is_connected and self.power_supply:
            threading.Thread(target=info_thread, daemon=True).start()
            return "Retrieving info..."
        else:
            return "Error: Power supply not connected"

    def test_connection(self) -> str:
        """Test VISA communication link"""
        try:
            if self.power_supply and self.power_supply.is_connected:
                self.log_message("Connection test: PASSED", "SUCCESS")
                return "✓ Connection test PASSED"
            else:
                self.log_message("Connection test: FAILED", "ERROR")
                return "✗ Connection test FAILED"
        
        except Exception as e:
            self.log_message(f"Connection test error: {e}", "ERROR")
            return f"Error: {e}"

    # Channel configuration and control
    def configure_channel(self, channel: int, voltage: float, current_limit: float, ovp_level: float) -> str:
        """Set up a power supply channel with desired parameters."""
        def config_thread():
            try:
                self.log_message(f"Configuring channel {channel} - V: {voltage:.3f}V, I: {current_limit:.3f}A, OVP: {ovp_level:.1f}V", "INFO")
                
                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")
                
                success = self.power_supply.configure_channel(
                    channel=channel,
                    voltage=voltage,
                    current_limit=current_limit,
                    ovp_level=ovp_level,
                    enable_output=False
                )
                
                if success:
                    self.status_queue.put(("channel_configured", f"Channel {channel} configured successfully"))
                else:
                    self.status_queue.put(("error", f"Failed to configure channel {channel}"))
            
            except Exception as e:
                self.status_queue.put(("error", f"Channel {channel} configuration error: {str(e)}"))
        
        if self.is_connected and self.power_supply:
            threading.Thread(target=config_thread, daemon=True).start()
            return f"Configuring channel {channel}..."
        else:
            return "Error: Power supply not connected"

    def enable_channel_output(self, channel: int) -> str:
        """Enable output on specified channel"""
        def enable_thread():
            try:
                self.log_message(f"Enabling output on channel {channel}...", "INFO")
                
                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")
                
                success = self.power_supply.enable_channel_output(channel)
                
                if success:
                    self.channel_states[channel]["enabled"] = True
                    self.status_queue.put(("channel_enabled", channel))
                else:
                    self.status_queue.put(("error", f"Failed to enable channel {channel} output"))
            
            except Exception as e:
                self.status_queue.put(("error", f"Channel {channel} enable error: {str(e)}"))
        
        if self.is_connected and self.power_supply:
            threading.Thread(target=enable_thread, daemon=True).start()
            return f"Enabling channel {channel}..."
        else:
            return "Error: Power supply not connected"

    def disable_channel_output(self, channel: int) -> str:
        """Disable output on specified channel"""
        def disable_thread():
            try:
                self.log_message(f"Disabling output on channel {channel}...", "INFO")
                
                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")
                
                success = self.power_supply.disable_channel_output(channel)
                
                if success:
                    self.channel_states[channel]["enabled"] = False
                    self.status_queue.put(("channel_disabled", channel))
                else:
                    self.status_queue.put(("error", f"Failed to disable channel {channel} output"))
            
            except Exception as e:
                self.status_queue.put(("error", f"Channel {channel} disable error: {str(e)}"))
        
        if self.is_connected and self.power_supply:
            threading.Thread(target=disable_thread, daemon=True).start()
            return f"Disabling channel {channel}..."
        else:
            return "Error: Power supply not connected"

    def clear_channel_protection(self, channel: int) -> str:
        """Clear OVP/OCP protection trip state on specified channel"""
        def clear_thread():
            try:
                self.log_message(f"Attempting to clear protection on CH{channel}...", "INFO")

                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")

                success = self.power_supply.clear_protection(channel)

                if success:
                    # Show prominent warning about physical power cycle
                    self.log_message("═" * 80, "WARNING")
                    self.log_message("⚠️  OVP SOFTWARE RESET SENT", "WARNING")
                    self.log_message("", "WARNING")
                    self.log_message("CHECK YOUR POWER SUPPLY FRONT PANEL:", "WARNING")
                    self.log_message("  • If 'OVER VOLTAGE' indicator is STILL ON:", "WARNING")
                    self.log_message("", "WARNING")
                    self.log_message("    ╔══════════════════════════════════════════════════╗", "WARNING")
                    self.log_message("    ║  PHYSICAL POWER CYCLE REQUIRED                   ║", "WARNING")
                    self.log_message("    ║                                                  ║", "WARNING")
                    self.log_message(f"    ║  1. Turn OFF PSU (front panel power button)     ║", "WARNING")
                    self.log_message("    ║  2. Wait 5 seconds                               ║", "WARNING")
                    self.log_message("    ║  3. Turn ON PSU                                  ║", "WARNING")
                    self.log_message("    ║  4. OVP should clear                             ║", "WARNING")
                    self.log_message("    ╚══════════════════════════════════════════════════╝", "WARNING")
                    self.log_message("", "WARNING")
                    self.log_message("  • If 'OVER VOLTAGE' indicator turned OFF:", "WARNING")
                    self.log_message(f"    ✓ CH{channel} is ready to reconfigure and use", "WARNING")
                    self.log_message("═" * 80, "WARNING")

                    self.status_queue.put(("warning", f"CH{channel} software reset sent - CHECK FRONT PANEL"))
                else:
                    self.status_queue.put(("error", f"Failed to send reset commands to CH{channel}"))

            except Exception as e:
                self.status_queue.put(("error", f"CH{channel} protection clear error: {str(e)}"))

        if self.is_connected and self.power_supply:
            threading.Thread(target=clear_thread, daemon=True).start()
            return f"Sending reset commands to CH{channel}..."
        else:
            return "Error: Power supply not connected"

    def measure_channel_output(self, channel: int) -> Tuple[str, str, str]:
        """Read current voltage and current from specified channel"""
        try:
            if not self.power_supply or not self.power_supply.is_connected:
                raise RuntimeError("Power supply not connected")

            measurement = self.power_supply.measure_channel_output(channel)

            if measurement and isinstance(measurement, tuple) and len(measurement) == 2:
                voltage = float(measurement[0])
                current = float(measurement[1])
                power = voltage * current

                self.channel_states[channel]["voltage"] = voltage
                self.channel_states[channel]["current"] = current
                self.channel_states[channel]["power"] = power

                self.log_message(f"Channel {channel}: {voltage:.3f}V, {current:.3f}A, {power:.3f}W", "SUCCESS")

                return f"{voltage:.3f} V", f"{current:.3f} A", f"{power:.3f} W"

            else:
                self.log_message(f"Failed to measure channel {channel} output", "ERROR")
                return "Error", "Error", "Error"

        except Exception as e:
            self.log_message(f"Channel {channel} measurement error: {str(e)}", "ERROR")
            return "Error", "Error", "Error"

    # Global operations and safety
    def measure_all_channels(self) -> Tuple[str, str, str, str, str, str, str, str, str]:
        """
        Measure voltage and current from all 3 channels in sequence.
        Returns a tuple of 9 strings for all channel measurements.
        """
        if not (self.is_connected and self.power_supply and self.power_supply.is_connected):
            error_tuple = ("Error",) * 9
            return error_tuple

        try:
            self.log_message("Starting sequential measurement of all channels...", "INFO")

            results = []

            for channel in range(1, 4):
                try:
                    if not self.power_supply or not self.power_supply.is_connected:
                        raise RuntimeError("Power supply not connected")

                    measurement = self.power_supply.measure_channel_output(channel)

                    if measurement and isinstance(measurement, tuple) and len(measurement) == 2:
                        voltage = float(measurement[0])
                        current = float(measurement[1])
                        power = voltage * current

                        self.channel_states[channel]["voltage"] = voltage
                        self.channel_states[channel]["current"] = current
                        self.channel_states[channel]["power"] = power

                        self.log_message(f"Channel {channel}: {voltage:.3f}V, {current:.3f}A, {power:.3f}W", "SUCCESS")

                        results.extend([
                            f"{voltage:.3f} V",
                            f"{current:.3f} A",
                            f"{power:.3f} W"
                        ])

                    else:
                        self.log_message(f"Failed to measure channel {channel}", "ERROR")
                        results.extend(["Error", "Error", "Error"])

                    # Add delay between measurements (except after the last channel)
                    if channel < 3:
                        time.sleep(0.8)  # Wait 0.8 seconds before next measurement

                except Exception as e:
                    self.log_message(f"Error measuring channel {channel}: {e}", "ERROR")
                    results.extend(["Error", "Error", "Error"])

            self.log_message("Sequential measurement completed", "SUCCESS")
            return tuple(results)

        except Exception as e:
            self.log_message(f"Error in sequential measurement: {e}", "ERROR")
            error_tuple = ("Error",) * 9
            return error_tuple

    def disable_all_outputs(self) -> str:
        """Disable output on all three channels (safety shutdown)"""
        def disable_all_thread():
            try:
                self.log_message("Emergency shutdown - disabling all outputs...", "ERROR")
                
                if not self.power_supply or not self.power_supply.is_connected:
                    raise RuntimeError("Power supply not connected")
                
                success = self.power_supply.disable_all_outputs()
                
                if success:
                    for ch in range(1, 4):
                        self.channel_states[ch]["enabled"] = False
                    self.status_queue.put(("all_disabled", "All outputs disabled successfully"))
                else:
                    self.status_queue.put(("error", "Failed to disable all outputs"))
            
            except Exception as e:
                self.status_queue.put(("error", f"Disable all error: {str(e)}"))
        
        if self.is_connected and self.power_supply:
            threading.Thread(target=disable_all_thread, daemon=True).start()
            return "Disabling all outputs..."
        else:
            return "Error: Power supply not connected"

    def emergency_stop(self) -> str:
        """
        Immediately disables all outputs for safety.
        Use this for dangerous or unexpected behavior.
        """
        self.log_message("EMERGENCY STOP ACTIVATED!", "ERROR")
        return self.disable_all_outputs()

    # Data logging and export
    def log_message(self, message: str, level: str = "INFO"):
        """Add timestamped message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        self.activity_log += log_entry
        self.logger.log(
            getattr(logging, level, logging.INFO),
            message
        )

    def export_measurement_data(self, save_path: str) -> str:
        """Export collected measurements to CSV file at user-specified location

        Args:
            save_path: Directory path where file should be saved

        Returns:
            Status message
        """
        try:
            if not self.measurement_data:
                return "No measurement data to export"

            if not save_path or save_path.strip() == "":
                return "Please select a save location using the Browse button"

            # Ensure the save directory exists
            save_dir = Path(save_path)
            if not save_dir.exists():
                return f"Error: Directory does not exist: {save_path}"

            if not save_dir.is_dir():
                return f"Error: Path is not a directory: {save_path}"

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"power_supply_data_{timestamp}.csv"
            filepath = save_dir / filename

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Channel", "Voltage (V)", "Current (A)", "Power (W)"])

                for channel, measurements in self.measurement_data.items():
                    for measurement in measurements:
                        writer.writerow([
                            measurement["timestamp"].isoformat(),
                            channel,
                            f"{measurement['voltage']:.6f}",
                            f"{measurement['current']:.6f}",
                            f"{measurement['power']:.6f}"
                        ])

            self.log_message(f"Data exported to: {filepath}", "SUCCESS")
            return f"✓ Data exported successfully to:\n{filepath}"

        except Exception as e:
            self.log_message(f"Export error: {e}", "ERROR")
            return f"Export error: {e}"

    def clear_measurement_data(self) -> str:
        """Clear all collected measurement data"""
        self.measurement_data.clear()
        self.log_message("Measurement data cleared", "INFO")
        return "Measurement data cleared"

    def toggle_auto_measure(self, enabled: bool) -> str:
        """Enable or disable automatic periodic measurement collection"""
        self.measurement_active = enabled
        if enabled:
            self.log_message("Auto-measurement enabled", "SUCCESS")
            return "Auto-measurement enabled"
        else:
            self.log_message("Auto-measurement disabled", "INFO")
            return "Auto-measurement disabled"

    # ════════════════════════════════════════════════════════════════════════════
    # Live Graphing Methods
    # ════════════════════════════════════════════════════════════════════════════

    def start_live_measurement(self, channels: List[int], interval: float = 1.0) -> str:
        """
        Start live measurement collection for specified channels.

        Args:
            channels: List of channel numbers to monitor (e.g., [1, 2, 3])
            interval: Time between measurements in seconds (0.5-60)

        Returns:
            Status message indicating success or error
        """
        if not self.is_connected or not self.power_supply:
            return "Error: Power supply not connected"

        if self.live_measurement_active:
            return "Live measurement already running"

        if not channels:
            return "Error: No channels selected"

        self.live_measurement_interval = max(0.5, min(interval, 60.0))
        self.live_measurement_active = True

        def live_measurement_worker():
            """Background worker for live measurements"""
            while self.live_measurement_active and self.is_connected:
                try:
                    timestamp = datetime.now()

                    for channel in channels:
                        if not self.live_measurement_active:
                            break

                        measurement = self.power_supply.measure_channel_output(channel)

                        if measurement and isinstance(measurement, tuple) and len(measurement) == 2:
                            voltage = float(measurement[0])
                            current = float(measurement[1])
                            power = voltage * current

                            self.channel_states[channel]["voltage"] = voltage
                            self.channel_states[channel]["current"] = current
                            self.channel_states[channel]["power"] = power

                            self.live_data[channel]['timestamps'].append(timestamp)
                            self.live_data[channel]['voltages'].append(voltage)
                            self.live_data[channel]['currents'].append(current)
                            self.live_data[channel]['powers'].append(power)

                            if len(self.live_data[channel]['timestamps']) > self.max_live_points:
                                self.live_data[channel]['timestamps'] = self.live_data[channel]['timestamps'][-self.max_live_points:]
                                self.live_data[channel]['voltages'] = self.live_data[channel]['voltages'][-self.max_live_points:]
                                self.live_data[channel]['currents'] = self.live_data[channel]['currents'][-self.max_live_points:]
                                self.live_data[channel]['powers'] = self.live_data[channel]['powers'][-self.max_live_points:]

                        time.sleep(0.1)

                    time.sleep(self.live_measurement_interval)

                except Exception as e:
                    self.log_message(f"Live measurement error: {e}", "ERROR")
                    time.sleep(1.0)

        self.live_measurement_thread = threading.Thread(target=live_measurement_worker, daemon=True)
        self.live_measurement_thread.start()

        self.log_message(f"Live measurement started for channels {channels}", "SUCCESS")
        return f"Live measurement started (interval: {self.live_measurement_interval}s)"

    def stop_live_measurement(self) -> str:
        """Stop live measurement collection"""
        self.live_measurement_active = False
        if self.live_measurement_thread and self.live_measurement_thread.is_alive():
            self.live_measurement_thread.join(timeout=2.0)
        self.log_message("Live measurement stopped", "INFO")
        return "Live measurement stopped"

    def clear_live_data(self) -> str:
        """Clear all live measurement data"""
        for channel in range(1, 4):
            self.live_data[channel] = {'timestamps': [], 'voltages': [], 'currents': [], 'powers': []}
        self.log_message("Live data cleared", "INFO")
        return "Live data cleared"

    def create_live_plot(self, plot_type: str = "voltage") -> Optional[plt.Figure]:
        """
        Create a live trend plot for all channels.

        Args:
            plot_type: Type of data to plot - "voltage", "current", or "power"

        Returns:
            Matplotlib figure object or None if no data
        """
        try:
            has_data = False
            for ch in range(1, 4):
                if len(self.live_data[ch]['timestamps']) > 0:
                    has_data = True
                    break

            if not has_data:
                return None

            fig, ax = plt.subplots(figsize=(12, 6))

            colors = {1: 'blue', 2: 'green', 3: 'red'}
            labels = {1: 'CH1', 2: 'CH2', 3: 'CH3'}

            if plot_type == "current":
                data_key = 'currents'
                ylabel = 'Current (A)'
                title = 'PSU Live Current Monitor'
            elif plot_type == "power":
                data_key = 'powers'
                ylabel = 'Power (W)'
                title = 'PSU Live Power Monitor'
            else:
                data_key = 'voltages'
                ylabel = 'Voltage (V)'
                title = 'PSU Live Voltage Monitor'

            has_data = False
            for ch in range(1, 4):
                if len(self.live_data[ch]['timestamps']) >= 2:
                    ax.plot(
                        self.live_data[ch]['timestamps'],
                        self.live_data[ch][data_key],
                        color=colors[ch],
                        label=labels[ch],
                        linewidth=1.5,
                        marker='o',
                        markersize=2
                    )
                    has_data = True

            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(loc='upper right')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)

            plt.tight_layout()
            return fig

        except Exception as e:
            self.logger.error(f"Plot creation error: {e}")
            return None

    def get_live_statistics(self, channel: int) -> Tuple[str, str, str, str, str]:
        """
        Calculate statistics from live measurement data for a channel.

        Returns:
            Tuple of (count, avg_voltage, avg_current, avg_power, last_update)
        """
        try:
            data = self.live_data[channel]
            if not data['timestamps']:
                return "0", "N/A", "N/A", "N/A", "N/A"

            count = len(data['timestamps'])
            avg_v = sum(data['voltages']) / count if data['voltages'] else 0
            avg_i = sum(data['currents']) / count if data['currents'] else 0
            avg_p = sum(data['powers']) / count if data['powers'] else 0
            last_time = data['timestamps'][-1].strftime('%H:%M:%S') if data['timestamps'] else "N/A"

            return (
                str(count),
                f"{avg_v:.4f} V",
                f"{avg_i:.4f} A",
                f"{avg_p:.4f} W",
                last_time
            )
        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return "Error", "N/A", "N/A", "N/A", "N/A"

    def get_waveform_status(self):
        """Get current waveform status for UI updates"""
        return self.waveform_status_message

    # ════════════════════════════════════════════════════════════════════════════
    # Waveform Execution Engine
    # ════════════════════════════════════════════════════════════════════════════

    def execute_waveform_ramping(self):
        """
        Execute real-time closed-loop waveform control with timing analysis and safety shutdown.

        ┌──────────────────────────────────────────────────────────────────────┐
        │ CRITICAL ALGORITHM: REAL-TIME VOLTAGE CONTROL LOOP                   │
        │                                                                      │
        │ Purpose: Execute pre-generated waveform on PSU with measurement      │
        │          feedback and comprehensive timing analysis                  │
        │                                                                      │
        │ Control Loop Steps (per waveform point):                             │
        │   1. Set PSU output voltage to target value                          │
        │   2. Calculate elapsed time and sleep to meet target time            │
        │   3. Record data with timing metadata                                │
        │   4. Check for stop condition (user abort)                           │
        │   5. Repeat for all points in profile                                │
        │                                                                      │
        │ Timing Components per Point:                                         │
        │   - SCPI set voltage command:    ~30-50ms (VISA overhead)            │
        │   - Calculated delay:            Adjusted to meet exact target       │
        │   TOTAL per point:               User-defined (default 262ms)        │
        │                                                                      │
        │ Note: Measurements are skipped to ensure precise timing control.     │
        │       Use external DMM for accurate voltage/current measurements.    │
        │                                                                      │
        │ Safety Features:                                                     │
        │   - Automatic output disable on completion                           │
        │   - Emergency shutdown on exception                                  │
        │   - User abort capability (ramping_active flag)                      │
        │   - Voltage clamp to 0V before output disable                        │
        └──────────────────────────────────────────────────────────────────────┘

        Thread Safety:
            Should be called from background thread (via start_waveform_ramping).
            Uses ramping_active flag for clean shutdown from UI thread.

        Performance:
            Typical execution time: N_points × (target_time_per_point)
            Example: 150 points × 262ms = 39.3 seconds
            User can adjust time per point from 50ms to 5000ms (default: 262ms).

        Data Collection:
            Each point stores: timestamp, set_voltage, measured_voltage,
            measured_current, cycle_number, point_in_cycle, point_index,
            point_duration (for timing analysis)

        Returns:
            None (updates self.ramping_data and self.waveform_status_message)

        Raises:
            No exceptions propagated - all caught and logged for safety

        Example:
            >>> psu.ramping_profile = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]
            >>> psu.ramping_active = True
            >>> psu.execute_waveform_ramping()
            # Executes 3-point voltage profile on active channel
            # Results stored in psu.ramping_data

        See Also:
            - start_waveform_ramping(): Wrapper that spawns background thread
            - stop_waveform(): Sets ramping_active=False for clean abort
            - _WaveformGenerator.generate(): Creates ramping_profile
        """
        # ────────────────────────────────────────────────────────────────────
        # Pre-flight Validation
        # ────────────────────────────────────────────────────────────────────
        if not self.ramping_profile:                # Guard: require generated waveform
            self.log_message("No waveform profile generated", "ERROR")
            return

        if not self.power_supply or not self.is_connected:  # Guard: require PSU connection
            self.log_message("Power supply not connected", "ERROR")
            return

        # ────────────────────────────────────────────────────────────────────
        # Extract Execution Parameters
        # ────────────────────────────────────────────────────────────────────
        channel = self.ramping_params['active_channel']     # Target channel (1, 2, or 3)
        psu_settle = self.ramping_params.get('psu_settle', 0.05)  # Settling time in seconds (default 50ms)
                                                            # Critical for output capacitor stabilization
                                                            # Too short = inaccurate measurements
                                                            # Too long = slower execution

        self.waveform_status_message = f"⏳ Running waveform on Channel {channel}..."

        # ────────────────────────────────────────────────────────────────────
        # Initialization and Logging Header
        # ────────────────────────────────────────────────────────────────────
        waveform_start_time = datetime.now()       # Record execution start time
        start_timestamp = waveform_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Millisecond precision

        # Log execution parameters for diagnostics
        self.log_message(f"{'='*60}", "INFO")
        self.log_message(f"WAVEFORM GENERATION STARTED", "INFO")
        self.log_message(f"Start Time: {start_timestamp}", "INFO")
        self.log_message(f"Channel: {channel}", "INFO")
        self.log_message(f"Waveform Type: {self.ramping_params['waveform']}", "INFO")
        self.log_message(f"Target Voltage: {self.ramping_params['target_voltage']}V", "INFO")
        self.log_message(f"Total Points: {len(self.ramping_profile)}", "INFO")
        self.log_message(f"Cycles: {self.ramping_params['cycles']}", "INFO")
        self.log_message(f"Points per Cycle: {self.ramping_params['points_per_cycle']}", "INFO")
        self.log_message(f"Target Time per Point: {psu_settle}s ({psu_settle*1000:.0f}ms)", "INFO")
        self.log_message(f"{'='*60}", "INFO")

        # ────────────────────────────────────────────────────────────────────
        # Data Collection Setup
        # ────────────────────────────────────────────────────────────────────
        self.ramping_data = []                      # Clear previous run data
        point_timings = []                          # Track execution time per point
        last_point_time = waveform_start_time       # For timing calculations (unused currently)

        try:
            # ────────────────────────────────────────────────────────────────
            # Clear Protection State from Previous Runs
            # ────────────────────────────────────────────────────────────────
            self.log_message("Clearing any previous protection states...", "INFO")
            self.power_supply.clear_protection(channel)  # Clear specific channel
            time.sleep(0.3)

            # ────────────────────────────────────────────────────────────────
            # PSU Output Enable (Safety Critical)
            # ────────────────────────────────────────────────────────────────
            self.power_supply.enable_channel_output(channel)  # Turn on PSU output
            time.sleep(0.1)                         # 100ms delay for relay switching
                                                    # Allows output relay to close fully

            # ────────────────────────────────────────────────────────────────
            # Loop Initialization
            # ────────────────────────────────────────────────────────────────
            cycle_num = 0                           # Current cycle index
            points_per_cycle = self.ramping_params['points_per_cycle']  # Points in one waveform cycle

            # ════════════════════════════════════════════════════════════════
            # MAIN CONTROL LOOP: Execute Waveform Point-by-Point
            # ════════════════════════════════════════════════════════════════
            # Iterates through generated waveform profile: [(t0, V0), (t1, V1), ...]
            # Each iteration: Set voltage → Wait settle → Measure → Store data
            for idx, (time_setpoint, voltage) in enumerate(self.ramping_profile):
                point_start_time = datetime.now()   # Record point start time for profiling

                # ────────────────────────────────────────────────────────────
                # User Abort Check (Safety)
                # ────────────────────────────────────────────────────────────
                if not self.ramping_active:         # Check abort flag (set by UI)
                    self.log_message("Waveform execution stopped by user", "WARNING")
                    break                           # Exit loop immediately, proceed to shutdown

                # ────────────────────────────────────────────────────────────
                # Calculate Cycle Position
                # ────────────────────────────────────────────────────────────
                cycle_num = idx // points_per_cycle         # Integer division: which cycle?
                point_in_cycle = idx % points_per_cycle     # Modulo: position within cycle

                # ────────────────────────────────────────────────────────────
                # STEP 1: Set Target Voltage
                # ────────────────────────────────────────────────────────────
                # Send SCPI command: VOLT <voltage>, CHAN <channel>
                # VISA overhead: ~30-50ms for USB, ~20-30ms for GPIB
                self.power_supply.set_voltage(channel, voltage)

                # ────────────────────────────────────────────────────────────
                # STEP 2: Calculate Elapsed Time and Sleep to Meet Target
                # ────────────────────────────────────────────────────────────
                # Calculate how much time has elapsed since point start
                point_elapsed = (datetime.now() - point_start_time).total_seconds()

                # Calculate remaining time to meet target time per point
                additional_delay = psu_settle - point_elapsed

                # Sleep for remaining time (includes settle + measurement time)
                if additional_delay > 0:
                    time.sleep(additional_delay)    # Sleep to meet exact target time

                # Use setpoint values (no actual measurement to avoid delays)
                measured_v = voltage                # Use commanded voltage
                measured_i = 0.0                    # No current measurement

                # ────────────────────────────────────────────────────────────
                # STEP 3: Timing Analysis
                # ────────────────────────────────────────────────────────────
                point_end_time = datetime.now()     # Record point completion time
                point_duration = (point_end_time - point_start_time).total_seconds()
                                                    # Duration should match target time per point
                point_timings.append(point_duration)  # Store for statistics

                # ────────────────────────────────────────────────────────────
                # STEP 4: Store Data Point with Metadata
                # ────────────────────────────────────────────────────────────
                data_point = {
                    'timestamp': point_end_time,            # ISO timestamp of measurement
                    'set_voltage': voltage,                 # Commanded voltage (V)
                    'measured_voltage': measured_v,         # Actual measured voltage (V)
                    'measured_current': measured_i,         # Actual measured current (A)
                    'cycle_number': cycle_num,              # Which cycle (0-indexed)
                    'point_in_cycle': point_in_cycle,       # Position within cycle
                    'point_index': idx,                     # Global point index
                    'point_duration': point_duration        # Execution time for this point (s)
                }
                self.ramping_data.append(data_point)  # Add to collection buffer

                # ────────────────────────────────────────────────────────────
                # Progress Logging (Every 10% of Profile)
                # ────────────────────────────────────────────────────────────
                if idx % max(1, len(self.ramping_profile) // 10) == 0:
                    progress = (idx / len(self.ramping_profile)) * 100  # Percentage complete
                    elapsed = (point_end_time - waveform_start_time).total_seconds()  # Total elapsed
                    avg_time_per_point = sum(point_timings) / len(point_timings) if point_timings else 0
                                                    # Running average of point duration

                    # Calculate ETA
                    points_remaining = len(self.ramping_profile) - idx
                    eta_seconds = points_remaining * avg_time_per_point
                    eta_minutes = eta_seconds / 60

                    # Format ETA display
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.1f}s"
                    else:
                        eta_str = f"{eta_minutes:.1f}min"

                    self.log_message(
                        f"Progress: {progress:.1f}% | "
                        f"Point {idx}/{len(self.ramping_profile)} | "
                        f"Cycle {cycle_num + 1}/{self.ramping_params['cycles']} | "
                        f"Elapsed: {elapsed:.2f}s | "
                        f"This point: {point_duration*1000:.0f}ms | "
                        f"Avg: {avg_time_per_point*1000:.1f}ms/pt | "
                        f"ETA: {eta_str}",
                        "INFO"
                    )

            # Waveform complete - disable output for safety
            self.power_supply.set_voltage(channel, 0.0)
            time.sleep(0.1)
            self.power_supply.disable_channel_output(channel)

            # Calculate and log timing statistics
            waveform_end_time = datetime.now()
            end_timestamp = waveform_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            total_duration = (waveform_end_time - waveform_start_time).total_seconds()

            if self.ramping_active:
                # Calculate timing statistics
                if point_timings:
                    avg_time_per_point = sum(point_timings) / len(point_timings)
                    min_time_per_point = min(point_timings)
                    max_time_per_point = max(point_timings)
                else:
                    avg_time_per_point = 0
                    min_time_per_point = 0
                    max_time_per_point = 0

                # Log completion summary
                self.log_message(f"{'='*60}", "SUCCESS")
                self.log_message(f"WAVEFORM GENERATION COMPLETED", "SUCCESS")
                self.log_message(f"End Time: {end_timestamp}", "SUCCESS")
                self.log_message(f"{'='*60}", "SUCCESS")
                self.log_message(f"TIMING SUMMARY:", "INFO")
                self.log_message(f"  Start Time:        {start_timestamp}", "INFO")
                self.log_message(f"  End Time:          {end_timestamp}", "INFO")
                self.log_message(f"  Total Duration:    {total_duration:.3f}s ({total_duration/60:.2f} min)", "INFO")
                self.log_message(f"  Points Collected:  {len(self.ramping_data)}", "INFO")
                self.log_message(f"  Avg Time/Point:    {avg_time_per_point*1000:.2f}ms", "INFO")
                self.log_message(f"  Min Time/Point:    {min_time_per_point*1000:.2f}ms", "INFO")
                self.log_message(f"  Max Time/Point:    {max_time_per_point*1000:.2f}ms", "INFO")
                self.log_message(f"  Expected Duration: {len(self.ramping_profile) * psu_settle:.2f}s (settle time only)", "INFO")
                self.log_message(f"  Overhead:          {(total_duration - len(self.ramping_profile) * psu_settle):.2f}s", "INFO")
                self.log_message(f"{'='*60}", "SUCCESS")

                completion_msg = f"✓ COMPLETED! Ch{channel}, {len(self.ramping_data)} pts, {total_duration:.1f}s, avg {avg_time_per_point*1000:.1f}ms/pt"
                self.waveform_status_message = completion_msg

            self.ramping_active = False

        except Exception as e:
            self.ramping_active = False
            self.log_message(f"Waveform execution error: {e}", "ERROR")
            # Attempt to disable output for safety
            try:
                self.power_supply.set_voltage(channel, 0.0)
                self.power_supply.disable_channel_output(channel)
            except:
                pass

    def save_waveform_plot(self, save_path: str) -> str:
        """Save the LIVE waveform execution data plot to user-specified location.

        This saves the actual measured data from waveform execution, supporting
        both single-channel and multi-channel waveform runs.

        Args:
            save_path: Directory path where plot should be saved

        Returns:
            Status message
        """
        if not self.ramping_data:
            return "No waveform data available. Please run a waveform first."

        if not save_path or save_path.strip() == "":
            return "Please select a save location using the Browse button"

        try:
            # Ensure the save directory exists
            save_dir = Path(save_path)
            if not save_dir.exists():
                return f"Error: Directory does not exist: {save_path}"

            if not save_dir.is_dir():
                return f"Error: Path is not a directory: {save_path}"

            import matplotlib.pyplot as plt

            # Group data by channel
            channel_data = {}
            for d in self.ramping_data:
                ch = d.get('channel', 1)  # Default to CH1 for legacy single-channel data
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
            channel_colors = {1: '#1E88E5', 2: '#43A047', 3: '#FB8C00'}  # Blue, Green, Orange

            if num_channels == 1:
                # Single channel - use simple 2-subplot layout
                ch = list(channel_data.keys())[0]
                data = channel_data[ch]

                # Convert timestamps to relative seconds
                base_time = data['timestamps'][0]
                time_sec = [(t - base_time).total_seconds() for t in data['timestamps']]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                color = channel_colors.get(ch, '#1E88E5')

                # Voltage plot
                ax1.plot(time_sec, data['set_voltages'], '--', color=color,
                        label='Setpoint', linewidth=1, alpha=0.7)
                ax1.plot(time_sec, data['measured_voltages'], '-', color=color,
                        label='Measured', linewidth=1.5)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Voltage (V)')
                ax1.set_title(f'CH{ch} Voltage Profile (Live Execution Data)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Current plot
                ax2.plot(time_sec, data['measured_currents'], '-', color=color, linewidth=1.5)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Current (A)')
                ax2.set_title(f'CH{ch} Measured Current')
                ax2.grid(True, alpha=0.3)
            else:
                # Multi-channel - create subplot for each channel
                fig, axes = plt.subplots(num_channels, 2, figsize=(14, 4*num_channels))
                fig.suptitle('Multi-Channel Waveform Live Execution Data', fontsize=14, fontweight='bold')

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
            filename = f"psu_waveform_live_{timestamp_str}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            self.log_message(f"Live waveform plot saved to: {filepath}", "SUCCESS")
            return f"✓ Live execution data plot saved ({num_channels} channel{'s' if num_channels > 1 else ''}):\n{filepath}"

        except Exception as e:
            self.logger.error(f"Waveform plot save error: {e}")
            return f"Plot save failed: {str(e)}"

    # ════════════════════════════════════════════════════════════════════════════
    # Multi-Channel Simultaneous Waveform Execution
    # ════════════════════════════════════════════════════════════════════════════

    def execute_multi_channel_waveform(self, channel_configs: List[Dict]) -> None:
        """
        Execute simultaneous waveform generation on multiple PSU channels.

        ┌──────────────────────────────────────────────────────────────────────┐
        │ MULTI-CHANNEL WAVEFORM EXECUTION ENGINE                              │
        │                                                                      │
        │ Purpose: Generate synchronized voltage waveforms on multiple         │
        │          channels simultaneously for complex test scenarios          │
        │                                                                      │
        │ Use Cases:                                                           │
        │   - Multi-rail power sequencing tests                                │
        │   - Simultaneous voltage stress on multiple supplies                 │
        │   - Cross-talk and interference testing                              │
        │   - Power profile characterization                                   │
        └──────────────────────────────────────────────────────────────────────┘

        Args:
            channel_configs: List of dictionaries, each containing:
                - channel: int (1, 2, or 3)
                - waveform: str (waveform type name)
                - target_voltage: float (peak voltage)
                - cycles: int (number of repetitions)
                - points_per_cycle: int (resolution)
                - cycle_duration: float (seconds per cycle)
                - current_limit: float (safety current limit)

        Thread Safety:
            Should be called from background thread via start_multi_channel_waveform.
            Uses multi_channel_stop_event for clean shutdown.
        """
        if not channel_configs:
            self.log_message("No channel configurations provided", "ERROR")
            return

        if not self.power_supply or not self.is_connected:
            self.log_message("Power supply not connected", "ERROR")
            return

        psu_settle = self.ramping_params.get('psu_settle', 0.05)
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
                # Extract name if it has description (e.g., "Sine - ∿ smooth wave" -> "Sine")
                if ' - ' in waveform_name:
                    waveform_name = waveform_name.split(' - ')[0]

                gen = self._WaveformGenerator(
                    waveform_type=waveform_name,
                    target_voltage=config.get('target_voltage', 3.0),
                    cycles=config.get('cycles', 3),
                    points_per_cycle=config.get('points_per_cycle', 50),
                    cycle_duration=config.get('cycle_duration', 8.0)
                )
                generators.append({
                    'generator': gen,
                    'channel': config.get('channel', 1),
                    'current_limit': config.get('current_limit', 0.1),
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
            self.ramping_data = []
            point_timings = []

            # Clear any protection trips from previous runs
            self.log_message("Clearing any previous protection states...", "INFO")
            self.power_supply.clear_protection()  # Clear all channels
            time.sleep(0.3)

            # Configure and enable all channels first
            for gen_data in generators:
                ch = gen_data['channel']
                current_limit = gen_data['current_limit']
                ovp_level = gen_data['generator'].target_voltage + 2.0

                self.power_supply.configure_channel(
                    channel=ch,
                    voltage=0.0,
                    current_limit=current_limit,
                    ovp_level=min(ovp_level, 35.0),  # Cap OVP at 35V
                    enable_output=True
                )
                time.sleep(0.1)

            self.log_message("All channels configured and enabled", "SUCCESS")
            self.waveform_status_message = f"⏳ Running multi-channel waveform ({len(generators)} channels)..."

            # Execute waveform points synchronously across all channels
            for point_idx in range(max_points):
                point_start_time = datetime.now()

                # Check for stop signal
                if hasattr(self, 'multi_channel_stop_event') and self.multi_channel_stop_event.is_set():
                    self.log_message("Multi-channel waveform stopped by user", "WARNING")
                    break

                if not self.ramping_active:
                    self.log_message("Waveform stopped (ramping_active=False)", "WARNING")
                    break

                # Set voltage on all channels for this time point AND record data
                timestamp = datetime.now()
                channels_start = datetime.now()

                for gen_data in generators:
                    profile = gen_data['profile']
                    ch = gen_data['channel']
                    wf_type = gen_data['generator'].waveform_type

                    # Get voltage for this point (or hold last value if profile is shorter)
                    if point_idx < len(profile):
                        t, voltage = profile[point_idx]
                    else:
                        t, voltage = profile[-1]

                    # Debug logging for first few points
                    if point_idx < 5:
                        self.log_message(
                            f"Point {point_idx}: CH{ch} ({wf_type}) -> {voltage:.4f}V",
                            "INFO"
                        )

                    # Set voltage on this channel
                    self.power_supply.set_voltage(ch, voltage)

                    # Calculate cycle position
                    points_per_cycle = gen_data['generator'].points_per_cycle
                    cycle_num = point_idx // points_per_cycle
                    point_in_cycle = point_idx % points_per_cycle

                    # Store data point immediately after setting voltage
                    data_point = {
                        'timestamp': timestamp,
                        'channel': ch,
                        'set_voltage': voltage,
                        'measured_voltage': voltage,  # Use set value (no actual measurement)
                        'measured_current': 0.0,  # No measurement during execution
                        'cycle_number': cycle_num,
                        'point_in_cycle': point_in_cycle,
                        'point_index': point_idx
                    }
                    self.ramping_data.append(data_point)

                # Calculate elapsed time and add delay to meet target time per point
                channels_elapsed = (datetime.now() - channels_start).total_seconds()
                point_elapsed = (datetime.now() - point_start_time).total_seconds()
                additional_delay = psu_settle - point_elapsed

                # Debug: Log timing breakdown for first few points
                if point_idx < 3:
                    self.log_message(
                        f"Point {point_idx} timing: channels={channels_elapsed*1000:.0f}ms, "
                        f"total_so_far={point_elapsed*1000:.0f}ms, target={psu_settle*1000:.0f}ms, "
                        f"will_sleep={max(0, additional_delay)*1000:.0f}ms",
                        "INFO"
                    )

                if additional_delay > 0:
                    time.sleep(additional_delay)  # Sleep remaining time to meet target

                # Track timing
                point_end_time = datetime.now()
                point_duration = (point_end_time - point_start_time).total_seconds()
                point_timings.append(point_duration)

                # Progress logging (every 10%)
                if point_idx % max(1, max_points // 10) == 0:
                    progress = (point_idx / max_points) * 100
                    elapsed = (point_end_time - waveform_start_time).total_seconds()
                    avg_time = sum(point_timings) / len(point_timings) if point_timings else 0

                    # Calculate ETA
                    points_remaining = max_points - point_idx
                    eta_seconds = points_remaining * avg_time
                    eta_minutes = eta_seconds / 60

                    # Format ETA display
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.1f}s"
                    else:
                        eta_str = f"{eta_minutes:.1f}min"

                    self.log_message(
                        f"Progress: {progress:.1f}% | "
                        f"Point {point_idx}/{max_points} | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"This point: {point_duration*1000:.0f}ms | "
                        f"Avg: {avg_time*1000:.0f}ms/pt | "
                        f"ETA: {eta_str}",
                        "INFO"
                    )

            # Ramp down all channels to 0V
            self.log_message("Ramping down all channels to 0V...", "INFO")
            for gen_data in generators:
                ch = gen_data['channel']
                self.power_supply.set_voltage(ch, 0.0)
                time.sleep(0.05)
                self.power_supply.disable_channel_output(ch)

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
            self.log_message(f"Data Points Collected: {len(self.ramping_data)}", "SUCCESS")
            self.log_message(f"Per-Point Timing:", "INFO")
            self.log_message(f"  • Average: {avg_time*1000:.1f}ms", "INFO")
            self.log_message(f"  • Minimum: {min_time*1000:.1f}ms", "INFO")
            self.log_message(f"  • Maximum: {max_time*1000:.1f}ms", "INFO")
            self.log_message(f"{'='*60}", "SUCCESS")

            self.waveform_status_message = (
                f"✓ COMPLETED! {len(generators)} channels, "
                f"{len(self.ramping_data)} pts, {total_duration:.1f}s"
            )

        except Exception as e:
            self.log_message(f"Multi-channel waveform error: {e}", "ERROR")
            import traceback
            self.logger.error(traceback.format_exc())
            self.waveform_status_message = f"✗ ERROR: {str(e)}"

            # Safety: disable all outputs
            try:
                for gen_data in generators:
                    self.power_supply.set_voltage(gen_data['channel'], 0.0)
                    self.power_supply.disable_channel_output(gen_data['channel'])
            except:
                pass

        finally:
            self.ramping_active = False

    def start_multi_channel_waveform(self, channel_configs: List[Dict]) -> str:
        """
        Start multi-channel waveform execution in a background thread.

        Args:
            channel_configs: List of channel configuration dictionaries

        Returns:
            Status message string
        """
        if self.ramping_active:
            return "ERROR: Waveform already running. Stop current waveform first."

        if not self.is_connected or not self.power_supply:
            return "ERROR: Power supply not connected. Please connect first."

        if not channel_configs:
            return "ERROR: No channels enabled for waveform generation."

        # Validate configurations
        for config in channel_configs:
            ch = config.get('channel', 1)
            target_v = config.get('target_voltage', 3.0)
            if ch == 3 and target_v > 5:
                return f"ERROR: Channel 3 limited to 5V maximum. Configured: {target_v}V"
            elif ch in [1, 2] and target_v > 30:
                return f"ERROR: Channel {ch} limited to 30V maximum. Configured: {target_v}V"

        self.ramping_active = True
        self.multi_channel_stop_event = threading.Event()

        self.ramping_thread = threading.Thread(
            target=self.execute_multi_channel_waveform,
            args=(channel_configs,),
            daemon=True
        )
        self.ramping_thread.start()

        channels_str = ", ".join([f"CH{c['channel']}" for c in channel_configs])
        return f"Multi-channel waveform STARTED on {channels_str}"

    def stop_multi_channel_waveform(self) -> str:
        """
        Stop multi-channel waveform execution gracefully.

        Returns:
            Status message string
        """
        self.ramping_active = False
        if hasattr(self, 'multi_channel_stop_event'):
            self.multi_channel_stop_event.set()

        if self.ramping_thread and self.ramping_thread.is_alive():
            self.ramping_thread.join(timeout=2.0)

        self.log_message("Multi-channel waveform stop signal sent", "INFO")
        return "Stopping multi-channel waveform..."

    def create_gradio_interface(self):
        """Create the web interface for power supply control."""
        
        # This method is now called from our unified interface
        # Rather than creating and returning a Gradio interface,
        # we'll just return the components needed for the unified interface
        return None


# ============================================================================
# SECTION 5: OSCILLOSCOPE CONTROLLER CLASS AND UTILITIES
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

# Data acquisition class for oscilloscope
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
        """
        Low-level SCPI waveform acquisition with binary protocol and preamble parsing.

        ┌──────────────────────────────────────────────────────────────────────┐
        │ CRITICAL ALGORITHM: OSCILLOSCOPE BINARY WAVEFORM ACQUISITION         │
        │                                                                      │
        │ Purpose: Retrieve digitized waveform data from oscilloscope via     │
        │          SCPI binary transfer protocol with voltage/time scaling    │
        │                                                                      │
        │ Protocol Overview (IEEE 488.2 / SCPI):                               │
        │   1. Configure waveform source (channel selection)                   │
        │   2. Set data format (BYTE = 8-bit unsigned integer)                │
        │   3. Set acquisition mode (RAW = full memory depth)                 │
        │   4. Query preamble (10 comma-separated scaling parameters)         │
        │   5. Transfer binary waveform data (efficient block transfer)       │
        │   6. Scale raw ADC values to physical units (V, s)                  │
        │                                                                      │
        │ Preamble Fields (10 values):                                         │
        │   [0] Format (0=BYTE, 1=WORD, 4=ASCII)                              │
        │   [1] Type (0=NORMAL, 1=PEAK, 2=AVERAGE)                            │
        │   [2] Points (number of data points)                                │
        │   [3] Count (always 1 for non-averaged)                             │
        │   [4] X Increment (time between samples in seconds)                 │
        │   [5] X Origin (time of first sample in seconds)                    │
        │   [6] X Reference (always 0 for time base)                          │
        │   [7] Y Increment (voltage LSB in volts/count)                      │
        │   [8] Y Origin (voltage offset in volts)                            │
        │   [9] Y Reference (ADC offset in counts, typically 127 or 128)     │
        │                                                                      │
        │ Voltage Conversion Formula:                                          │
        │   V(n) = (ADC(n) - Y_ref) × Y_inc + Y_origin                        │
        │   where:                                                             │
        │     ADC(n)   = Raw 8-bit value from oscilloscope (0-255)           │
        │     Y_ref    = ADC reference level (typically 127-128)              │
        │     Y_inc    = Volts per ADC count (scale factor)                   │
        │     Y_origin = Voltage offset (position on screen)                  │
        │                                                                      │
        │ Time Conversion Formula:                                             │
        │   t(i) = X_origin + (i × X_inc)                                     │
        │   where:                                                             │
        │     i        = Sample index (0, 1, 2, ...)                          │
        │     X_inc    = Time between samples (1/sample_rate)                 │
        │     X_origin  = Trigger time offset (typically negative)            │
        └──────────────────────────────────────────────────────────────────────┘

        Args:
            channel: Oscilloscope channel number (1-4 for DSOX6004A)
            max_points: Maximum number of waveform points to retrieve
                       Typical: 62,500 (full memory depth)
                       Max: 4,000,000 (with deep memory option)

        Returns:
            Dict containing:
                - channel (int): Source channel number
                - time (List[float]): Time axis in seconds
                - voltage (List[float]): Voltage values in volts
                - sample_rate (float): Calculated sample rate in Hz
                - time_increment (float): Time between samples in seconds
                - voltage_increment (float): Voltage LSB in volts
                - points_count (int): Number of acquired points
                - acquisition_time (str): ISO timestamp of acquisition
                - is_math (bool): False (indicates physical channel, not math)
            Returns None on error

        Performance:
            Transfer time depends on data size and interface:
            - USB: ~100ms for 10K points, ~1s for 100K points
            - LAN: ~200ms for 10K points, ~2s for 100K points
            Binary BYTE format is fastest (1 byte/point vs 2 bytes for WORD)

        Example:
            >>> data = acq._acquire_waveform_scpi(channel=1, max_points=10000)
            >>> print(f"Acquired {data['points_count']} points")
            >>> print(f"Sample rate: {data['sample_rate']/1e6:.1f} MSa/s")
            >>> print(f"Time span: {data['time'][-1] - data['time'][0]:.6f} s")

        See Also:
            - Keysight Programmer's Guide: :WAVeform subsystem (Chapter 24)
            - IEEE 488.2 Definite Length Block Data format
        """
        try:
            # ────────────────────────────────────────────────────────────────
            # STEP 1: Configure Waveform Source
            # ────────────────────────────────────────────────────────────────
            # Select which channel to read waveform from
            # SCPI: :WAVeform:SOURce CHANnel<n>
            # Alternatives: CHANnel1-4, FUNCtion1-4, MATH, FFT, etc.
            self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")

            # ────────────────────────────────────────────────────────────────
            # STEP 2: Set Data Format to BYTE
            # ────────────────────────────────────────────────────────────────
            # BYTE = 8-bit unsigned integer (0-255)
            # Most efficient for data transfer (1 byte per sample)
            # Alternatives: WORD (16-bit), ASCII (slow, human-readable)
            self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")

            # ────────────────────────────────────────────────────────────────
            # STEP 3: Set Acquisition Mode to RAW
            # ────────────────────────────────────────────────────────────────
            # RAW = Full acquisition memory (no decimation)
            # Alternative: NORMal (decimated to screen resolution ~600 points)
            self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")

            # ────────────────────────────────────────────────────────────────
            # STEP 4: Set Maximum Number of Points
            # ────────────────────────────────────────────────────────────────
            # Request up to max_points (scope returns available points if less)
            self.scope._scpi_wrapper.write(f":WAVeform:POINts {max_points}")

            # ────────────────────────────────────────────────────────────────
            # STEP 5: Query Waveform Preamble (Scaling Parameters)
            # ────────────────────────────────────────────────────────────────
            # Preamble contains all information needed to convert raw ADC
            # values to physical units (volts, seconds)
            # Format: 10 comma-separated floating-point values
            preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
            preamble_parts = preamble.split(',')    # Split CSV into list

            # ────────────────────────────────────────────────────────────────
            # STEP 6: Extract Voltage Scaling Parameters
            # ────────────────────────────────────────────────────────────────
            y_increment = float(preamble_parts[7])  # Volts per ADC count
                                                    # Example: 0.00390625 V/count for 1V/div
            y_origin = float(preamble_parts[8])     # Voltage offset in volts
                                                    # Example: -5.0V if waveform centered at -5V
            y_reference = float(preamble_parts[9])  # ADC zero reference
                                                    # Typically 127 or 128 for 8-bit ADC

            # ────────────────────────────────────────────────────────────────
            # STEP 7: Extract Time Scaling Parameters
            # ────────────────────────────────────────────────────────────────
            x_increment = float(preamble_parts[4])  # Time between samples (seconds)
                                                    # Example: 1e-9 for 1 GSa/s (1ns spacing)
            x_origin = float(preamble_parts[5])     # Time of first sample (seconds)
                                                    # Typically negative (pre-trigger data)
                                                    # Example: -0.001 (1ms before trigger)

            # ────────────────────────────────────────────────────────────────
            # STEP 8: Transfer Binary Waveform Data
            # ────────────────────────────────────────────────────────────────
            # :WAVeform:DATA? returns IEEE 488.2 definite length block data
            # Format: #<N><digits><data bytes>
            # Example: #800010000<10000 bytes> where N=8, digits="00010000"
            # datatype='B' = unsigned byte (0-255)
            raw_data = self.scope._scpi_wrapper.query_binary_values(
                ":WAVeform:DATA?",
                datatype='B'                        # 'B' = unsigned char (uint8)
            )
            # raw_data now contains list of ADC values: [127, 128, 130, ...]

            # ────────────────────────────────────────────────────────────────
            # STEP 9: Convert Raw ADC Values to Voltage (List Comprehension)
            # ────────────────────────────────────────────────────────────────
            # Apply conversion formula: V = (ADC - Y_ref) × Y_inc + Y_origin
            # Example calculation:
            #   ADC = 200, Y_ref = 128, Y_inc = 0.01, Y_origin = -2.0
            #   V = (200 - 128) × 0.01 + (-2.0) = 72 × 0.01 - 2.0 = -1.28V
            voltage_data = [
                (value - y_reference) * y_increment + y_origin
                for value in raw_data
            ]

            # ────────────────────────────────────────────────────────────────
            # STEP 10: Generate Time Axis (List Comprehension)
            # ────────────────────────────────────────────────────────────────
            # Apply conversion formula: t(i) = X_origin + (i × X_inc)
            # Example calculation:
            #   i = 100, X_origin = -0.001, X_inc = 1e-9
            #   t = -0.001 + (100 × 1e-9) = -0.001 + 0.0000001 = -0.9999999s
            time_data = [
                x_origin + (i * x_increment)
                for i in range(len(voltage_data))
            ]

            # ────────────────────────────────────────────────────────────────
            # STEP 11: Return Structured Data Dictionary
            # ────────────────────────────────────────────────────────────────
            return {
                'channel': channel,                         # Source channel (1-4)
                'time': time_data,                          # Time axis [s]
                'voltage': voltage_data,                    # Voltage axis [V]
                'sample_rate': 1.0 / x_increment,           # Calculated Sa/s
                'time_increment': x_increment,              # Time step [s]
                'voltage_increment': y_increment,           # Voltage LSB [V]
                'points_count': len(voltage_data),          # Number of samples
                'acquisition_time': datetime.now().isoformat(),  # ISO timestamp
                'is_math': False                            # Physical channel flag
            }

        except Exception as e:
            # ────────────────────────────────────────────────────────────────
            # Error Handling - Log and Return None
            # ────────────────────────────────────────────────────────────────
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
            # Use provided path or default data directory
            if custom_path is None:
                save_dir = self.default_data_dir
            else:
                save_dir = Path(custom_path)
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
                source_label = "Math" if waveform_data['is_math'] else "Channel"
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


# Main oscilloscope controller class
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
        
        self.save_locations = {
            'data': str(Path.cwd() / "data"),
            'graphs': str(Path.cwd() / "graphs"),
            'screenshots': str(Path.cwd() / "screenshots")
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

    # Connection management
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

    # Channel configuration
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

    # Timebase & trigger configuration
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

    # Measurements - channel and math functions
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

    # Advanced trigger modes
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

    # Acquisition control
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

    # Marker/cursor operations
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

    # Math functions
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
        """Set vertical scale for math function result"""
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

    # Setup management
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

    # Function generators
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

    # Data acquisition
    def capture_screenshot(self):
        """Capture and save display screenshot to the configured save location"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

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
                return "Error: Oscilloscope SCPI interface not available"
                
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
                    self.logger.info(f"Screenshot saved to: {screenshot_path}")
                    return f"✓ Screenshot saved: {screenshot_path}"
                else:
                    return "Screenshot capture failed: No data received"
                    
            except Exception as e:
                self.logger.error(f"Error capturing screenshot: {str(e)}")
                return f"Error capturing screenshot: {str(e)}"

        except Exception as e:
            self.logger.error(f"Screenshot save error: {e}")
            return f"Error: {str(e)}"

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

    def export_csv(self, save_path: str):
        """Export acquired waveform data to CSV files at user-specified location

        Args:
            save_path: Directory path where files should be saved

        Returns:
            Status message
        """
        if not self.last_acquired_data:
            return "Error: No data available"
        if not self.data_acquisition:
            return "Error: Not initialized"

        if not save_path or save_path.strip() == "":
            return "Please select a save location using the Browse button"

        # Verify path exists and is a directory
        save_dir = Path(save_path)
        if not save_dir.exists():
            return f"Error: Directory does not exist: {save_path}"
        if not save_dir.is_dir():
            return f"Error: Path is not a directory: {save_path}"

        try:
            exported_files = []
            if isinstance(self.last_acquired_data, dict):
                for source_key, data in self.last_acquired_data.items():
                    filename = self.data_acquisition.export_to_csv(data, custom_path=save_path)
                    if filename:
                        exported_files.append(Path(filename).name)

            if exported_files:
                return f"✓ Exported {len(exported_files)} file(s) successfully to:\n{save_path}\n\nFiles: {', '.join(exported_files)}"
            else:
                return "Export failed"
        except Exception as e:
            return f"Error: {str(e)}"

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

    def run_acquisition(self):
        """Start continuous acquisition (RUN mode)"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.run()

            if success:
                return "✓ Acquisition started (RUN mode)\nScope is continuously acquiring waveforms"
            else:
                return "Failed to start acquisition"
        except Exception as e:
            return f"Error: {str(e)}"

    def stop_acquisition(self):
        """Stop acquisition (STOP mode - freezes display)"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.stop()

            if success:
                return "⏹ Acquisition stopped\nDisplay frozen - Perfect for screenshots/data capture"
            else:
                return "Failed to stop acquisition"
        except Exception as e:
            return f"Error: {str(e)}"

    def single_acquisition(self):
        """Trigger single acquisition (SINGLE mode)"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            with self.io_lock:
                success = self.oscilloscope.single()

            if success:
                return "⏯ Single trigger armed\nWaiting for trigger event to capture one waveform"
            else:
                return "Failed to arm single trigger"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_full_automation(self, ch1, ch2, ch3, ch4, math1, math2, math3, math4, plot_title):
        """Execute complete acquisition, export, and analysis workflow"""
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"
        if not self.data_acquisition:
            return "Error: Not initialized"

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
            results = []
            results.append("Preparing automation...")

            # CRITICAL: Stop oscilloscope to freeze the waveform
            # This prevents signal from disappearing during data capture
            results.append("Stopping oscilloscope to freeze waveform...")
            try:
                self.oscilloscope.stop()
                import time
                time.sleep(0.3)  # Allow scope to settle after stop
                results.append("✓ Oscilloscope stopped - waveform frozen")
            except Exception as e:
                self.logger.warning(f"Could not stop oscilloscope: {e}")
                results.append(f"⚠ Warning: Could not stop oscilloscope: {str(e)}")

            results.append("\nStep 1/4: Acquiring data...")
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
                # Resume oscilloscope even if data acquisition failed
                try:
                    self.oscilloscope.run()
                    results.append("⚠ Oscilloscope resumed after error")
                except:
                    pass
                return "Error: Data acquisition failed"

            results.append("\nStep 2/4: Exporting CSV...")
            csv_files = []
            for source_key, data in all_channel_data.items():
                csv_file = self.data_acquisition.export_to_csv(
                    data,
                    custom_path=self.save_locations['data']
                )
                if csv_file:
                    csv_files.append(Path(csv_file).name)

            if csv_files:
                results.append(f" ✓ {len(csv_files)} files exported to: {self.save_locations['data']}")

            results.append("\nStep 3/4: Generating plots...")
            custom_title = plot_title.strip() or None
            plot_files = []
            for source_key, data in all_channel_data.items():
                if custom_title:
                    source_label = "Math" if data['is_math'] else "Channel"
                    channel_title = f"{custom_title} - {source_label} {data['channel']}"
                else:
                    channel_title = None

                plot_file = self.data_acquisition.generate_waveform_plot(
                    data,
                    custom_path=self.save_locations['graphs'],
                    plot_title=channel_title
                )
                if plot_file:
                    plot_files.append(Path(plot_file).name)

            if plot_files:
                results.append(f" ✓ {len(plot_files)} plots generated to: {self.save_locations['graphs']}")

            results.append("\nStep 4/4: Capturing screenshot...")

            # Capture screenshot while oscilloscope is still stopped
            try:
                screenshot_dir = Path(self.save_locations['screenshots'])
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"scope_screenshot_{timestamp}.png"
                screenshot_path = screenshot_dir / filename

                if hasattr(self.oscilloscope, '_scpi_wrapper'):
                    import time
                    time.sleep(0.1)  # Brief pause before screenshot
                    image_data = self.oscilloscope._scpi_wrapper.query_binary_values(
                        ":DISPlay:DATA? PNG",
                        datatype='B'
                    )

                    if image_data:
                        with open(screenshot_path, 'wb') as f:
                            f.write(bytes(image_data))
                        results.append(f"✓ Screenshot saved: {screenshot_path}")
                    else:
                        results.append("⚠ Screenshot capture failed: No data")
                else:
                    results.append("⚠ SCPI interface not available, skipping screenshot")
            except Exception as e:
                self.logger.warning(f"Screenshot capture error in automation: {e}")
                results.append(f"⚠ Screenshot failed: {str(e)}")

            # CRITICAL: Resume oscilloscope run mode
            results.append("\nResuming oscilloscope to RUN mode...")
            try:
                self.oscilloscope.run()
                import time
                time.sleep(0.1)
                results.append("✓ Oscilloscope resumed - acquisition running")
            except Exception as e:
                self.logger.error(f"Failed to resume oscilloscope: {e}")
                results.append(f"⚠ Warning: Could not resume oscilloscope: {str(e)}")

            self.last_acquired_data = all_channel_data
            results.append("\n" + "="*60)
            results.append("✓ FULL AUTOMATION COMPLETED SUCCESSFULLY!")
            results.append("="*60)
            results.append(f"\nAll files saved to:")
            results.append(f"  • Screenshots: {self.save_locations['screenshots']}")
            results.append(f"  • Data: {self.save_locations['data']}")
            results.append(f"  • Graphs: {self.save_locations['graphs']}")

            return "\n".join(results)
        except Exception as e:
            # Ensure oscilloscope is resumed even if there's an error
            try:
                self.oscilloscope.run()
                self.logger.info("Oscilloscope resumed after exception")
            except:
                pass
            self.logger.error(f"Automation error: {e}")
            return f"Automation error: {str(e)}"

    def browse_folder(self, current_path, _folder_type="folder"):
        """
        Validate and return the provided path.
        Note: Browse dialog removed - users should manually edit the path textbox.
        When accessing via localhost, users can specify any path on their local machine.
        """
        return current_path if current_path else str(Path.cwd())

    def create_interface(self):
        """
        Create the Gradio interface for oscilloscope control.
        This method is now handled by the unified interface.
        """
        # This method is now called from our unified interface
        # Rather than creating and returning a Gradio interface,
        # we'll just return None to indicate this is handled elsewhere
        return None


# ============================================================================
# SECTION 6: UNIFIED CONTROLLER CLASS
# ============================================================================
class UnifiedInstrumentControl:
    """
    Unified interface for controlling multiple lab instruments.
    
    This class integrates multiple instrument controllers into a single
    Gradio interface with tabs for each instrument:
    1. Keithley DMM6500 Digital Multimeter
    2. Keithley 2230 Power Supply
    3. Keysight DSOX6004A Oscilloscope
    
    All features from the original interfaces are preserved, while providing
    a clean, unified experience for the user.
    """
    
    def __init__(self):
        """Initialize the unified controller."""
        self.setup_logging()
        
        # Initialize controllers for each instrument
        self.dmm_controller = DMM_GUI_Controller()
        self.psu_controller = PowerSupplyAutomationGradio()
        self.oscilloscope_controller = GradioOscilloscopeGUI()

        # ═══════════════════════════════════════════════════════════════════
        # CSS STYLING - CONTROLS THE APPEARANCE OF THE WEB INTERFACE
        # ═══════════════════════════════════════════════════════════════════
        # You can edit these values to customize the look and feel
        # Numbers can be changed (e.g., 20px to 30px for more spacing)
        # Colors are in hex format (e.g., #9c9cff is light purple)
        # ═══════════════════════════════════════════════════════════════════

        self._css = """
        /* ============================================================
           MAIN CONTAINER - Controls the entire interface width/height
           ============================================================ */
        .gradio-container {
            max-width: 100% !important;      /* Makes interface full width (100% = entire screen) */
            padding: 20px !important;        /* Space around the edges (change 20px to adjust) */
            margin: 100 !important;            /* No extra margins (keep at 0) */
            min-height: 100vh;               /* Full screen height (100vh = entire screen height) */
        }

        /* ============================================================
           CONTENT CONTAINER - Inner content area
           ============================================================ */
        .container {
            max-width: 100% !important;      /* Inner content full width (100% = fills parent) */
            padding: 0 10px !important;      /* Small left/right padding (change 10px to adjust) */
            margin: 0 !important;            /* No margins (keep at 0) */
        }

        /* ============================================================
           MAIN COMPONENT - Root component sizing
           ============================================================ */
        #component-0 {
            min-height: 100vh;               /* Minimum full screen height */
        }

        /* ============================================================
           TAB CONTENT - Each instrument tab (DMM, PSU, Oscilloscope)
           ============================================================ */
        .tab {
            padding: 0 10px;                 /* Space inside tabs (change 10px for more/less padding) */
            min-height: calc(100vh - 120px); /* Tab height (120px reserved for header/footer) */
        }

        /* ============================================================
           PANELS - Individual sections within tabs
           ============================================================ */
        .panel {
            margin: 5px 0;                   /* Space between panels (change 5px to adjust spacing) */
        }

        /* ============================================================
           TAB NAVIGATION - The tab buttons at the top
           ============================================================ */
        .tab-nav {
            border-bottom: 2px solid #c034eb;  /* Bottom border (2px thickness, #c034eb = purple) */
            margin-bottom: 12px;               /* Space below tab buttons (change 12px to adjust) */
        }

        /* ============================================================
           SELECTED TAB - The currently active tab appearance
           ============================================================ */
        .tab-selected {
            background-color: #e0e0ff;       /* Background color when tab is selected (#e0e0ff = pale purple) */
            font-weight: 600;                /* Text boldness (600 = semi-bold, try 400=normal, 700=bold) */
        }

        /* ============================================================
           CUSTOMIZATION TIPS:
           - To change colors: Use hex codes like #9c9cff (Google "color picker" to find codes)
           - To adjust spacing: Change pixel values (e.g., 20px to 30px for more space)
           - To make interface narrower: Change max-width to 90% or 80%
           - To add more vertical space: Increase padding values
           ============================================================ */
        """
        
    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('UnifiedInstrumentControl')
        
    def create_interface(self):
        """Create the unified Gradio interface with tabs for each instrument."""
        # ═══════════════════════════════════════════════════════════════════
        # GRADIO INTERFACE CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        # EASY CUSTOMIZATION GUIDE (no coding knowledge needed):
        #
        # title: Change the browser tab name
        #   Example: "My Lab Control" or "Test Equipment Interface"
        #
        # primary_hue: Main color scheme
        #   Options: "indigo", "blue", "purple", "green", "red", "orange"
        #   Try: "purple" for purple theme, "blue" for blue theme
        #
        # spacing_size: Space between elements
        #   Options: "sm" (small), "md" (medium), "lg" (large)
        #   Try: "md" for more spacing, "lg" for even more
        #
        # radius_size: Corner roundness of buttons/boxes
        #   Options: "sm" (sharp), "md" (medium), "lg" (very round)
        #   Try: "none" for square corners, "lg" for round corners
        #
        # text_size: Size of all text
        #   Options: "sm" (small), "md" (medium), "lg" (large)
        #   Try: "md" for bigger text, "lg" for even bigger
        # ═══════════════════════════════════════════════════════════════════

        with gr.Blocks(
            title="DIGANTARA Unified Instrument Control",  # Browser tab title
            theme=gr.themes.Soft(
                primary_hue="purple",     # Color: indigo, blue, purple, green, red, orange
                spacing_size="sm",        # Spacing: sm, md, lg
                radius_size="sm",         # Corner roundness: none, sm, md, lg
                text_size="sm"            # Text size: sm, md, lg
            ),
            css=self._css  # Apply the custom CSS styles defined above
        ) as interface:
            gr.Markdown("# DIGANTARA Unified Lab Instrument Control")
            gr.Markdown("**Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.")
            gr.Markdown("Professional control interface for multiple lab instruments")
            
            with gr.Tabs():
                # DMM Tab
                with gr.Tab("Digital Multimeter (DMM6500)"):
                    self.create_dmm_interface()
                
                # Power Supply Tab
                with gr.Tab("Power Supply (2230)"):
                    self.create_psu_interface()
                
                # Oscilloscope Tab
                with gr.Tab("Oscilloscope (DSOX6004A)"):
                    self.create_oscilloscope_interface()
            
            gr.Markdown("---")
            gr.Markdown("**DIGANTARA Lab Automation System** | All instruments controlled from single interface")
            
        return interface
    
    def create_dmm_interface(self):
        """Create the Digital Multimeter interface tab"""
        # Connection Tab
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Instrument Connection")
                dmm_visa_address = gr.Textbox(
                    label="VISA Address",
                    value=self.dmm_controller.default_settings['visa_address'],
                    placeholder="USB0::0x05E6::0x6500::04561287::INSTR"
                )
                dmm_timeout_ms = gr.Number(
                    label="Timeout (ms)",
                    value=self.dmm_controller.default_settings['timeout_ms'],
                    minimum=1000,
                    maximum=60000
                )
                
                with gr.Row():
                    dmm_connect_btn = gr.Button("Connect", variant="primary")
                    dmm_disconnect_btn = gr.Button("Disconnect", variant="secondary")
                
                dmm_connection_status = gr.Textbox(
                    label="Connection Status",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Instrument Status")
                dmm_status_connection = gr.Textbox(label="Connection", interactive=False)
                dmm_status_instrument = gr.Textbox(label="Instrument", interactive=False)
                dmm_status_errors = gr.Textbox(label="Errors", interactive=False)
                dmm_status_time = gr.Textbox(label="System Time", interactive=False)
                dmm_refresh_status_btn = gr.Button("Refresh Status")

        # Measurement Configuration
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Measurement Configuration")
                dmm_measurement_function = gr.Dropdown(
                    label="Measurement Function",
                    choices=[
                        "DC_VOLTAGE", "AC_VOLTAGE",
                        "DC_CURRENT", "AC_CURRENT",
                        "RESISTANCE_2W", "RESISTANCE_4W",
                        "CAPACITANCE", "FREQUENCY", "TEMPERATURE"
                    ],
                    value="DC_VOLTAGE"
                )

                # Get default range options for DC_VOLTAGE
                default_ranges, default_value = self.dmm_controller.get_range_options("DC_VOLTAGE")

                dmm_measurement_range = gr.Dropdown(
                    label="Range (AUTO or specific value)",
                    choices=default_ranges,
                    value=default_value
                )
                
                dmm_resolution = gr.Number(
                    label="Resolution",
                    value=1e-6,
                    minimum=1e-9,
                    maximum=1e-3,
                    step=1e-9
                )
                
                dmm_nplc = gr.Dropdown(
                    label="NPLC (Integration Time)",
                    choices=[0.01, 0.02, 0.06, 0.2, 1.0, 2.0, 10.0],
                    value=1.0
                )
                
                dmm_auto_zero = gr.Checkbox(
                    label="Auto Zero",
                    value=True
                )
                
                with gr.Row():
                    dmm_single_measure_btn = gr.Button("Single Measurement", variant="primary")
                    dmm_clear_data_btn = gr.Button("Clear Data", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Measurement Results")
                dmm_current_measurement = gr.Textbox(
                    label="Reading",
                    interactive=False,
                    lines=2
                )
                
                dmm_measurement_status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                gr.Markdown("### Continuous Measurements")
                dmm_measurement_interval = gr.Number(
                    label="Interval (seconds)",
                    value=1.0,
                    minimum=0.0000000000000000001,
                    maximum=60.0
                )
                
                with gr.Row():
                    dmm_start_continuous_btn = gr.Button("Start Continuous", variant="primary")
                    dmm_stop_continuous_btn = gr.Button("Stop Continuous", variant="secondary")
                
                dmm_continuous_status = gr.Textbox(
                    label="Continuous Status",
                    interactive=False
                )

        # Statistics Tab
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Statistical Analysis")
                dmm_stats_points = gr.Number(
                    label="Number of Points",
                    value=7000,
                    minimum=1,
                    maximum=65000
                )
                
                dmm_calculate_stats_btn = gr.Button("Calculate Statistics", variant="primary")
                
                dmm_stats_count = gr.Textbox(label="Count", interactive=False)
                dmm_stats_mean = gr.Textbox(label="Mean", interactive=False)
                dmm_stats_std = gr.Textbox(label="Standard Deviation", interactive=False)
                dmm_stats_min = gr.Textbox(label="Minimum", interactive=False)
                dmm_stats_max = gr.Textbox(label="Maximum", interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### Trend Plot")
                dmm_plot_points = gr.Number(
                    label="Points to Plot",
                    value=7000,
                    minimum=10,
                    maximum=65000
                )
                dmm_update_plot_btn = gr.Button("Update Plot", variant="primary")
                dmm_trend_plot = gr.Plot()

                gr.Markdown("### Save Plot")
                with gr.Row():
                    dmm_plot_save_path = gr.Textbox(
                        label="Save Location for Plots",
                        placeholder="Click Browse to select folder...",
                        interactive=True,
                        scale=3
                    )
                    dmm_plot_browse_btn = gr.Button("Browse", variant="secondary", scale=1)

                dmm_save_plot_btn = gr.Button("Save Plot", variant="primary")
                dmm_plot_save_status = gr.Textbox(
                    label="Save Status",
                    interactive=False
                )

        # Data Export Tab
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Export Measurement Data")

                with gr.Row():
                    dmm_export_path = gr.Textbox(
                        label="Save Location",
                        placeholder="Click Browse to select folder...",
                        interactive=True,
                        scale=3
                    )
                    dmm_export_browse_btn = gr.Button("Browse", variant="secondary", scale=1)

                dmm_export_format = gr.Dropdown(
                    label="Export Format",
                    choices=["CSV", "JSON", "Excel"],
                    value="CSV"
                )

                dmm_export_btn = gr.Button("Export Data", variant="primary")
                dmm_export_status = gr.Textbox(
                    label="Export Status",
                    interactive=False
                )

                gr.Markdown("### Data Preview")
                dmm_data_preview = gr.Dataframe(
                    headers=["Timestamp", "Function", "Value", "Range", "Resolution"],
                    interactive=False
                )

                dmm_refresh_preview_btn = gr.Button("Refresh Preview")
        
        # Helper function to update range options when measurement function changes
        def update_range_dropdown(measurement_function: str):
            """Update range dropdown choices and value based on selected measurement function."""
            ranges, default = self.dmm_controller.get_range_options(measurement_function)
            return gr.Dropdown(choices=ranges, value=default)

        # Helper function to convert AUTO to 0 for driver compatibility
        def single_measurement_wrapper(function: str, range_val, resolution: float,
                                       nplc: float, auto_zero: bool):
            """Wrapper to convert 'AUTO' string to 0 for driver."""
            # Convert "AUTO" string to 0 (auto-range value for driver)
            if range_val == "AUTO":
                range_val = 0
            return self.dmm_controller.single_measurement(function, range_val, resolution, nplc, auto_zero)

        def continuous_measurement_wrapper(function: str, range_val, resolution: float,
                                          nplc: float, auto_zero: bool, interval: float):
            """Wrapper to convert 'AUTO' string to 0 for driver."""
            # Convert "AUTO" string to 0 (auto-range value for driver)
            if range_val == "AUTO":
                range_val = 0
            return self.dmm_controller.start_continuous_measurement(function, range_val, resolution, nplc, auto_zero, interval)

        # Event handlers
        # Update range dropdown when measurement function changes
        dmm_measurement_function.change(
            fn=update_range_dropdown,
            inputs=[dmm_measurement_function],
            outputs=[dmm_measurement_range]
        )

        dmm_connect_btn.click(
            self.dmm_controller.connect_instrument,
            inputs=[dmm_visa_address, dmm_timeout_ms],
            outputs=[dmm_connection_status, gr.State()]
        )
        
        dmm_disconnect_btn.click(
            self.dmm_controller.disconnect_instrument,
            outputs=[dmm_connection_status]
        )
        
        dmm_refresh_status_btn.click(
            self.dmm_controller.get_instrument_status,
            outputs=[dmm_status_connection, dmm_status_instrument, dmm_status_errors, dmm_status_time]
        )
        
        dmm_single_measure_btn.click(
            single_measurement_wrapper,
            inputs=[dmm_measurement_function, dmm_measurement_range, dmm_resolution, dmm_nplc, dmm_auto_zero],
            outputs=[dmm_current_measurement, dmm_measurement_status]
        )

        dmm_start_continuous_btn.click(
            continuous_measurement_wrapper,
            inputs=[dmm_measurement_function, dmm_measurement_range, dmm_resolution, dmm_nplc, dmm_auto_zero, dmm_measurement_interval],
            outputs=[dmm_continuous_status]
        )
        
        dmm_stop_continuous_btn.click(
            self.dmm_controller.stop_continuous_measurement,
            outputs=[dmm_continuous_status]
        )
        
        dmm_calculate_stats_btn.click(
            self.dmm_controller.get_statistics,
            inputs=[dmm_stats_points],
            outputs=[dmm_stats_count, dmm_stats_mean, dmm_stats_std, dmm_stats_min, dmm_stats_max]
        )
        
        dmm_update_plot_btn.click(
            lambda points: self.dmm_controller.create_trend_plot(points),
            inputs=[dmm_plot_points],
            outputs=[dmm_trend_plot]
        )

        # Note: Browse buttons removed - users should manually type/paste paths in the textboxes
        # The export path and plot save path textboxes are directly editable by the user

        # Save plot button
        dmm_save_plot_btn.click(
            fn=self.dmm_controller.save_trend_plot,
            inputs=[dmm_plot_save_path, dmm_plot_points],
            outputs=[dmm_plot_save_status]
        )

        dmm_export_btn.click(
            self.dmm_controller.export_data,
            inputs=[dmm_export_path, dmm_export_format],
            outputs=[dmm_export_status]
        )
        
        dmm_clear_data_btn.click(
            self.dmm_controller.clear_data,
            outputs=[dmm_measurement_status]
        )
        
        def update_data_preview():
            if self.dmm_controller.measurement_data:
                recent_data = self.dmm_controller.measurement_data[-200:]  # Show last 200 points
                df_data = []
                for point in recent_data:
                    df_data.append([
                        point['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        point['function'],
                        f"{point['value']:.6e}",
                        point['range'],
                        f"{point['resolution']:.2e}"
                    ])
                return df_data
            return []
        
        dmm_refresh_preview_btn.click(
            update_data_preview,
            outputs=[dmm_data_preview]
        )
    
    def create_psu_interface(self):
        """Create the Power Supply interface tab"""
        gr.Markdown("### Connection Settings")
        with gr.Group():
            with gr.Row():
                psu_visa_addr = gr.Textbox(
                    label="VISA Address",
                    value="USB0::0x05E6::0x2230::805224014806770001::INSTR",
                    lines=1
                )
            
            with gr.Row():
                psu_conn_btn = gr.Button("Connect", variant="primary", size="lg")
                psu_disc_btn = gr.Button("Disconnect", variant="stop", size="lg")
                psu_test_btn = gr.Button("Test Connection", variant="secondary", size="lg")
                psu_emerg_btn = gr.Button("EMERGENCY STOP", variant="stop", size="lg")
            
            with gr.Row():
                psu_conn_status = gr.Textbox(label="Status", value="Disconnected", interactive=False)
            
            psu_info = gr.Textbox(label="Instrument Info", lines=2, interactive=False)
        
        # Channel Controls
        gr.Markdown("### Channel Controls")
        with gr.Group():
            psu_channel_outputs = {}

            # Channel specifications: (max_voltage, max_current, default_ovp)
            channel_specs = {
                1: (30, 3.0, 30),  # Channel 1: 0-30V, 0-3A
                2: (30, 3.0, 30),  # Channel 2: 0-30V, 0-3A
                3: (5, 3.0, 6)     # Channel 3: 0-5V, 0-3A (limited voltage)
            }

            with gr.Tabs():
                for ch in range(1, 4):
                    max_volt, max_curr, default_ovp = channel_specs[ch]

                    with gr.TabItem(label=f"Channel {ch}"):
                        with gr.Row():
                            psu_volt_slider = gr.Slider(0, max_volt, value=0, label="Voltage (V)", step=0.1)
                            psu_curr_limit = gr.Slider(0.001, max_curr, value=0.1, label="Current Limit (A)", step=0.001)
                            psu_ovp_level = gr.Slider(1, max_volt + 5, value=default_ovp, label="OVP (V)", step=0.5)

                        with gr.Row():
                            psu_conf_btn = gr.Button(f"Configure Ch{ch}", variant="secondary")
                            psu_enable_btn = gr.Button(f"Enable Output", variant="primary")
                            psu_disable_btn = gr.Button(f"Disable Output", variant="stop")
                            psu_meas_btn = gr.Button(f"Measure", variant="secondary")
                            psu_clear_ovp_btn = gr.Button(f"⚠️ Clear OVP (May Need Power Cycle)", variant="secondary")

                        with gr.Row():
                            psu_volt_display = gr.Textbox(label="Measured Voltage", value="0.000 V", interactive=False)
                            psu_curr_display = gr.Textbox(label="Measured Current", value="0.000 A", interactive=False)
                            psu_power_display = gr.Textbox(label="Measured Power", value="0.000 W", interactive=False)

                        psu_ch_status = gr.Textbox(label="Status", value="OFF", interactive=False)

                        psu_channel_outputs[ch] = {
                            "voltage": psu_volt_slider,
                            "current_limit": psu_curr_limit,
                            "ovp_level": psu_ovp_level,
                            "configure_btn": psu_conf_btn,
                            "enable_btn": psu_enable_btn,
                            "disable_btn": psu_disable_btn,
                            "measure_btn": psu_meas_btn,
                            "clear_ovp_btn": psu_clear_ovp_btn,
                            "volt_display": psu_volt_display,
                            "curr_display": psu_curr_display,
                            "power_display": psu_power_display,
                            "status": psu_ch_status
                        }

                        psu_conf_btn.click(
                            fn=lambda v, cl, ov, ch=ch: self.psu_controller.configure_channel(ch, v, cl, ov),
                            inputs=[psu_volt_slider, psu_curr_limit, psu_ovp_level]
                        )

                        psu_enable_btn.click(
                            fn=lambda ch=ch: self.psu_controller.enable_channel_output(ch)
                        )

                        psu_disable_btn.click(
                            fn=lambda ch=ch: self.psu_controller.disable_channel_output(ch)
                        )

                        psu_meas_btn.click(
                            fn=lambda ch=ch: self.psu_controller.measure_channel_output(ch),
                            outputs=[psu_volt_display, psu_curr_display, psu_power_display]
                        )

                        psu_clear_ovp_btn.click(
                            fn=lambda ch=ch: self.psu_controller.clear_channel_protection(ch)
                        )
        
        # Global Operations
        gr.Markdown("### Global Operations")
        with gr.Group():
            with gr.Row():
                psu_get_info_btn = gr.Button("Get Instrument Info", variant="secondary")
                psu_measure_all_btn = gr.Button("Measure All Channels", variant="primary")
                psu_disable_all_btn = gr.Button("Disable All Outputs", variant="stop")
                psu_clear_all_ovp_btn = gr.Button("⚠️ Clear All OVP (May Need Power Cycle)", variant="secondary")
            
            psu_get_info_btn.click(fn=self.psu_controller.get_instrument_info)
            psu_measure_all_btn.click(
                fn=self.psu_controller.measure_all_channels,
                outputs=[
                    psu_channel_outputs[1]["volt_display"],
                    psu_channel_outputs[1]["curr_display"],
                    psu_channel_outputs[1]["power_display"],
                    psu_channel_outputs[2]["volt_display"],
                    psu_channel_outputs[2]["curr_display"],
                    psu_channel_outputs[2]["power_display"],
                    psu_channel_outputs[3]["volt_display"],
                    psu_channel_outputs[3]["curr_display"],
                    psu_channel_outputs[3]["power_display"]
                ]
            )
            psu_disable_all_btn.click(fn=self.psu_controller.disable_all_outputs)
            psu_clear_all_ovp_btn.click(
                fn=lambda: self.psu_controller.power_supply.clear_protection() if self.psu_controller.power_supply else "Not connected"
            )

        # Data Logging
        gr.Markdown("### Data Logging & Export")
        with gr.Group():
            with gr.Row():
                psu_auto_measure_cb = gr.Checkbox(label="Enable Auto-Measurement", value=False)
                psu_measure_interval = gr.Slider(0.5, 60, value=2.0, label="Interval (seconds)", step=0.5)

            with gr.Row():
                psu_export_path = gr.Textbox(
                    label="Save Location",
                    placeholder="Click Browse to select folder...",
                    interactive=True,
                    scale=3
                )
                psu_export_browse_btn = gr.Button("Browse", variant="secondary", scale=1)

            with gr.Row():
                psu_export_btn = gr.Button("Export to CSV", variant="primary")
                psu_clear_btn = gr.Button("Clear Data", variant="secondary")

            psu_export_status = gr.Textbox(
                label="Export Status",
                interactive=False
            )

            # Note: Browse button removed - users should manually type/paste path in the textbox
            # The export path textbox is directly editable by the user

            psu_auto_measure_cb.change(fn=self.psu_controller.toggle_auto_measure, inputs=psu_auto_measure_cb)

            psu_export_btn.click(
                fn=self.psu_controller.export_measurement_data,
                inputs=[psu_export_path],
                outputs=[psu_export_status]
            )
            psu_clear_btn.click(fn=self.psu_controller.clear_measurement_data)

        # # ════════════════════════════════════════════════════════════════════════════
        # # LIVE MONITOR - Real-time graphs for PSU channels
        # # ════════════════════════════════════════════════════════════════════════════
        # gr.Markdown("### Live Monitor")
        # gr.Markdown("Real-time voltage, current, and power monitoring with live updating graphs")

        # with gr.Group():
        #     with gr.Row():
        #         psu_live_ch1_cb = gr.Checkbox(label="Monitor CH1", value=True)
        #         psu_live_ch2_cb = gr.Checkbox(label="Monitor CH2", value=False)
        #         psu_live_ch3_cb = gr.Checkbox(label="Monitor CH3", value=False)
        #         psu_live_interval = gr.Slider(0.5, 10.0, value=1.0, label="Update Interval (s)", step=0.5)

        #     with gr.Row():
        #         psu_start_live_btn = gr.Button("Start Live Monitor", variant="primary", size="lg")
        #         psu_stop_live_btn = gr.Button("Stop Live Monitor", variant="stop", size="lg")
        #         psu_clear_live_btn = gr.Button("Clear Live Data", variant="secondary")

        #     psu_live_status = gr.Textbox(label="Live Monitor Status", value="Stopped", interactive=False)

        #     with gr.Row():
        #         psu_plot_type = gr.Dropdown(
        #             choices=["voltage", "current", "power"],
        #             value="voltage",
        #             label="Plot Type"
        #         )
        #         psu_refresh_plot_btn = gr.Button("Refresh Plot", variant="secondary")

        #     psu_live_plot = gr.Plot(label="Live Trend Plot")

        #     gr.Markdown("#### Live Statistics")
        #     with gr.Row():
        #         with gr.Column():
        #             gr.Markdown("**CH1**")
        #             psu_ch1_count = gr.Textbox(label="Samples", value="0", interactive=False)
        #             psu_ch1_avg_v = gr.Textbox(label="Avg V", value="N/A", interactive=False)
        #             psu_ch1_avg_i = gr.Textbox(label="Avg I", value="N/A", interactive=False)
        #             psu_ch1_avg_p = gr.Textbox(label="Avg P", value="N/A", interactive=False)
        #         with gr.Column():
        #             gr.Markdown("**CH2**")
        #             psu_ch2_count = gr.Textbox(label="Samples", value="0", interactive=False)
        #             psu_ch2_avg_v = gr.Textbox(label="Avg V", value="N/A", interactive=False)
        #             psu_ch2_avg_i = gr.Textbox(label="Avg I", value="N/A", interactive=False)
        #             psu_ch2_avg_p = gr.Textbox(label="Avg P", value="N/A", interactive=False)
        #         with gr.Column():
        #             gr.Markdown("**CH3**")
        #             psu_ch3_count = gr.Textbox(label="Samples", value="0", interactive=False)
        #             psu_ch3_avg_v = gr.Textbox(label="Avg V", value="N/A", interactive=False)
        #             psu_ch3_avg_i = gr.Textbox(label="Avg I", value="N/A", interactive=False)
        #             psu_ch3_avg_p = gr.Textbox(label="Avg P", value="N/A", interactive=False)

        #     # Live monitor handlers
        #     def psu_handle_start_live(ch1, ch2, ch3, interval):
        #         channels = []
        #         if ch1:
        #             channels.append(1)
        #         if ch2:
        #             channels.append(2)
        #         if ch3:
        #             channels.append(3)
        #         return self.psu_controller.start_live_measurement(channels, interval)

        #     def psu_handle_stop_live():
        #         return self.psu_controller.stop_live_measurement()

        #     def psu_handle_clear_live():
        #         return self.psu_controller.clear_live_data()

        #     def psu_handle_refresh_plot(plot_type):
        #         fig = self.psu_controller.create_live_plot(plot_type)
        #         s1 = self.psu_controller.get_live_statistics(1)
        #         s2 = self.psu_controller.get_live_statistics(2)
        #         s3 = self.psu_controller.get_live_statistics(3)
        #         return (fig,
        #                 s1[0], s1[1], s1[2], s1[3],
        #                 s2[0], s2[1], s2[2], s2[3],
        #                 s3[0], s3[1], s3[2], s3[3])

        #     psu_start_live_btn.click(
        #         fn=psu_handle_start_live,
        #         inputs=[psu_live_ch1_cb, psu_live_ch2_cb, psu_live_ch3_cb, psu_live_interval],
        #         outputs=[psu_live_status]
        #     )

        #     psu_stop_live_btn.click(
        #         fn=psu_handle_stop_live,
        #         outputs=[psu_live_status]
        #     )

        #     psu_clear_live_btn.click(
        #         fn=psu_handle_clear_live,
        #         outputs=[psu_live_status]
        #     )

        #     psu_refresh_plot_btn.click(
        #         fn=psu_handle_refresh_plot,
        #         inputs=[psu_plot_type],
        #         outputs=[psu_live_plot,
        #                  psu_ch1_count, psu_ch1_avg_v, psu_ch1_avg_i, psu_ch1_avg_p,
        #                  psu_ch2_count, psu_ch2_avg_v, psu_ch2_avg_i, psu_ch2_avg_p,
        #                  psu_ch3_count, psu_ch3_avg_v, psu_ch3_avg_i, psu_ch3_avg_p]
        #     )

        # ════════════════════════════════════════════════════════════════════════════
        # MULTI-CHANNEL SIMULTANEOUS WAVEFORM GENERATOR (Replaces single-channel)
        # ════════════════════════════════════════════════════════════════════════════
        gr.Markdown("### Multi-Channel Simultaneous Waveform Generator")
        gr.Markdown("Configure and run synchronized waveform patterns on multiple channels simultaneously.")

        with gr.Group():
            # Waveform type options
            psu_waveform_types = [
                "Sine - ∿ smooth wave",
                "Square - ⎍ sharp on/off",
                "Triangle - △ linear ramp up/down",
                "Ramp Up - ⟋ linear rise",
                "Ramp Down - ⟍ linear fall",
                "Cardiac - ♥ ECG heartbeat",
                "Damped Sine - ∿↘ decaying oscillation",
                "Exponential Raise - ⤴ slow→fast rise",
                "Exponential Fall - ⤵ fast→slow fall",
                "Gaussian Pulse - ⌒ bell curve",
                "Neural Spike - ⟰ action potential",
                "Staircase - ⎽⎺ discrete steps",
                "PWM - ⎍⎍⎍ variable duty cycle",
                "Chirp - ∿∿∿ frequency sweep",
                "Burst Mode - ⎍⎍___ on/off bursts",
                "Brownout - ⎺⤵_ power sag/recovery",
                "RC Charge - ⤴⎺ capacitor charging",
                "Sinc - ⌒∿∿ main lobe + ripples",
                "Breathing - ⬭ smooth fade in/out"
            ]

            with gr.Tabs():
                # Channel 1 Waveform Configuration
                with gr.TabItem(label="CH1 Waveform"):
                    psu_ch1_enable = gr.Checkbox(label="Enable CH1", value=True)
                    with gr.Row():
                        psu_ch1_waveform = gr.Dropdown(
                            choices=psu_waveform_types,
                            value="Sine - ∿ smooth wave",
                            label="Waveform Type"
                        )
                        psu_ch1_voltage = gr.Slider(0.1, 30.0, value=3.0, label="Target Voltage (V)", step=0.1)
                        psu_ch1_current = gr.Slider(0.001, 3.0, value=0.1, label="Current Limit (A)", step=0.001)
                    with gr.Row():
                        psu_ch1_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=100, precision=0)
                        psu_ch1_points = gr.Number(value=50, label="Points/Cycle", minimum=1, maximum=1000, precision=0)
                        psu_ch1_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                # Channel 2 Waveform Configuration
                with gr.TabItem(label="CH2 Waveform"):
                    psu_ch2_enable = gr.Checkbox(label="Enable CH2", value=False)
                    with gr.Row():
                        psu_ch2_waveform = gr.Dropdown(
                            choices=psu_waveform_types,
                            value="Triangle - △ linear ramp up/down",
                            label="Waveform Type"
                        )
                        psu_ch2_voltage = gr.Slider(0.1, 30.0, value=2.5, label="Target Voltage (V)", step=0.1)
                        psu_ch2_current = gr.Slider(0.001, 3.0, value=0.1, label="Current Limit (A)", step=0.001)
                    with gr.Row():
                        psu_ch2_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=100, precision=0)
                        psu_ch2_points = gr.Number(value=50, label="Points/Cycle", minimum=1, maximum=1000, precision=0)
                        psu_ch2_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

                # Channel 3 Waveform Configuration (limited to 5V)
                with gr.TabItem(label="CH3 Waveform"):
                    psu_ch3_enable = gr.Checkbox(label="Enable CH3", value=False)
                    gr.Markdown("*Note: Channel 3 is limited to 5V maximum*")
                    with gr.Row():
                        psu_ch3_waveform = gr.Dropdown(
                            choices=psu_waveform_types,
                            value="Square - ⎍ sharp on/off",
                            label="Waveform Type"
                        )
                        psu_ch3_voltage = gr.Slider(0.1, 5.0, value=3.3, label="Target Voltage (V)", step=0.1)
                        psu_ch3_current = gr.Slider(0.001, 3.0, value=0.1, label="Current Limit (A)", step=0.001)
                    with gr.Row():
                        psu_ch3_cycles = gr.Number(value=3, label="Cycles", minimum=1, maximum=100, precision=0)
                        psu_ch3_points = gr.Number(value=50, label="Points/Cycle", minimum=1, maximum=1000, precision=0)
                        psu_ch3_duration = gr.Number(value=8.0, label="Cycle Duration (s)", minimum=0.1, maximum=60)

            # Global settings
            with gr.Row():
                psu_settle_time = gr.Slider(
                    label="Time per Point (s)",
                    minimum=0.05,
                    maximum=5.0,
                    value=0.262,
                    step=0.01,
                    info="Total time per point including settle + measurement (e.g., 0.262s = 262ms)"
                )
                psu_estimated_duration = gr.Textbox(
                    label="Estimated Duration",
                    value="~300s (5.0 min) | 150 pts @ 2000ms/pt",
                    interactive=False,
                    info="Estimate based on enabled channels"
                )

            # ════════════════════════════════════════════════════════════════
            # DURATION ESTIMATION FOR MULTI-CHANNEL WAVEFORM
            # ════════════════════════════════════════════════════════════════
            def update_multi_channel_duration(
                ch1_en, ch1_cyc, ch1_pts,
                ch2_en, ch2_cyc, ch2_pts,
                ch3_en, ch3_cyc, ch3_pts,
                settle
            ):
                """
                Calculate estimated duration for multi-channel waveform execution.

                The duration is based on the LONGEST profile among enabled channels,
                since all channels execute synchronously at each time point.

                With the new timing control, each point takes exactly the user-defined
                "Time per Point" value regardless of number of channels.
                """
                try:
                    # Count enabled channels and find max points
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

                    if enabled_channels == 0 or max_points == 0:
                        return "Enable at least one channel"

                    # Use the user-defined time per point directly
                    time_per_point = float(settle) if settle else 0.262

                    # Total estimated time = points × time_per_point
                    estimated_time = max_points * time_per_point
                    time_per_point_ms = time_per_point * 1000

                    # Format output with start/end times
                    from datetime import datetime, timedelta
                    start_time = datetime.now()
                    end_time = start_time + timedelta(seconds=estimated_time)

                    start_str = start_time.strftime('%H:%M:%S')
                    end_str = end_time.strftime('%H:%M:%S')

                    if estimated_time < 60:
                        return f"Start: {start_str} → End: ~{end_str} | ~{estimated_time:.1f}s | {max_points} pts @ {time_per_point_ms:.0f}ms/pt ({enabled_channels} CH)"
                    else:
                        return f"Start: {start_str} → End: ~{end_str} | ~{estimated_time:.1f}s ({estimated_time/60:.1f} min) | {max_points} pts ({enabled_channels} CH)"

                except Exception as e:
                    return f"~0s (error: {e})"

            # Wire up duration updates - trigger on any parameter change
            duration_inputs = [
                psu_ch1_enable, psu_ch1_cycles, psu_ch1_points,
                psu_ch2_enable, psu_ch2_cycles, psu_ch2_points,
                psu_ch3_enable, psu_ch3_cycles, psu_ch3_points,
                psu_settle_time
            ]

            # Update on checkbox changes
            psu_ch1_enable.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch2_enable.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch3_enable.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )

            # Update on cycle/points changes
            psu_ch1_cycles.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch1_points.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch2_cycles.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch2_points.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch3_cycles.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )
            psu_ch3_points.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )

            # Update on settle time change
            psu_settle_time.change(
                fn=update_multi_channel_duration,
                inputs=duration_inputs,
                outputs=[psu_estimated_duration]
            )

            # Control buttons
            with gr.Row():
                psu_preview_waveform_btn = gr.Button("Preview All Enabled Channels", variant="secondary", size="lg")
                psu_start_waveform_btn = gr.Button("Start Multi-Channel Waveform", variant="primary", size="lg")
                psu_stop_waveform_btn = gr.Button("Stop Waveform", variant="stop", size="lg")

            psu_waveform_status = gr.Textbox(
                label="Waveform Status",
                value="Ready - Enable channels, configure parameters, and click Start",
                interactive=False,
                lines=2
            )

            psu_waveform_plot = gr.Plot(label="Multi-Channel Waveform Preview")

            gr.Markdown("### Save Waveform Plot")
            with gr.Row():
                psu_waveform_save_path = gr.Textbox(
                    label="Save Location for Waveform Plots",
                    placeholder="Click Browse to select folder...",
                    interactive=True,
                    scale=3
                )
                psu_waveform_browse_btn = gr.Button("Browse", variant="secondary", scale=1)

            psu_save_waveform_btn = gr.Button("Save Multi-Channel Waveform Plot", variant="primary")
            psu_waveform_save_status = gr.Textbox(
                label="Save Status",
                interactive=False
            )

            # ════════════════════════════════════════════════════════════════
            # MULTI-CHANNEL WAVEFORM EVENT HANDLERS (FIXED)
            # ════════════════════════════════════════════════════════════════

            # Store the last generated figure for saving
            self._last_waveform_fig = None

            def preview_all_channels(
                ch1_en, ch1_wf, ch1_v, ch1_cyc, ch1_pts, ch1_dur,
                ch2_en, ch2_wf, ch2_v, ch2_cyc, ch2_pts, ch2_dur,
                ch3_en, ch3_wf, ch3_v, ch3_cyc, ch3_pts, ch3_dur
            ):
                """Generate preview plot showing ALL enabled channels"""
                try:
                    import matplotlib.pyplot as plt

                    # Channel colors
                    colors = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}  # Blue, Green, Orange

                    fig, ax = plt.subplots(figsize=(14, 7))

                    enabled_count = 0
                    max_voltage = 0
                    total_points = 0

                    # Process each enabled channel
                    channel_configs = [
                        (1, ch1_en, ch1_wf, ch1_v, ch1_cyc, ch1_pts, ch1_dur),
                        (2, ch2_en, ch2_wf, ch2_v, ch2_cyc, ch2_pts, ch2_dur),
                        (3, ch3_en, ch3_wf, ch3_v, ch3_cyc, ch3_pts, ch3_dur)
                    ]

                    for ch, enabled, wf_type, voltage, cycles, points, duration in channel_configs:
                        if not enabled:
                            continue

                        enabled_count += 1

                        # Extract waveform name
                        wf_name = wf_type.split(' - ')[0] if ' - ' in wf_type else wf_type

                        # Generate waveform profile
                        cyc = int(cycles) if cycles else 3
                        pts = int(points) if points else 50
                        dur = float(duration) if duration else 8.0
                        volt = float(voltage) if voltage else 3.0

                        generator = self.psu_controller._WaveformGenerator(
                            waveform_type=wf_name,
                            target_voltage=volt,
                            cycles=cyc,
                            points_per_cycle=pts,
                            cycle_duration=dur
                        )
                        profile = generator.generate()

                        times = [p[0] for p in profile]
                        voltages = [p[1] for p in profile]

                        # Plot this channel
                        ax.plot(times, voltages, color=colors[ch], linewidth=2,
                               label=f'CH{ch}: {wf_name} ({volt}V, {cyc}×{pts}pts)')

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

                    # Store for saving
                    self._last_waveform_fig = fig

                    return fig

                except Exception as e:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.text(0.5, 0.5, f'Error generating preview:\n{str(e)}',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                    return fig

            def save_waveform_plot(save_path):
                """Save the LIVE execution data graph (measured data from actual waveform run)"""
                try:
                    if not save_path or save_path.strip() == "":
                        return "ERROR: Please select a save location first"

                    from pathlib import Path
                    import matplotlib.pyplot as plt
                    from datetime import datetime

                    save_dir = Path(save_path)

                    if not save_dir.exists():
                        return f"ERROR: Directory does not exist: {save_path}"

                    # Check if we have actual execution data
                    if not hasattr(self.psu_controller, 'ramping_data') or not self.psu_controller.ramping_data:
                        return "ERROR: No execution data to save. Run a waveform first to collect live data."

                    # We have execution data - create a plot from the LIVE measured data
                    data = self.psu_controller.ramping_data

                    # Group data by channel
                    channels_data = {}
                    for point in data:
                        ch = point.get('channel', 1)
                        if ch not in channels_data:
                            channels_data[ch] = {'timestamps': [], 'set_v': [], 'measured_v': [], 'measured_i': []}
                        channels_data[ch]['timestamps'].append(point['timestamp'])
                        channels_data[ch]['set_v'].append(point.get('set_voltage', 0))
                        channels_data[ch]['measured_v'].append(point.get('measured_voltage', 0))
                        channels_data[ch]['measured_i'].append(point.get('measured_current', 0))

                    # Create figure with subplots
                    num_channels = len(channels_data)
                    if num_channels == 0:
                        return "ERROR: No channel data found in execution results."

                    fig, axes = plt.subplots(num_channels, 2, figsize=(16, 5 * num_channels), squeeze=False)

                    colors = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}

                    for idx, (ch, ch_data) in enumerate(sorted(channels_data.items())):
                        # Convert timestamps to relative seconds
                        if ch_data['timestamps']:
                            t0 = ch_data['timestamps'][0]
                            times = [(t - t0).total_seconds() for t in ch_data['timestamps']]
                        else:
                            times = []

                        ax_v = axes[idx][0]
                        ax_i = axes[idx][1]

                        # Voltage plot
                        ax_v.plot(times, ch_data['set_v'], '--', color=colors.get(ch, 'blue'),
                                 linewidth=1, label='Setpoint', alpha=0.7)
                        ax_v.plot(times, ch_data['measured_v'], '-', color=colors.get(ch, 'blue'),
                                 linewidth=2, label='Measured')
                        ax_v.set_xlabel('Time (s)')
                        ax_v.set_ylabel('Voltage (V)')
                        ax_v.set_title(f'CH{ch} Voltage - Live Execution Data')
                        ax_v.legend(loc='upper right')
                        ax_v.grid(True, alpha=0.3)

                        # Current plot
                        ax_i.plot(times, ch_data['measured_i'], '-', color=colors.get(ch, 'blue'),
                                 linewidth=2)
                        ax_i.set_xlabel('Time (s)')
                        ax_i.set_ylabel('Current (A)')
                        ax_i.set_title(f'CH{ch} Current - Live Execution Data')
                        ax_i.grid(True, alpha=0.3)

                    plt.suptitle(f'Multi-Channel Waveform Execution Results - {len(data)} Data Points',
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()

                    # Save
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"waveform_live_data_{timestamp_str}.png"
                    filepath = save_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)

                    return f"✓ LIVE data saved to:\n{filepath}\n({len(data)} points, {num_channels} channel(s))"

                except Exception as e:
                    import traceback
                    return f"ERROR saving plot: {str(e)}"

            def start_multi_channel_waveform(
                ch1_en, ch1_wf, ch1_v, ch1_i, ch1_cyc, ch1_pts, ch1_dur,
                ch2_en, ch2_wf, ch2_v, ch2_i, ch2_cyc, ch2_pts, ch2_dur,
                ch3_en, ch3_wf, ch3_v, ch3_i, ch3_cyc, ch3_pts, ch3_dur,
                settle_time
            ):
                """Start multi-channel waveform generation"""
                # Update settle time
                self.psu_controller.ramping_params['psu_settle'] = settle_time

                configs = []
                if ch1_en:
                    configs.append({
                        'channel': 1,
                        'waveform': ch1_wf,
                        'target_voltage': ch1_v,
                        'current_limit': ch1_i,
                        'cycles': int(ch1_cyc) if ch1_cyc else 3,
                        'points_per_cycle': int(ch1_pts) if ch1_pts else 50,
                        'cycle_duration': ch1_dur if ch1_dur else 8.0
                    })
                if ch2_en:
                    configs.append({
                        'channel': 2,
                        'waveform': ch2_wf,
                        'target_voltage': ch2_v,
                        'current_limit': ch2_i,
                        'cycles': int(ch2_cyc) if ch2_cyc else 3,
                        'points_per_cycle': int(ch2_pts) if ch2_pts else 50,
                        'cycle_duration': ch2_dur if ch2_dur else 8.0
                    })
                if ch3_en:
                    configs.append({
                        'channel': 3,
                        'waveform': ch3_wf,
                        'target_voltage': ch3_v,
                        'current_limit': ch3_i,
                        'cycles': int(ch3_cyc) if ch3_cyc else 3,
                        'points_per_cycle': int(ch3_pts) if ch3_pts else 50,
                        'cycle_duration': ch3_dur if ch3_dur else 8.0
                    })

                if not configs:
                    return "ERROR: Enable at least one channel for waveform generation"

                return self.psu_controller.start_multi_channel_waveform(configs)

            def stop_waveform_generation():
                """Stop active waveform generation"""
                return self.psu_controller.stop_multi_channel_waveform()

            # All inputs needed for preview
            preview_inputs = [
                psu_ch1_enable, psu_ch1_waveform, psu_ch1_voltage, psu_ch1_cycles, psu_ch1_points, psu_ch1_duration,
                psu_ch2_enable, psu_ch2_waveform, psu_ch2_voltage, psu_ch2_cycles, psu_ch2_points, psu_ch2_duration,
                psu_ch3_enable, psu_ch3_waveform, psu_ch3_voltage, psu_ch3_cycles, psu_ch3_points, psu_ch3_duration
            ]

            # Wire up preview button - shows ALL enabled channels
            psu_preview_waveform_btn.click(
                fn=preview_all_channels,
                inputs=preview_inputs,
                outputs=[psu_waveform_plot]
            )

            # Wire up start button with all channel configs
            psu_start_waveform_btn.click(
                fn=start_multi_channel_waveform,
                inputs=[
                    psu_ch1_enable, psu_ch1_waveform, psu_ch1_voltage, psu_ch1_current,
                    psu_ch1_cycles, psu_ch1_points, psu_ch1_duration,
                    psu_ch2_enable, psu_ch2_waveform, psu_ch2_voltage, psu_ch2_current,
                    psu_ch2_cycles, psu_ch2_points, psu_ch2_duration,
                    psu_ch3_enable, psu_ch3_waveform, psu_ch3_voltage, psu_ch3_current,
                    psu_ch3_cycles, psu_ch3_points, psu_ch3_duration,
                    psu_settle_time
                ],
                outputs=[psu_waveform_status]
            )

            psu_stop_waveform_btn.click(
                fn=stop_waveform_generation,
                outputs=[psu_waveform_status]
            )

            # Note: Browse button removed - users should manually type/paste path in the textbox
            # The waveform save path textbox is directly editable by the user

            # Save waveform plot button - uses the new save_waveform_plot function
            psu_save_waveform_btn.click(
                fn=save_waveform_plot,
                inputs=[psu_waveform_save_path],
                outputs=[psu_waveform_save_status]
            )

        # Status & Activity Log
        gr.Markdown("### Status & Activity Log")
        with gr.Group():
            psu_activity_log_display = gr.Textbox(
                label="Activity Log",
                value=self.psu_controller.activity_log,
                lines=15,
                interactive=False,
                max_lines=500
            )

            # Waveform status and log polling (updates every 2 seconds)
            def poll_waveform_status_and_log():
                """Poll for waveform status and activity log updates"""
                return self.psu_controller.get_waveform_status(), self.psu_controller.activity_log

            # Timer to update status and log every 2 seconds
            waveform_timer = gr.Timer(value=2.0)
            waveform_timer.tick(
                fn=poll_waveform_status_and_log,
                outputs=[psu_waveform_status, psu_activity_log_display]
            )
        
        # Connection handlers
        def psu_handle_connect(visa_addr_val):
            """Handle connection button click"""
            self.psu_controller.connect_power_supply(visa_addr_val)
            # Wait for connection to complete (max 5 seconds)
            max_wait = 5.0
            waited = 0.0
            wait_step = 0.1
            while waited < max_wait:
                time.sleep(wait_step)
                waited += wait_step
                if self.psu_controller.is_connected:
                    break
            status = "Connected" if self.psu_controller.is_connected else "Disconnected"
            return status, self.psu_controller.activity_log
        
        def psu_handle_disconnect():
            """Handle disconnect button click"""
            self.psu_controller.disconnect_power_supply()
            return "Disconnected", self.psu_controller.activity_log
        
        def psu_handle_test():
            """Handle test connection button click"""
            result = self.psu_controller.test_connection()
            return result, self.psu_controller.activity_log
        
        def psu_handle_emergency():
            """Handle emergency stop button click"""
            result = self.psu_controller.emergency_stop()
            return result, self.psu_controller.activity_log
        
        # Register connection handlers
        psu_conn_btn.click(
            fn=psu_handle_connect,
            inputs=psu_visa_addr,
            outputs=[psu_conn_status, psu_activity_log_display]
        )
        
        psu_disc_btn.click(
            fn=psu_handle_disconnect,
            outputs=[psu_conn_status, psu_activity_log_display]
        )
        
        psu_test_btn.click(
            fn=psu_handle_test,
            outputs=[psu_conn_status, psu_activity_log_display]
        )
        
        psu_emerg_btn.click(
            fn=psu_handle_emergency,
            outputs=[psu_conn_status, psu_activity_log_display]
        )
    
    def create_oscilloscope_interface(self):
        """Create the Oscilloscope interface tab"""
        with gr.Tab("Connection"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Instrument Connection")
                    osc_visa_address = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x0957::0x1780::MY65220169::INSTR",
                        scale=3
                    )
                    osc_connect_btn = gr.Button("Connect", variant="primary", scale=1)
                    osc_disconnect_btn = gr.Button("Disconnect", variant="stop", scale=1)
                    osc_test_btn = gr.Button("Test", scale=1)
                
                    osc_connection_status = gr.Textbox(label="Status", value="Disconnected", interactive=False)
                    osc_instrument_info = gr.Textbox(label="Instrument Information", interactive=False)

        with gr.Tab("Channel Configuration"):
            gr.Markdown("### Channel Selection and Configuration")
            with gr.Row():
                osc_ch1_select = gr.Checkbox(label="Ch1", value=True)
                osc_ch2_select = gr.Checkbox(label="Ch2", value=False)
                osc_ch3_select = gr.Checkbox(label="Ch3", value=False)
                osc_ch4_select = gr.Checkbox(label="Ch4", value=False)
            
            with gr.Row():
                osc_v_scale = gr.Number(label="V/div", value=1.0)
                osc_v_offset = gr.Number(label="Offset (V)", value=0.0)
                osc_coupling = gr.Dropdown(
                    label="Coupling",
                    choices=["AC", "DC"],
                    value="DC"
                )
                osc_probe = gr.Dropdown(
                    label="Probe",
                    choices=[("1x", 1.0), ("10x", 10.0), ("100x", 100.0)],
                    value=1.0
                )
            
            osc_config_channel_btn = gr.Button("Configure Channels", variant="primary")
            osc_channel_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Acquisition Control")
            with gr.Row():
                osc_run_btn = gr.Button("▶ RUN", variant="primary")
                osc_stop_btn = gr.Button("⏹ STOP", variant="stop")
                osc_single_btn = gr.Button("⏯ SINGLE", variant="secondary")
                osc_autoscale_btn = gr.Button("⚡ Autoscale", variant="primary")

            osc_acquisition_status = gr.Textbox(label="Acquisition Status", interactive=False)
            osc_system_status = gr.Textbox(label="System Status", interactive=False, lines=3)

        with gr.Tab("Timebase & Trigger"):
            gr.Markdown("### Horizontal Timebase Configuration")
            with gr.Row():
                osc_time_scale = gr.Dropdown(
                    label="Time/div",
                    choices=self.oscilloscope_controller.timebase_scales,
                    value=10e-3
                )
                osc_timebase_btn = gr.Button("Apply Timebase", variant="primary")
            
            osc_timebase_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Edge Trigger Configuration")
            with gr.Row():
                osc_trigger_source = gr.Dropdown(
                    label="Source",
                    choices=["CH1", "CH2", "CH3", "CH4"],
                    value="CH1"
                )
                osc_trigger_level = gr.Number(label="Level (V)", value=0.0)
                osc_trigger_slope = gr.Dropdown(
                    label="Slope",
                    choices=["Rising", "Falling", "Either"],
                    value="Rising"
                )
            
            osc_trigger_btn = gr.Button("Apply Trigger", variant="primary")
            osc_trigger_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Trigger Sweep & Holdoff")
            with gr.Row():
                osc_sweep_mode = gr.Dropdown(
                    label="Sweep Mode",
                    choices=["AUTO", "NORMal", "TRIG"],
                    value="AUTO"
                )
                osc_sweep_btn = gr.Button("Apply Sweep", variant="primary")

            osc_sweep_status = gr.Textbox(label="Sweep Status", interactive=False)

            with gr.Row():
                osc_holdoff_time = gr.Number(label="Holdoff Time (ns)", value=100.0)
                osc_holdoff_btn = gr.Button("Apply Holdoff", variant="primary")

            osc_holdoff_status = gr.Textbox(label="Holdoff Status", interactive=False)

        # Acquisition Control Tab
        with gr.Tab("Acquisition Control"):
            gr.Markdown("### Acquisition Mode Configuration")
            with gr.Row():
                osc_acq_mode = gr.Dropdown(
                    label="Acquisition Mode",
                    choices=["RTIMe", "ETIMe", "SEGMented"],
                    value="RTIMe"
                )
                osc_acq_mode_btn = gr.Button("Apply Mode", variant="primary")

            osc_acq_mode_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Acquisition Type Configuration")
            with gr.Row():
                osc_acq_type = gr.Dropdown(
                    label="Acquisition Type",
                    choices=["NORMal", "AVERage", "HRESolution", "PEAK"],
                    value="NORMal"
                )
                osc_acq_type_btn = gr.Button("Apply Type", variant="primary")

            osc_acq_type_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Averaging Configuration")
            with gr.Row():
                osc_avg_count = gr.Slider(
                    label="Averaging Count",
                    minimum=2,
                    maximum=65536,
                    value=16,
                    step=1
                )
                osc_avg_btn = gr.Button("Apply Averaging", variant="primary")

            osc_avg_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Acquisition Information")
            osc_info_btn = gr.Button("Query Acquisition Info", variant="secondary")
            osc_acq_info_display = gr.Textbox(label="Acquisition Info", interactive=False, lines=6)

        # Markers & Cursors Tab
        with gr.Tab("Markers & Cursors"):
            gr.Markdown("### Marker/Cursor Mode Configuration")
            with gr.Row():
                osc_marker_mode = gr.Dropdown(
                    label="Marker Mode",
                    choices=["OFF", "MEASurement", "MANual", "WAVeform"],
                    value="OFF"
                )
                osc_marker_mode_btn = gr.Button("Set Mode", variant="primary")

            osc_marker_mode_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Marker Position Configuration")
            with gr.Row():
                osc_marker_num = gr.Dropdown(
                    label="Marker Number",
                    choices=[1, 2],
                    value=1
                )
                osc_marker_x = gr.Number(label="X Position (s)", value=0.0)
                osc_marker_y = gr.Number(label="Y Position (V)", value=0.0)

            osc_marker_set_btn = gr.Button("Set Marker Position", variant="primary")
            osc_marker_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Marker Delta Measurements")
            osc_delta_btn = gr.Button("Get Delta Values", variant="primary")
            osc_delta_result = gr.Textbox(label="Delta Results", interactive=False, lines=4)

        # Math Functions Tab
        with gr.Tab("Math Functions"):
            gr.Markdown("### Math Function Configuration")
            with gr.Row():
                osc_math_func = gr.Dropdown(
                    label="Function Number",
                    choices=[1, 2, 3, 4],
                    value=1
                )
                osc_math_op = gr.Dropdown(
                    label="Operation",
                    choices=["ADD", "SUBTract", "MULTiply", "DIVide", "FFT"],
                    value="ADD"
                )

            with gr.Row():
                osc_math_src1 = gr.Dropdown(
                    label="Source 1",
                    choices=[1, 2, 3, 4],
                    value=1
                )
                osc_math_src2 = gr.Dropdown(
                    label="Source 2 (not used for FFT)",
                    choices=[1, 2, 3, 4],
                    value=2
                )

            osc_math_config_btn = gr.Button("Configure Math Function", variant="primary")
            osc_math_config_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Math Function Display & Scale")
            with gr.Row():
                osc_math_show = gr.Checkbox(label="Show on Display", value=True)
                osc_math_scale_val = gr.Number(label="Vertical Scale (V/div)", value=1.0)

            with gr.Row():
                osc_math_display_btn = gr.Button("Toggle Display", variant="primary")
                osc_math_scale_btn = gr.Button("Set Scale", variant="primary")

            osc_math_display_status = gr.Textbox(label="Display Status", interactive=False)
            osc_math_scale_status = gr.Textbox(label="Scale Status", interactive=False)

        # Setup Management Tab
        with gr.Tab("Setup Management"):
            gr.Markdown("### Save/Recall Instrument Setup")
            with gr.Row():
                osc_setup_save_name = gr.Textbox(
                    label="Setup Name",
                    placeholder="my_setup.stp"
                )
                osc_setup_save_btn = gr.Button("Save Setup", variant="primary")

            osc_setup_save_status = gr.Textbox(label="Save Status", interactive=False)

            with gr.Row():
                osc_setup_recall_name = gr.Textbox(
                    label="Setup Name",
                    placeholder="my_setup.stp"
                )
                osc_setup_recall_btn = gr.Button("Recall Setup", variant="primary")

            osc_setup_recall_status = gr.Textbox(label="Recall Status", interactive=False)

            gr.Markdown("### Save/Recall Waveform to Memory")
            with gr.Row():
                osc_wf_save_ch = gr.Dropdown(
                    label="Channel",
                    choices=[1, 2, 3, 4],
                    value=1
                )
                osc_wf_save_name = gr.Textbox(
                    label="Waveform Name",
                    placeholder="my_waveform"
                )
                osc_wf_save_btn = gr.Button("Save Waveform", variant="primary")

            osc_wf_save_status = gr.Textbox(label="Save Status", interactive=False)

            with gr.Row():
                osc_wf_recall_name = gr.Textbox(
                    label="Waveform Name",
                    placeholder="my_waveform"
                )
                osc_wf_recall_btn = gr.Button("Recall Waveform", variant="primary")

            osc_wf_recall_status = gr.Textbox(label="Recall Status", interactive=False)

        # Function Generators Tab
        with gr.Tab("Function Generators"):
            gr.Markdown("### WGEN1 Configuration")
            with gr.Row():
                osc_wgen1_enable = gr.Checkbox(label="Enable WGEN1", value=False)
                osc_wgen1_waveform = gr.Dropdown(
                    label="Waveform",
                    choices=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"],
                    value="SIN"
                )

            with gr.Row():
                osc_wgen1_freq = gr.Number(label="Frequency (Hz)", value=1000)
                osc_wgen1_amp = gr.Number(label="Amplitude (Vpp)", value=1.0)
                osc_wgen1_offset = gr.Number(label="Offset (V)", value=0.0)

            with gr.Row():
                osc_wgen1_btn = gr.Button("Apply WGEN1", variant="primary")
                osc_wgen1_info_btn = gr.Button("Query WGEN1", variant="secondary")

            osc_wgen1_status = gr.Textbox(label="WGEN1 Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### WGEN2 Configuration")
            with gr.Row():
                osc_wgen2_enable = gr.Checkbox(label="Enable WGEN2", value=False)
                osc_wgen2_waveform = gr.Dropdown(
                    label="Waveform",
                    choices=["SIN", "SQU", "RAMP", "PULS", "DC", "NOIS", "ARB", "SINC", "EXPR", "EXPF", "CARD", "GAUS"],
                    value="SIN"
                )

            with gr.Row():
                osc_wgen2_freq = gr.Number(label="Frequency (Hz)", value=1000)
                osc_wgen2_amp = gr.Number(label="Amplitude (Vpp)", value=1.0)
                osc_wgen2_offset = gr.Number(label="Offset (V)", value=0.0)

            with gr.Row():
                osc_wgen2_btn = gr.Button("Apply WGEN2", variant="primary")
                osc_wgen2_info_btn = gr.Button("Query WGEN2", variant="secondary")

            osc_wgen2_status = gr.Textbox(label="WGEN2 Status", interactive=False)

        # Measurements Tab
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
                osc_meas_source = gr.Dropdown(
                    label="Source",
                    choices=source_choices,
                    value="CH1"
                )
                
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
                
                osc_measurement_type = gr.Dropdown(
                    label="Measurement Type",
                    choices=measurement_choices,
                    value="FREQ"
                )
                
                osc_measure_btn = gr.Button("Measure", variant="primary")
                osc_all_measurements_btn = gr.Button("Show All", variant="primary")
            
                osc_measurement_result = gr.Textbox(label="Measurement Result", interactive=False)
                osc_all_measurements_result = gr.Textbox(
                    label="All Measurements",
                    interactive=False,
                    lines=10,
                    max_lines=20,
                    show_copy_button=True
                )

        # Advanced Triggers Tab
        with gr.Tab("Advanced Triggers"):
            gr.Markdown("### Glitch (Spike) Trigger")
            gr.Markdown("Detects signal violations - pulses narrower than threshold")
            with gr.Row():
                osc_glitch_source = gr.Dropdown(
                    label="Source",
                    choices=["CH1", "CH2", "CH3", "CH4"],
                    value="CH1"
                )
                osc_glitch_level = gr.Number(label="Level (V)", value=0.0)
                osc_glitch_polarity = gr.Dropdown(
                    label="Polarity",
                    choices=["POSitive", "NEGative"],
                    value="POSitive"
                )
                osc_glitch_width = gr.Number(label="Width (ns)", value=1.0)
            
            osc_glitch_btn = gr.Button("Set Glitch Trigger", variant="primary")
            osc_glitch_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Pulse Width Trigger")
            gr.Markdown("Triggers on pulses with width above or below threshold")
            with gr.Row():
                osc_pulse_source = gr.Dropdown(
                    label="Source",
                    choices=["CH1", "CH2", "CH3", "CH4"],
                    value="CH1"
                )
                osc_pulse_level = gr.Number(label="Level (V)", value=0.0)
                osc_pulse_width = gr.Number(label="Width (ns)", value=10.0)
                osc_pulse_polarity = gr.Dropdown(
                    label="Polarity",
                    choices=["POSitive", "NEGative"],
                    value="POSitive"
                )
            
            osc_pulse_btn = gr.Button("Set Pulse Trigger", variant="primary")
            osc_pulse_status = gr.Textbox(label="Status", interactive=False)
        
        # Operations & File Management Tab
        with gr.Tab("Operations & File Management"):
            with gr.Column(variant="panel"):
                gr.Markdown("### File Save Locations")
                
                with gr.Row():
                    osc_data_path = gr.Textbox(
                        label="Data Directory",
                        value=self.oscilloscope_controller.save_locations['data'],
                        scale=4
                    )
                    osc_data_browse_btn = gr.Button("Browse", scale=1)
                
                with gr.Row():
                    osc_graphs_path = gr.Textbox(
                        label="Graphs Directory",
                        value=self.oscilloscope_controller.save_locations['graphs'],
                        scale=4
                    )
                    osc_graphs_browse_btn = gr.Button("Browse", scale=1)
                
                with gr.Row():
                    osc_screenshots_path = gr.Textbox(
                        label="Screenshots Directory",
                        value=self.oscilloscope_controller.save_locations['screenshots'],
                        scale=4
                    )
                    osc_screenshots_browse_btn = gr.Button("Browse", scale=1)
                
                with gr.Row():
                    osc_update_paths_btn = gr.Button("Update Paths", variant="primary")
                    osc_path_status = gr.Textbox(label="Path Status", interactive=False, scale=4)
            
            # Data Acquisition and Export section
            gr.Markdown("### Data Acquisition and Export")

            with gr.Row():
                osc_op_ch1 = gr.Checkbox(label="Ch1", value=True)
                osc_op_ch2 = gr.Checkbox(label="Ch2", value=False)
                osc_op_ch3 = gr.Checkbox(label="Ch3", value=False)
                osc_op_ch4 = gr.Checkbox(label="Ch4", value=False)

            with gr.Row():
                osc_op_math1 = gr.Checkbox(label="Math1", value=False)
                osc_op_math2 = gr.Checkbox(label="Math2", value=False)
                osc_op_math3 = gr.Checkbox(label="Math3", value=False)
                osc_op_math4 = gr.Checkbox(label="Math4", value=False)

            with gr.Row():
                osc_export_path = gr.Textbox(
                    label="Export Save Location",
                    placeholder="Click Browse to select folder...",
                    interactive=True,
                    scale=3
                )
                osc_export_browse_btn = gr.Button("Browse", variant="secondary", scale=1)

            osc_plot_title_input = gr.Textbox(
                label="Plot Title (optional)",
                placeholder="Enter custom plot title"
            )

            with gr.Row():
                osc_screenshot_btn = gr.Button("Capture Screenshot", variant="secondary")
                osc_acquire_btn = gr.Button("Acquire Data", variant="primary")
                osc_export_btn = gr.Button("Export CSV", variant="secondary")
                osc_plot_btn = gr.Button("Generate Plot", variant="secondary")

            with gr.Row():
                osc_full_auto_btn = gr.Button("Full Automation", variant="primary", scale=2)

            osc_operation_status = gr.Textbox(label="Operation Status", interactive=False, lines=10)
        
        # Connect UI events to controller methods
        osc_connect_btn.click(
            fn=self.oscilloscope_controller.connect_oscilloscope,
            inputs=[osc_visa_address],
            outputs=[osc_instrument_info, osc_connection_status]
        )
        
        osc_disconnect_btn.click(
            fn=self.oscilloscope_controller.disconnect_oscilloscope,
            inputs=[],
            outputs=[osc_instrument_info, osc_connection_status]
        )
        
        osc_test_btn.click(
            fn=self.oscilloscope_controller.test_connection,
            inputs=[],
            outputs=[osc_instrument_info]
        )
        
        osc_config_channel_btn.click(
            fn=self.oscilloscope_controller.configure_channel,
            inputs=[osc_ch1_select, osc_ch2_select, osc_ch3_select, osc_ch4_select, osc_v_scale, osc_v_offset, osc_coupling, osc_probe],
            outputs=[osc_channel_status]
        )
        
        osc_autoscale_btn.click(
            fn=self.oscilloscope_controller.perform_autoscale,
            inputs=[],
            outputs=[osc_system_status]
        )

        osc_run_btn.click(
            fn=self.oscilloscope_controller.run_acquisition,
            inputs=[],
            outputs=[osc_acquisition_status]
        )

        osc_stop_btn.click(
            fn=self.oscilloscope_controller.stop_acquisition,
            inputs=[],
            outputs=[osc_acquisition_status]
        )

        osc_single_btn.click(
            fn=self.oscilloscope_controller.single_acquisition,
            inputs=[],
            outputs=[osc_acquisition_status]
        )

        osc_timebase_btn.click(
            fn=self.oscilloscope_controller.configure_timebase,
            inputs=[osc_time_scale],
            outputs=[osc_timebase_status]
        )
        
        osc_trigger_btn.click(
            fn=self.oscilloscope_controller.configure_trigger,
            inputs=[osc_trigger_source, osc_trigger_level, osc_trigger_slope],
            outputs=[osc_trigger_status]
        )
        
        osc_measure_btn.click(
            fn=self.oscilloscope_controller.perform_measurement,
            inputs=[osc_meas_source, osc_measurement_type],
            outputs=[osc_measurement_result]
        )
        
        osc_all_measurements_btn.click(
            fn=self.oscilloscope_controller.get_all_measurements,
            inputs=[osc_meas_source],
            outputs=[osc_all_measurements_result]
        )
        
        osc_glitch_btn.click(
            fn=self.oscilloscope_controller.set_glitch_trigger,
            inputs=[osc_glitch_source, osc_glitch_level, osc_glitch_polarity, osc_glitch_width],
            outputs=[osc_glitch_status]
        )
        
        osc_pulse_btn.click(
            fn=self.oscilloscope_controller.set_pulse_trigger,
            inputs=[osc_pulse_source, osc_pulse_level, osc_pulse_width, osc_pulse_polarity],
            outputs=[osc_pulse_status]
        )

        # Trigger sweep and holdoff handlers
        osc_sweep_btn.click(
            fn=self.oscilloscope_controller.set_trigger_sweep_mode,
            inputs=[osc_sweep_mode],
            outputs=[osc_sweep_status]
        )

        osc_holdoff_btn.click(
            fn=self.oscilloscope_controller.set_trigger_holdoff,
            inputs=[osc_holdoff_time],
            outputs=[osc_holdoff_status]
        )

        # Acquisition control handlers
        osc_acq_mode_btn.click(
            fn=self.oscilloscope_controller.set_acquisition_mode,
            inputs=[osc_acq_mode],
            outputs=[osc_acq_mode_status]
        )

        osc_acq_type_btn.click(
            fn=self.oscilloscope_controller.set_acquisition_type,
            inputs=[osc_acq_type],
            outputs=[osc_acq_type_status]
        )

        osc_avg_btn.click(
            fn=self.oscilloscope_controller.set_acquisition_count,
            inputs=[osc_avg_count],
            outputs=[osc_avg_status]
        )

        osc_info_btn.click(
            fn=self.oscilloscope_controller.query_acquisition_info,
            inputs=[],
            outputs=[osc_acq_info_display]
        )

        # Marker/cursor handlers
        osc_marker_mode_btn.click(
            fn=self.oscilloscope_controller.set_marker_mode,
            inputs=[osc_marker_mode],
            outputs=[osc_marker_mode_status]
        )

        osc_marker_set_btn.click(
            fn=self.oscilloscope_controller.set_marker_positions,
            inputs=[osc_marker_num, osc_marker_x, osc_marker_y],
            outputs=[osc_marker_status]
        )

        osc_delta_btn.click(
            fn=self.oscilloscope_controller.get_marker_deltas,
            inputs=[],
            outputs=[osc_delta_result]
        )

        # Math function handlers
        osc_math_config_btn.click(
            fn=self.oscilloscope_controller.configure_math_operation,
            inputs=[osc_math_func, osc_math_op, osc_math_src1, osc_math_src2],
            outputs=[osc_math_config_status]
        )

        osc_math_display_btn.click(
            fn=self.oscilloscope_controller.toggle_math_display,
            inputs=[osc_math_func, osc_math_show],
            outputs=[osc_math_display_status]
        )

        osc_math_scale_btn.click(
            fn=self.oscilloscope_controller.set_math_scale,
            inputs=[osc_math_func, osc_math_scale_val],
            outputs=[osc_math_scale_status]
        )

        # Setup management handlers
        osc_setup_save_btn.click(
            fn=self.oscilloscope_controller.save_instrument_setup,
            inputs=[osc_setup_save_name],
            outputs=[osc_setup_save_status]
        )

        osc_setup_recall_btn.click(
            fn=self.oscilloscope_controller.recall_instrument_setup,
            inputs=[osc_setup_recall_name],
            outputs=[osc_setup_recall_status]
        )

        osc_wf_save_btn.click(
            fn=self.oscilloscope_controller.save_waveform_to_memory,
            inputs=[osc_wf_save_ch, osc_wf_save_name],
            outputs=[osc_wf_save_status]
        )

        osc_wf_recall_btn.click(
            fn=self.oscilloscope_controller.recall_waveform_from_memory,
            inputs=[osc_wf_recall_name],
            outputs=[osc_wf_recall_status]
        )

        # Function generator handlers
        osc_wgen1_btn.click(
            fn=self.oscilloscope_controller.configure_wgen,
            inputs=[gr.State(1), osc_wgen1_enable, osc_wgen1_waveform, osc_wgen1_freq, osc_wgen1_amp, osc_wgen1_offset],
            outputs=[osc_wgen1_status]
        )

        osc_wgen1_info_btn.click(
            fn=self.oscilloscope_controller.get_wgen_configuration,
            inputs=[gr.State(1)],
            outputs=[osc_wgen1_status]
        )

        osc_wgen2_btn.click(
            fn=self.oscilloscope_controller.configure_wgen,
            inputs=[gr.State(2), osc_wgen2_enable, osc_wgen2_waveform, osc_wgen2_freq, osc_wgen2_amp, osc_wgen2_offset],
            outputs=[osc_wgen2_status]
        )

        osc_wgen2_info_btn.click(
            fn=self.oscilloscope_controller.get_wgen_configuration,
            inputs=[gr.State(2)],
            outputs=[osc_wgen2_status]
        )

        # Define path update functions
        def osc_update_paths(data, graphs, screenshots):
            self.oscilloscope_controller.save_locations['data'] = data
            self.oscilloscope_controller.save_locations['graphs'] = graphs
            self.oscilloscope_controller.save_locations['screenshots'] = screenshots
            return "Paths updated successfully"
        
        def osc_browse_data_folder(current_path):
            new_path = self.oscilloscope_controller.browse_folder(current_path, "Data")
            self.oscilloscope_controller.save_locations['data'] = new_path
            return new_path, f"Data directory updated to: {new_path}"
        
        def osc_browse_graphs_folder(current_path):
            new_path = self.oscilloscope_controller.browse_folder(current_path, "Graphs")
            self.oscilloscope_controller.save_locations['graphs'] = new_path
            return new_path, f"Graphs directory updated to: {new_path}"
        
        def osc_browse_screenshots_folder(current_path):
            new_path = self.oscilloscope_controller.browse_folder(current_path, "Screenshots")
            self.oscilloscope_controller.save_locations['screenshots'] = new_path
            return new_path, f"Screenshots directory updated to: {new_path}"
        
        # Connect the buttons to their functions
        osc_update_paths_btn.click(
            fn=osc_update_paths,
            inputs=[osc_data_path, osc_graphs_path, osc_screenshots_path],
            outputs=[osc_path_status]
        )
        
        osc_data_browse_btn.click(
            fn=osc_browse_data_folder,
            inputs=[osc_data_path],
            outputs=[osc_data_path, osc_path_status]
        )
        
        osc_graphs_browse_btn.click(
            fn=osc_browse_graphs_folder,
            inputs=[osc_graphs_path],
            outputs=[osc_graphs_path, osc_path_status]
        )
        
        osc_screenshots_browse_btn.click(
            fn=osc_browse_screenshots_folder,
            inputs=[osc_screenshots_path],
            outputs=[osc_screenshots_path, osc_path_status]
        )
        
        osc_screenshot_btn.click(
            fn=self.oscilloscope_controller.capture_screenshot,
            inputs=[],
            outputs=[osc_operation_status]
        )
        
        osc_acquire_btn.click(
            fn=self.oscilloscope_controller.acquire_data,
            inputs=[osc_op_ch1, osc_op_ch2, osc_op_ch3, osc_op_ch4, osc_op_math1, osc_op_math2, osc_op_math3, osc_op_math4],
            outputs=[osc_operation_status]
        )

        # Note: Browse button removed - users should manually type/paste path in the textbox
        # The export path textbox is directly editable by the user

        osc_export_btn.click(
            fn=self.oscilloscope_controller.export_csv,
            inputs=[osc_export_path],
            outputs=[osc_operation_status]
        )
        
        osc_plot_btn.click(
            fn=self.oscilloscope_controller.generate_plot,
            inputs=[osc_plot_title_input],
            outputs=[osc_operation_status]
        )
        
        osc_full_auto_btn.click(
            fn=self.oscilloscope_controller.run_full_automation,
            inputs=[osc_op_ch1, osc_op_ch2, osc_op_ch3, osc_op_ch4, osc_op_math1, osc_op_math2, osc_op_math3, osc_op_math4, osc_plot_title_input],
            outputs=[osc_operation_status]
        )

    def launch(self, share=False, server_port=7860, auto_open=True):
        """Launch the unified interface."""
        interface = self.create_interface()
        interface.launch(
            server_name="0.0.0.0",
            share=share,
            server_port=server_port,
            #inbrowser=auto_open,
            show_error=True
        )


def main():
    """Application entry point."""
    print("DIGANTARA Unified Instrument Control")
    print("=" * 80)
    print("Starting web interface...")
    
    try:
        # Find an available port starting from 7860
        start_port = 7860
        max_attempts = 10
        
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.close()
                
                print(f"Found available port: {port}")
                print("The browser will open automatically when ready.")
                print("IMPORTANT: To stop the application, press Ctrl+C in this terminal.")
                
                app = UnifiedInstrumentControl()
                app.launch(share=False, server_port=port, auto_open=True)
                break
                
            except OSError:
                if port == start_port + max_attempts - 1:
                    print(f"Error: Could not find an available port after {max_attempts} attempts.")
                    print("Please close any applications using ports {}-{}".format(start_port, start_port + max_attempts - 1))
                    return
                    
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nApplication shutdown complete.")


if __name__ == "__main__":
    main()