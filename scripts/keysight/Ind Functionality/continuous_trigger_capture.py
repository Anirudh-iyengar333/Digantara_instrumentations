#!/usr/bin/env python3
"""
================================================================================
CONTINUOUS TRIGGER CAPTURE AUTOMATION FOR OSCILLOSCOPE
================================================================================

PURPOSE:
--------
This program automates the process of capturing screenshots and data from a
Keysight oscilloscope every time a trigger event occurs. Think of it as a
camera that takes a picture and records measurements each time an electrical
signal crosses a specific threshold.

WHAT IT DOES (In Simple Terms):
--------------------------------
1. Connects to an oscilloscope (a device that displays electrical signals)
2. Waits for a signal to cross a specified level (called a "trigger")
3. When triggered, it:
   - Takes a screenshot of the oscilloscope display
   - Saves the actual data values to a CSV file
   - Records measurements (voltage, frequency, etc.)
4. Repeats this process for as many captures as you specify
5. Provides a web-based interface so you can control everything from your browser

TYPICAL USE CASE:
-----------------
Testing electronics where you need to capture many events automatically,
such as monitoring power-on sequences, signal quality over time, or
intermittent faults.

Author: Senior Instrumentation Engineer
Organization: Digantara Research and Technologies Pvt. Ltd.
Date: 2025-01-22
Version: 2.0.0
================================================================================
"""

# ============================================================================
# IMPORT SECTION
# This section brings in pre-written code libraries that provide functionality
# we need (like file handling, web interface, data processing, etc.)
# ============================================================================

import sys                  # System-specific functions
import os                   # Operating system functions (file paths, etc.)
import time                 # Time-related functions (delays, timestamps)
import logging              # For recording what the program is doing
import threading            # Allows running multiple tasks at once
from pathlib import Path    # Modern way to handle file paths
from datetime import datetime  # For timestamps
from typing import Optional, Dict, Any, List, Tuple  # Type hints for clarity
from dataclasses import dataclass  # Easy way to create data containers
import signal               # For handling system interrupts (Ctrl+C)
import atexit               # For cleanup when program exits
import json                 # For saving reports in structured format

import numpy as np          # Numerical processing library
import pandas as pd         # Data analysis library (for CSV files)
import gradio as gr         # Web interface library

# ============================================================================
# SETUP IMPORT PATH
# This allows us to import our custom oscilloscope control module
# ============================================================================

# Find the parent directory (going up 4 levels from this script to reach project root)
script_dir = Path(__file__).resolve().parent.parent.parent.parent

# Add it to the system path so Python can find our modules
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Try to import our oscilloscope control module
try:
    from instrument_control.keysight_oscilloscope import KeysightDSOX6004A
except ImportError as e:
    print(f"Error importing oscilloscope module: {e}")
    sys.exit(1)  # Exit if we can't find the module

# ============================================================================
# DATA STRUCTURES
# These define the format for storing configuration and results
# ============================================================================

@dataclass
class CaptureConfig:
    """
    CAPTURE CONFIGURATION
    ---------------------
    This holds all the settings for a capture session.

    Think of this as a recipe card that tells the program:
    - How many captures to take
    - How long to wait between captures
    - Which channels to record
    - Where to save files
    - What types of files to save
    """

    # Basic capture parameters
    num_captures: int           # How many trigger events to capture
    time_interval: float        # Seconds to wait between captures
    channels: List[int]         # Which oscilloscope channels to record (1-4)
    base_filename: str          # Base name for saved files
    save_directory: str         # Folder where files will be saved

    # File saving options
    capture_screenshots: bool = True      # Save PNG images of screen?
    save_waveforms: bool = True           # Save individual CSV files per channel?
    save_combined_csv: bool = False       # Save multi-channel CSV file?
    trigger_timeout: float = 10.0         # How long to wait for a trigger (seconds)

    def validate(self) -> Tuple[bool, str]:
        """
        VALIDATION FUNCTION
        -------------------
        Checks if the configuration makes sense before starting.
        Returns: (True, "OK") if valid, or (False, "error message") if not
        """

        # Check that we're capturing at least 1 event
        if self.num_captures < 1:
            return False, "Number of captures must be at least 1"

        # Check that time interval isn't negative
        if self.time_interval < 0:
            return False, "Time interval cannot be negative"

        # Check that at least one channel is selected
        if not self.channels:
            return False, "At least one channel must be selected"

        # Check that filename isn't empty
        if not self.base_filename:
            return False, "Base filename cannot be empty"

        # Check that at least one save option is enabled
        if not (self.capture_screenshots or self.save_waveforms or self.save_combined_csv):
            return False, "Must enable at least one saving option"

        # If we get here, everything is valid
        return True, "Configuration valid"


@dataclass
class CaptureResult:
    """
    CAPTURE RESULT
    --------------
    This stores the outcome of a single trigger capture.

    After each trigger event, we store:
    - When it happened
    - What files were saved
    - What measurements were taken
    - Whether it succeeded or failed
    """

    index: int                              # Capture number (0, 1, 2, ...)
    timestamp: datetime                     # When this capture happened
    screenshot_file: Optional[str] = None   # Path to screenshot (if saved)
    waveform_files: List[str] = None        # Paths to CSV files (if saved)
    measurements: Dict[str, Any] = None     # Measured values (voltage, frequency, etc.)
    success: bool = True                    # Did the capture succeed?
    error_message: Optional[str] = None     # Error description (if failed)


# ============================================================================
# TRIGGER CAPTURE ENGINE
# This is the "brain" that performs the actual capture operations
# ============================================================================

class TriggerCaptureEngine:
    """
    TRIGGER CAPTURE ENGINE
    ----------------------
    This class handles the core functionality of capturing data from the
    oscilloscope. It runs in the background and manages the entire capture
    process automatically.

    HOW IT WORKS:
    1. Configures the oscilloscope to wait for a trigger
    2. When a trigger occurs, captures screenshot and data
    3. Saves files with timestamps
    4. Waits for the next trigger
    5. Repeats until all captures are done
    """

    def __init__(self, oscilloscope: KeysightDSOX6004A):
        """
        INITIALIZATION
        --------------
        Sets up the capture engine with a connection to the oscilloscope.

        Parameters:
            oscilloscope: The connected oscilloscope object
        """
        self.scope = oscilloscope  # Store the oscilloscope connection
        self.logger = logging.getLogger(self.__class__.__name__)  # Setup logging

        # State tracking variables (keep track of what we're doing)
        self.is_running = False          # Are we currently capturing?
        self.stop_requested = False      # Has the user asked to stop?
        self.current_capture = 0         # Which capture are we on?
        self.total_captures = 0          # How many captures total?

        # Results storage (where we keep track of what we've captured)
        self.capture_results: List[CaptureResult] = []  # List of all results
        self.capture_thread: Optional[threading.Thread] = None  # Background worker

        # Thread safety (prevents conflicts when multiple tasks access same data)
        self.lock = threading.RLock()  # A "lock" to prevent data corruption

    def start_capture_session(self, config: CaptureConfig) -> bool:
        """
        START CAPTURE SESSION
        ---------------------
        Begins the automated capture process.

        This function:
        1. Validates the configuration
        2. Creates the save directory
        3. Starts a background worker thread to do the captures

        Parameters:
            config: Configuration object with all settings

        Returns:
            True if started successfully, False if there was a problem
        """

        # Step 1: Validate configuration
        valid, msg = config.validate()
        if not valid:
            self.logger.error(f"Invalid configuration: {msg}")
            return False

        # Step 2: Check oscilloscope connection
        if not self.scope.is_connected:
            self.logger.error("Oscilloscope not connected")
            return False

        # Step 3: Create save directory if it doesn't exist
        save_dir = Path(config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: Initialize state (reset everything)
        with self.lock:  # Use lock to prevent conflicts
            if self.is_running:
                self.logger.warning("Capture already in progress")
                return False

            # Reset all state variables
            self.is_running = True
            self.stop_requested = False
            self.current_capture = 0
            self.total_captures = config.num_captures
            self.capture_results = []

        # Step 5: Start background thread to perform captures
        # This runs in parallel so the interface doesn't freeze
        self.capture_thread = threading.Thread(
            target=self._capture_loop,  # The function to run
            args=(config,),             # Arguments to pass
            daemon=True                 # Dies when main program exits
        )
        self.capture_thread.start()

        self.logger.info(f"Started capture session: {config.num_captures} captures")
        return True

    def stop_capture_session(self):
        """
        STOP CAPTURE SESSION
        --------------------
        Stops the ongoing capture process gracefully.

        This tells the background worker to stop after the current capture
        completes, then waits for it to finish.
        """
        self.logger.info("Stopping capture session...")

        # Set the stop flag
        with self.lock:
            self.stop_requested = True

        # Wait for the background thread to finish (up to 5 seconds)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)

        # Mark as stopped
        with self.lock:
            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """
        GET CURRENT STATUS
        ------------------
        Returns information about the current capture session.

        This is used by the web interface to show progress updates.

        Returns:
            Dictionary with status information:
            - is_running: Are we currently capturing?
            - current_capture: Which capture are we on?
            - total_captures: How many total?
            - completed_captures: How many done?
            - successful_captures: How many succeeded?
            - failed_captures: How many failed?
            - progress_percentage: Progress as a percentage (0-100)
        """
        with self.lock:
            return {
                'is_running': self.is_running,
                'current_capture': self.current_capture,
                'total_captures': self.total_captures,
                'completed_captures': len(self.capture_results),
                'successful_captures': sum(1 for r in self.capture_results if r.success),
                'failed_captures': sum(1 for r in self.capture_results if not r.success),
                'progress_percentage': (self.current_capture / max(self.total_captures, 1)) * 100
            }

    def get_file_list(self) -> List[str]:
        """
        GET LIST OF CAPTURED FILES
        --------------------------
        Returns a list of all files that have been saved.
        """
        files = []
        for result in self.capture_results:
            if result.screenshot_file:
                files.append(result.screenshot_file)
            if result.waveform_files:
                files.extend(result.waveform_files)
        return files

    def _capture_loop(self, config: CaptureConfig):
        """
        MAIN CAPTURE LOOP (BACKGROUND WORKER)
        --------------------------------------
        This runs in a separate thread and performs all the captures.

        PROCESS:
        1. Setup oscilloscope for single trigger mode
        2. For each capture:
           a. Wait for trigger
           b. Capture screenshot
           c. Save waveform data
           d. Record measurements
           e. Wait for specified interval
        3. Clean up when done

        Parameters:
            config: The capture configuration
        """
        try:
            # Initial setup: Configure oscilloscope
            self._setup_oscilloscope_for_capture(config)

            # Main loop: Perform each capture
            for capture_idx in range(config.num_captures):
                loop_start = time.time()  # Record when we started this iteration

                # Check if user requested stop
                if self.stop_requested:
                    self.logger.info("Capture stopped by user")
                    break

                # Update current capture index
                with self.lock:
                    self.current_capture = capture_idx + 1

                self.logger.info(f"Starting capture {self.current_capture}/{config.num_captures}")

                # Perform the actual capture
                result = self._perform_single_capture(config, capture_idx)

                # Store the result
                with self.lock:
                    self.capture_results.append(result)

                # Log the outcome
                if not result.success:
                    self.logger.error(f"Capture {capture_idx + 1} failed: {result.error_message}")
                else:
                    self.logger.info(f"Capture {capture_idx + 1} completed successfully")

                # Wait before next capture (if not the last one)
                if capture_idx < config.num_captures - 1 and not self.stop_requested:
                    # Calculate how long to sleep to maintain target interval
                    elapsed = time.time() - loop_start  # How long did capture take?
                    sleep_time = max(0.0, config.time_interval - elapsed)  # Remaining time

                    self.logger.info(
                        f"Capture {capture_idx + 1} took {elapsed:.3f}s, "
                        f"sleeping {sleep_time:.3f}s to maintain {config.time_interval}s interval"
                    )

                    if sleep_time > 0:
                        time.sleep(sleep_time)

            self.logger.info("Capture session completed")

        except Exception as e:
            self.logger.error(f"Capture loop error: {e}")
        finally:
            # Always mark as not running when we exit
            with self.lock:
                self.is_running = False

    def _setup_oscilloscope_for_capture(self, config: CaptureConfig):
        """
        SETUP OSCILLOSCOPE
        ------------------
        Configures the oscilloscope to be ready for single-trigger captures.

        This:
        1. Stops any ongoing acquisition
        2. Sets acquisition type to normal
        3. Enables the selected channels
        4. Sets trigger mode to wait for valid triggers

        Parameters:
            config: Capture configuration
        """
        try:
            # Stop any ongoing acquisition
            self.scope._scpi_wrapper.write(":STOP")
            time.sleep(0.1)  # Brief pause for command to take effect

            # Set acquisition type to normal (standard mode)
            self.scope._scpi_wrapper.write(":ACQuire:TYPE NORMal")

            # Enable selected channels (turn them on for display and capture)
            for channel in config.channels:
                self.scope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay ON")

            # Set trigger sweep to NORMAL (wait for valid trigger before acquiring)
            self.scope._scpi_wrapper.write(":TRIGger:SWEep NORMal")

            self.logger.info("Oscilloscope configured for single trigger capture")

        except Exception as e:
            self.logger.error(f"Failed to setup oscilloscope: {e}")
            raise

    def _perform_single_capture(self, config: CaptureConfig, capture_idx: int) -> CaptureResult:
        """
        PERFORM SINGLE CAPTURE
        ----------------------
        Captures data from one trigger event.

        PROCESS:
        1. Set oscilloscope to SINGLE mode (capture one trigger and stop)
        2. Wait for trigger to occur
        3. Capture screenshot (if enabled)
        4. Save waveform data (if enabled)
        5. Get measurements

        Parameters:
            config: Capture configuration
            capture_idx: Index of this capture (0, 1, 2, ...)

        Returns:
            CaptureResult object with all the details
        """
        timestamp = datetime.now()  # Record when this capture started

        # Create result object to store outcome
        result = CaptureResult(
            index=capture_idx,
            timestamp=timestamp,
            waveform_files=[]
        )

        try:
            # Step 1: Set to SINGLE mode and arm trigger
            self.logger.debug("Setting SINGLE mode and waiting for trigger...")
            self.scope._scpi_wrapper.write(":SINGle")

            # Step 2: Wait for trigger to occur (with timeout)
            trigger_acquired = self._wait_for_trigger(config.trigger_timeout)

            if not trigger_acquired:
                result.success = False
                result.error_message = "Trigger timeout - no trigger detected"
                return result

            self.logger.debug("Trigger acquired successfully")

            # Brief delay to ensure display is fully updated
            time.sleep(0.1)

            # Step 3: Capture screenshot if enabled
            if config.capture_screenshots:
                screenshot_file = self._capture_screenshot(config, capture_idx, timestamp)
                if screenshot_file:
                    result.screenshot_file = screenshot_file
                    self.logger.debug(f"Screenshot saved: {screenshot_file}")

            # Step 4: Save waveform data if enabled
            if config.save_waveforms or config.save_combined_csv:
                waveform_files = self._save_waveform_data(config, capture_idx, timestamp)
                if waveform_files:
                    result.waveform_files = waveform_files
                    self.logger.debug(f"Waveforms saved: {len(waveform_files)} files")

            # Step 5: Get measurements from channels
            result.measurements = self._get_measurements(config.channels)

            result.success = True
            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Capture failed: {e}")
            return result

    def _wait_for_trigger(self, timeout: float) -> bool:
        """
        WAIT FOR TRIGGER
        ----------------
        Waits for the oscilloscope to detect a trigger event.

        This repeatedly checks the oscilloscope status until either:
        - A trigger occurs (return True)
        - Timeout expires (return False)

        Parameters:
            timeout: Maximum seconds to wait

        Returns:
            True if trigger occurred, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Query the operation status register
                # This tells us if the oscilloscope is currently acquiring
                oper_status = self.scope._scpi_wrapper.query(":OPERegister:CONDition?")

                # Bit 3 (value 8) indicates "Run" status
                # When this bit is 0, the acquisition is complete (trigger happened)
                if int(oper_status) & 8 == 0:
                    return True  # Trigger acquired!

                time.sleep(0.1)  # Check every 100ms

            except Exception as e:
                self.logger.warning(f"Error checking trigger status: {e}")
                time.sleep(0.1)

        return False  # Timeout - no trigger detected

    def _capture_screenshot(self, config: CaptureConfig, capture_idx: int,
                          timestamp: datetime) -> Optional[str]:
        """
        CAPTURE SCREENSHOT
        ------------------
        Takes a screenshot of the oscilloscope display and saves it as PNG.

        Process:
        1. Generate unique filename with timestamp
        2. Request screenshot data from oscilloscope
        3. Save as PNG file

        Parameters:
            config: Capture configuration
            capture_idx: Capture number
            timestamp: When this capture occurred

        Returns:
            Full path to saved file, or None if failed
        """
        try:
            # Generate filename with timestamp (includes milliseconds for uniqueness)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{config.base_filename}_screenshot_{capture_idx:04d}_{timestamp_str}.png"
            filepath = Path(config.save_directory) / filename

            # Request screenshot data from oscilloscope in PNG format
            self.logger.debug("Capturing screenshot...")
            image_data = self.scope._scpi_wrapper.query_binary_values(
                ":DISPlay:DATA? PNG",  # Command to get display as PNG
                datatype='B'           # Binary data type
            )

            if image_data:
                # Save the binary image data to file
                with open(filepath, 'wb') as f:
                    f.write(bytes(image_data))
                return str(filepath)
            else:
                self.logger.warning("No screenshot data received")
                return None

        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None

    def _save_waveform_data(self, config: CaptureConfig, capture_idx: int,
                           timestamp: datetime) -> List[str]:
        """
        SAVE WAVEFORM DATA
        ------------------
        Retrieves the actual waveform data (time and voltage values) from
        the oscilloscope and saves to CSV files.

        This can save:
        1. Individual CSV files per channel (if save_waveforms is True)
        2. Combined multi-channel CSV (if save_combined_csv is True)

        Parameters:
            config: Capture configuration
            capture_idx: Capture number
            timestamp: When this capture occurred

        Returns:
            List of file paths that were saved
        """
        saved_files: List[str] = []
        channel_traces: Dict[int, Tuple[List[float], List[float]]] = {}

        # Loop through each selected channel
        for channel in config.channels:
            try:
                # Configure oscilloscope for waveform export
                self.scope._scpi_wrapper.write(f":WAVeform:SOURce CHANnel{channel}")
                self.scope._scpi_wrapper.write(":WAVeform:FORMat BYTE")
                self.scope._scpi_wrapper.write(":WAVeform:POINts:MODE RAW")
                self.scope._scpi_wrapper.write(":WAVeform:POINts 62500")  # Maximum points

                # Get waveform preamble (contains scaling information)
                preamble = self.scope._scpi_wrapper.query(":WAVeform:PREamble?")
                preamble_parts = preamble.split(',')

                # Extract scaling factors from preamble
                # These convert raw byte values to actual voltage/time values
                y_increment = float(preamble_parts[7])  # Volts per bit
                y_origin = float(preamble_parts[8])     # Voltage offset
                y_reference = float(preamble_parts[9])  # Reference level
                x_increment = float(preamble_parts[4])  # Seconds per point
                x_origin = float(preamble_parts[5])     # Time offset

                # Get raw waveform data (as bytes)
                raw_data = self.scope._scpi_wrapper.query_binary_values(
                    ":WAVeform:DATA?",
                    datatype='B'
                )

                # Convert raw bytes to actual voltage values
                voltage_data = [(val - y_reference) * y_increment + y_origin for val in raw_data]

                # Create time axis (one time value for each voltage point)
                time_data = [x_origin + i * x_increment for i in range(len(voltage_data))]

                # Store for potential combined CSV
                channel_traces[channel] = (time_data, voltage_data)

                # Save individual channel CSV if requested
                if config.save_waveforms:
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_CH{channel}_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / filename

                    # Create DataFrame (table) with time and voltage columns
                    df = pd.DataFrame({
                        'Time (s)': time_data,
                        'Voltage (V)': voltage_data
                    })

                    # Save to CSV with header comments
                    with open(filepath, 'w') as f:
                        # Write metadata as comments
                        f.write(f"# Channel: {channel}\n")
                        f.write(f"# Capture Index: {capture_idx}\n")
                        f.write(f"# Timestamp: {timestamp.isoformat()}\n")
                        f.write(f"# Sample Rate: {1.0/x_increment:.2e} Hz\n")
                        f.write(f"# Points: {len(voltage_data)}\n")
                        f.write("\n")
                        # Write actual data
                        df.to_csv(f, index=False)

                    saved_files.append(str(filepath))

            except Exception as e:
                self.logger.error(f"Failed to save waveform for channel {channel}: {e}")

        # Save combined multi-channel CSV if requested
        if config.save_combined_csv and channel_traces:
            try:
                # Find shortest trace (in case channels have different lengths)
                min_len = min(len(tr[0]) for tr in channel_traces.values())

                if min_len > 0:
                    # Use time axis from first channel
                    first_channel = config.channels[0]
                    time_ref = channel_traces[first_channel][0][:min_len]

                    # Build data dictionary with time and all channel voltages
                    data = {'Time (s)': time_ref}
                    for ch, (t_data, v_data) in channel_traces.items():
                        data[f'CH{ch} (V)'] = v_data[:min_len]

                    # Create DataFrame and save
                    df_multi = pd.DataFrame(data)
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{config.base_filename}_MULTI_{capture_idx:04d}_{timestamp_str}.csv"
                    filepath = Path(config.save_directory) / filename

                    # Save with metadata header
                    with open(filepath, 'w') as f:
                        f.write(f"# Channels: {', '.join(str(ch) for ch in config.channels)}\n")
                        f.write(f"# Capture Index: {capture_idx}\n")
                        f.write(f"# Timestamp: {timestamp.isoformat()}\n")
                        f.write(f"# Points: {min_len}\n")
                        f.write("\n")
                        df_multi.to_csv(f, index=False)

                    saved_files.append(str(filepath))

            except Exception as e:
                self.logger.error(f"Failed to save combined waveform CSV: {e}")

        return saved_files

    def _get_measurements(self, channels: List[int]) -> Dict[str, Any]:
        """
        GET MEASUREMENTS
        ----------------
        Retrieves standard measurements from each channel.

        Measurements include:
        - FREQ: Frequency (Hz)
        - PERiod: Period (seconds)
        - VPP: Peak-to-peak voltage
        - VAVG: Average voltage
        - VRMS: RMS (root mean square) voltage

        Parameters:
            channels: List of channel numbers to measure

        Returns:
            Dictionary with measurements for each channel
        """
        measurements = {}

        for channel in channels:
            try:
                ch_measurements = {}

                # List of measurement types to retrieve
                measurement_types = ['FREQ', 'PERiod', 'VPP', 'VAVG', 'VRMS']

                for meas_type in measurement_types:
                    try:
                        value = self.scope.measure_single(channel, meas_type)
                        if value is not None:
                            ch_measurements[meas_type] = value
                    except:
                        pass  # Skip measurements that fail

                measurements[f'CH{channel}'] = ch_measurements

            except Exception as e:
                self.logger.warning(f"Failed to get measurements for channel {channel}: {e}")

        return measurements


# ============================================================================
# GRAPHICAL USER INTERFACE (WEB-BASED)
# This creates a web interface so users can control the system from a browser
# ============================================================================

class TriggerCaptureGUI:
    """
    TRIGGER CAPTURE WEB INTERFACE
    -----------------------------
    This creates a user-friendly web interface using Gradio.

    The interface has tabs for:
    1. Connection - Connect to oscilloscope
    2. Channel Configuration - Setup vertical settings
    3. Timebase & Trigger - Setup horizontal and trigger settings
    4. Capture Setup - Configure capture parameters
    5. Capture Control - Start/stop capture and monitor progress
    6. Results - View files and generate reports
    """

    def __init__(self):
        """
        INITIALIZATION
        --------------
        Sets up the GUI application.
        """
        self.oscilloscope = None      # Will hold oscilloscope connection
        self.capture_engine = None    # Will hold capture engine
        self.logger = self._setup_logging()  # Setup logging

        # Default save directory
        self.default_save_dir = Path.cwd() / "trigger_captures"
        self.default_save_dir.mkdir(parents=True, exist_ok=True)

        # Timebase scale options (from nanoseconds to seconds)
        self.timebase_scales: List[Tuple[str, float]] = [
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
            ("10 s", 10.0), ("20 s", 20.0), ("50 s", 50.0),
        ]

        # Setup cleanup handlers (ensure proper shutdown)
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """
        SETUP LOGGING
        -------------
        Configures the logging system to record program activities.
        """
        logger = logging.getLogger('TriggerCapture')
        logger.setLevel(logging.INFO)

        # Console handler (prints to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter (how log messages look)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _signal_handler(self, signum, frame):
        """
        SIGNAL HANDLER
        --------------
        Handles system signals like Ctrl+C to shutdown gracefully.
        """
        print(f"\nReceived signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """
        CLEANUP
        -------
        Performs cleanup when the program exits.
        This ensures the oscilloscope is properly disconnected.
        """
        try:
            if self.capture_engine:
                self.capture_engine.stop_capture_session()

            if self.oscilloscope and self.oscilloscope.is_connected:
                self.oscilloscope.disconnect()

            self.logger.info("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")

    # ========================================================================
    # CONNECTION MANAGEMENT
    # Functions to connect/disconnect from oscilloscope
    # ========================================================================

    def connect_oscilloscope(self, visa_address: str) -> Tuple[str, str]:
        """
        CONNECT TO OSCILLOSCOPE
        -----------------------
        Establishes connection to the oscilloscope via USB or network.

        Parameters:
            visa_address: Device address (e.g., "USB0::0x0957::0x1780::...")

        Returns:
            (info_message, status) - Information text and connection status
        """
        try:
            if not visa_address:
                return "Error: VISA address required", "Disconnected"

            # Create oscilloscope object
            self.oscilloscope = KeysightDSOX6004A(visa_address)

            # Attempt connection
            if self.oscilloscope.connect():
                # Create capture engine
                self.capture_engine = TriggerCaptureEngine(self.oscilloscope)

                # Get instrument info
                info = self.oscilloscope.get_instrument_info()
                if info:
                    info_text = (
                        f"Connected to {info['manufacturer']} {info['model']}\n"
                        f"Serial: {info['serial_number']}\n"
                        f"Firmware: {info['firmware_version']}"
                    )
                    self.logger.info(f"Connected to {info['model']}")
                    return info_text, "Connected"
            else:
                return "Connection failed", "Disconnected"

        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return f"Error: {str(e)}", "Disconnected"

    def disconnect_oscilloscope(self) -> Tuple[str, str]:
        """
        DISCONNECT FROM OSCILLOSCOPE
        ----------------------------
        Closes connection to the oscilloscope.
        """
        try:
            if self.capture_engine:
                self.capture_engine.stop_capture_session()

            if self.oscilloscope:
                self.oscilloscope.disconnect()

            self.oscilloscope = None
            self.capture_engine = None

            self.logger.info("Disconnected from oscilloscope")
            return "Disconnected successfully", "Disconnected"

        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            return f"Error: {str(e)}", "Disconnected"

    def test_connection(self) -> str:
        """
        TEST CONNECTION
        ---------------
        Verifies that the oscilloscope is still connected and responding.
        """
        if self.oscilloscope and getattr(self.oscilloscope, "is_connected", False):
            return "Connection test: PASSED"
        return "Connection test: FAILED - Not connected"

    # ========================================================================
    # CHANNEL CONFIGURATION
    # Functions to setup oscilloscope channels
    # ========================================================================

    def configure_channel(
        self,
        ch1: bool, ch2: bool, ch3: bool, ch4: bool,
        v_scale: float,
        v_offset: float,
        coupling: str,
        probe: float
    ) -> str:
        """
        CONFIGURE CHANNELS
        ------------------
        Sets up the vertical parameters for selected channels.

        Parameters:
            ch1, ch2, ch3, ch4: Which channels to configure (True/False)
            v_scale: Volts per division (e.g., 1.0 means 1V per division)
            v_offset: Vertical offset in volts
            coupling: "AC" or "DC" coupling
            probe: Probe attenuation (1x, 10x, or 100x)

        Returns:
            Status message
        """
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"

        channel_states = {1: ch1, 2: ch2, 3: ch3, 4: ch4}

        try:
            success_count = 0
            disabled_count = 0

            for channel, enabled in channel_states.items():
                if enabled:
                    # Configure enabled channels
                    try:
                        success = self.oscilloscope.configure_channel(
                            channel=channel,
                            vertical_scale=v_scale,
                            vertical_offset=v_offset,
                            coupling=coupling,
                            probe_attenuation=probe
                        )
                        if success:
                            success_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to configure channel {channel}: {e}")
                else:
                    # Disable unchecked channels
                    try:
                        self.oscilloscope._scpi_wrapper.write(f":CHANnel{channel}:DISPlay OFF")
                        disabled_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to disable channel {channel}: {e}")

            return f"Configured: {success_count} enabled, {disabled_count} disabled"
        except Exception as e:
            return f"Configuration error: {str(e)}"

    # ========================================================================
    # CAPTURE CONTROL
    # Functions to start/stop captures and get status
    # ========================================================================

    def start_capture(
        self,
        num_captures: int,
        time_interval: float,
        ch1: bool, ch2: bool, ch3: bool, ch4: bool,
        base_filename: str,
        save_directory: str,
        capture_screenshots: bool,
        save_waveforms: bool,
        save_combined_csv: bool,
        trigger_timeout: float
    ) -> str:
        """
        START CAPTURE SESSION
        ---------------------
        Begins the automated capture process with specified parameters.

        Parameters:
            num_captures: How many trigger events to capture
            time_interval: Seconds between captures
            ch1-ch4: Which channels to capture
            base_filename: Base name for files
            save_directory: Where to save files
            capture_screenshots: Save PNG screenshots?
            save_waveforms: Save individual CSV files?
            save_combined_csv: Save multi-channel CSV?
            trigger_timeout: Max seconds to wait for trigger

        Returns:
            Status message
        """
        if not self.capture_engine:
            return "Error: Not connected to oscilloscope"

        # Parse selected channels
        channels = []
        if ch1: channels.append(1)
        if ch2: channels.append(2)
        if ch3: channels.append(3)
        if ch4: channels.append(4)

        if not channels:
            return "Error: No channels selected"

        # Create configuration
        config = CaptureConfig(
            num_captures=num_captures,
            time_interval=time_interval,
            channels=channels,
            base_filename=base_filename,
            save_directory=save_directory,
            capture_screenshots=capture_screenshots,
            save_waveforms=save_waveforms,
            save_combined_csv=save_combined_csv,
            trigger_timeout=trigger_timeout
        )

        # Validate configuration
        valid, msg = config.validate()
        if not valid:
            return f"Configuration error: {msg}"

        # Start capture
        if self.capture_engine.start_capture_session(config):
            self.logger.info(f"Started capture: {num_captures} triggers")
            return (
                f"STARTED CAPTURE SESSION\n"
                f"-----------------------------------------\n"
                f"Captures: {num_captures}\n"
                f"Interval: {time_interval}s\n"
                f"Channels: {channels}\n"
                f"Screenshots: {'Yes' if capture_screenshots else 'No'}\n"
                f"Waveforms: {'Yes' if save_waveforms else 'No'}\n"
                f"Directory: {save_directory}"
            )
        else:
            return "Failed to start capture"

    def stop_capture(self) -> str:
        """
        STOP CAPTURE SESSION
        --------------------
        Stops the ongoing capture session.
        """
        if not self.capture_engine:
            return "Error: Not connected"

        self.capture_engine.stop_capture_session()
        time.sleep(0.1)

        status = self.capture_engine.get_status()
        return (
            f"CAPTURE STOPPED\n"
            f"-----------------------------------------\n"
            f"Completed: {status['completed_captures']}/{status['total_captures']}\n"
            f"Successful: {status['successful_captures']}\n"
            f"Failed: {status['failed_captures']}"
        )

    def get_status(self) -> str:
        """
        GET CURRENT STATUS
        ------------------
        Returns current capture progress and statistics.
        """
        if not self.capture_engine:
            return "Status: Not connected"

        status = self.capture_engine.get_status()

        if status['is_running']:
            status_icon = "RUNNING"
            progress_bar = self._create_progress_bar(status['progress_percentage'])
        else:
            status_icon = "IDLE"
            progress_bar = ""

        return (
            f"{status_icon}\n"
            f"-----------------------------------------\n"
            f"Progress: {status['current_capture']}/{status['total_captures']} "
            f"({status['progress_percentage']:.1f}%)\n"
            f"{progress_bar}\n"
            f"Successful: {status['successful_captures']}\n"
            f"Failed: {status['failed_captures']}\n"
            f"Total Files: {len(self.capture_engine.get_file_list())}"
        )

    def get_file_list(self) -> str:
        """
        GET FILE LIST
        -------------
        Returns a formatted list of all captured files.
        """
        if not self.capture_engine:
            return "No files - not connected"

        files = self.capture_engine.get_file_list()

        if not files:
            return "No files captured yet"

        # Group files by type
        screenshots = [f for f in files if 'screenshot' in f]
        waveforms = [f for f in files if 'screenshot' not in f]

        output = ["CAPTURED FILES", "-----------------------------------------"]

        if screenshots:
            output.append(f"\nScreenshots ({len(screenshots)}):")
            for f in screenshots[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(screenshots) > 5:
                output.append(f"  ... and {len(screenshots)-5} more")

        if waveforms:
            output.append(f"\nWaveforms ({len(waveforms)}):")
            for f in waveforms[-5:]:  # Show last 5
                output.append(f"  • {Path(f).name}")
            if len(waveforms) > 5:
                output.append(f"  ... and {len(waveforms)-5} more")

        return "\n".join(output)

    def get_capture_summary(self) -> str:
        """
        GET CAPTURE SUMMARY
        -------------------
        Generates a comprehensive summary of the capture session.
        """
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "No capture data available"

        results = self.capture_engine.capture_results

        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        # Get timing info
        if results:
            duration = (results[-1].timestamp - results[0].timestamp).total_seconds()
            avg_time_per_capture = duration / max(total - 1, 1) if total > 1 else 0
        else:
            duration = 0
            avg_time_per_capture = 0

        # Count files
        total_screenshots = sum(1 for r in results if r.screenshot_file)
        total_waveforms = sum(len(r.waveform_files) for r in results if r.waveform_files)

        summary = [
            "CAPTURE SESSION SUMMARY",
            "-----------------------------------------",
            "",
            "STATISTICS:",
            f"  • Total Captures: {total}",
            f"  • Successful: {successful} ({(successful/total*100):.1f}%)",
            f"  • Failed: {failed}",
            f"  • Success Rate: {(successful/total*100):.1f}%",
            "",
            "TIMING:",
            f"  • Total Duration: {duration:.1f} seconds",
            f"  • Avg Time/Capture: {avg_time_per_capture:.2f} seconds",
            "",
            "FILES SAVED:",
            f"  • Screenshots: {total_screenshots}",
            f"  • Waveform Files: {total_waveforms}",
            f"  • Total Files: {total_screenshots + total_waveforms}",
        ]

        # Add recent measurements if available
        recent_results = [r for r in results[-5:] if r.success and r.measurements]
        if recent_results:
            summary.extend(["", "RECENT MEASUREMENTS:"])

            for result in recent_results:
                summary.append(f"\n  Capture {result.index + 1}:")
                for ch, meas in result.measurements.items():
                    if meas and 'FREQ' in meas:
                        freq = meas.get('FREQ', 0)
                        vpp = meas.get('VPP', 0)
                        summary.append(f"    {ch}: {freq:.2e} Hz, {vpp:.3f} Vpp")

        return "\n".join(summary)

    def save_summary_report(self, save_directory: str) -> str:
        """
        SAVE SUMMARY REPORT
        -------------------
        Saves a detailed summary report to a JSON file.
        """
        if not self.capture_engine or not self.capture_engine.capture_results:
            return "No data to save"

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_summary_{timestamp}.json"
            filepath = Path(save_directory) / filename

            # Prepare summary data
            results = self.capture_engine.capture_results
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'total_captures': len(results),
                'successful_captures': sum(1 for r in results if r.success),
                'failed_captures': sum(1 for r in results if not r.success),
                'captures': []
            }

            for result in results:
                capture_data = {
                    'index': result.index,
                    'timestamp': result.timestamp.isoformat(),
                    'success': result.success,
                    'screenshot': result.screenshot_file,
                    'waveforms': result.waveform_files or [],
                    'measurements': result.measurements or {},
                    'error': result.error_message
                }
                summary_data['captures'].append(capture_data)

            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)

            return f"Summary saved: {filename}"

        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
            return f"Error saving summary: {str(e)}"

    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """
        CREATE PROGRESS BAR
        -------------------
        Creates a text-based progress bar for status display.
        """
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    # ========================================================================
    # OSCILLOSCOPE CONFIGURATION
    # Functions to setup trigger and timebase
    # ========================================================================

    def configure_trigger(self, source: str, level: float, slope: str) -> str:
        """
        CONFIGURE TRIGGER
        -----------------
        Sets the trigger parameters.

        Parameters:
            source: Which channel to trigger on ("CH1", "CH2", etc.)
            level: Voltage level that causes trigger
            slope: "Rising", "Falling", or "Either"
        """
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            channel = int(source.replace("CH", ""))

            # Set trigger parameters
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SOURce CHANnel{channel}")
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:LEVel {level}")

            slope_map = {"Rising": "POS", "Falling": "NEG", "Either": "EITH"}
            self.oscilloscope._scpi_wrapper.write(f":TRIGger:EDGE:SLOPe {slope_map[slope]}")

            # Set to NORMAL mode for stable triggering
            self.oscilloscope._scpi_wrapper.write(":TRIGger:SWEep NORMal")

            return f"Trigger configured: {source} @ {level} V, {slope}"

        except Exception as e:
            return f"Error: {str(e)}"

    def configure_timebase(self, time_scale: float) -> str:
        """
        CONFIGURE TIMEBASE
        ------------------
        Sets the horizontal time scale (seconds per division).
        """
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            success = self.oscilloscope.configure_timebase(time_scale)
            if success:
                return f"Timebase set: {time_scale} s/div"
            return "Timebase configuration failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def run_autoscale(self) -> str:
        """
        RUN AUTOSCALE
        -------------
        Executes automatic scaling (like pressing "Auto" button on scope).
        """
        if not self.oscilloscope or not self.oscilloscope.is_connected:
            return "Error: Not connected"

        try:
            self.oscilloscope._scpi_wrapper.write(":AUToscale")
            time.sleep(3)  # Wait for autoscale to complete
            return "Autoscale completed"
        except Exception as e:
            return f"Error: {str(e)}"

    def perform_autoscale(self) -> str:
        """Alias for run_autoscale"""
        return self.run_autoscale()

    # ========================================================================
    # ACQUISITION CONTROL
    # Functions to configure acquisition settings
    # ========================================================================

    def set_acquisition_mode(self, mode_type: str) -> str:
        """Set acquisition mode (RTIMe, ETIMe, SEGMented)"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"

        try:
            success = self.oscilloscope.set_acquire_mode(mode_type)
            if success:
                return f"Acquisition mode: {mode_type}"
            return "Failed to set acquisition mode"
        except Exception as e:
            return f"Error: {str(e)}"

    def set_acquisition_type(self, acq_type: str) -> str:
        """Set acquisition type (NORMal, AVERage, HRESolution, PEAK)"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"

        try:
            success = self.oscilloscope.set_acquire_type(acq_type)
            if success:
                return f"Acquisition type: {acq_type}"
            return "Failed to set acquisition type"
        except Exception as e:
            return f"Error: {str(e)}"

    def set_acquisition_count(self, average_count: int) -> str:
        """Set number of acquisitions to average"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"

        try:
            if not (2 <= average_count <= 65536):
                return f"Error: Count must be 2-65536, got {average_count}"

            success = self.oscilloscope.set_acquire_count(int(average_count))
            if success:
                return f"Acquisition count: {int(average_count)} averages"
            return "Failed to set acquisition count"
        except Exception as e:
            return f"Error: {str(e)}"

    def query_acquisition_info(self) -> str:
        """Query current acquisition parameters"""
        if not self.oscilloscope or not getattr(self.oscilloscope, "is_connected", False):
            return "Error: Not connected"

        try:
            info_lines = []

            mode = self.oscilloscope.get_acquire_mode()
            acq_type = self.oscilloscope.get_acquire_type()
            count = self.oscilloscope.get_acquire_count()
            sample_rate = self.oscilloscope.get_sample_rate()
            points = self.oscilloscope.get_acquire_points()

            if mode is not None:
                info_lines.append(f"Mode: {mode}")
            if acq_type is not None:
                info_lines.append(f"Type: {acq_type}")
            if count is not None:
                info_lines.append(f"Count: {count}")
            if sample_rate is not None:
                info_lines.append(f"Sample Rate: {sample_rate:.3e} Hz")
            if points is not None:
                info_lines.append(f"Acquired Points: {points}")

            return "\n".join(info_lines) if info_lines else "No acquisition info available"
        except Exception as e:
            return f"Error: {str(e)}"

    # ========================================================================
    # CREATE WEB INTERFACE
    # Builds the Gradio web interface
    # ========================================================================

    def create_interface(self) -> gr.Blocks:
        """
        CREATE GRADIO INTERFACE
        -----------------------
        Builds the complete web interface with all tabs and controls.
        """

        # Custom CSS for full-page layout
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
            title="Trigger Capture Automation",
            css=css,
            theme=gr.themes.Soft(
                primary_hue="purple",
                spacing_size="sm",
                radius_size="sm",
                text_size="sm"
            )
        ) as interface:

            # Header
            gr.Markdown("# Continuous Trigger Capture System")
            gr.Markdown("Automated Screenshot & Data Capture on Each Trigger Event")
            gr.Markdown("Developed by: Anirudh Iyengar | Digantara Research and Technologies")

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

                connection_info = gr.Textbox(
                    label="Connection Info",
                    lines=3,
                    interactive=False
                )
                connection_status = gr.Textbox(
                    label="Status",
                    value="Disconnected",
                    interactive=False
                )

                # Wire up buttons
                connect_btn.click(
                    self.connect_oscilloscope,
                    inputs=[visa_address],
                    outputs=[connection_info, connection_status]
                )
                disconnect_btn.click(
                    self.disconnect_oscilloscope,
                    outputs=[connection_info, connection_status]
                )
                test_btn.click(
                    self.test_connection,
                    outputs=[connection_info]
                )

            # ================================================================
            # CHANNEL CONFIGURATION TAB
            # ================================================================
            with gr.Tab("Channel Configuration"):
                gr.Markdown("### Channel Selection and Configuration")

                with gr.Row():
                    ch1_cfg = gr.Checkbox(label="Ch1", value=True)
                    ch2_cfg = gr.Checkbox(label="Ch2", value=False)
                    ch3_cfg = gr.Checkbox(label="Ch3", value=False)
                    ch4_cfg = gr.Checkbox(label="Ch4", value=False)

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
                autoscale_channels_btn = gr.Button("Autoscale", variant="secondary")
                channel_system_status = gr.Textbox(label="System Status", interactive=False, lines=3)

                # Wire up buttons
                config_channel_btn.click(
                    self.configure_channel,
                    inputs=[ch1_cfg, ch2_cfg, ch3_cfg, ch4_cfg, v_scale, v_offset, coupling, probe],
                    outputs=[channel_status]
                )
                autoscale_channels_btn.click(
                    self.perform_autoscale,
                    outputs=[channel_system_status]
                )

            # ================================================================
            # TIMEBASE & TRIGGER TAB
            # ================================================================
            with gr.Tab("Timebase & Trigger"):
                gr.Markdown("### Configure oscilloscope before starting capture")

                gr.Markdown("### Trigger Configuration")
                with gr.Row():
                    trigger_source = gr.Dropdown(
                        label="Trigger Source",
                        choices=["CH1", "CH2", "CH3", "CH4"],
                        value="CH1"
                    )
                    trigger_level = gr.Number(
                        label="Trigger Level (V)",
                        value=0.0,
                        step=0.1
                    )
                    trigger_slope = gr.Dropdown(
                        label="Trigger Slope",
                        choices=["Rising", "Falling", "Either"],
                        value="Rising"
                    )
                    trigger_btn = gr.Button("Apply Trigger", variant="primary")

                trigger_status = gr.Textbox(label="Trigger Status", interactive=False)

                gr.Markdown("### Timebase")
                with gr.Row():
                    time_scale = gr.Dropdown(
                        label="Time/div",
                        choices=self.timebase_scales,
                        value=10e-3
                    )
                    timebase_btn = gr.Button("Apply Timebase", variant="primary")
                    autoscale_btn = gr.Button("Autoscale", variant="secondary")

                timebase_status = gr.Textbox(label="Status", interactive=False)

                # Wire up buttons
                trigger_btn.click(
                    self.configure_trigger,
                    inputs=[trigger_source, trigger_level, trigger_slope],
                    outputs=[trigger_status]
                )
                timebase_btn.click(
                    self.configure_timebase,
                    inputs=[time_scale],
                    outputs=[timebase_status]
                )
                autoscale_btn.click(
                    self.run_autoscale,
                    outputs=[timebase_status]
                )

            # ================================================================
            # CAPTURE SETUP TAB
            # ================================================================
            with gr.Tab("Capture Setup"):
                gr.Markdown("### Capture Parameters")

                with gr.Row():
                    num_captures = gr.Number(
                        label="Number of Captures",
                        value=10,
                        minimum=1,
                        maximum=10000,
                        step=1,
                        info="How many trigger events to capture"
                    )
                    time_interval = gr.Number(
                        label="Interval Between Captures (s)",
                        value=1.0,
                        minimum=0,
                        step=0.1,
                        info="Target interval between captures"
                    )
                    trigger_timeout = gr.Number(
                        label="Trigger Timeout (s)",
                        value=10.0,
                        minimum=1,
                        maximum=60,
                        info="Max time to wait for trigger"
                    )

                gr.Markdown("### Channel Selection")
                with gr.Row():
                    ch1_select = gr.Checkbox(label="Channel 1", value=True)
                    ch2_select = gr.Checkbox(label="Channel 2", value=True)
                    ch3_select = gr.Checkbox(label="Channel 3", value=False)
                    ch4_select = gr.Checkbox(label="Channel 4", value=False)

                gr.Markdown("### Save Options")
                with gr.Row():
                    capture_screenshots = gr.Checkbox(
                        label="Capture Screenshots",
                        value=True,
                        info="Save oscilloscope display after each trigger"
                    )
                    save_waveforms = gr.Checkbox(
                        label="Save Waveform Data",
                        value=True,
                        info="Export waveform data to CSV files"
                    )
                    save_combined_csv = gr.Checkbox(
                        label="Save Combined Multi-Channel CSV",
                        value=False,
                        info="Save all channels into one CSV per capture"
                    )

                gr.Markdown("### File Settings")
                with gr.Row():
                    base_filename = gr.Textbox(
                        label="Base Filename",
                        value="trigger_capture",
                        scale=2
                    )
                    save_directory = gr.Textbox(
                        label="Save Directory",
                        value=str(self.default_save_dir),
                        scale=3
                    )

                # Time estimate calculator
                def calculate_time(n, interval):
                    total = n * (interval + 2)  # Add ~2s for capture/save overhead
                    return f"Estimated time: ~{total:.1f} seconds ({total/60:.1f} minutes)"

                time_estimate = gr.Textbox(
                    label="Time Estimate",
                    interactive=False
                )

                num_captures.change(
                    calculate_time,
                    inputs=[num_captures, time_interval],
                    outputs=[time_estimate]
                )
                time_interval.change(
                    calculate_time,
                    inputs=[num_captures, time_interval],
                    outputs=[time_estimate]
                )

            # ================================================================
            # CAPTURE CONTROL TAB
            # ================================================================
            with gr.Tab("Capture Control"):
                gr.Markdown("### Control Panel")

                with gr.Row():
                    start_btn = gr.Button(
                        "Start Capture",
                        variant="primary",
                        scale=2,
                        elem_classes=["large-button"]
                    )
                    stop_btn = gr.Button(
                        "Stop",
                        variant="stop",
                        scale=1
                    )

                capture_output = gr.Textbox(
                    label="Capture Output",
                    lines=8,
                    interactive=False
                )

                gr.Markdown("### Live Status")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", scale=1)
                    auto_refresh = gr.Checkbox(
                        label="Auto Refresh (2s)",
                        value=False
                    )

                status_display = gr.Textbox(
                    label="Current Status",
                    lines=7,
                    interactive=False
                )

                # Wire up buttons
                start_btn.click(
                    self.start_capture,
                    inputs=[
                        num_captures, time_interval,
                        ch1_select, ch2_select, ch3_select, ch4_select,
                        base_filename, save_directory,
                        capture_screenshots, save_waveforms, save_combined_csv, trigger_timeout
                    ],
                    outputs=[capture_output]
                )

                stop_btn.click(
                    self.stop_capture,
                    outputs=[capture_output]
                )

                refresh_btn.click(
                    self.get_status,
                    outputs=[status_display]
                )

                # Auto-refresh functionality
                def auto_update(enable):
                    if enable:
                        return self.get_status()
                    return "Auto-refresh disabled"

                auto_refresh.change(
                    auto_update,
                    inputs=[auto_refresh],
                    outputs=[status_display],
                    every=2
                )

            # ================================================================
            # RESULTS TAB
            # ================================================================
            with gr.Tab("Results"):
                gr.Markdown("### Captured Files")

                with gr.Row():
                    show_files_btn = gr.Button("Show Files", variant="primary")
                    summary_btn = gr.Button("Generate Summary", variant="primary")
                    save_report_btn = gr.Button("Save Report", variant="secondary")

                file_list_display = gr.Textbox(
                    label="File List",
                    lines=15,
                    interactive=False
                )

                summary_display = gr.Textbox(
                    label="Capture Summary",
                    lines=20,
                    interactive=False
                )

                report_status = gr.Textbox(
                    label="Report Status",
                    interactive=False
                )

                # Wire up buttons
                show_files_btn.click(
                    self.get_file_list,
                    outputs=[file_list_display]
                )

                summary_btn.click(
                    self.get_capture_summary,
                    outputs=[summary_display]
                )

                save_report_btn.click(
                    lambda: self.save_summary_report(str(self.default_save_dir)),
                    outputs=[report_status]
                )

            # Footer
            gr.Markdown("---")
            gr.Markdown("**Trigger Capture System** v2.0 | Professional Oscilloscope Automation")

        return interface

    def launch(self, share=False, server_port=7866):
        """
        LAUNCH INTERFACE
        ----------------
        Starts the web server and opens the interface in a browser.

        Parameters:
            share: Create public link? (default False)
            server_port: Port number for web server (default 7866)
        """
        interface = self.create_interface()

        print("\n" + "="*60)
        print("TRIGGER CAPTURE AUTOMATION SYSTEM")
        print("Screenshot & Data Capture on Trigger Events")
        print("="*60)
        print(f"Starting web interface on port {server_port}...")

        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=server_port,
                share=share,
                show_error=True,
                inbrowser=True
            )
        except Exception as e:
            print(f"Failed to launch: {e}")
            self.cleanup()


# ============================================================================
# MAIN ENTRY POINT
# This is where the program starts when you run it
# ============================================================================

def main():
    """
    MAIN FUNCTION
    -------------
    Entry point when running this script directly.

    This function:
    1. Prints startup message
    2. Creates the GUI application
    3. Launches the web interface
    4. Handles shutdown gracefully
    """
    print("Initializing Trigger Capture System...")
    print("Developed by: Anirudh Iyengar")
    print("Organization: Digantara Research and Technologies")
    print("-" * 60)

    # Create the GUI application
    app = TriggerCaptureGUI()

    try:
        # Launch the web interface
        app.launch(share=False, server_port=7866)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Clean up resources
        app.cleanup()
        print("Application terminated")


# This ensures main() only runs when script is executed directly,
# not when imported as a module
if __name__ == "__main__":
    main()
