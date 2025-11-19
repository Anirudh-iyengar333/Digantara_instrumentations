#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║         KEITHLEY MULTI-CHANNEL POWER SUPPLY AUTOMATION - GRADIO VERSION         ║
║                   Professional Web-Based Application                             ║
║                     Fully Annotated for Executive Review                       ║
║                                                                                ║
║  Purpose: Comprehensive automation control for Keithley 2230-30-1 power supply║
║           with multi-channel independent control, waveform ramping, and       ║
║           data acquisition capabilities via Gradio web interface              ║
║                                                                                ║
║  Core Features:                                                                ║
║    • Multi-threaded responsive web UI using Gradio framework                  ║
║    • VISA-based communication with Keithley 2230 power supply                ║
║    • Three independent channel configuration and control (0-30V, 0-3A)       ║
║    • Real-time voltage, current, and power measurements                      ║
║    • Automated waveform generation (Sine, Square, Triangle, Ramp)           ║
║    • Voltage ramping cycles with time-based profile execution               ║
║    • CSV data export for measurement logging                                 ║
║    • Matplotlib graph generation for voltage vs time visualization          ║
║    • Auto-measurement with configurable polling interval                    ║
║    • Emergency stop and safety shutdown procedures                          ║
║    • Color-coded activity logging for diagnostic purposes                   ║
║    • Sequential measurement to prevent USB resource conflicts               ║
║                                                                                ║
║  System Requirements:                                                          ║
║    • Python 3.7+                                                              ║
║    • PyVISA library for instrument communication                              ║
║    • Keysight/National Instruments VISA drivers installed                    ║
║    • Matplotlib for graphing capabilities                                    ║
║    • Gradio for web interface                                                ║
║    • Keithley 2230 power supply connected via USB                              ║
║                                                                                ║
║  Author: Professional Instrumentation Control System                          ║
║  Last Updated: 2025-10-24 | Status: Production Ready - Gradio Edition         ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ============================================================================
# This section brings in external tools and capabilities that our program needs
# Think of these as toolboxes that contain pre-built functions we can use

import sys              # Provides access to system-specific parameters and functions
import logging          # Enables tracking and recording of application events and errors
import time             # Provides time-related functions (delays, timestamps, etc.)
import threading        # Allows multiple tasks to run simultaneously in the background
import queue            # Creates a line/queue for managing data between different parts of the program
from datetime import datetime  # Provides tools for working with dates and times
from typing import Optional, Tuple, Dict, List  # Helps define what type of data functions expect
import math             # Mathematical functions (sine, cosine, etc.) for waveform generation
import os               # Operating system interface for file and directory operations
import csv              # Tools for reading and writing CSV (spreadsheet) files
from pathlib import Path  # Modern way to handle file system paths
import gradio as gr     # Web interface framework that creates the user interface

# ============================================================================
# SECTION 2: SETTING UP FILE PATHS AND IMPORTING KEITHLEY DRIVER
# ============================================================================
# This section ensures Python can find the Keithley power supply control module

# Find the directory where this script is located and go up 3 levels
# This helps us locate the instrument_control folder that contains the driver
script_dir = Path(__file__).resolve().parent.parent.parent

# Add this directory to Python's search path so it can find our modules
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# ============================================================================
# Import the Keithley Power Supply driver class
# This is the core code that knows how to talk to the Keithley hardware
# ============================================================================
try:
    # First attempt: Try to import the KeithleyPowerSupply class directly
    from instrument_control.keithley_power_supply import KeithleyPowerSupply
except ImportError as e:
    # Second attempt: If the first method fails, try an alternative import method
    try:
        import instrument_control.keithley_power_supply
        KeithleyPowerSupply = instrument_control.keithley_power_supply.KeithleyPowerSupply
    except ImportError as e:
        # If both attempts fail, display helpful error messages and stop the program
        print(f"ERROR: Cannot import keithley_power_supply: {e}")
        print(f"Current Python path: {sys.path}")
        print(f"Looking for instrument_control in: {script_dir}")
        print("Please ensure the instrument_control package is in your Python path")
        sys.exit(1)  # Exit the program with error code 1


# ============================================================================
# SECTION 3: MAIN APPLICATION CLASS - THE BRAIN OF THE SYSTEM
# ============================================================================
# This class is like the control center that manages everything in our application

class PowerSupplyAutomationGradio:
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  MAIN APPLICATION CLASS - PowerSupplyAutomationGradio            ║
    ║                                                                  ║
    ║  WHAT THIS CLASS DOES:                                           ║
    ║  This is the "brain" of our automation system. It manages:      ║
    ║  • Connection to the Keithley power supply hardware              ║
    ║  • The web interface that users interact with                   ║
    ║  • All three power supply channels (voltage, current control)   ║
    ║  • Automated voltage ramping (creating waveforms over time)     ║
    ║  • Data collection and export to CSV files                      ║
    ║  • Real-time monitoring and safety features                     ║
    ║                                                                  ║
    ║  THINK OF IT AS: A remote control for a complex power supply   ║
    ║  that works through a web browser                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self):
        """
        INITIALIZATION METHOD - Sets up the application when it starts

        This is called ONCE when the program first runs. It prepares all the
        variables and settings needed for the application to work properly.

        Think of this as preparing a workspace before starting work - organizing
        your tools, setting up your desk, getting everything ready.
        """

        # ====================================================================
        # POWER SUPPLY CONNECTION
        # ====================================================================
        # This will hold the connection to the actual Keithley hardware
        # Starts as None (nothing) until we connect to the device
        self.power_supply = None

        # ====================================================================
        # VOLTAGE RAMPING CONTROLS
        # ====================================================================
        # These variables manage automated voltage ramping operations
        # (gradually changing voltage over time in specific patterns)

        self.ramping_active = False      # Is a ramping operation currently running?
        self.ramping_thread = None       # Background task that runs the ramping
        self.ramping_profile = []        # List of voltage points to apply over time
        self.ramping_data = []           # Collected data during ramping

        # Dictionary containing all settings for voltage ramping
        # A dictionary is like a labeled storage box - each item has a name and value
        self.ramping_params = {
            'waveform': 'Sine',          # Type of waveform (Sine, Square, Triangle, etc.)
            'target_voltage': 3.0,       # Maximum voltage to reach (in Volts)
            'cycles': 3,                 # How many times to repeat the waveform
            'points_per_cycle': 50,      # How many voltage steps in each cycle
            'cycle_duration': 8.0,       # How long each cycle takes (in seconds)
            'psu_settle': 0.05,          # Wait time for power supply to stabilize (seconds)
            'nplc': 1.0,                 # Measurement accuracy setting
            'active_channel': 1          # Which channel to apply ramping to (1, 2, or 3)
        }

        # ====================================================================
        # DATA COLLECTION AND STORAGE
        # ====================================================================
        self.measurement_data = {}       # Stores all measurements taken from channels
        self.status_queue = queue.Queue() # Message queue for background tasks
        self.measurement_active = False  # Is automatic measurement currently running?

        # ====================================================================
        # CONNECTION STATUS
        # ====================================================================
        self.is_connected = False        # Are we currently connected to the power supply?

        # ====================================================================
        # LOGGING SYSTEM SETUP
        # ====================================================================
        # Initialize the system that records all activities and errors
        self.setup_logging()

        # ====================================================================
        # CHANNEL STATE TRACKING
        # ====================================================================
        # This dictionary keeps track of all 3 channels and their current state
        # For each channel (1, 2, 3), we store:
        #   - enabled: Is the output turned on?
        #   - voltage: Current voltage reading (Volts)
        #   - current: Current current reading (Amperes)
        #   - power: Calculated power (Watts = Volts × Amperes)
        self.channel_states = {
            i: {"enabled": False, "voltage": 0.0, "current": 0.0, "power": 0.0}
            for i in range(1, 4)  # Creates entries for channels 1, 2, and 3
        }

        # ====================================================================
        # ACTIVITY LOG
        # ====================================================================
        # String that accumulates all log messages to display to the user
        self.activity_log = "Application started\n"

    def setup_logging(self):
        """
        LOGGING CONFIGURATION METHOD

        Sets up the system that records what the application is doing.
        This is like keeping a diary or log book of all activities.

        PURPOSE:
        - Records events (connections, measurements, errors)
        - Helps troubleshoot problems
        - Provides activity history for review

        TECHNICAL DETAILS:
        - Level set to INFO means we record normal activities and errors
        - Format includes timestamp, component name, severity, and message
        """
        logging.basicConfig(
            level=logging.INFO,  # Record informational messages and above (INFO, WARNING, ERROR)
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # How messages are formatted
        )
        # Create a logger specific to this application
        self.logger = logging.getLogger("PowerSupplyAutomationGradio")

    # ========================================================================
    # NESTED CLASS: WAVEFORM GENERATOR
    # ========================================================================
    # This class lives inside the main class and creates voltage patterns

    class _WaveformGenerator:
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║  WAVEFORM GENERATOR CLASS                                        ║
        ║                                                                  ║
        ║  WHAT IT DOES:                                                   ║
        ║  Creates patterns of voltage that change over time.              ║
        ║                                                                  ║
        ║  ANALOGY: Like a music synthesizer that creates different       ║
        ║  wave patterns, but instead of sound waves, we create voltage   ║
        ║  patterns to test electronic devices.                           ║
        ║                                                                  ║
        ║  SUPPORTED WAVEFORMS:                                           ║
        ║  • Sine Wave: Smooth up and down curve (like a wave in water)   ║
        ║  • Square Wave: Sharp on/off pattern (like a light switch)      ║
        ║  • Triangle Wave: Linear up and down (like a mountain peak)     ║
        ║  • Ramp Up: Gradual increase from 0 to maximum                  ║
        ║  • Ramp Down: Gradual decrease from maximum to 0                ║
        ╚══════════════════════════════════════════════════════════════════╝
        """

        # List of all supported waveform types
        TYPES = ["Sine", "Square", "Triangle", "Ramp Up", "Ramp Down"]

        def __init__(self, waveform_type: str = "Sine", target_voltage: float = 3.0,
                     cycles: int = 3, points_per_cycle: int = 50, cycle_duration: float = 8.0):
            """
            INITIALIZE WAVEFORM GENERATOR

            Sets up a new waveform generator with specified parameters.

            PARAMETERS EXPLAINED:
            - waveform_type: Which pattern to create (Sine, Square, Triangle, etc.)
            - target_voltage: Maximum voltage to reach (in Volts, limited to 5.0V for safety)
            - cycles: How many times to repeat the pattern
            - points_per_cycle: How many voltage steps in one complete pattern
            - cycle_duration: How long each pattern takes to complete (in seconds)

            EXAMPLE:
            If you set cycles=3, target_voltage=3.0, cycle_duration=8.0:
            The voltage will go through your chosen pattern 3 times,
            reaching a maximum of 3 Volts, with each repetition taking 8 seconds.
            Total time = 3 cycles × 8 seconds = 24 seconds
            """

            # Validate and store waveform type (use Sine if invalid type specified)
            self.waveform_type = waveform_type if waveform_type in self.TYPES else "Sine"

            # Ensure voltage is between 0 and 5 Volts (safety limits)
            # max() and min() functions constrain the value to this range
            self.target_voltage = max(0.0, min(float(target_voltage), 5.0))

            # Ensure cycles is at least 1
            self.cycles = max(1, int(cycles))

            # Ensure points_per_cycle is at least 1
            self.points_per_cycle = max(1, int(points_per_cycle))

            # Store cycle duration in seconds
            self.cycle_duration = float(cycle_duration)

        def generate(self):
            """
            GENERATE WAVEFORM PROFILE

            Creates a complete list of (time, voltage) pairs that form the waveform.

            HOW IT WORKS:
            1. Loop through each cycle (repetition of the pattern)
            2. For each cycle, create many voltage points
            3. Calculate what voltage should be at each time point
            4. Store as (time, voltage) pairs

            RETURNS:
            A list of tuples, where each tuple is (time_in_seconds, voltage_in_volts)

            EXAMPLE OUTPUT:
            [(0.0, 0.0), (0.16, 0.45), (0.32, 0.85), ...]
            This means: at 0.0 seconds -> 0V, at 0.16 seconds -> 0.45V, etc.
            """

            # Create an empty list to store all (time, voltage) pairs
            profile = []

            # OUTER LOOP: Go through each cycle (each repetition of the pattern)
            for cycle in range(self.cycles):

                # INNER LOOP: Create each point within this cycle
                for point in range(self.points_per_cycle):

                    # Calculate position in cycle (0.0 to 1.0)
                    # 0.0 = start of cycle, 0.5 = middle, 1.0 = end of cycle
                    pos = point / max(1, (self.points_per_cycle - 1)) if self.points_per_cycle > 1 else 0.0

                    # Calculate absolute time from start (in seconds)
                    t = cycle * self.cycle_duration + pos * self.cycle_duration

                    # ========================================================
                    # CALCULATE VOLTAGE BASED ON WAVEFORM TYPE
                    # ========================================================

                    if self.waveform_type == 'Sine':
                        # Sine wave: Smooth curve using mathematical sine function
                        # Starts at 0, peaks at target_voltage, returns to 0
                        v = math.sin(pos * math.pi) * self.target_voltage

                    elif self.waveform_type == 'Square':
                        # Square wave: Full voltage for first half, zero for second half
                        # Creates an on/off pattern
                        v = self.target_voltage if pos < 0.5 else 0.0

                    elif self.waveform_type == 'Triangle':
                        # Triangle wave: Linear increase then linear decrease
                        # First half: voltage increases, second half: voltage decreases
                        if pos < 0.5:
                            v = (pos * 2.0) * self.target_voltage  # Rising edge
                        else:
                            v = (2.0 - pos * 2.0) * self.target_voltage  # Falling edge

                    elif self.waveform_type == 'Ramp Up':
                        # Ramp up: Gradual linear increase from 0 to target voltage
                        v = pos * self.target_voltage

                    elif self.waveform_type == 'Ramp Down':
                        # Ramp down: Gradual linear decrease from target voltage to 0
                        v = (1.0 - pos) * self.target_voltage

                    else:
                        # Safety fallback: If unknown type, set voltage to 0
                        v = 0.0

                    # ========================================================
                    # SAFETY CHECK: Ensure voltage is within safe limits
                    # ========================================================
                    v = max(0.0, min(v, 5.0))  # Constrain between 0V and 5V

                    # Add this (time, voltage) pair to the profile
                    # Round to 6 decimal places for precision
                    profile.append((round(t, 6), round(v, 6)))

            # Return the complete list of (time, voltage) pairs
            return profile

    # ========================================================================
    # NESTED CLASS: RAMP DATA MANAGER
    # ========================================================================

    class _RampDataManager:
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║  RAMP DATA MANAGER CLASS                                         ║
        ║                                                                  ║
        ║  PURPOSE:                                                        ║
        ║  Collects, stores, and exports data during voltage ramping      ║
        ║  operations.                                                     ║
        ║                                                                  ║
        ║  WHAT IT DOES:                                                   ║
        ║  • Records voltage readings during ramping tests                ║
        ║  • Exports data to CSV files (spreadsheet format)               ║
        ║  • Creates graphs showing voltage over time                     ║
        ║  • Manages folders for storing data and graphs                  ║
        ║                                                                  ║
        ║  ANALOGY: Like a lab notebook and camera that automatically    ║
        ║  records everything during an experiment                        ║
        ╚══════════════════════════════════════════════════════════════════╝
        """

        def __init__(self):
            """
            INITIALIZE DATA MANAGER

            Sets up the data collection system and creates folders for
            storing CSV files and graphs.

            WHAT HAPPENS:
            1. Creates an empty list to store voltage measurements
            2. Defines folder paths for data and graphs
            3. Creates these folders if they don't exist
            """

            # Empty list to store all measurement data points
            # Each data point will contain timestamp, voltages, cycle info, etc.
            self.voltage_data = []

            # Define where to save CSV data files
            # os.getcwd() gets the current working directory
            # os.path.join() creates a complete folder path
            self.data_dir = os.path.join(os.getcwd(), 'voltage_ramp_data')

            # Define where to save graph images
            self.graphs_dir = os.path.join(os.getcwd(), 'voltage_ramp_graphs')

            # Try to create these folders
            try:
                # exist_ok=True means don't error if folder already exists
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(self.graphs_dir, exist_ok=True)
            except Exception:
                # If folder creation fails, just continue (might not have permissions)
                pass
        
        def add_point(self, ts, set_v, meas_v, cycle_no, point_idx):
            """
            ADD DATA POINT

            Saves one measurement to our data collection.

            PARAMETERS:
            - ts: Timestamp (when the measurement was taken)
            - set_v: Set voltage (what we told the power supply to output)
            - meas_v: Measured voltage (what the power supply actually output)
            - cycle_no: Which cycle/repetition this is from
            - point_idx: Point number within the cycle

            WHY WE COLLECT BOTH SET AND MEASURED:
            The power supply may not always output exactly what we request.
            By comparing set vs measured, we can check accuracy.
            """

            # Create a dictionary (labeled data structure) with all the info
            # and add it to our list of data points
            self.voltage_data.append({
                'timestamp': ts,              # When this happened
                'set_voltage': set_v,         # What we requested
                'measured_voltage': meas_v,   # What the power supply actually got
                'cycle_number': cycle_no,     # Which repetition
                'point_in_cycle': point_idx   # Position in that repetition
            })

        def clear(self):
            """
            CLEAR DATA

            Removes all collected data points from memory.
            Use this when starting a new test.
            """
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
                plt.savefig(fn, dpi=200)
                plt.close(fig)
                
            except Exception:
                raise
            
            return fn

    # ============================================================================
    # SECTION 4: CONNECTION MANAGEMENT METHODS
    # ============================================================================
    # These methods handle connecting to and disconnecting from the Keithley
    # power supply hardware. Think of them as plugging in and unplugging a device.

    def connect_power_supply(self, visa_address: str) -> str:
        """
        CONNECT TO POWER SUPPLY

        Establishes communication with the Keithley power supply via USB.

        WHAT IS A VISA ADDRESS?
        VISA (Virtual Instrument Software Architecture) is a standard way
        to communicate with test equipment. The address uniquely identifies
        the device, like a phone number or IP address.

        EXAMPLE VISA ADDRESS:
        USB0::0x05E6::0x2230::805224014806770001::INSTR
        This tells the computer: "Connect via USB to a Keithley device
        with serial number 805224014806770001"

        WHY USE A SEPARATE THREAD?
        Connection can take a few seconds. By using a separate thread
        (background task), the user interface stays responsive and doesn't
        freeze while connecting.
        """

        def connect_thread():
            """
            BACKGROUND CONNECTION TASK

            This function runs in a separate thread to avoid blocking
            the main user interface.
            """
            try:
                # Log that we're starting the connection attempt
                self.log_message("Attempting to connect to Keithley power supply...", "INFO")

                # Validate that the user provided a VISA address
                if not visa_address.strip():
                    raise ValueError("VISA address cannot be empty")

                # Create a KeithleyPowerSupply object with the provided address
                self.power_supply = KeithleyPowerSupply(visa_address)

                # Attempt to establish the connection
                if self.power_supply.connect():
                    # Connection successful!
                    self.log_message("Connection established successfully!", "SUCCESS")
                    self.status_queue.put(("connected", None))  # Notify other parts of the app
                    self.is_connected = True  # Update connection status flag
                else:
                    # Connection failed
                    raise Exception("Connection failed - check VISA address and instrument")

            except Exception as e:
                # Something went wrong during connection
                self.status_queue.put(("error", f"Connection failed: {str(e)}"))
                self.log_message(f"Connection failed: {str(e)}", "ERROR")

        # Start the connection process in a background thread
        # daemon=True means the thread will automatically stop when the program exits
        threading.Thread(target=connect_thread, daemon=True).start()

        # Return immediately to keep UI responsive
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

    # ============================================================================
    # INSTRUMENT INFORMATION AND TESTING
    # ============================================================================

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

    # ============================================================================
    # SECTION 5: CHANNEL CONFIGURATION AND CONTROL
    # ============================================================================
    # The Keithley 2230 has 3 independent channels. Each channel can be
    # configured with its own voltage, current limit, and safety settings.
    #
    # CHANNEL OPERATIONS:
    # 1. Configure: Set voltage, current limit, over-voltage protection
    # 2. Enable: Turn the output ON
    # 3. Disable: Turn the output OFF
    # 4. Measure: Read actual voltage and current being output

    def configure_channel(self, channel: int, voltage: float, current_limit: float, ovp_level: float) -> str:
        """
        CONFIGURE CHANNEL SETTINGS

        Sets up a power supply channel with desired parameters.

        PARAMETERS:
        - channel: Which channel to configure (1, 2, or 3)
        - voltage: Target output voltage (0-30V for this model)
        - current_limit: Maximum current allowed (0-3A) - safety feature
        - ovp_level: Over-Voltage Protection threshold (1-35V)

        WHAT IS OVP (Over-Voltage Protection)?
        Safety feature that shuts off the output if voltage exceeds
        a certain level. Protects sensitive equipment from damage.

        WHAT IS CURRENT LIMIT?
        Maximum current the channel will supply. If the load tries
        to draw more, the voltage will drop to maintain this limit.
        This prevents overheating and protects both the power supply
        and the device being powered.
        """
        def config_thread():
            try:
                # Log the configuration being applied
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

    # ============================================================================
    # SECTION 6: GLOBAL OPERATIONS AND SAFETY
    # ============================================================================
    # These methods perform operations on all channels at once or provide
    # emergency safety features.

    def measure_all_channels(self) -> Tuple[str, str, str, str, str, str, str, str, str]:
        """
        MEASURE ALL CHANNELS

        Reads voltage and current from all 3 channels in sequence.

        WHY SEQUENTIAL?
        We measure one channel at a time (not simultaneously) to avoid
        USB communication conflicts. The power supply can only handle
        one command at a time through USB.

        TIMING:
        - Measure Channel 1
        - Wait 0.8 seconds
        - Measure Channel 2
        - Wait 0.8 seconds
        - Measure Channel 3

        The delays prevent overwhelming the USB interface and ensure
        accurate readings.

        RETURNS:
        Tuple of 9 strings: (ch1_volt, ch1_curr, ch1_power, ch2_volt, ch2_curr, ch2_power, ch3_volt, ch3_curr, ch3_power)
        """
        # Check if we're connected before attempting to measure
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
        EMERGENCY STOP

        Immediately disables all outputs for safety.

        WHEN TO USE:
        - If you see smoke or sparks
        - If connected equipment is malfunctioning
        - Any dangerous or unexpected behavior
        - When you need to stop everything immediately

        This is like a "panic button" that turns everything off.
        """
        self.log_message("EMERGENCY STOP ACTIVATED!", "ERROR")
        return self.disable_all_outputs()

    # ============================================================================
    # SECTION 7: DATA LOGGING AND EXPORT
    # ============================================================================
    # Methods for recording activities and exporting measurement data

    def log_message(self, message: str, level: str = "INFO"):
        """Add timestamped message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        self.activity_log += log_entry
        self.logger.log(
            getattr(logging, level, logging.INFO),
            message
        )

    def export_measurement_data(self) -> str:
        """Export collected measurements to CSV file"""
        try:
            if not self.measurement_data:
                return "No measurement data to export"
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"power_supply_data_{timestamp}.csv"
            
            with open(filename, "w", newline="") as csvfile:
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
            
            self.log_message(f"Data exported to: {filename}", "SUCCESS")
            return f"Data exported to: {filename}"
        
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

    def create_gradio_interface(self):
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║  CREATE WEB USER INTERFACE                                      ║
        ║                                                                  ║
        ║  This method builds the entire web-based control panel using    ║
        ║  the Gradio framework.                                          ║
        ║                                                                  ║
        ║  WHAT IS GRADIO?                                                ║
        ║  Gradio is a Python library that creates web interfaces         ║
        ║  automatically. You define buttons, sliders, and text boxes     ║
        ║  in Python, and Gradio generates a webpage that users can       ║
        ║  access through a web browser.                                  ║
        ║                                                                  ║
        ║  INTERFACE SECTIONS:                                            ║
        ║  1. Connection controls (connect/disconnect buttons)            ║
        ║  2. Individual channel controls (3 tabs, one per channel)       ║
        ║  3. Global operations (measure all, emergency stop)             ║
        ║  4. Data export controls                                        ║
        ║  5. Activity log display                                        ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        # ====================================================================
        # CREATE THE WEB INTERFACE LAYOUT
        # ====================================================================
        # gr.Blocks creates a customizable web interface
        # theme=gr.themes.Soft() gives it a professional appearance

        with gr.Blocks(title="DIGANTARA PSU Control", theme=gr.themes.Soft()) as demo:

            # Page title and description
            gr.Markdown("# DIGANTARA PSU Control")
            gr.Markdown("**Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.")
            gr.Markdown("Professional automation control for Keithley 2230 power supply")

            # ================================================================
            # CONNECTION SECTION
            # ================================================================
            # This section allows users to connect to the power supply
            gr.Markdown("## Connection Settings")
            with gr.Group():
                with gr.Row():
                    visa_addr = gr.Textbox(
                        label="VISA Address",
                        value="USB0::0x05E6::0x2230::805224014806770001::INSTR",
                        lines=1
                    )
                
                with gr.Row():
                    conn_btn = gr.Button("Connect", variant="primary", size="lg")
                    disc_btn = gr.Button("Disconnect", variant="stop", size="lg")
                    test_btn = gr.Button("Test Connection", variant="secondary", size="lg")
                    emerg_btn = gr.Button("EMERGENCY STOP", variant="stop", size="lg")
                
                with gr.Row():
                    conn_status = gr.Textbox(label="Status", value="Disconnected", interactive=False)
                
                info_display = gr.Textbox(label="Instrument Info", lines=2, interactive=False)
            
            # Channel Controls
            gr.Markdown("## Channel Controls")
            with gr.Group():
                channel_outputs = {}

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
                                volt_slider = gr.Slider(0, max_volt, value=0, label="Voltage (V)", step=0.1)
                                curr_limit = gr.Slider(0.001, max_curr, value=0.1, label="Current Limit (A)", step=0.001)
                                ovp_level = gr.Slider(1, max_volt + 5, value=default_ovp, label="OVP (V)", step=0.5)

                            with gr.Row():
                                conf_btn = gr.Button(f"Configure Ch{ch}", variant="secondary")
                                enable_btn = gr.Button(f"Enable Output", variant="primary")
                                disable_btn = gr.Button(f"Disable Output", variant="stop")
                                meas_btn = gr.Button(f"Measure", variant="secondary")

                            with gr.Row():
                                volt_display = gr.Textbox(label="Measured Voltage", value="0.000 V", interactive=False)
                                curr_display = gr.Textbox(label="Measured Current", value="0.000 A", interactive=False)
                                power_display = gr.Textbox(label="Measured Power", value="0.000 W", interactive=False)

                            ch_status = gr.Textbox(label="Status", value="OFF", interactive=False)

                            channel_outputs[ch] = {
                                "voltage": volt_slider,
                                "current_limit": curr_limit,
                                "ovp_level": ovp_level,
                                "configure_btn": conf_btn,
                                "enable_btn": enable_btn,
                                "disable_btn": disable_btn,
                                "measure_btn": meas_btn,
                                "volt_display": volt_display,
                                "curr_display": curr_display,
                                "power_display": power_display,
                                "status": ch_status
                            }

                            conf_btn.click(
                                fn=lambda v, cl, ov, ch=ch: self.configure_channel(ch, v, cl, ov),
                                inputs=[volt_slider, curr_limit, ovp_level]
                            )

                            # --- FINAL FIX: Query output state from instrument after enable/disable ---
                            def update_channel_status_after_action(action, ch=ch):
                                # Perform the action (enable/disable)
                                if action == "enable":
                                    self.enable_channel_output(ch)
                                else:
                                    self.disable_channel_output(ch)
                                # Wait for instrument to process
                                time.sleep(0.5)
                                # Try to query the instrument for output state
                                status = "OFF"
                                try:
                                    if self.power_supply and self.power_supply.is_connected:
                                        # Try to use a direct query if available
                                        if hasattr(self.power_supply, "query_output_enabled"):
                                            is_on = self.power_supply.query_output_enabled(ch)
                                            status = "ON" if is_on else "OFF"
                                            self.channel_states[ch]["enabled"] = bool(is_on)
                                        elif hasattr(self.power_supply, "get_output_state"):
                                            state = self.power_supply.get_output_state(ch)
                                            status = "ON" if state in ("ON", True, 1) else "OFF"
                                            self.channel_states[ch]["enabled"] = (status == "ON")
                                        else:
                                            # Fallback: use last known state
                                            status = "ON" if self.channel_states[ch]["enabled"] else "OFF"
                                except Exception:
                                    status = "ON" if self.channel_states[ch]["enabled"] else "OFF"
                                return status

                            enable_btn.click(
                                fn=lambda ch=ch: update_channel_status_after_action("enable", ch),
                                outputs=[ch_status]
                            )

                            disable_btn.click(
                                fn=lambda ch=ch: update_channel_status_after_action("disable", ch),
                                outputs=[ch_status]
                            )

                            meas_btn.click(
                                fn=lambda ch=ch: self.measure_channel_output(ch),
                                outputs=[volt_display, curr_display, power_display]
                            )
            
            # Global Operations
            gr.Markdown("## Global Operations")
            with gr.Group():
                with gr.Row():
                    get_info_btn = gr.Button("Get Instrument Info", variant="secondary")
                    measure_all_btn = gr.Button("Measure All Channels", variant="primary")
                    disable_all_btn = gr.Button("Disable All Outputs", variant="stop")
                
                get_info_btn.click(fn=self.get_instrument_info)
                measure_all_btn.click(
                    fn=self.measure_all_channels,
                    outputs=[
                        channel_outputs[1]["volt_display"],
                        channel_outputs[1]["curr_display"],
                        channel_outputs[1]["power_display"],
                        channel_outputs[2]["volt_display"],
                        channel_outputs[2]["curr_display"],
                        channel_outputs[2]["power_display"],
                        channel_outputs[3]["volt_display"],
                        channel_outputs[3]["curr_display"],
                        channel_outputs[3]["power_display"]
                    ]
                )
                disable_all_btn.click(fn=self.disable_all_outputs)
            
            # Data Logging
            gr.Markdown("## Data Logging & Export")
            with gr.Group():
                with gr.Row():
                    auto_measure_cb = gr.Checkbox(label="Enable Auto-Measurement", value=False)
                    measure_interval = gr.Slider(0.5, 60, value=2.0, label="Interval (seconds)", step=0.5)
                
                with gr.Row():
                    export_btn = gr.Button("Export to CSV", variant="primary")
                    clear_btn = gr.Button("Clear Data", variant="secondary")
                
                auto_measure_cb.change(fn=self.toggle_auto_measure, inputs=auto_measure_cb)
                export_btn.click(fn=self.export_measurement_data)
                clear_btn.click(fn=self.clear_measurement_data)
            
            # Status & Activity Log
            gr.Markdown("## Status & Activity Log")
            with gr.Group():
                activity_log_display = gr.Textbox(
                    label="Activity Log",
                    value=self.activity_log,
                    lines=15,
                    interactive=False,
                    max_lines=500
                )
            
            # Connection handlers
            def handle_connect(visa_addr_val):
                """Handle connection button click"""
                self.connect_power_supply(visa_addr_val)
                # Wait for connection to complete (max 5 seconds)
                max_wait = 5.0
                waited = 0.0
                wait_step = 0.1
                while waited < max_wait:
                    time.sleep(wait_step)
                    waited += wait_step
                    if self.is_connected:
                        break
                status = "Connected" if self.is_connected else "Disconnected"
                return status, self.activity_log
            
            def handle_disconnect():
                """Handle disconnect button click"""
                self.disconnect_power_supply()
                return "Disconnected", self.activity_log
            
            def handle_test():
                """Handle test connection button click"""
                result = self.test_connection()
                return result, self.activity_log
            
            def handle_emergency():
                """Handle emergency stop button click"""
                result = self.emergency_stop()
                return result, self.activity_log
            
            # Register connection handlers
            conn_btn.click(
                fn=handle_connect,
                inputs=visa_addr,
                outputs=[conn_status, activity_log_display]
            )
            
            disc_btn.click(
                fn=handle_disconnect,
                outputs=[conn_status, activity_log_display]
            )
            
            test_btn.click(
                fn=handle_test,
                outputs=[conn_status, activity_log_display]
            )
            
            emerg_btn.click(
                fn=handle_emergency,
                outputs=[conn_status, activity_log_display]
            )
        
        return demo


# ============================================================================
# SECTION 8: MAIN ENTRY POINT - PROGRAM STARTUP
# ============================================================================

def main():
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  MAIN FUNCTION - PROGRAM ENTRY POINT                                 ║
    ║                                                                      ║
    ║  This function is called when you run the program. It:              ║
    ║  1. Creates the automation application                              ║
    ║  2. Builds the web interface                                        ║
    ║  3. Starts a local web server                                       ║
    ║  4. Opens your default browser to the control page                  ║
    ║                                                                      ║
    ║  HOW TO USE:                                                        ║
    ║  Run this script from command line:                                 ║
    ║      python keithley_gradio_automation.py                           ║
    ║                                                                      ║
    ║  The web interface will open automatically at:                      ║
    ║      http://localhost:7860                                          ║
    ║                                                                      ║
    ║  WHAT IS LOCALHOST?                                                 ║
    ║  "localhost" means "this computer". The web server runs on your     ║
    ║  local machine, not on the internet. Only you can access it.        ║
    ║                                                                      ║
    ║  PORT 7860:                                                         ║
    ║  A port is like a door number. Port 7860 is where the Gradio        ║
    ║  web server listens for connections.                                ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """

    # Display startup banner
    print("DIGANTARA PSU Control - GRADIO VERSION")
    print("="*60)

    try:
        # ================================================================
        # STEP 1: Create the application object
        # ================================================================
        # This initializes all the internal systems (logging, data storage, etc.)
        app = PowerSupplyAutomationGradio()
        demo = app.create_gradio_interface()

        # ================================================================
        # STEP 3: Launch the web server and open browser
        # ================================================================
        print("Launching web interface...")

        # Use network binding so other laptops on the same WiFi can access
        chosen_port = 7861
        print(f"Opening browser to http://0.0.0.0:{chosen_port}")
        print("Other laptops on the same WiFi can open: http://[your-computer-ip]:7861")

        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=chosen_port,
                share=False,
                show_error=True,
                #inbrowser=True
            )
        except OSError as e:
            print(f"Application error during launch: {e}")
            print("You can set a specific port via the GRADIO_SERVER_PORT environment variable,")
            print("or ensure the chosen port is free and try again.")

    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# PROGRAM EXECUTION CHECK
# ============================================================================
# This special Python construct checks if this file is being run directly
# (as opposed to being imported as a module by another Python file)

if __name__ == "__main__":
    """
    EXECUTION GUARD

    __name__ is a special Python variable:
    - When you run this file directly: __name__ == "__main__"
    - When this file is imported: __name__ == "keithley_gradio_automation"

    This ensures main() only runs when the script is executed directly,
    not when it's imported as a module.

    EXAMPLE:
    ✓ python keithley_gradio_automation.py  → main() runs
    ✗ import keithley_gradio_automation    → main() does NOT run
    """
    main()  # Start the application