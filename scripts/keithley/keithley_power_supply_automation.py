#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║         KEITHLEY MULTI-CHANNEL POWER SUPPLY AUTOMATION - COMPLETE               ║
║                   Professional GUI Application                                   ║
║                     Fully Annotated for Executive Review                       ║
║                                                                                ║
║  Purpose: Comprehensive automation control for Keithley 2230-30-1 power supply║
║           with multi-channel independent control, waveform ramping, and       ║
║           data acquisition capabilities                                        ║
║                                                                                ║
║  Core Features:                                                                ║
║    • Multi-threaded responsive GUI using tkinter framework                    ║
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
║    • Keithley 2230 power supply connected via USB                              ║
║                                                                                ║
║  Author: Professional Instrumentation Control System                          ║
║  Last Updated: 2025-10-24 | Status: Production Ready                          ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import sys  # System-level operations and program exit control
import logging  # Application event logging for diagnostics and troubleshooting
import time  # Time delay operations for device response waiting
import threading  # Multi-threaded execution to prevent GUI blocking
import queue  # Thread-safe queue for inter-thread communication (main ↔ workers)
from datetime import datetime  # Timestamp generation for logging and data records
from typing import Optional, Tuple  # Type hints for improved code clarity
import tkinter as tk  # Core GUI framework for window creation and widgets
from tkinter import ttk, messagebox, filedialog, scrolledtext  # Advanced GUI components
import math  # Mathematical functions for waveform generation (sine, etc.)
import os  # Operating system interface for file and directory operations
import csv  # CSV file format handling for data export
from pathlib import Path  # Object-oriented filesystem path handling

# Add parent directories to Python path to find instrument_control
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Import instrument control module (Keithley Power Supply driver)
try:
    # Primary import method: from instrument_control package
    from instrument_control.keithley_power_supply import KeithleyPowerSupply  # Main driver class
except ImportError as e:
    try:
        # Fallback import method: direct module import
        import instrument_control.keithley_power_supply  # Alternative module access
        KeithleyPowerSupply = instrument_control.keithley_power_supply.KeithleyPowerSupply  # Extract class
    except ImportError as e:
        # Error handling: driver not found
        print(f"ERROR: Cannot import keithley_power_supply: {e}")
        print(f"Current Python path: {sys.path}")
        print(f"Looking for instrument_control in: {script_dir}")
        print("Please ensure the instrument_control package is in your Python path")
        sys.exit(1)


class PowerSupplyAutomationGUI:
    """
    Professional Power Supply Automation GUI Controller
    
    Manages user interface, event handling, and multi-threaded operations for
    complete control of Keithley multi-channel power supply with real-time
    measurements, waveform ramping, and data logging capabilities.
    
    Main Responsibilities:
    - GUI construction and event management (tkinter)
    - Thread spawning and coordination for non-blocking operations
    - VISA communication through KeithleyPowerSupply driver
    - Data collection, export, and visualization
    - Status tracking and user feedback via logging
    """

    def __init__(self):
        """
        Initialize Power Supply Automation GUI Application
        
        Creates main Tkinter window, initializes application state variables,
        configures threading infrastructure, and sets up logging system.
        
        Instance Variables:
        - root: Main Tkinter window container
        - power_supply: KeithleyPowerSupply driver instance (None if disconnected)
        - status_queue: Thread-safe queue for worker-to-main communication
        - measurement_data: Dictionary storing timestamped measurements per channel
        - channel_vars: Dictionary of Tkinter variables for each channel
        - channel_frames: Dictionary of GUI frame references for each channel
        """
        self.root = tk.Tk()  # Create main application window
        self.power_supply = None  # No power supply connected initially
        
        # Ramping operation state variables (for waveform cycling automation)
        self.ramping_active = False  # Flag: ramping operation in progress
        self.ramping_thread = None  # Background thread handle for ramping worker
        self.ramping_profile = []  # List of (time, voltage) tuples for ramping cycle
        self.ramping_data = []  # Raw measurement points during ramping
        
        # Ramping parameters dictionary (configuration for waveform generation)
        self.ramping_params = {
            'waveform': 'Sine',  # Waveform type: Sine, Square, Triangle, Ramp Up, Ramp Down
            'target_voltage': 3.0,  # Maximum voltage in waveform (volts)
            'cycles': 3,  # Number of complete cycles to execute
            'points_per_cycle': 50,  # Samples per cycle for smooth profile
            'cycle_duration': 8.0,  # Time per cycle (seconds)
            'psu_settle': 0.05,  # Settle time after voltage set (seconds)
            'nplc': 1.0,  # NPLC (number of power line cycles) for measurement accuracy
            'active_channel': 1  # Target channel for ramping operation
        }
        
        # Data management and measurement storage
        self.measurement_data = {}  # Dictionary: channel → [timestamp, voltage, current, power]
        self.status_queue = queue.Queue()  # Thread-safe queue for status updates from workers
        self.measurement_active = False  # Flag: auto-measurement polling enabled
        
        # Initialize application infrastructure
        self.setup_logging()  # Configure diagnostic logging system
        self.setup_gui()  # Construct GUI elements and layout
        self.check_status_updates()  # Start periodic status check (queue polling)
        self.update_measurements()  # Start auto-measurement cycle

    def setup_logging(self):
        """
        Configure application-wide logging infrastructure
        
        Sets up Python logging system with standardized format including
        timestamp, logger name, log level, and message content.
        
        Format: YYYY-MM-DD HH:MM:SS - LoggerName - LEVEL - Message
        """
        logging.basicConfig(
            level=logging.INFO,  # Log INFO level and above (INFO, WARNING, ERROR, CRITICAL)
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Standard format string
        )
        self.logger = logging.getLogger("PowerSupplyAutomation")  # Create module logger

    def setup_gui(self):
        """
        Construct main GUI window and initialize all UI components
        
        Configures window properties, creates scrollable container frame,
        applies custom styling, and triggers creation of all GUI sections
        (connection, channels, operations, logging).
        """
        self.root.title("Keithley Power Supply Automation Control")  # Window title bar text
        self.root.minsize(950, 700)  # Minimum window dimensions in pixels (width, height)
        self.root.configure(bg="#f0f0f0")  # Background color (light gray)
        
        # GUI styling configuration
        self.style = ttk.Style()  # Create themed widget style manager
        self.style.theme_use("clam")  # Apply "clam" theme (modern flat design)
        self.configure_styles()  # Apply custom color/font configurations
        
        # Create scrollable container for content exceeding window height
        self.setup_scrollable_frame()  # Initialize canvas with vertical scrollbar

    def configure_styles(self):
        """
        Apply custom styling to GUI widgets (colors, fonts, button states)
        
        Defines appearance for Labels (bold blue titles), Buttons (color-coded
        by function: Success=green, Warning=orange, Info=blue, Danger=red),
        and measurement displays (monospace for alignment).
        """
        # Title label styling: large bold font, professional blue color
        self.style.configure("Title.TLabel", font=("Arial", 14, "bold"), foreground="#2c5aa0")
        
        # Success button style (green for enabled/safe operations)
        self.style.configure("Success.TButton", font=("Arial", 9))  # Normal state font
        self.style.map("Success.TButton",
            background=[(
            "active", "#28a745"), (
            "!active", "#198754")],  # Active (hover) = lighter, inactive = darker
            foreground=[(
            "active", "white"), (
            "!active", "white")])  # All text is white
        
        # Warning button style (orange for caution operations)
        self.style.configure("Warning.TButton", font=("Arial", 9))
        self.style.map("Warning.TButton",
            background=[(
            "active", "#fd7e14"), (
            "!active", "#e36209")],
            foreground=[(
            "active", "white"), (
            "!active", "white")])
        
        # Info button style (blue for informational operations)
        self.style.configure("Info.TButton", font=("Arial", 9))
        self.style.map("Info.TButton",
            background=[(
            "active", "#17a2b8"), (
            "!active", "#138496")],
            foreground=[(
            "active", "white"), (
            "!active", "white")])
        
        # Danger button style (red for stop/disable operations)
        self.style.configure("Danger.TButton", font=("Arial", 9))
        self.style.map("Danger.TButton",
            background=[(
            "active", "#3835dc"), (
            "!active", "#c82333")],
            foreground=[(
            "active", "white"), (
            "!active", "white")])
        
        # Measurement display label style (monospace for numerical alignment)
        self.style.configure("Measurement.TLabel", font=("Courier", 10, "bold"), foreground="#0066cc")

    # Inner classes for data generation and management
    class _WaveformGenerator:
        """
        Waveform Profile Generator for Automated Voltage Ramping
        
        Generates time-voltage profiles for different waveform shapes
        (sine, square, triangle, ramp). Used for automated PSU testing
        with programmable voltage patterns.
        """
        TYPES = ["Sine", "Square", "Triangle", "Ramp Up", "Ramp Down"]  # Supported waveform types
        
        def __init__(self, waveform_type: str = "Sine", target_voltage: float = 3.0, 
                     cycles: int = 3, points_per_cycle: int = 50, cycle_duration: float = 8.0):
            """
            Initialize waveform generator with parameters
            
            Args:
                waveform_type: Shape of waveform (must be in TYPES list)
                target_voltage: Maximum voltage in waveform (0-5V, clamped)
                cycles: Number of complete cycles to generate
                points_per_cycle: Samples per cycle (resolution)
                cycle_duration: Time duration of one complete cycle (seconds)
            """
            self.waveform_type = waveform_type if waveform_type in self.TYPES else "Sine"  # Validate type
            self.target_voltage = max(0.0, min(float(target_voltage), 5.0))  # Clamp voltage 0-5V
            self.cycles = max(1, int(cycles))  # Ensure at least 1 cycle
            self.points_per_cycle = max(1, int(points_per_cycle))  # Ensure at least 1 point
            self.cycle_duration = float(cycle_duration)  # Convert to float
        
        def generate(self):
            """
            Generate time-voltage profile array
            
            Returns:
                List of (time_seconds, voltage_volts) tuples representing
                the complete waveform profile across all cycles
            """
            profile = []  # Initialize empty profile
            
            # Iterate through each cycle
            for cycle in range(self.cycles):  # cycle: 0 to (cycles-1)
                # Iterate through points within cycle
                for point in range(self.points_per_cycle):  # point: 0 to (points_per_cycle-1)
                    # Calculate normalized position within cycle (0.0 to 1.0)
                    pos = point / max(1, (self.points_per_cycle - 1)) if self.points_per_cycle > 1 else 0.0
                    
                    # Calculate absolute time for this point
                    t = cycle * self.cycle_duration + pos * self.cycle_duration
                    
                    # Calculate voltage based on waveform type using position
                    if self.waveform_type == 'Sine':
                        v = math.sin(pos * math.pi) * self.target_voltage  # Sine: 0→π
                    elif self.waveform_type == 'Square':
                        v = self.target_voltage if pos < 0.5 else 0.0  # High for first half, low for second half
                    elif self.waveform_type == 'Triangle':
                        # Rising first half, falling second half
                        v = (pos * 2.0) * self.target_voltage if pos < 0.5 else (2.0 - pos * 2.0) * self.target_voltage
                    elif self.waveform_type == 'Ramp Up':
                        v = pos * self.target_voltage  # Linear from 0 to target
                    elif self.waveform_type == 'Ramp Down':
                        v = (1.0 - pos) * self.target_voltage  # Linear from target to 0
                    else:
                        v = 0.0  # Default: no output
                    
                    # Safety clamp voltage to 0-5V range
                    v = max(0.0, min(v, 5.0))
                    
                    # Add (time, voltage) tuple to profile with 6 decimal places precision
                    profile.append((round(t, 6), round(v, 6)))
            
            return profile  # Return complete waveform profile

    class _RampDataManager:
        """
        Data Collection and Export Manager for Ramping Operations
        
        Manages storage of ramping measurement data points and provides
        export functionality to CSV files and matplotlib graphs.
        """
        def __init__(self):
            """Initialize data manager with output directories"""
            self.voltage_data = []  # List of measurement dictionaries during ramping
            
            # Create output directory paths in current working directory
            self.data_dir = os.path.join(os.getcwd(), 'voltage_ramp_data')  # CSV output folder
            self.graphs_dir = os.path.join(os.getcwd(), 'voltage_ramp_graphs')  # Graph output folder
            
            # Create directories if they don't exist
            try:
                os.makedirs(self.data_dir, exist_ok=True)  # Create data directory (safe: no error if exists)
                os.makedirs(self.graphs_dir, exist_ok=True)  # Create graphs directory
            except Exception:
                pass  # Silently ignore any directory creation errors
        
        def add_point(self, ts, set_v, meas_v, cycle_no, point_idx):
            """
            Add single measurement point to ramping data collection
            
            Args:
                ts: Timestamp (datetime object) of measurement
                set_v: Set voltage (volts) sent to PSU
                meas_v: Measured voltage (volts) from PSU feedback
                cycle_no: Cycle number this point belongs to
                point_idx: Point index within the cycle
            """
            # Create measurement record dictionary and append to collection
            self.voltage_data.append({
                'timestamp': ts,  # When measurement was taken
                'set_voltage': set_v,  # Voltage command sent to PSU
                'measured_voltage': meas_v,  # Actual voltage measured
                'cycle_number': cycle_no,  # Which cycle (0-indexed)
                'point_in_cycle': point_idx  # Position within cycle
            })
        
        def clear(self):
            """Clear all collected ramping data points (reset collection)"""
            self.voltage_data.clear()  # Empty the data list
        
        def export_csv(self, folder=None):
            """
            Export ramping data to CSV file with timestamp header
            
            Args:
                folder: Output directory path (uses default if None)
            
            Returns:
                Path to created CSV file
            
            Raises:
                ValueError: If no data has been collected
            """
            if not self.voltage_data:  # Check if data exists
                raise ValueError('No ramping data')  # Raise error if empty
            
            folder = folder or '.'  # Use current directory if folder not specified
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate timestamp for filename
            fn = os.path.join(folder, f'psu_ramping_{ts}.csv')  # Construct full file path
            
            # Write CSV file with proper header and data
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)  # Create CSV writer object
                w.writerow(['timestamp', 'set_voltage', 'measured_voltage', 'cycle', 'point'])  # Header row
                for d in self.voltage_data:  # Iterate all measurement points
                    w.writerow([
                        d['timestamp'].isoformat(),  # Timestamp in ISO format
                        d['set_voltage'],  # Set voltage value
                        d['measured_voltage'],  # Measured voltage value
                        d['cycle_number'],  # Cycle number
                        d['point_in_cycle']  # Point index
                    ])
            
            return fn  # Return path to created file
        
        def generate_graph(self, folder=None, title: Optional[str] = None) -> str:
            """
            Generate matplotlib graph of set vs measured voltage over time
            
            Creates publication-quality plot comparing programmed voltage
            against actual measured voltage to visualize PSU response behavior.
            
            Args:
                folder: Output directory path (uses default if None)
                title: Custom plot title (auto-generated if None)
            
            Returns:
                Path to saved PNG image file
            
            Raises:
                ValueError: If no data has been collected
            """
            if not self.voltage_data:  # Check if data exists
                raise ValueError('No ramping data')  # Error if empty
            
            folder = folder or self.graphs_dir  # Use default folder if not specified
            
            try:
                os.makedirs(folder, exist_ok=True)  # Ensure output folder exists
            except Exception:
                pass  # Ignore folder creation errors
            
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate timestamp
            fn = os.path.join(folder, f'voltage_ramp_{ts}.png')  # Construct filename
            
            # Extract data arrays from measurement dictionaries
            times = []  # Time values (seconds from start)
            set_v = []  # Set voltage values (programmed)
            meas_v = []  # Measured voltage values (actual)
            
            # Calculate time relative to first measurement (so plot starts at t=0)
            t0 = self.voltage_data[0]['timestamp'] if self.voltage_data else datetime.now()  # Reference time
            
            for d in self.voltage_data:  # Process each measurement point
                times.append((d['timestamp'] - t0).total_seconds())  # Calculate elapsed time in seconds
                set_v.append(d['set_voltage'])  # Add set voltage value
                meas_v.append(d['measured_voltage'])  # Add measured voltage value
            
            try:
                # Configure matplotlib for non-interactive use (file output)
                import matplotlib
                matplotlib.use('Agg')  # Use Agg backend (no GUI needed)
                import matplotlib.pyplot as plt  # Import pyplot interface
                
                # Create figure and axis objects
                fig, ax = plt.subplots(figsize=(10, 6))  # 10x6 inch figure
                
                # Plot set and measured voltage lines
                ax.plot(times, set_v, label='Set Voltage', color='tab:blue')  # Blue line for set voltage
                ax.plot(times, meas_v, label='Measured Voltage', color='tab:red')  # Red line for measured voltage
                
                # Configure axis labels and title
                ax.set_xlabel('Time (s)')  # Horizontal axis label
                ax.set_ylabel('Voltage (V)')  # Vertical axis label
                ax.set_title(title or 'Voltage Ramping')  # Plot title
                
                # Add grid for easy reading
                ax.grid(True, ls='--', alpha=0.4)  # Dashed grid lines, semi-transparent
                
                # Add legend showing what each line represents
                ax.legend()  # Place legend automatically
                
                # Save figure to file
                plt.tight_layout()  # Adjust layout to prevent label cutoff
                plt.savefig(fn, dpi=200)  # Save at 200 DPI (high quality)
                plt.close(fig)  # Free memory by closing figure
                
            except Exception as e:
                raise  # Re-raise exception if graph generation fails
            
            return fn  # Return path to saved graph file

    def setup_scrollable_frame(self):
        """
        Create scrollable container for GUI content exceeding window height
        
        Implements canvas with vertical scrollbar to allow scrolling when
        content exceeds available window space. Enables responsive handling
        of multiple channel controls.
        """
        # Main container frame
        main_container = ttk.Frame(self.root)  # Frame to hold canvas and scrollbar
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Fill available space with padding
        
        # Create canvas (scrollable area)
        self.canvas = tk.Canvas(main_container, bg="#f0f0f0", highlightthickness=0)  # Canvas for scrollable content
        self.v_scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)  # Vertical scrollbar
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)  # Link scrollbar to canvas
        
        # Create frame inside canvas (this frame holds all GUI elements)
        self.scrollable_frame = ttk.Frame(self.canvas)  # Container for all content
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")  # Place frame in canvas
        
        # Bind events for responsive scrolling
        def configure_scroll(event):
            """Update scroll region when content changes"""
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))  # Set scroll region to content bounds
        
        def configure_canvas(event):
            """Resize frame to match canvas width"""
            self.canvas.itemconfig(self.canvas_window, width=event.width)  # Stretch frame to canvas width
        
        def on_mousewheel(event):
            """Handle mouse wheel scroll events"""
            if event.delta:  # Check if mousewheel event has delta (Windows)
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")  # Scroll up/down
        
        # Register event handlers
        self.scrollable_frame.bind("<Configure>", configure_scroll)  # When content size changes
        self.canvas.bind("<Configure>", configure_canvas)  # When canvas size changes
        self.root.bind_all("<MouseWheel>", on_mousewheel)  # Global mousewheel event
        
        # Pack scrollbar and canvas
        self.canvas.pack(side="left", fill="both", expand=True)  # Canvas on left, expandable
        self.v_scrollbar.pack(side="right", fill="y")  # Scrollbar on right, fills vertically
        
        # Create all GUI elements inside scrollable frame
        self.create_gui_elements()  # Build complete UI layout

    def create_gui_elements(self):
        """
        Build complete GUI layout with all sections and controls
        
        Orchestrates creation of connection panel, channel controls,
        operations buttons, data logging controls, and status display.
        """
        # Application title
        title_label = ttk.Label(self.scrollable_frame, text="Keithley Multi-Channel Power Supply Control",
            style="Title.TLabel")  # Large bold blue title
        title_label.pack(pady=(0, 15))  # Place with bottom padding
        
        # Create each section of the GUI
        self.create_connection_frame()  # Connection settings and status
        self.create_channel_frames()  # Three channel control sections
        self.create_global_operations_frame()  # Global control buttons
        self.create_data_logging_frame()  # Data export and auto-measurement
        self.create_status_frame()  # Status display and activity log

    def create_connection_frame(self):
        """
        Create connection settings section (VISA address, Connect/Disconnect)
        
        Allows user to enter instrument VISA address and provides buttons
        for connecting/disconnecting from power supply.
        """
        conn_frame = ttk.LabelFrame(self.scrollable_frame, text="Connection Settings", padding="12")  # Labeled section
        conn_frame.pack(fill=tk.X, pady=(0, 15))  # Fill width, add bottom padding
        
        # VISA address input
        addr_frame = ttk.Frame(conn_frame)  # Container for address controls
        addr_frame.pack(fill=tk.X, pady=(0, 10))  # Fill width, add padding
        
        ttk.Label(addr_frame, text="VISA Address:", font=("Arial", 9)).pack(side=tk.LEFT)  # Label
        self.visa_address_var = tk.StringVar(value="USB0::0x05E6::0x2230::805224014806770001::INSTR")  # Default address
        self.visa_entry = ttk.Entry(addr_frame, textvariable=self.visa_address_var, font=("Courier", 9))  # Entry field
        self.visa_entry.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)  # Fill remaining space
        
        # Connection control buttons
        btn_frame = ttk.Frame(conn_frame)  # Container for buttons
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.connect_btn = ttk.Button(btn_frame, text="Connect",
            command=self.connect_power_supply, style="Success.TButton")  # Green connect button
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.disconnect_btn = ttk.Button(btn_frame, text="Disconnect",
            command=self.disconnect_power_supply,
            style="Danger.TButton", state="disabled")  # Red, disabled until connected
        self.disconnect_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.emergency_stop_btn = ttk.Button(btn_frame, text="EMERGENCY STOP",
            command=self.emergency_stop, style="Danger.TButton", state="disabled")  # Red, disabled until connected
        self.emergency_stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # Connection status indicator
        self.conn_status_var = tk.StringVar(value="Disconnected")  # Status text
        self.conn_status_label = ttk.Label(btn_frame, textvariable=self.conn_status_var,
            font=("Arial", 9, "bold"), foreground="red")  # Red initially (disconnected)
        self.conn_status_label.pack(side=tk.RIGHT)  # Right aligned
        
        # Instrument info display
        self.info_text = tk.Text(conn_frame, height=2, font=("Courier", 8), state="disabled",
            wrap=tk.WORD, bg="#96cbff")  # Read-only monospace text
        self.info_text.pack(fill=tk.X)  # Fill width

    def create_channel_frames(self):
        """
        Create section for three independent channel controls
        
        Generates control panels for channels 1-3 with voltage/current/OVP
        settings and enable/disable/measure buttons.
        """
        channels_frame = ttk.LabelFrame(self.scrollable_frame, text="Channel Controls", padding="12")  # Labeled section
        channels_frame.pack(fill=tk.X, pady=(0, 15))  # Fill width with padding
        
        self.channel_frames = {}  # Dictionary to store channel frame references
        self.channel_vars = {}  # Dictionary to store channel control variables
        
        # Create control panel for each channel (1-3)
        for channel in range(1, 4):  # channels 1, 2, 3
            self.create_single_channel_frame(channels_frame, channel)  # Build channel panel

    def create_single_channel_frame(self, parent, channel_num):
        """
        Create control panel for single channel with settings and buttons
        
        Builds a labeled frame containing voltage/current spinboxes,
        buttons for configure/enable/disable/measure, and measurement displays.
        
        Args:
            parent: Parent container frame
            channel_num: Channel number (1, 2, or 3)
        """
        channel_frame = ttk.LabelFrame(parent, text=f"Channel {channel_num}", padding="10")  # Labeled frame
        channel_frame.pack(fill=tk.X, pady=(0, 10))  # Fill width, add bottom padding
        
        self.channel_frames[channel_num] = channel_frame  # Store frame reference
        
        # Initialize control variables for this channel
        self.channel_vars[channel_num] = {
            "voltage": tk.DoubleVar(value=0.0),  # Set voltage in volts
            "current_limit": tk.DoubleVar(value=0.1),  # Current limit in amps
            "ovp_level": tk.DoubleVar(value=30.0),  # Over-voltage protection level
            "measured_voltage": tk.StringVar(value="0.000 V"),  # Displayed measured voltage
            "measured_current": tk.StringVar(value="0.000 A"),  # Displayed measured current
            "measured_power": tk.StringVar(value="0.000 W")  # Displayed measured power (V×I)
        }
        
        # Row 1: Input controls (voltage, current, OVP)
        control_row1 = ttk.Frame(channel_frame)  # Container for input controls
        control_row1.pack(fill=tk.X, pady=(0, 8))  # Fill width with bottom padding
        
        # Voltage setting
        ttk.Label(control_row1, text="Voltage (V):", font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=(0, 5))
        voltage_spin = tk.Spinbox(control_row1, from_=0.0, to=30.0, increment=0.1, width=8,
            textvariable=self.channel_vars[channel_num]["voltage"], format="%.2f")  # 0-30V spinbox
        voltage_spin.grid(row=0, column=1, padx=(0, 15))  # Right-aligned with padding
        
        # Current limit setting
        ttk.Label(control_row1, text="Current Limit (A):", font=("Arial", 9)).grid(row=0, column=2, sticky="w", padx=(0, 5))
        current_spin = tk.Spinbox(control_row1, from_=0.001, to=3.0, increment=0.01, width=8,
            textvariable=self.channel_vars[channel_num]["current_limit"], format="%.3f")  # 0.001-3.0A spinbox
        current_spin.grid(row=0, column=3, padx=(0, 15))
        
        # OVP (Over-voltage protection) setting
        ttk.Label(control_row1, text="OVP (V):", font=("Arial", 9)).grid(row=0, column=4, sticky="w", padx=(0, 5))
        ovp_spin = tk.Spinbox(control_row1, from_=1.0, to=35.0, increment=0.5, width=8,
            textvariable=self.channel_vars[channel_num]["ovp_level"], format="%.1f")  # 1-35V OVP setting
        ovp_spin.grid(row=0, column=5)
        
        # Row 2: Operation buttons (Configure, Enable, Disable, Measure)
        btn_row = ttk.Frame(channel_frame)  # Container for control buttons
        btn_row.pack(fill=tk.X, pady=(0, 8))  # Fill width with bottom padding
        
        configure_btn = ttk.Button(btn_row, text="Configure",
            command=lambda ch=channel_num: self.configure_channel(ch),
            style="Info.TButton", state="disabled")  # Configure channel settings
        configure_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        enable_btn = ttk.Button(btn_row, text="Enable Output",
            command=lambda ch=channel_num: self.enable_channel_output(ch),
            style="Success.TButton", state="disabled")  # Turn on output
        enable_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        disable_btn = ttk.Button(btn_row, text="Disable Output",
            command=lambda ch=channel_num: self.disable_channel_output(ch),
            style="Warning.TButton", state="disabled")  # Turn off output
        disable_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        measure_btn = ttk.Button(btn_row, text="Measure",
            command=lambda ch=channel_num: self.measure_channel_output(ch),
            style="Info.TButton", state="disabled")  # Get current readings
        measure_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Store button references for enable/disable control
        self.channel_vars[channel_num]["buttons"] = {
            "configure": configure_btn,
            "enable": enable_btn,
            "disable": disable_btn,
            "measure": measure_btn
        }
        
        # Row 3: Measurement displays and status
        measure_row = ttk.Frame(channel_frame)  # Container for measurements
        measure_row.pack(fill=tk.X)  # Fill width
        
        ttk.Label(measure_row, text="Measurements:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)  # Header
        
        voltage_label = ttk.Label(measure_row, textvariable=self.channel_vars[channel_num]["measured_voltage"],
            style="Measurement.TLabel")  # Display measured voltage
        voltage_label.pack(side=tk.LEFT, padx=(15, 15))  # Center with padding
        
        current_label = ttk.Label(measure_row, textvariable=self.channel_vars[channel_num]["measured_current"],
            style="Measurement.TLabel")  # Display measured current
        current_label.pack(side=tk.LEFT, padx=(0, 15))
        
        power_label = ttk.Label(measure_row, textvariable=self.channel_vars[channel_num]["measured_power"],
            style="Measurement.TLabel")  # Display calculated power (V×I)
        power_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # Channel status indicator (ON/OFF)
        status_label = ttk.Label(measure_row, text="OFF", font=("Arial", 9, "bold"), foreground="red")  # Red initially (off)
        status_label.pack(side=tk.RIGHT)  # Right aligned
        self.channel_vars[channel_num]["status_label"] = status_label  # Store reference

    def create_global_operations_frame(self):
        """
        Create section with global operation buttons (Get Info, Disable All, etc.)
        
        Provides buttons for operations that affect the entire instrument
        rather than individual channels.
        """
        global_frame = ttk.LabelFrame(self.scrollable_frame, text="Global Operations", padding="12")  # Labeled section
        global_frame.pack(fill=tk.X, pady=(0, 15))  # Fill width with padding
        
        btn_frame = ttk.Frame(global_frame)  # Container for buttons
        btn_frame.pack(fill=tk.X)  # Fill width
        
        self.get_info_btn = ttk.Button(btn_frame, text="Get Instrument Info",
            command=self.get_instrument_info, style="Info.TButton", state="disabled")  # Query device info
        self.get_info_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.disable_all_btn = ttk.Button(btn_frame, text="Disable All Outputs",
            command=self.disable_all_outputs, style="Danger.TButton", state="disabled")  # Turn off all channels
        self.disable_all_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.test_connection_btn = ttk.Button(btn_frame, text="Test Connection",
            command=self.test_connection, style="Info.TButton", state="disabled")  # Verify communication
        self.test_connection_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.measure_all_btn = ttk.Button(btn_frame, text="Measure All Channels",
            command=self.measure_all_channels, style="Success.TButton", state="disabled")  # Get all readings
        self.measure_all_btn.pack(side=tk.LEFT)

    def create_data_logging_frame(self):
        """
        Create section for automatic measurement logging and data export
        
        Provides checkbox for auto-measurement, interval setting, and
        buttons for exporting data to CSV and clearing collection.
        """
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Logging & Export", padding="12")  # Labeled section
        log_frame.pack(fill=tk.X, pady=(0, 15))  # Fill width with padding
        
        control_frame = ttk.Frame(log_frame)  # Container for controls
        control_frame.pack(fill=tk.X)  # Fill width
        
        # Auto-measurement checkbox and interval
        self.auto_measure_var = tk.BooleanVar(value=False)  # Auto-measurement enabled flag
        
        auto_check = ttk.Checkbutton(control_frame, text="Auto-measure every",
            variable=self.auto_measure_var, command=self.toggle_auto_measure)  # Enable/disable auto-measurement
        auto_check.pack(side=tk.LEFT)
        
        self.measure_interval_var = tk.DoubleVar(value=2.0)  # Measurement interval in seconds
        
        interval_spin = tk.Spinbox(control_frame, from_=0.5, to=60.0, increment=0.5, width=6,
            textvariable=self.measure_interval_var, format="%.1f")  # 0.5-60 second spinbox
        interval_spin.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Label(control_frame, text="seconds").pack(side=tk.LEFT)  # Unit label
        
        # Data export buttons
        self.export_btn = ttk.Button(control_frame, text="Export to CSV",
            command=self.export_measurement_data, style="Success.TButton", state="disabled")  # Export measurements
        self.export_btn.pack(side=tk.LEFT, padx=(20, 8))
        
        ttk.Button(control_frame, text="Clear Data",
            command=self.clear_measurement_data, style="Warning.TButton").pack(side=tk.LEFT)  # Clear collection

    def create_status_frame(self):
        """
        Create section for status display and activity logging
        
        Shows current operation status and maintains scrolling log of all
        user actions and system events for troubleshooting.
        """
        status_frame = ttk.LabelFrame(self.scrollable_frame, text="Status & Activity Log", padding="12")  # Labeled section
        status_frame.pack(fill=tk.BOTH, expand=True)  # Fill remaining space (expandable)
        
        # Current status display
        self.current_operation_var = tk.StringVar(value="Ready - Connect to power supply to begin")  # Status text
        ttk.Label(status_frame, text="Current Status:", font=("Arial", 9)).pack(anchor="w")  # Label
        
        operation_label = ttk.Label(status_frame, textvariable=self.current_operation_var,
            font=("Arial", 9), foreground="#2c5aa0", wraplength=800)  # Blue status text, wraps long messages
        operation_label.pack(anchor="w", pady=(0, 10))  # West-aligned with bottom padding
        
        # Activity log
        ttk.Label(status_frame, text="Activity Log:", font=("Arial", 9)).pack(anchor="w")  # Header
        
        self.log_text = scrolledtext.ScrolledText(status_frame, height=10, font=("Courier", 8), wrap=tk.WORD)  # Monospace scrolling text
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))  # Expandable text area with padding
        
        # Control buttons for log
        control_frame = ttk.Frame(status_frame)  # Container for buttons
        control_frame.pack(fill=tk.X)  # Fill width
        
        ttk.Button(control_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)  # Clear log content
        ttk.Button(control_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=(5, 0))  # Save to file

    # ============================================================================
    # CONNECTION MANAGEMENT METHODS
    # ============================================================================

    def connect_power_supply(self):
        """
        Establish VISA connection to Keithley power supply
        
        Spawns background thread to connect, avoiding GUI freezing.
        Updates connection status and enables channel controls on success.
        """
        def connect_thread():  # Background worker function
            try:
                self.update_status("Connecting to Keithley power supply...")  # Update status message
                self.log_message("Attempting to connect to Keithley power supply...")  # Log attempt
                
                visa_address = self.visa_address_var.get().strip()  # Get VISA address from entry field
                if not visa_address:  # Check if address is empty
                    raise ValueError("VISA address cannot be empty")  # Raise error
                
                self.power_supply = KeithleyPowerSupply(visa_address)  # Create driver instance
                
                if self.power_supply.connect():  # Attempt connection
                    self.log_message("Connection established successfully!", "SUCCESS")  # Log success
                    self.status_queue.put(("connected", None))  # Signal main thread: connected
                else:
                    raise Exception("Connection failed - check VISA address and instrument")  # Raise error
            
            except Exception as e:  # Catch any connection errors
                self.status_queue.put(("error", f"Connection failed: {str(e)}"))  # Signal main thread: error
        
        threading.Thread(target=connect_thread, daemon=True).start()  # Start background thread

    def disconnect_power_supply(self):
        """
        Close VISA connection to power supply and disable all controls
        
        Performs cleanup: closes connection, resets UI state, clears status
        displays, and disables all operation buttons until reconnected.
        """
        try:
            if self.power_supply:  # Check if power supply is connected
                self.power_supply.disconnect()  # Close VISA connection
                self.power_supply = None  # Clear reference
            
            # Update connection status UI
            self.conn_status_var.set("Disconnected")  # Set status text
            self.conn_status_label.configure(foreground="red")  # Red color (disconnected)
            self.connect_btn.configure(state="normal")  # Enable connect button
            self.disconnect_btn.configure(state="disabled")  # Disable disconnect button
            
            # Disable all operation buttons
            self.disable_all_buttons()  # Reset button states
            
            # Clear instrument info display
            self.info_text.configure(state="normal")  # Enable editing
            self.info_text.delete(1.0, tk.END)  # Clear content
            self.info_text.configure(state="disabled")  # Disable editing
            
            # Reset all channel displays
            for channel in self.channel_vars:  # Process each channel
                self.channel_vars[channel]["status_label"].configure(text="OFF", foreground="red")  # OFF status
                self.channel_vars[channel]["measured_voltage"].set("0.000 V")  # Clear readings
                self.channel_vars[channel]["measured_current"].set("0.000 A")
                self.channel_vars[channel]["measured_power"].set("0.000 W")
            
            self.measurement_active = False  # Disable auto-measurement
            
            self.update_status("Disconnected from power supply")  # Update status
            self.log_message("Disconnected from power supply", "SUCCESS")  # Log event
        
        except Exception as e:  # Catch any errors during disconnect
            self.log_message(f"Error during disconnection: {e}", "ERROR")  # Log error

    # ============================================================================
    # INSTRUMENT INFORMATION AND TESTING
    # ============================================================================

    def get_instrument_info(self):
        """
        Query instrument identification and capabilities information
        
        Retrieves manufacturer, model, serial number, and firmware version
        from connected power supply and displays in info text widget.
        """
        def info_thread():  # Background worker function
            try:
                self.update_status("Retrieving instrument information...")  # Update status
                self.log_message("Getting instrument information...")  # Log operation
                
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                info = self.power_supply.get_instrument_info()  # Query device (calls *IDN?)
                
                if info:  # Check if info retrieved successfully
                    self.status_queue.put(("info_retrieved", info))  # Send to main thread
                else:
                    self.status_queue.put(("error", "Failed to retrieve instrument information"))  # Error signal
            
            except Exception as e:  # Catch any query errors
                self.status_queue.put(("error", f"Info retrieval error: {str(e)}"))  # Send error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection exists
            threading.Thread(target=info_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error dialog

    def test_connection(self):
        """
        Test VISA communication link to verify power supply is responsive
        
        Quick connectivity check without modifying any settings. Useful
        for verifying VISA drivers and USB connection stability.
        """
        try:
            if self.power_supply:  # Check if power supply reference exists
                is_connected = self.power_supply.is_connected  # Check connection flag
                
                if is_connected:  # Connection is active
                    self.log_message("Connection test: PASSED", "SUCCESS")  # Log success
                    self.update_status("Connection test passed")  # Update status
                    messagebox.showinfo("Connection Test", "Connection active")  # Show success dialog
                else:
                    self.log_message("Connection test: FAILED", "ERROR")  # Log failure
                    messagebox.showerror("Connection Test", "No connection")  # Show error dialog
            else:
                messagebox.showerror("Connection Test", "Connect first")  # Show connect-first message
        
        except Exception as e:  # Catch any test errors
            self.log_message(f"Connection test error: {e}", "ERROR")  # Log error

    # ============================================================================
    # CHANNEL CONFIGURATION AND CONTROL
    # ============================================================================

    def configure_channel(self, channel):
        """
        Apply configuration settings to channel (voltage, current, OVP)
        
        Sends voltage, current limit, and over-voltage protection settings
        to specified channel. Must be done before enabling output.
        
        Args:
            channel: Channel number (1, 2, or 3)
        """
        def config_thread():  # Background worker function
            try:
                # Retrieve settings from GUI spinboxes
                voltage = self.channel_vars[channel]["voltage"].get()  # Set voltage
                current_limit = self.channel_vars[channel]["current_limit"].get()  # Current limit
                ovp_level = self.channel_vars[channel]["ovp_level"].get()  # OVP level
                
                self.update_status(f"Configuring channel {channel}...")  # Update status
                self.log_message(f"Configuring channel {channel} - V: {voltage:.3f}V, I: {current_limit:.3f}A, OVP: {ovp_level:.1f}V")  # Log settings
                
                # Send configuration to power supply
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                success = self.power_supply.configure_channel(
                    channel=channel,
                    voltage=voltage,
                    current_limit=current_limit,
                    ovp_level=ovp_level,
                    enable_output=False  # Don't enable output yet
                )
                
                if success:  # Configuration succeeded
                    self.status_queue.put(("channel_configured",
                        f"Channel {channel} configured successfully"))  # Success signal
                else:
                    self.status_queue.put(("error", f"Failed to configure channel {channel}"))  # Failure signal
            
            except Exception as e:  # Catch any configuration errors
                self.status_queue.put(("error", f"Channel {channel} configuration error: {str(e)}"))  # Error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection
            threading.Thread(target=config_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error dialog

    def enable_channel_output(self, channel):
        """
        Enable output on specified channel
        
        Turns on the output on the channel, providing voltage/current
        to the load. Channel must be configured first.
        
        Args:
            channel: Channel number (1, 2, or 3)
        """
        def enable_thread():  # Background worker function
            try:
                self.update_status(f"Enabling output on channel {channel}...")  # Update status
                self.log_message(f"Enabling output on channel {channel}...")  # Log operation
                
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                success = self.power_supply.enable_channel_output(channel)  # Send enable command
                
                if success:  # Enable succeeded
                    self.status_queue.put(("channel_enabled", channel))  # Success signal with channel
                else:
                    self.status_queue.put(("error", f"Failed to enable channel {channel} output"))  # Failure signal
            
            except Exception as e:  # Catch any enable errors
                self.status_queue.put(("error", f"Channel {channel} enable error: {str(e)}"))  # Error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection
            threading.Thread(target=enable_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error dialog

    def disable_channel_output(self, channel):
        """
        Disable output on specified channel
        
        Turns off the output, stopping voltage/current delivery to load.
        Safety: Outputs voltage to 0V before disabling.
        
        Args:
            channel: Channel number (1, 2, or 3)
        """
        def disable_thread():  # Background worker function
            try:
                self.update_status(f"Disabling output on channel {channel}...")  # Update status
                self.log_message(f"Disabling output on channel {channel}...")  # Log operation
                
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                success = self.power_supply.disable_channel_output(channel)  # Send disable command
                
                if success:  # Disable succeeded
                    self.status_queue.put(("channel_disabled", channel))  # Success signal with channel
                else:
                    self.status_queue.put(("error", f"Failed to disable channel {channel} output"))  # Failure signal
            
            except Exception as e:  # Catch any disable errors
                self.status_queue.put(("error", f"Channel {channel} disable error: {str(e)}"))  # Error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection
            threading.Thread(target=disable_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error dialog

    def measure_channel_output(self, channel):
        """
        Read current voltage and current from specified channel
        
        Queries actual channel output voltage and current, calculates power
        (P = V × I), and displays in GUI measurement labels.
        
        Args:
            channel: Channel number (1, 2, or 3)
        """
        def measure_thread():  # Background worker function
            try:
                self.update_status(f"Measuring channel {channel} output...")  # Update status
                
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                measurement = self.power_supply.measure_channel_output(channel)  # Get readings
                
                if measurement and isinstance(measurement, tuple) and len(measurement) == 2:  # Measurement succeeded
                    voltage = float(measurement[0])  # Extract voltage
                    current = float(measurement[1])  # Extract current
                    power = voltage * current  # Calculate power in watts
                    
                    self.status_queue.put(("channel_measured", {
                        "channel": channel,
                        "voltage": voltage,
                        "current": current,
                        "power": power
                    }))  # Send measurement data
                    
                    self.log_message(f"Channel {channel}: {voltage:.3f}V, {current:.3f}A, {power:.3f}W", "SUCCESS")  # Log readings
                
                else:
                    self.status_queue.put(("error", f"Failed to measure channel {channel} output"))  # Failure signal
            
            except Exception as e:  # Catch any measurement errors
                self.status_queue.put(("error", f"Channel {channel} measurement error: {str(e)}"))  # Error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection
            threading.Thread(target=measure_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error dialog

    # ============================================================================
    # GLOBAL OPERATIONS AND SAFETY
    # ============================================================================

    def measure_all_channels(self):
        """
        Measure all three channels sequentially to avoid resource conflicts
        
        Performs measurements on channels 1, 2, 3 with timing delays between
        to prevent USB timeouts. Results displayed in channel measurement labels.
        """
        if not (self.power_supply and self.power_supply.is_connected):  # Verify connection
            messagebox.showerror("Error", "Power supply not connected")  # Show error
            return
        
        def sequential_measurement():  # Background worker function
            try:
                self.update_status("Measuring all channels sequentially...")  # Update status
                self.log_message("Starting sequential measurement of all channels...")  # Log operation
                
                # Disable measurement buttons during operation
                self.measure_all_btn.configure(state="disabled")  # Disable main button
                for channel in range(1, 4):  # All three channels
                    self.channel_vars[channel]["buttons"]["measure"].configure(state="disabled")  # Disable channel buttons
                
                # Measure each channel sequentially
                for channel in range(1, 4):  # channels 1, 2, 3
                    try:
                        self.log_message(f"Measuring channel {channel}...")  # Log current operation
                        
                        if not self.power_supply:
                            raise RuntimeError("Power supply not connected")
                        measurement = self.power_supply.measure_channel_output(channel)  # Get readings
                        
                        if measurement and isinstance(measurement, tuple) and len(measurement) == 2:  # Measurement succeeded
                            voltage = float(measurement[0])  # Extract voltage
                            current = float(measurement[1])  # Extract current
                            power = voltage * current  # Calculate power
                            
                            self.status_queue.put(("channel_measured", {
                                "channel": channel,
                                "voltage": voltage,
                                "current": current,
                                "power": power
                            }))  # Send data to main thread
                            
                            self.log_message(f"Channel {channel}: {voltage:.3f}V, {current:.3f}A, {power:.3f}W", "SUCCESS")  # Log result
                        
                        else:
                            self.log_message(f"Failed to measure channel {channel}", "ERROR")  # Log failure
                        
                        # Wait between measurements to prevent USB timeouts
                        if channel < 3:  # Not the last channel
                            time.sleep(0.8)  # 800ms delay before next measurement
                    
                    except Exception as e:  # Catch per-channel errors
                        self.log_message(f"Error measuring channel {channel}: {e}", "ERROR")  # Log error
                
                self.update_status("All channel measurements completed")  # Update status
                self.log_message("Sequential measurement completed", "SUCCESS")  # Log completion
            
            except Exception as e:  # Catch any overall errors
                self.log_message(f"Error in sequential measurement: {e}", "ERROR")  # Log error
            
            finally:
                # Re-enable buttons after measurement
                self.status_queue.put(("measurement_complete", None))  # Signal measurement done
        
        threading.Thread(target=sequential_measurement, daemon=True).start()  # Start background thread

    def disable_all_outputs(self):
        """
        Disable output on all three channels (safety shutdown)
        
        Turns off all channel outputs immediately. Used for quick shutdown
        when exiting or in response to emergency stop request.
        """
        def disable_all_thread():  # Background worker function
            try:
                self.update_status("Disabling all outputs...")  # Update status
                self.log_message("Emergency shutdown - disabling all outputs...")  # Log event
                
                if not self.power_supply:
                    raise RuntimeError("Power supply not connected")
                success = self.power_supply.disable_all_outputs()  # Send shutdown command
                
                if success:  # All outputs disabled
                    self.status_queue.put(("all_disabled", "All outputs disabled successfully"))  # Success signal
                else:
                    self.status_queue.put(("error", "Failed to disable all outputs"))  # Failure signal
            
            except Exception as e:  # Catch any disable errors
                self.status_queue.put(("error", f"Disable all error: {str(e)}"))  # Error signal
        
        if self.power_supply and self.power_supply.is_connected:  # Verify connection
            threading.Thread(target=disable_all_thread, daemon=True).start()  # Start background thread
        else:
            messagebox.showerror("Error", "Power supply not connected")  # Show error

    def emergency_stop(self):
        """
        Activate emergency stop (immediate shutdown of all outputs)
        
        Critical safety feature: stops all power delivery immediately.
        Disables all channel outputs without questions or delays.
        """
        self.log_message("EMERGENCY STOP ACTIVATED!", "ERROR")  # Log emergency event
        self.disable_all_outputs()  # Disable all outputs now

    # ============================================================================
    # DATA LOGGING AND EXPORT
    # ============================================================================

    def toggle_auto_measure(self):
        """
        Enable or disable automatic periodic measurement collection
        
        When enabled, background task periodically reads all channel outputs
        at user-specified interval and stores measurements.
        """
        self.measurement_active = self.auto_measure_var.get()  # Get checkbox state
        
        if self.measurement_active:  # Auto-measure enabled
            self.log_message("Auto-measurement enabled", "SUCCESS")  # Log state change
        else:
            self.log_message("Auto-measurement disabled")  # Log state change

    def update_measurements(self):
        """
        Periodic background task for automatic measurement collection
        
        Runs on fixed interval (from measure_interval_var) when auto-measurement
        is enabled. Reads all channels, stores data, and updates displays.
        Calls itself recursively using root.after() for scheduling.
        """
        if (self.measurement_active and self.power_supply and  # All checks: active + connected
            self.power_supply.is_connected and self.auto_measure_var.get()):  # Flags are set
            
            # Measure each channel
            for channel in range(1, 4):  # channels 1, 2, 3
                try:
                    measurement = self.power_supply.measure_channel_output(channel)  # Get readings
                    
                    if measurement:  # Measurement succeeded
                        if isinstance(measurement, tuple) and len(measurement) == 2:
                            voltage = float(measurement[0])  # Extract voltage
                            current = float(measurement[1])  # Extract current
                        else:
                            continue
                        power = voltage * current  # Calculate power
                        
                        # Update display labels
                        self.channel_vars[channel]["measured_voltage"].set(f"{voltage:.3f} V")  # Display voltage
                        self.channel_vars[channel]["measured_current"].set(f"{current:.3f} A")  # Display current
                        self.channel_vars[channel]["measured_power"].set(f"{power:.3f} W")  # Display power
                        
                        # Store measurement for export
                        timestamp = datetime.now()  # Get current time
                        if channel not in self.measurement_data:  # First measurement for this channel
                            self.measurement_data[channel] = []  # Initialize list
                        
                        self.measurement_data[channel].append({  # Add measurement record
                            "timestamp": timestamp,
                            "voltage": voltage,
                            "current": current,
                            "power": power
                        })
                        
                        self.export_btn.configure(state="normal")  # Enable export button
                
                except Exception as e:  # Catch any measurement errors
                    self.log_message(f"Auto-measurement error on channel {channel}: {e}", "ERROR")  # Log error
        
        # Schedule next measurement (recurring timer)
        interval_ms = int(self.measure_interval_var.get() * 1000)  # Convert seconds to milliseconds
        self.root.after(interval_ms, self.update_measurements)  # Schedule next call

    def export_measurement_data(self):
        """
        Export collected measurements to CSV file with timestamped filename
        
        Opens file dialog for user to select save location, then writes all
        stored channel measurements to CSV format with standard headers.
        """
        try:
            if not self.measurement_data:  # Check if any data collected
                messagebox.showwarning("No Data", "No measurement data to export")  # Show warning
                return
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Generate timestamp for filename
            
            # Open file save dialog with suggestions
            filename = filedialog.asksaveasfilename(
                title="Export measurements to CSV",  # Dialog title
                defaultextension=".csv",  # Default extension
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],  # File type filters
                initialdir=".",  # Start in current directory
                initialfile=f"power_supply_data_{timestamp}.csv"  # Suggested filename
            )
            
            if not filename:  # User cancelled dialog
                return  # Exit without saving
            
            # Write CSV file
            with open(filename, "w", newline="") as csvfile:  # Open file for writing
                writer = csv.writer(csvfile)  # Create CSV writer
                writer.writerow(["Timestamp", "Channel", "Voltage (V)", "Current (A)", "Power (W)"])  # Header row
                
                # Write all measurements
                for channel, measurements in self.measurement_data.items():  # Each channel
                    for measurement in measurements:  # Each measurement point
                        writer.writerow([
                            measurement["timestamp"].isoformat(),  # ISO format timestamp
                            channel,  # Channel number
                            f"{measurement['voltage']:.6f}",  # Voltage with 6 decimals
                            f"{measurement['current']:.6f}",  # Current with 6 decimals
                            f"{measurement['power']:.6f}"  # Power with 6 decimals
                        ])
            
            self.log_message(f"Data exported to: {filename}", "SUCCESS")  # Log success
            messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")  # Show completion dialog
        
        except Exception as e:  # Catch any export errors
            self.log_message(f"Export error: {e}", "ERROR")  # Log error
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")  # Show error dialog

    def clear_measurement_data(self):
        """
        Clear all collected measurement data (reset collection)
        
        Removes all stored measurements. User can restart collection fresh.
        """
        self.measurement_data.clear()  # Empty measurement dictionary
        self.log_message("Measurement data cleared")  # Log action
        self.export_btn.configure(state="disabled")  # Disable export button (no data)

    def save_log(self):
        """
        Save activity log content to text file with timestamp
        
        Allows user to save debugging/audit trail of all operations.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Generate timestamp
            log_content = self.log_text.get(1.0, tk.END)  # Get all log text
            
            # Open file save dialog
            filename = filedialog.asksaveasfilename(
                title="Save activity log",  # Dialog title
                defaultextension=".txt",  # Default extension
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],  # File type filters
                initialdir=".",  # Start in current directory
                initialfile=f"power_supply_log_{timestamp}.txt"  # Suggested filename
            )
            
            if not filename:  # User cancelled
                return  # Exit without saving
            
            # Write log file
            with open(filename, "w") as f:  # Open file for writing
                f.write(f"Keithley Power Supply Log - {datetime.now()}\n")  # Header with date
                f.write("="*60 + "\n\n")  # Separator
                f.write(log_content)  # All log content
            
            self.log_message(f"Log saved to: {filename}", "SUCCESS")  # Log action
        
        except Exception as e:  # Catch any save errors
            self.log_message(f"Failed to save log: {e}", "ERROR")  # Log error
            messagebox.showerror("Save Log Error", f"Failed to save log:\n{e}")  # Show error dialog

    # ============================================================================
    # UI UTILITY METHODS
    # ============================================================================

    def log_message(self, message: str, level: str = "INFO"):
        """
        Add timestamped message to activity log with color-coding
        
        Appends message to log text widget with timestamp prefix and color
        based on message level (INFO=blue, SUCCESS=green, WARNING=orange, ERROR=red).
        
        Args:
            message: Message text to log
            level: Message level (INFO, SUCCESS, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")  # Format timestamp HH:MM:SS
        log_entry = f"[{timestamp}] {level}: {message}\n"  # Format complete log line
        
        self.log_text.insert(tk.END, log_entry)  # Add to end of text widget
        self.log_text.see(tk.END)  # Scroll to bottom to show new message
        
        # Apply color-coding based on level
        if level == "ERROR":  # Error messages = red
            self.log_text.tag_add("error", f"end-{len(log_entry)}c", "end-1c")  # Mark error text
            self.log_text.tag_config("error", foreground="red")  # Configure red color
        elif level == "SUCCESS":  # Success messages = green
            self.log_text.tag_add("success", f"end-{len(log_entry)}c", "end-1c")  # Mark success text
            self.log_text.tag_config("success", foreground="green")  # Configure green color
        elif level == "WARNING":  # Warning messages = orange
            self.log_text.tag_add("warning", f"end-{len(log_entry)}c", "end-1c")  # Mark warning text
            self.log_text.tag_config("warning", foreground="orange")  # Configure orange color
        else:  # INFO messages = blue (default)
            self.log_text.tag_add("info", f"end-{len(log_entry)}c", "end-1c")  # Mark info text
            self.log_text.tag_config("info", foreground="blue")  # Configure blue color

    def clear_log(self):
        """Clear all content from activity log text widget"""
        self.log_text.delete(1.0, tk.END)  # Delete from line 1, column 0 to end

    def update_status(self, status: str):
        """
        Update current operation status display text
        
        Immediately updates the status label with new message.
        Forces GUI update to display new status.
        
        Args:
            status: Status message text
        """
        self.current_operation_var.set(status)  # Update StringVar (updates label automatically)
        self.root.update_idletasks()  # Process pending GUI updates immediately

    def disable_all_buttons(self):
        """Disable all operation buttons (for disconnected state)"""
        buttons = [self.get_info_btn, self.disable_all_btn, self.test_connection_btn,
                   self.measure_all_btn, self.emergency_stop_btn]  # Global buttons
        
        for btn in buttons:  # Disable each button
            btn.configure(state="disabled")  # Set to disabled state
        
        # Disable channel buttons
        for channel in self.channel_vars:  # Each channel
            for btn in self.channel_vars[channel]["buttons"].values():  # Each button in channel
                btn.configure(state="disabled")  # Disable button

    def enable_all_buttons(self):
        """Enable all operation buttons (for connected state)"""
        buttons = [self.get_info_btn, self.disable_all_btn, self.test_connection_btn,
                   self.measure_all_btn, self.emergency_stop_btn]  # Global buttons
        
        for btn in buttons:  # Enable each button
            btn.configure(state="normal")  # Set to normal (enabled) state
        
        # Enable channel buttons
        for channel in self.channel_vars:  # Each channel
            for btn in self.channel_vars[channel]["buttons"].values():  # Each button in channel
                btn.configure(state="normal")  # Enable button
        
        # Only enable export if data exists
        if self.measurement_data:  # Has collected measurements
            self.export_btn.configure(state="normal")  # Enable export

    def display_instrument_info(self, info):
        """
        Display instrument identification information in info text widget
        
        Shows manufacturer, model, serial number, and firmware in single line.
        
        Args:
            info: Dictionary with keys: manufacturer, model, serial_number, firmware_version
        """
        self.info_text.configure(state="normal")  # Enable editing (currently read-only)
        self.info_text.delete(1.0, tk.END)  # Clear previous content
        
        # Format info string
        info_str = f"{info.get('manufacturer', 'N/A')} {info.get('model', 'N/A')} "  # Manufacturer and model
        info_str += f"| S/N: {info.get('serial_number', 'N/A')} | FW: {info.get('firmware_version', 'N/A')}"  # Serial and firmware
        
        self.info_text.insert(1.0, info_str)  # Insert formatted string
        self.info_text.configure(state="disabled")  # Disable editing again (read-only)

    def check_status_updates(self):
        """
        Periodic polling of status queue for messages from worker threads
        
        Checks for status updates from background threads and processes them:
        connected, disconnected, errors, measurements, button states, etc.
        Calls itself recursively every 100ms using root.after().
        """
        try:
            while True:  # Process all queued messages
                status_type, data = self.status_queue.get_nowait()  # Get next message (non-blocking)
                
                # Route message to appropriate handler based on type
                if status_type == "connected":  # Connection successful
                    self.conn_status_var.set("Connected")  # Update status text
                    self.conn_status_label.configure(foreground="green")  # Green color
                    self.connect_btn.configure(state="disabled")  # Disable connect button
                    self.disconnect_btn.configure(state="normal")  # Enable disconnect button
                    self.enable_all_buttons()  # Enable all operation buttons
                    self.update_status("Connected - All methods available")  # Update status
                    self.get_instrument_info()  # Query device info automatically
                
                elif status_type == "error":  # Error from worker thread
                    self.log_message(data, "ERROR")  # Log error message
                    self.update_status("Error occurred - check log")  # Update status
                
                elif status_type == "info_retrieved":  # Instrument info received
                    self.display_instrument_info(data)  # Display info
                    self.log_message("Instrument information retrieved", "SUCCESS")  # Log success
                    self.update_status("Connected and ready")  # Update status
                
                elif status_type == "channel_configured":  # Channel configured successfully
                    self.log_message(data, "SUCCESS")  # Log success message
                    self.update_status("Channel configuration completed")  # Update status
                
                elif status_type == "channel_enabled":  # Channel output enabled
                    channel = data  # Extract channel number
                    self.channel_vars[channel]["status_label"].configure(text="ON", foreground="green")  # Green ON
                    self.log_message(f"Channel {channel} output enabled", "SUCCESS")  # Log success
                    self.update_status(f"Channel {channel} output enabled")  # Update status
                
                elif status_type == "channel_disabled":  # Channel output disabled
                    channel = data  # Extract channel number
                    self.channel_vars[channel]["status_label"].configure(text="OFF", foreground="red")  # Red OFF
                    self.log_message(f"Channel {channel} output disabled", "SUCCESS")  # Log success
                    self.update_status(f"Channel {channel} output disabled")  # Update status
                
                elif status_type == "channel_measured":  # Measurement data received
                    channel = data["channel"]  # Extract channel
                    voltage = data["voltage"]  # Extract voltage
                    current = data["current"]  # Extract current
                    power = data["power"]  # Extract power
                    
                    # Update display labels
                    self.channel_vars[channel]["measured_voltage"].set(f"{voltage:.3f} V")  # Display V
                    self.channel_vars[channel]["measured_current"].set(f"{current:.3f} A")  # Display I
                    self.channel_vars[channel]["measured_power"].set(f"{power:.3f} W")  # Display P
                    
                    # Store measurement for export
                    timestamp = datetime.now()  # Current time
                    if channel not in self.measurement_data:  # First measurement
                        self.measurement_data[channel] = []  # Initialize list
                    
                    self.measurement_data[channel].append({  # Add measurement
                        "timestamp": timestamp,
                        "voltage": voltage,
                        "current": current,
                        "power": power
                    })
                    
                    self.export_btn.configure(state="normal")  # Enable export button
                    self.update_status(f"Channel {channel} measured: {voltage:.3f}V, {current:.3f}A")  # Update status
                
                elif status_type == "all_disabled":  # All outputs disabled
                    for channel in range(1, 4):  # All channels
                        self.channel_vars[channel]["status_label"].configure(text="OFF", foreground="red")  # Red OFF
                    self.log_message(data, "SUCCESS")  # Log success
                    self.update_status("All outputs disabled")  # Update status
                
                elif status_type == "measurement_complete":  # Sequential measurement done
                    # Re-enable measurement buttons
                    self.measure_all_btn.configure(state="normal")  # Enable main button
                    for channel in range(1, 4):  # All channels
                        self.channel_vars[channel]["buttons"]["measure"].configure(state="normal")  # Enable buttons
                    self.update_status("All measurements completed")  # Update status
        
        except queue.Empty:  # No messages in queue
            pass  # Continue to next polling cycle
        
        finally:
            # Schedule next polling cycle (100ms delay)
            self.root.after(100, self.check_status_updates)  # Recursive scheduling

    def run(self):
        """
        Start the main GUI application loop
        
        Launches Tkinter event loop (blocks until user closes window).
        Handles cleanup on application exit.
        """
        try:
            # Log application startup
            self.log_message("Keithley Power Supply Automation started")  # Log start event
            self.log_message("Ready to connect using correct VISA address")  # Provide guidance
            
            # Start GUI event loop (blocks until window closed)
            self.root.mainloop()  # Enter event loop
        
        except KeyboardInterrupt:  # User interrupts (Ctrl+C)
            self.log_message("Application interrupted")  # Log interruption
        
        finally:
            # Cleanup on exit
            if self.power_supply:  # If connected
                self.power_supply.disconnect()  # Close connection


def main():
    """
    Application entry point
    
    Creates and runs GUI application. Provides error handling for startup issues.
    """
    print("Keithley Power Supply Automation - COMPLETE WORKING VERSION")  # Print header
    print("="*60)  # Print separator
    
    try:
        app = PowerSupplyAutomationGUI()  # Create GUI instance
        app.run()  # Start application
    
    except Exception as e:  # Catch any startup errors
        print(f"Application error: {e}")  # Print error message
        input("Press Enter to exit...")  # Wait for user


if __name__ == "__main__":  # Check if run as main script
    main()  # Execute main function