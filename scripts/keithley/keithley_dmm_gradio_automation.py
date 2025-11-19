#!/usr/bin/env python3
"""
Keithley DMM Automation GUI

A comprehensive Gradio-based web interface for controlling Keithley DMM6500/DMM7510
digital multimeters. This application provides an intuitive interface for precision
measurements, data logging, statistical analysis, and instrument configuration.

Features:
- Real-time measurements with live updates
- Statistical analysis and trending
- Data export capabilities
- Instrument status monitoring
- Configuration save/recall
- Error logging and diagnostics

Author: Professional Instrument Control Team
Version: 1.0.0
License: MIT
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import io
import base64
from pathlib import Path

# Import the DMM control classes from instrument_control package
import sys
from pathlib import Path

# Add parent directory to path to import from instrument_control
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from instrument_control.keithley_dmm import (
        KeithleyDMM6500,
        MeasurementFunction,
        TriggerSource,
        DisplayState,
        KeithleyDMM6500Error
    )
except ImportError as e:
    print(f"ERROR: Failed to import DMM control library: {e}")
    print("Make sure the instrument_control package is in the Python path")
    sys.exit(1)


class DMM_GUI_Controller:
    """Main controller class for the DMM Gradio interface."""
    
    def __init__(self):
        """Initialize the GUI controller."""
        self.dmm: Optional[KeithleyDMM6500] = None
        self.is_connected = False
        self.measurement_thread: Optional[threading.Thread] = None
        self.continuous_measurement = False
        self.measurement_data = []
        self.max_data_points = 1000
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DMM_GUI')
        
        # Default settings
        self.default_settings = {
            'visa_address': 'USB0::0x05E6::0x6500::04561287::INSTR',
            'timeout_ms': 30000,
            'measurement_function': 'DC_VOLTAGE',
            'measurement_range': 10.0,
            'resolution': 1e-6,
            'nplc': 1.0,
            'auto_zero': True,
            'measurement_interval': 1.0
        }
    
    def connect_instrument(self, visa_address: str, timeout_ms: int) -> Tuple[str, bool]:
        """
        Connect to the DMM instrument.
        
        Returns:
            Tuple of (status_message, connection_success)
        """
        try:
            if self.is_connected:
                return "Already connected to instrument", True
            
            self.dmm = KeithleyDMM6500(visa_address, timeout_ms)
            
            if self.dmm.connect():
                self.is_connected = True
                info = self.dmm.get_instrument_info()
                if info:
                    msg = f"Connected: {info['manufacturer']} {info['model']} (S/N: {info['serial_number']})"
                else:
                    msg = "Connected to DMM successfully"
                self.logger.info(msg)
                return msg, True
            else:
                return "Failed to connect to instrument", False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return f"Connection error: {str(e)}", False
    
    def disconnect_instrument(self) -> str:
        """Disconnect from the DMM instrument."""
        try:
            if self.continuous_measurement:
                self.stop_continuous_measurement()
            
            if self.dmm and self.is_connected:
                self.dmm.disconnect()
                self.is_connected = False
                return "Disconnected from instrument"
            else:
                return "No instrument connected"
                
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            return f"Disconnection error: {str(e)}"
    
    def single_measurement(self, function: str, range_val: float, resolution: float, 
                         nplc: float, auto_zero: bool) -> Tuple[str, str]:
        """
        Perform a single measurement.
        
        Returns:
            Tuple of (measurement_result, status_message)
        """
        if not self.is_connected or not self.dmm:
            return "N/A", "Not connected to instrument"
        
        try:
            # Map function string to enum
            func_map = {
                'DC_VOLTAGE': MeasurementFunction.DC_VOLTAGE,
                'AC_VOLTAGE': MeasurementFunction.AC_VOLTAGE,
                'DC_CURRENT': MeasurementFunction.DC_CURRENT,
                'AC_CURRENT': MeasurementFunction.AC_CURRENT,
                'RESISTANCE_2W': MeasurementFunction.RESISTANCE_2W,
                'RESISTANCE_4W': MeasurementFunction.RESISTANCE_4W,
                'CAPACITANCE': MeasurementFunction.CAPACITANCE,
                'FREQUENCY': MeasurementFunction.FREQUENCY,
                'TEMPERATURE': MeasurementFunction.TEMPERATURE
            }
            
            measurement_func = func_map.get(function)
            if not measurement_func:
                return "N/A", f"Unknown measurement function: {function}"
            
            # Perform measurement based on function type
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
                result = self.dmm.measure_temperature()
            
            if result is not None:
                # Add to measurement data
                timestamp = datetime.now()
                self.measurement_data.append({
                    'timestamp': timestamp,
                    'function': function,
                    'value': result,
                    'range': range_val,
                    'resolution': resolution
                })
                
                # Limit data points
                if len(self.measurement_data) > self.max_data_points:
                    self.measurement_data = self.measurement_data[-self.max_data_points:]
                
                # Format result based on function
                unit_map = {
                    'DC_VOLTAGE': 'V', 'AC_VOLTAGE': 'V',
                    'DC_CURRENT': 'A', 'AC_CURRENT': 'A',
                    'RESISTANCE_2W': 'Ω', 'RESISTANCE_4W': 'Ω',
                    'CAPACITANCE': 'F', 'FREQUENCY': 'Hz',
                    'TEMPERATURE': '°C'
                }
                unit = unit_map.get(function, '')

                # Format with proper SI prefixes (no scientific notation)
                formatted_result = self._format_with_si_prefix(result, unit)

                return formatted_result, "Measurement successful"
            else:
                return "N/A", "Measurement failed"
                
        except Exception as e:
            self.logger.error(f"Measurement error: {e}")
            return "N/A", f"Measurement error: {str(e)}"
    
    def start_continuous_measurement(self, function: str, range_val: float, resolution: float, 
                                   nplc: float, auto_zero: bool, interval: float) -> str:
        """Start continuous measurements in a separate thread."""
        if not self.is_connected:
            return "Not connected to instrument"
        
        if self.continuous_measurement:
            return "Continuous measurement already running"
        
        self.continuous_measurement = True
        self.measurement_thread = threading.Thread(
            target=self._continuous_measurement_worker,
            args=(function, range_val, resolution, nplc, auto_zero, interval),
            daemon=True
        )
        self.measurement_thread.start()
        return "Continuous measurement started"
    
    def stop_continuous_measurement(self) -> str:
        """Stop continuous measurements."""
        self.continuous_measurement = False
        if self.measurement_thread and self.measurement_thread.is_alive():
            self.measurement_thread.join(timeout=2)
        return "Continuous measurement stopped"
    
    def _continuous_measurement_worker(self, function: str, range_val: float, resolution: float, 
                                     nplc: float, auto_zero: bool, interval: float):
        """Worker thread for continuous measurements."""
        while self.continuous_measurement and self.is_connected:
            try:
                self.single_measurement(function, range_val, resolution, nplc, auto_zero)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Continuous measurement error: {e}")
                break
    
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
    
    def _get_unit(self, function: str) -> str:
        """Get the unit for a measurement function."""
        unit_map = {
            'DC_VOLTAGE': 'V', 'AC_VOLTAGE': 'V',
            'DC_CURRENT': 'A', 'AC_CURRENT': 'A',
            'RESISTANCE_2W': 'Ω', 'RESISTANCE_4W': 'Ω',
            'CAPACITANCE': 'F', 'FREQUENCY': 'Hz',
            'TEMPERATURE': '°C'
        }
        return unit_map.get(function, '')

    def _format_with_si_prefix(self, value: float, base_unit: str) -> str:
        """
        Format a value with appropriate SI prefix (no scientific notation).

        Args:
            value: The numerical value to format
            base_unit: The base unit (V, A, Ω, F, Hz, °C)

        Returns:
            Formatted string with SI prefix (e.g., "1.234 mV", "5.67 kΩ")
        """
        # Temperature doesn't use SI prefixes
        if base_unit == '°C':
            return f"{value:.3f} {base_unit}"

        # SI prefixes from largest to smallest
        prefixes = [
            (1e12, 'T'),   # Tera
            (1e9, 'G'),    # Giga
            (1e6, 'M'),    # Mega
            (1e3, 'k'),    # kilo
            (1, ''),       # base unit
            (1e-3, 'm'),   # milli
            (1e-6, 'µ'),   # micro (using proper µ symbol)
            (1e-9, 'n'),   # nano
            (1e-12, 'p'),  # pico
            (1e-15, 'f'),  # femto
        ]

        abs_value = abs(value)

        # Handle zero
        if abs_value == 0:
            return f"0.000 {base_unit}"

        # Find the appropriate prefix
        for scale, prefix in prefixes:
            if abs_value >= scale:
                scaled_value = value / scale
                # Use appropriate decimal places based on magnitude
                if abs(scaled_value) >= 100:
                    formatted = f"{scaled_value:.2f}"
                elif abs(scaled_value) >= 10:
                    formatted = f"{scaled_value:.3f}"
                else:
                    formatted = f"{scaled_value:.4f}"

                return f"{formatted} {prefix}{base_unit}"

        # If value is extremely small, use femto
        scaled_value = value / 1e-15
        return f"{scaled_value:.4f} f{base_unit}"

    def export_data(self, format_type: str = "CSV") -> Optional[str]:
        """Export measurement data to file."""
        if not self.measurement_data:
            return None
        
        try:
            df = pd.DataFrame(self.measurement_data)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "CSV":
                filename = f"dmm_data_{timestamp_str}.csv"
                filepath = f"/mnt/user-data/outputs/{filename}"
                df.to_csv(filepath, index=False)
                return filepath
            elif format_type == "JSON":
                filename = f"dmm_data_{timestamp_str}.json"
                filepath = f"/mnt/user-data/outputs/{filename}"
                df.to_json(filepath, orient='records', date_format='iso')
                return filepath
            elif format_type == "Excel":
                filename = f"dmm_data_{timestamp_str}.xlsx"
                filepath = f"/mnt/user-data/outputs/{filename}"
                df.to_excel(filepath, index=False)
                return filepath
        except Exception as e:
            self.logger.error(f"Data export error: {e}")
            return None
    
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


def create_dmm_interface():
    """Create the main Gradio interface for DMM control."""
    controller = DMM_GUI_Controller()
    
    with gr.Blocks(title="Keithley DMM Control Center", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Keithley DMM Control Center")
        gr.Markdown("**Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.")
        gr.Markdown("Professional digital multimeter control and data acquisition interface")
        
        with gr.Tabs():
            # Connection Tab
            with gr.Tab("Connection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Instrument Connection")
                        visa_address = gr.Textbox(
                            label="VISA Address",
                            value=controller.default_settings['visa_address'],
                            placeholder="USB0::0x05E6::0x6500::04561287::INSTR"
                        )
                        timeout_ms = gr.Number(
                            label="Timeout (ms)",
                            value=controller.default_settings['timeout_ms'],
                            minimum=1000,
                            maximum=60000
                        )
                        
                        with gr.Row():
                            connect_btn = gr.Button("Connect", variant="primary")
                            disconnect_btn = gr.Button("Disconnect", variant="secondary")
                        
                        connection_status = gr.Textbox(
                            label="Connection Status",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Instrument Status")
                        status_connection = gr.Textbox(label="Connection", interactive=False)
                        status_instrument = gr.Textbox(label="Instrument", interactive=False)
                        status_errors = gr.Textbox(label="Errors", interactive=False)
                        status_time = gr.Textbox(label="System Time", interactive=False)
                        refresh_status_btn = gr.Button("Refresh Status")
            
            # Measurement Tab
            with gr.Tab("Measurements"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Measurement Configuration")
                        measurement_function = gr.Dropdown(
                            label="Measurement Function",
                            choices=[
                                "DC_VOLTAGE", "AC_VOLTAGE",
                                "DC_CURRENT", "AC_CURRENT",
                                "RESISTANCE_2W", "RESISTANCE_4W",
                                "CAPACITANCE", "FREQUENCY", "TEMPERATURE"
                            ],
                            value="DC_VOLTAGE"
                        )
                        
                        measurement_range = gr.Number(
                            label="Range",
                            value=10.0,
                            minimum=0.001,
                            maximum=1000.0
                        )
                        
                        resolution = gr.Number(
                            label="Resolution",
                            value=1e-6,
                            minimum=1e-9,
                            maximum=1e-3,
                            step=1e-9
                        )
                        
                        nplc = gr.Dropdown(
                            label="NPLC (Integration Time)",
                            choices=[0.01, 0.02, 0.06, 0.2, 1.0, 2.0, 10.0],
                            value=1.0
                        )
                        
                        auto_zero = gr.Checkbox(
                            label="Auto Zero",
                            value=True
                        )
                        
                        with gr.Row():
                            single_measure_btn = gr.Button("Single Measurement", variant="primary")
                            clear_data_btn = gr.Button("Clear Data", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Measurement Results")
                        current_measurement = gr.Textbox(
                            label="Current Reading",
                            interactive=False,
                            lines=2
                        )
                        
                        measurement_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        gr.Markdown("### Continuous Measurements")
                        measurement_interval = gr.Number(
                            label="Interval (seconds)",
                            value=1.0,
                            minimum=0.1,
                            maximum=60.0
                        )
                        
                        with gr.Row():
                            start_continuous_btn = gr.Button("Start Continuous", variant="primary")
                            stop_continuous_btn = gr.Button("Stop Continuous", variant="secondary")
                        
                        continuous_status = gr.Textbox(
                            label="Continuous Status",
                            interactive=False
                        )
            
            # Statistics Tab
            with gr.Tab("Statistics & Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Statistical Analysis")
                        stats_points = gr.Number(
                            label="Number of Points",
                            value=100,
                            minimum=1,
                            maximum=1000
                        )
                        
                        calculate_stats_btn = gr.Button("Calculate Statistics", variant="primary")
                        
                        stats_count = gr.Textbox(label="Count", interactive=False)
                        stats_mean = gr.Textbox(label="Mean", interactive=False)
                        stats_std = gr.Textbox(label="Standard Deviation", interactive=False)
                        stats_min = gr.Textbox(label="Minimum", interactive=False)
                        stats_max = gr.Textbox(label="Maximum", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Trend Plot")
                        plot_points = gr.Number(
                            label="Points to Plot",
                            value=100,
                            minimum=10,
                            maximum=1000
                        )
                        update_plot_btn = gr.Button("Update Plot", variant="primary")
                        trend_plot = gr.Plot()
            
            # Data Export Tab
            with gr.Tab("Data Export"):
                with gr.Column():
                    gr.Markdown("### Export Measurement Data")
                    
                    export_format = gr.Dropdown(
                        label="Export Format",
                        choices=["CSV", "JSON", "Excel"],
                        value="CSV"
                    )
                    
                    export_btn = gr.Button("Export Data", variant="primary")
                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False
                    )
                    
                    gr.Markdown("### Data Preview")
                    data_preview = gr.Dataframe(
                        headers=["Timestamp", "Function", "Value", "Range", "Resolution"],
                        interactive=False
                    )
                    
                    refresh_preview_btn = gr.Button("Refresh Preview")
        
        # Event handlers
        connect_btn.click(
            controller.connect_instrument,
            inputs=[visa_address, timeout_ms],
            outputs=[connection_status, gr.State()]
        )
        
        disconnect_btn.click(
            controller.disconnect_instrument,
            outputs=[connection_status]
        )
        
        refresh_status_btn.click(
            controller.get_instrument_status,
            outputs=[status_connection, status_instrument, status_errors, status_time]
        )
        
        single_measure_btn.click(
            controller.single_measurement,
            inputs=[measurement_function, measurement_range, resolution, nplc, auto_zero],
            outputs=[current_measurement, measurement_status]
        )
        
        start_continuous_btn.click(
            controller.start_continuous_measurement,
            inputs=[measurement_function, measurement_range, resolution, nplc, auto_zero, measurement_interval],
            outputs=[continuous_status]
        )
        
        stop_continuous_btn.click(
            controller.stop_continuous_measurement,
            outputs=[continuous_status]
        )
        
        calculate_stats_btn.click(
            controller.get_statistics,
            inputs=[stats_points],
            outputs=[stats_count, stats_mean, stats_std, stats_min, stats_max]
        )
        
        update_plot_btn.click(
            controller.create_trend_plot,
            inputs=[plot_points],
            outputs=[trend_plot]
        )
        
        export_btn.click(
            controller.export_data,
            inputs=[export_format],
            outputs=[export_status]
        )
        
        clear_data_btn.click(
            controller.clear_data,
            outputs=[measurement_status]
        )
        
        def update_data_preview():
            if controller.measurement_data:
                recent_data = controller.measurement_data[-20:]  # Show last 20 points
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
        
        refresh_preview_btn.click(
            update_data_preview,
            outputs=[data_preview]
        )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_dmm_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=True,
        show_error=True,
        #inbrowser=True
    )
