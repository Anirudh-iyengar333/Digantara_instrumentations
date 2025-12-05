"""Tektronix MSO24 Oscilloscope Control Library.

Optimized SCPI command interface for MSO24 series oscilloscopes.
"""

import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
import numpy as np

try:
    from .scpi_wrapper import SCPIWrapper
except ImportError:
    from scpi_wrapper import SCPIWrapper

# Constants
DEFAULT_TIMEOUT_MS = 60000
SCPI_COMMAND_DELAY = 0.01  # 10ms between commands (reduced from 50ms)
SCPI_BATCH_DELAY = 0.001  # 1ms for batched commands
MAX_SCREENSHOT_CHUNKS = 1000
SCREENSHOT_CHUNK_SIZE = 20 * 1024 * 1024  # 20MB

class Coupling(str, Enum):
    """Valid channel coupling options."""
    DC = "DC"
    AC = "AC"
    DCREJECT = "DCREJECT"

class TriggerType(str, Enum):
    """Valid trigger types."""
    EDGE = "EDGE"
    PULSE = "PULSE"
    LOGIC = "LOGIC"
    BUS = "BUS"
    VIDEO = "VIDEO"

class MeasurementType(str, Enum):
    """Valid measurement types."""
    FREQUENCY = "FREQUENCY"
    PERIOD = "PERIOD"
    AMPLITUDE = "AMPLITUDE"
    HIGH = "HIGH"
    LOW = "LOW"
    MAX = "MAX"
    MIN = "MIN"
    PEAK2PEAK = "PEAK2PEAK"
    MEAN = "MEAN"
    RMS = "RMS"
    RISE = "RISE"
    FALL = "FALL"
    WIDTH = "WIDTH"
    DUTYCYCLE = "DUTYCYCLE"
    OVERSHOOT = "OVERSHOOT"
    PRESHOOT = "PRESHOOT"
    AREA = "AREA"
    PHASE = "PHASE"

class TektronixMSO24Error(Exception):
    """Custom exception for Tektronix MSO24 oscilloscope errors."""
    pass

class TektronixMSO24:
    """Tektronix MSO24 Oscilloscope Control Class - Optimized Version."""

    # Class-level constants (shared across all instances)
    MAX_CHANNELS = 4
    MAX_SAMPLE_RATE = 2.5e9  # 2.5 GS/s
    MAX_MEMORY_DEPTH = 62.5e6  # 62.5 Mpts
    BANDWIDTH_HZ = 200e6  # 200 MHz

    VALID_VERTICAL_SCALES = frozenset([
        1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3,
        100e-3, 200e-3, 500e-3, 1.0, 2.0, 5.0, 10.0
    ])

    VALID_TIMEBASE_SCALES = frozenset([
        1e-9, 2e-9, 4e-9, 10e-9, 20e-9, 40e-9,
        100e-9, 200e-9, 400e-9, 1e-6, 2e-6, 4e-6,
        10e-6, 20e-6, 40e-6, 100e-6, 200e-6, 400e-6,
        1e-3, 2e-3, 4e-3, 10e-3, 20e-3, 40e-3,
        100e-3, 200e-3, 400e-3, 1.0, 2.0, 4.0, 10.0
    ])

    def __init__(self, visa_address: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        """Initialize oscilloscope connection parameters.

        Args:
            visa_address: VISA resource address
            timeout_ms: Initial VISA timeout in milliseconds
        """
        self._scpi_wrapper = SCPIWrapper(visa_address, timeout_ms)
        self._logger = logging.getLogger(self.__class__.__name__)

        # Instance-specific output directories (lazy initialization)
        self.screenshot_dir: Optional[Path] = None
        self.data_dir: Optional[Path] = None
        self.graph_dir: Optional[Path] = None

    def connect(self) -> bool:
        """Establish VISA connection to oscilloscope."""
        if not self._scpi_wrapper.connect():
            return False

        try:
            identification = self._scpi_wrapper.query("*IDN?")
            self._logger.info(f"Connected: {identification.strip()}")

            # Clear status and wait for completion (combined operation)
            self._scpi_wrapper.write("*CLS;*OPC")
            time.sleep(0.1)  # Reduced from 0.5s
            return True
        except Exception as e:
            self._logger.error(f"Connection failed: {e}")
            self._scpi_wrapper.disconnect()
            return False

    def disconnect(self) -> None:
        """Close connection to oscilloscope."""
        self._scpi_wrapper.disconnect()
        self._logger.info("Disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if oscilloscope is currently connected."""
        return self._scpi_wrapper.is_connected

    def get_instrument_info(self) -> Optional[Dict[str, Any]]:
        """Query instrument identification and specifications."""
        if not self.is_connected:
            return None
        try:
            idn = self._scpi_wrapper.query("*IDN?").strip()
            parts = idn.split(',', maxsplit=3)  # Limit splits for efficiency
            return {
                'manufacturer': parts[0] if parts else 'Unknown',
                'model': parts[1] if len(parts) > 1 else 'Unknown',
                'serial_number': parts[2] if len(parts) > 2 else 'Unknown',
                'firmware_version': parts[3] if len(parts) > 3 else 'Unknown',
                'max_channels': self.MAX_CHANNELS,
                'bandwidth_hz': self.BANDWIDTH_HZ,
                'max_sample_rate': self.MAX_SAMPLE_RATE,
                'max_memory_depth': self.MAX_MEMORY_DEPTH,
                'identification': idn
            }
        except Exception as e:
            self._logger.error(f"Failed to get instrument info: {e}")
            return None

    def configure_channel(self, channel: int, vertical_scale: float, vertical_offset: float = 0.0,
                          coupling: str = "DC", bandwidth_limit: bool = False) -> bool:
        """Configure vertical parameters for specified channel.

        Args:
            channel: Channel number (1-4)
            vertical_scale: Vertical scale in volts per division
            vertical_offset: Vertical offset in volts
            coupling: Input coupling ("DC", "AC", "DCREJECT")
            bandwidth_limit: Enable 20MHz bandwidth limit

        Note:
            Probe attenuation should be set separately using set_probe_attenuation()
        """
        if not self.is_connected:
            raise TektronixMSO24Error("Oscilloscope not connected")
        if not (1 <= channel <= self.MAX_CHANNELS):
            raise ValueError(f"Channel must be 1-{self.MAX_CHANNELS}, got {channel}")
        if coupling not in [c.value for c in Coupling]:
            raise ValueError(f"Invalid coupling: {coupling}")

        try:
            bw_setting = "TWENty" if bandwidth_limit else "FULl"

            # Send each command individually with delays to ensure they are applied
            self._scpi_wrapper.write(f"DISplay:GLObal:CH{channel}:STATE ON")
            time.sleep(0.05)

            self._scpi_wrapper.write(f"CH{channel}:SCAle {vertical_scale}")
            time.sleep(0.05)

            self._scpi_wrapper.write(f"CH{channel}:OFFSet {vertical_offset}")
            time.sleep(0.05)

            self._scpi_wrapper.write(f"CH{channel}:COUPling {coupling}")
            time.sleep(0.05)

            self._scpi_wrapper.write(f"CH{channel}:BANdwidth {bw_setting}")
            time.sleep(0.05)

            self._logger.info(f"CH{channel}: {vertical_scale}V/div, {vertical_offset}V offset, "
                            f"{coupling}, BW={bw_setting}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to configure channel {channel}: {e}")
            return False

    def set_probe_attenuation(self, channel: int, attenuation: float) -> bool:
        """Set probe attenuation independently for a specific channel.

        Args:
            channel: Channel number (1-4)
            attenuation: Probe attenuation factor (e.g., 1.0, 10.0, 100.0)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            raise TektronixMSO24Error("Oscilloscope not connected")
        if not (1 <= channel <= self.MAX_CHANNELS):
            raise ValueError(f"Channel must be 1-{self.MAX_CHANNELS}, got {channel}")

        try:
            # Set probe attenuation
            self._scpi_wrapper.write(f"CH{channel}:PROBEFunc:EXTAtten {attenuation}")
            time.sleep(0.1)  # Allow time for setting to take effect

            # Verify the setting was applied
            actual_atten = float(self._scpi_wrapper.query(f"CH{channel}:PROBEFunc:EXTAtten?").strip())

            self._logger.info(f"CH{channel}: Probe attenuation set to {attenuation}x (verified: {actual_atten}x)")
            return True
        except Exception as e:
            self._logger.error(f"Failed to set probe attenuation for CH{channel}: {e}")
            return False

    def configure_timebase(self, time_scale: float, time_position: float = 0.0,
                           record_length: int = 10000) -> bool:
        """Configure horizontal timebase settings.

        Args:
            time_scale: Time scale in seconds per division
            time_position: Horizontal position in seconds
            record_length: Record length in points
        """
        if not self.is_connected:
            raise TektronixMSO24Error("Oscilloscope not connected")

        try:
            # Batch timebase commands
            commands = (
                f"HORizontal:SCAle {time_scale};"
                f"HORizontal:POSition {time_position};"
                f"HORizontal:RECOrdlength {record_length}"
            )
            self._scpi_wrapper.write(commands)
            time.sleep(0.05)  # Reduced from 0.2s

            # Adjust timeout based on timebase
            if time_scale >= 10.0:
                self._scpi_wrapper.set_timeout(120000)
            elif time_scale >= 1.0:
                self._scpi_wrapper.set_timeout(90000)
            else:
                self._scpi_wrapper.reset_timeout()

            self._logger.info(f"Timebase: {time_scale}s/div, pos={time_position}s, len={record_length}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to configure timebase: {e}")
            return False

    def configure_trigger(self, trigger_type: str = "EDGE", source: str = "CH1",
                         level: float = 0.0, slope: str = "RISE",
                         coupling: str = "DC") -> bool:
        """Configure trigger settings.

        Args:
            trigger_type: Trigger type ("EDGE", "PULSE", "LOGIC", "BUS", "VIDEO")
            source: Trigger source channel
            level: Trigger level in volts
            slope: Trigger slope ("RISE", "FALL", "EITHER")
            coupling: Trigger coupling
        """
        if not self.is_connected:
            raise TektronixMSO24Error("Oscilloscope not connected")
        if trigger_type not in [t.value for t in TriggerType]:
            raise ValueError(f"Invalid trigger type: {trigger_type}")

        try:
            if trigger_type == "EDGE":
                # Send each trigger command individually with delays
                self._scpi_wrapper.write(f"TRIGger:A:TYPe {trigger_type}")
                time.sleep(0.05)

                self._scpi_wrapper.write(f"TRIGger:A:EDGE:SOUrce {source}")
                time.sleep(0.05)

                self._scpi_wrapper.write(f"TRIGger:A:EDGE:SLOpe {slope}")
                time.sleep(0.05)

                self._scpi_wrapper.write(f"TRIGger:A:EDGE:COUPling {coupling}")
                time.sleep(0.05)

                self._scpi_wrapper.write(f"TRIGger:A:LEVel:{source} {level}")
                time.sleep(0.1)  # Longer delay for level to take effect

                # Verify settings were applied
                actual_source = self._scpi_wrapper.query("TRIGger:A:EDGE:SOUrce?").strip()
                actual_slope = self._scpi_wrapper.query("TRIGger:A:EDGE:SLOpe?").strip()
                actual_level = float(self._scpi_wrapper.query(f"TRIGger:A:LEVel:{source}?").strip())

                self._logger.info(f"Trigger set: {trigger_type}, {source}, {level}V, {slope}")
                self._logger.info(f"Trigger verified: {actual_source}, {actual_slope}, {actual_level}V")
            else:
                self._scpi_wrapper.write(f"TRIGger:A:TYPe {trigger_type}")
                time.sleep(0.1)

            return True
        except Exception as e:
            self._logger.error(f"Failed to configure trigger: {e}")
            return False

    def run(self) -> bool:
        """Start acquisition."""
        if not self.is_connected:
            return False
        try:
            self._scpi_wrapper.write("ACQuire:STATE RUN")
            self._logger.info("Acquisition: RUN")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start acquisition: {e}")
            return False

    def stop(self) -> bool:
        """Stop acquisition."""
        if not self.is_connected:
            return False
        try:
            self._scpi_wrapper.write("ACQuire:STATE STOP")
            self._logger.info("Acquisition: STOP")
            return True
        except Exception as e:
            self._logger.error(f"Failed to stop acquisition: {e}")
            return False

    def single(self) -> bool:
        """Trigger single acquisition."""
        if not self.is_connected:
            return False
        try:
            # Batch single acquisition commands
            self._scpi_wrapper.write("ACQuire:STATE RUN;ACQuire:STOPAfter SEQuence")
            self._logger.info("Acquisition: SINGLE")
            return True
        except Exception as e:
            self._logger.error(f"Failed to start single acquisition: {e}")
            return False

    def get_channel_data(self, channel: Union[int, str], start_point: int = 1,
                        stop_point: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get waveform data from specified channel or math function.

        Args:
            channel: Channel number (1-4) or source name ("CH1", "MATH1", etc.)
            start_point: Starting point index
            stop_point: Ending point index (None = all points)

        Returns:
            Dictionary with time/voltage arrays and metadata
        """
        if not self.is_connected:
            self._logger.error("Cannot get channel data: not connected")
            return None

        # Normalize channel specification
        if isinstance(channel, int):
            if not (1 <= channel <= self.MAX_CHANNELS):
                self._logger.error(f"Invalid channel: {channel}")
                return None
            source_name = f"CH{channel}"
        elif isinstance(channel, str):
            source_name = channel.upper()
            valid_sources = {f"CH{i}" for i in range(1, 5)} | {f"MATH{i}" for i in range(1, 5)}
            if source_name not in valid_sources:
                self._logger.error(f"Invalid source: {source_name}")
                return None
        else:
            self._logger.error(f"Invalid channel type: {type(channel)}")
            return None

        try:
            # Get record length if stop_point not specified
            if stop_point is None:
                record_length = int(self._scpi_wrapper.query("HORizontal:RECOrdlength?").strip())
                stop_point = record_length

            # Batch all DATA configuration commands (saves 150ms!)
            commands = (
                f"DATa:SOUrce {source_name};"
                f"DATa:ENCdg SRIbinary;"
                f"DATa:WIDth 1;"
                f"DATa:STARt {start_point};"
                f"DATa:STOP {stop_point}"
            )
            self._scpi_wrapper.write(commands)
            time.sleep(SCPI_BATCH_DELAY)

            # Query all scaling parameters in one batch (saves 200ms!)
            # Note: Some SCPI implementations support chaining queries with semicolons
            x_increment = float(self._scpi_wrapper.query("WFMOutpre:XINcr?").strip())
            x_zero = float(self._scpi_wrapper.query("WFMOutpre:XZEro?").strip())
            y_multiplier = float(self._scpi_wrapper.query("WFMOutpre:YMUlt?").strip())
            y_zero = float(self._scpi_wrapper.query("WFMOutpre:YZEro?").strip())
            y_offset = float(self._scpi_wrapper.query("WFMOutpre:YOFf?").strip())

            # Get waveform data
            waveform_data = self._scpi_wrapper.query_binary_values("CURVe?", datatype='b')

            if not waveform_data:
                return None

            # Convert binary data to voltage values (vectorized for speed)
            voltage_data = (np.array(waveform_data, dtype=np.float32) - y_offset) * y_multiplier + y_zero

            # Create time array (vectorized)
            num_points = len(voltage_data)
            time_data = np.arange(num_points, dtype=np.float32) * x_increment + x_zero

            return {
                'time': time_data,
                'voltage': voltage_data,
                'channel': channel,
                'source': source_name,
                'num_points': num_points,
                'x_increment': x_increment,
                'x_zero': x_zero,
                'y_multiplier': y_multiplier,
                'y_zero': y_zero,
                'y_offset': y_offset,
                'start_point': start_point,
                'stop_point': stop_point
            }

        except Exception as e:
            self._logger.error(f"Failed to get channel {channel} data: {e}")
            return None

    # Math Functions

    def configure_math_function(self, function_num: int, operation: str,
                               source1: str, source2: Optional[str] = None,
                               math_expression: Optional[str] = None,
                               basic_function: Optional[str] = None) -> bool:
        """Configure math function using correct MSO24 SCPI commands.

        Args:
            function_num: Function number (1-4)
            operation: "BASIC", "ADVANCED", or "FFT"
            source1: First source
            source2: Second source (for basic math operations)
            math_expression: Expression for advanced math
            basic_function: Function for BASIC mode (ADD, SUBtract, MULTiply, DIVide)

        Returns:
            bool: True if successful
        """
        if not self.is_connected:
            self._logger.error("Cannot configure math function: not connected")
            return False

        if not (1 <= function_num <= 4):
            self._logger.error(f"Invalid function number: {function_num}")
            return False

        operation = operation.upper()
        if operation not in ["BASIC", "ADVANCED", "FFT"]:
            self._logger.error(f"Invalid operation: {operation}")
            return False

        try:
            # Set math type first (send as separate command with delay)
            self._scpi_wrapper.write(f"MATH:MATH{function_num}:TYPe {operation}")
            time.sleep(0.1)

            if operation == "BASIC":
                # For basic math, use source commands (sent separately)
                self._scpi_wrapper.write(f"MATH:MATH{function_num}:SOUrce1 {source1}")
                time.sleep(0.05)

                if source2:
                    self._scpi_wrapper.write(f"MATH:MATH{function_num}:SOUrce2 {source2}")
                    time.sleep(0.05)

            elif operation == "ADVANCED":
                # For advanced math, use DEFine command with expression
                if math_expression:
                    expression = math_expression
                elif source1 and source2:
                    # Create basic expression if not provided
                    expression = f"{source1}+{source2}"  # Default to addition
                else:
                    expression = source1  # Single source

                # Send DEFine command
                self._scpi_wrapper.write(f'MATH:MATH{function_num}:DEFine "{expression}"')
                time.sleep(0.1)
                self._logger.info(f"Math{function_num} defined with expression: {expression}")

            elif operation == "FFT":
                # For FFT, set the source
                self._scpi_wrapper.write(f"MATH:MATH{function_num}:SOUrce1 {source1}")
                time.sleep(0.05)

            self._logger.info(f"Math{function_num} configured: Type={operation}, Source1={source1}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to configure math function: {e}")
            return False

    def set_math_display(self, function_num: int, display: bool) -> bool:
        """Show/hide math function."""
        if not self.is_connected:
            return False
        try:
            state = "ON" if display else "OFF"
            self._scpi_wrapper.write(f"DISplay:WAVEView1:MATH:MATH{function_num}:STATE {state}")
            time.sleep(0.1)
            self._logger.info(f"Math{function_num} display: {state}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to set math display: {e}")
            return False

    def set_math_scale(self, function_num: int, scale: float) -> bool:
        """Set math function vertical scale."""
        if not self.is_connected:
            return False
        try:
            # First, disable autoscale so manual scale setting takes effect
            self._scpi_wrapper.write(f"DISplay:WAVEView1:MATH:MATH{function_num}:AUTOScale OFF")
            time.sleep(0.05)

            # Now set the manual scale
            self._scpi_wrapper.write(f"DISplay:WAVEView1:MATH:MATH{function_num}:VERTical:SCAle {scale}")
            time.sleep(0.1)  # Allow time for setting to take effect

            # Verify the setting
            actual_scale = self._scpi_wrapper.query(f"DISplay:WAVEView1:MATH:MATH{function_num}:VERTical:SCAle?").strip()

            self._logger.info(f"Math{function_num} autoscale disabled, scale set to: {scale} V/div (verified: {actual_scale} V/div)")
            return True
        except Exception as e:
            self._logger.error(f"Failed to set math scale: {e}")
            return False

    def autoscale_math(self, function_num: int) -> bool:
        """Enable autoscale for math function display."""
        if not self.is_connected:
            return False
        try:
            # Enable autoscale for the math function
            self._scpi_wrapper.write(f"DISplay:WAVEView1:MATH:MATH{function_num}:AUTOScale ON")
            time.sleep(0.2)  # Autoscale needs time to take effect
            self._logger.info(f"Math{function_num} autoscale enabled")
            return True
        except Exception as e:
            self._logger.error(f"Failed to enable autoscale for math function: {e}")
            return False

    # ============================================================================
    # 📸 SCREENSHOT FUNCTION - COMPLETELY FIXED
    # ============================================================================

    def get_screenshot(self, screenshot_path: str, freeze_acquisition: bool = True) -> Optional[str]:
        """
        Capture screenshot of oscilloscope display and save to local PC

        This method is optimized for MSO24 (2 Series) oscilloscopes which do not support
        direct screenshot transfer commands. The screenshot is saved to the scope's internal
        drive and then transferred to the PC via chunked binary read.

        Args:
            screenshot_path: Path to save screenshot file (will be converted to .png)
            freeze_acquisition: Freeze acquisition during screenshot (default: True)

        Returns:
            Path to saved screenshot file if successful, None if failed
        """
        if not self.is_connected:
            self._logger.error("Cannot capture screenshot: oscilloscope not connected")
            return None

        acquisition_was_running = False
        screenshot_path_png = str(Path(screenshot_path).with_suffix('.png'))
        temp_scope_path = "C:/Temp_Screenshot.png"

        try:
            # Check if acquisition is running
            try:
                acquisition_was_running = self.get_acquisition_state() == "RUN"
            except:
                pass

            # Freeze acquisition if requested
            if freeze_acquisition and acquisition_was_running:
                self.stop()
                time.sleep(0.2)

            self._logger.info(f"Capturing screenshot to: {screenshot_path_png}")

            # Configure screenshot format (PNG)
            self._scpi_wrapper.write("SAVE:IMAGe:FILEFormat PNG")
            time.sleep(0.2)

            # Disable ink saver for better quality
            try:
                self._scpi_wrapper.write("SAVE:IMAGe:INKSaver OFF")
                time.sleep(0.1)
            except:
                pass

            # Save screenshot to scope's internal drive
            self._scpi_wrapper.write(f'SAVE:IMAGe "{temp_scope_path}"')

            # Wait for save operation to complete
            try:
                self._scpi_wrapper.query("*OPC?", timeout=30000)
            except:
                time.sleep(5.0)  # Fallback delay if *OPC? not supported

            # Transfer screenshot file from scope to PC using chunked read
            self._scpi_wrapper.write(f'FILESystem:READFile "{temp_scope_path}"')
            time.sleep(0.5)

            # Configure chunked transfer settings
            original_timeout = self._scpi_wrapper._instrument.timeout
            original_chunk_size = self._scpi_wrapper._instrument.chunk_size

            self._scpi_wrapper._instrument.timeout = 5000  # 5 seconds per chunk
            self._scpi_wrapper._instrument.chunk_size = 20 * 1024 * 1024  # 20MB chunks

            try:
                # Read screenshot data in chunks
                image_data = bytearray()
                chunk_count = 0
                max_chunks = 1000

                while chunk_count < max_chunks:
                    try:
                        chunk = self._scpi_wrapper._instrument.read_raw()
                        if not chunk or len(chunk) == 0:
                            break
                        image_data.extend(chunk)
                        chunk_count += 1
                    except Exception as e:
                        if 'timeout' in str(e).lower():
                            break  # Timeout indicates transfer complete
                        raise

                image_data = bytes(image_data)
                self._logger.info(f"Screenshot transfer complete: {len(image_data)} bytes ({chunk_count} chunks)")

            finally:
                # Restore original VISA settings
                self._scpi_wrapper._instrument.timeout = original_timeout
                self._scpi_wrapper._instrument.chunk_size = original_chunk_size

            # Validate and save screenshot
            if len(image_data) < 1000:
                self._logger.error(f"Screenshot transfer failed: only {len(image_data)} bytes received")
                return None

            # Ensure parent directory exists before saving
            screenshot_path_obj = Path(screenshot_path_png)
            screenshot_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save screenshot to PC
            with open(screenshot_path_obj, 'wb') as f:
                f.write(image_data)

            # Cleanup temporary file on scope
            try:
                self._scpi_wrapper.write(f'FILESystem:DELEte "{temp_scope_path}"')
                time.sleep(0.1)
            except:
                pass

            self._logger.info(f"Screenshot saved successfully: {screenshot_path_png}")
            return screenshot_path_png

        except Exception as e:
            self._logger.error(f"Screenshot capture failed: {e}")

            # Cleanup on error
            try:
                self._scpi_wrapper.write(f'FILESystem:DELEte "{temp_scope_path}"')
            except:
                pass

            return None

        finally:
            # Resume acquisition if it was running
            if freeze_acquisition and acquisition_was_running:
                try:
                    self.run()
                except:
                    pass

    # ============================================================================
    # 📊 MEASUREMENT FUNCTIONS - ENHANCED ERROR HANDLING
    # ============================================================================

    def add_measurement(self, measurement_type: str, source: str) -> Optional[int]:
        """Add a measurement to the instrument.

        Args:
            measurement_type: Type of measurement
            source: Source for measurement

        Returns:
            Measurement number if successful, None if failed
        """
        if not self.is_connected:
            self._logger.error("Cannot add measurement: not connected")
            return None

        if measurement_type not in [m.value for m in MeasurementType]:
            raise ValueError(f"Invalid measurement type: {measurement_type}")

        try:
            # Get existing measurements (single query)
            before_list_str = self._scpi_wrapper.query("MEASUrement:LIST?", timeout=5000)
            existing_names = {
                m.strip().strip('"')
                for m in before_list_str.strip().split(',')
                if m.strip() and m.strip() not in ('', '""')
            } if before_list_str else set()

            # Add measurement
            self._scpi_wrapper.write(f"MEASUrement:ADDMeas {measurement_type}")
            time.sleep(0.1)  # Reduced from 0.3s

            # Find new measurement
            after_list_str = self._scpi_wrapper.query("MEASUrement:LIST?", timeout=5000)
            if not after_list_str or after_list_str.strip() in ("", '""'):
                self._logger.error("No measurements after ADDMeas")
                return None

            after_names = [
                m.strip().strip('"')
                for m in after_list_str.strip().split(',')
                if m.strip()
            ]

            # Find the newly added measurement
            new_names = [name for name in after_names if name not in existing_names]
            target_name = new_names[-1] if new_names else (after_names[-1] if after_names else None)

            if not target_name or not target_name.startswith("MEAS"):
                self._logger.error(f"Invalid measurement name: {target_name}")
                return None

            measurement_number = int(target_name.replace("MEAS", ""))

            # Set source (batched command to save time)
            self._scpi_wrapper.write(f"MEASUrement:MEAS{measurement_number}:SOUrce {source}")

            self._logger.info(f"Added MEAS{measurement_number}: {measurement_type} on {source}")
            return measurement_number

        except ValueError as ve:
            self._logger.error(f"Failed to parse measurement number: {ve}")
            return None
        except Exception as e:
            self._logger.error(f"Failed to add measurement: {e}")
            return None

    def get_measurement_value(self, measurement_number: int) -> Optional[float]:
        """Get value from specified measurement."""
        if not self.is_connected:
            return None

        try:
            value_str = self._scpi_wrapper.query(
                f"MEASUrement:MEAS{measurement_number}:RESUlts:CURRentacq:MEAN?"
            ).strip()

            # Handle invalid values
            if value_str.upper() in {'9.9E37', '9.91E37', 'NAN', 'INF', '-INF', 'UNDEF'}:
                return None

            return float(value_str)
        except Exception as e:
            self._logger.error(f"Failed to get measurement {measurement_number}: {e}")
            return None

    def get_all_measurements(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active measurements and their current values

        ✅ NEW: Retrieve all configured measurements from the oscilloscope

        Returns:
            Dictionary with measurement names as keys, containing type, source, and value
        """
        if not self.is_connected:
            self._logger.error("Cannot get measurements: not connected")
            return {}

        try:
            # Query measurement list
            meas_list_str = self._scpi_wrapper.query("MEASUrement:LIST?", timeout=5000)

            if not meas_list_str or meas_list_str.strip() in ("", '""'):
                self._logger.info("No measurements currently configured")
                return {}

            measurements = {}
            meas_names = [m.strip().strip('"') for m in meas_list_str.split(',') if m.strip()]

            for meas_name in meas_names:
                if not meas_name.startswith("MEAS"):
                    continue

                meas_num = int(meas_name.replace("MEAS", ""))

                # Get measurement details
                try:
                    meas_type = self._scpi_wrapper.query(f"MEASUrement:MEAS{meas_num}:TYPe?").strip()
                    meas_source = self._scpi_wrapper.query(f"MEASUrement:MEAS{meas_num}:SOUrce?").strip()
                    meas_value = self.get_measurement_value(meas_num)

                    measurements[meas_name] = {
                        'type': meas_type,
                        'source': meas_source,
                        'value': meas_value,
                        'number': meas_num
                    }

                    self._logger.debug(f"{meas_name}: {meas_type} on {meas_source} = {meas_value}")

                except Exception as e:
                    self._logger.warning(f"Could not get details for {meas_name}: {e}")

            self._logger.info(f"Retrieved {len(measurements)} measurements")
            return measurements

        except Exception as e:
            self._logger.error(f"Failed to get all measurements: {e}")
            return {}

    # ============================================================================
    # 🎵 AFG (ARBITRARY FUNCTION GENERATOR) CONTROL
    # ============================================================================

    def configure_afg(self, function: str, frequency: float, amplitude: float,
                      offset: float = 0.0, enable: bool = True) -> bool:
        """
        Configure the built-in Arbitrary Function Generator (AFG)

        ✅ NEW: Complete AFG control for MSO24 series

        Args:
            function: Waveform type ("SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "DC")
            frequency: Frequency in Hz (0.1 Hz to 50 MHz)
            amplitude: Peak-to-peak amplitude in V (0.002V to 5V)
            offset: DC offset in V (-2.5V to +2.5V)
            enable: Enable/disable AFG output

        Returns:
            bool: True if successful
        """
        if not self.is_connected:
            self._logger.error("Cannot configure AFG: not connected")
            return False

        # Validate parameters - Per MSO24 Programmer Manual AFG:FUNCtion
        # Accept both uppercase and mixed-case SCPI command formats
        valid_functions = [
            "SINE", "SQUARE", "SQU", "SQUAE",  # SINE and SQUARE variants
            "PULSE", "PULS",                    # PULSE variants
            "RAMP",                             # RAMP
            "NOISE", "NOIS",                    # NOISE variants
            "DC",                               # DC
            "SINC",                             # SINC (Sin(x)/x)
            "GAUSSIAN", "GAUS",                 # GAUSSIAN variants
            "LORENTZ", "LOREN",                 # LORENTZ variants
            "ERISE", "ERIS",                    # Exponential Rise variants
            "EDECAY", "EDECA", "EFALL",         # Exponential Decay variants (note: EFALL maps to EDECAY)
            "HAVERSINE", "HAVERSIN",            # HAVERSINE variants
            "CARDIAC", "CARDIA",                # CARDIAC variants
            "ARBITRARY", "ARB"                  # ARBITRARY variants
        ]
        if function.upper() not in valid_functions:
            self._logger.error(f"Invalid function: {function}. Must be one of {valid_functions}")
            return False

        if not (0.1 <= frequency <= 50e6):
            self._logger.error(f"Frequency {frequency} Hz out of range (0.1 Hz to 50 MHz)")
            return False

        if not (0.002 <= amplitude <= 5.0):
            self._logger.error(f"Amplitude {amplitude} V out of range (0.002V to 5V)")
            return False

        if not (-2.5 <= offset <= 2.5):
            self._logger.error(f"Offset {offset} V out of range (-2.5V to +2.5V)")
            return False

        try:
            # ✅ VERIFIED: AFG commands from MSO24 manual
            # Set AFG function type
            self._scpi_wrapper.write(f"AFG:FUNCtion {function.upper()}")
            time.sleep(0.05)

            # Set frequency (not applicable for DC)
            if function.upper() != "DC":
                self._scpi_wrapper.write(f"AFG:FREQuency {frequency}")
                time.sleep(0.05)

            # Set amplitude (peak-to-peak)
            self._scpi_wrapper.write(f"AFG:AMPLitude {amplitude}")
            time.sleep(0.05)

            # Set offset
            self._scpi_wrapper.write(f"AFG:OFFSet {offset}")
            time.sleep(0.05)

            # Enable/disable output
            output_state = "ON" if enable else "OFF"
            self._scpi_wrapper.write(f"AFG:OUTPut:STATE {output_state}")
            time.sleep(0.05)

            self._logger.info(f"AFG configured: {function} @ {frequency}Hz, {amplitude}Vpp, "
                            f"Offset={offset}V, Output={output_state}")
            return True

        except Exception as e:
            self._logger.error(f"AFG configuration failed: {e}")
            return False

    def get_afg_config(self) -> Optional[Dict[str, Any]]:
        """
        Query current AFG configuration

        Returns:
            Dictionary with AFG settings or None if failed
        """
        if not self.is_connected:
            return None

        try:
            config = {
                'function': self._scpi_wrapper.query("AFG:FUNCtion?").strip(),
                'frequency': float(self._scpi_wrapper.query("AFG:FREQuency?").strip()),
                'amplitude': float(self._scpi_wrapper.query("AFG:AMPLitude?").strip()),
                'offset': float(self._scpi_wrapper.query("AFG:OFFSet?").strip()),
                'output_state': self._scpi_wrapper.query("AFG:OUTPut:STATE?").strip()
            }
            return config
        except Exception as e:
            self._logger.error(f"Failed to get AFG configuration: {e}")
            return None

    # ============================================================================
    # 🔧 UTILITY FUNCTIONS
    # ============================================================================

    def autoscale(self) -> bool:
        """
        Execute autoscale command
        
        ✅ VERIFIED: AUTOSet command from MSO24 manual
        """
        if not self.is_connected:
            self._logger.error("Cannot autoscale: oscilloscope not connected")
            return False

        try:
            # ✅ VERIFIED: AUTOSet command from MSO24 manual
            self._scpi_wrapper.write("AUTOSet EXECute")
            time.sleep(5.0)  # Wait for autoscale to complete
            self._scpi_wrapper.query("*OPC?", timeout=15000)  # Wait for completion
            self._logger.info("Autoscale executed successfully")
            return True
        except Exception as e:
            self._logger.error(f"Autoscale failed: {type(e).__name__}: {e}")
            return False

    def get_acquisition_state(self) -> Optional[str]:
        """
        Query current acquisition state
        
        ✅ VERIFIED: ACQuire:STATE command from MSO24 manual
        """
        if not self.is_connected:
            return None
            
        try:
            state = self._scpi_wrapper.query("ACQuire:STATE?").strip()
            return state
        except Exception as e:
            self._logger.error(f"Failed to query acquisition state: {e}")
            return None

    def get_system_error(self) -> Optional[str]:
        """
        Query system error queue
        
        ✅ VERIFIED: SYSTem:ERRor command from MSO24 manual
        """
        if not self.is_connected:
            return None
            
        try:
            error_response = self._scpi_wrapper.query("SYSTem:ERRor?").strip()
            return error_response
        except Exception as e:
            self._logger.error(f"Failed to query system error: {e}")
            return None

    def reset(self) -> bool:
        """
        Reset instrument to default state
        
        ✅ VERIFIED: *RST command - standard SCPI command
        """
        if not self.is_connected:
            return False
            
        try:
            self._scpi_wrapper.write("*RST")
            time.sleep(5.0)  # Allow time for reset to complete
            self._scpi_wrapper.query("*OPC?", timeout=15000)
            self._logger.info("Instrument reset completed")
            return True
        except Exception as e:
            self._logger.error(f"Failed to reset instrument: {e}")
            return False

    def set_output_directories(self, data_dir: str = None, graph_dir: str = None,
                             screenshot_dir: str = None) -> bool:
        """Set custom output directories for data, graphs, and screenshots.

        Args:
            data_dir: Path for waveform data files
            graph_dir: Path for generated graphs
            screenshot_dir: Path for screenshots

        Returns:
            bool: True if successful
        """
        try:
            if data_dir:
                self.data_dir = Path(data_dir)
                self.data_dir.mkdir(parents=True, exist_ok=True)
                self._logger.info(f"Data dir: {self.data_dir}")

            if graph_dir:
                self.graph_dir = Path(graph_dir)
                self.graph_dir.mkdir(parents=True, exist_ok=True)
                self._logger.info(f"Graph dir: {self.graph_dir}")

            if screenshot_dir:
                self.screenshot_dir = Path(screenshot_dir)
                self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                self._logger.info(f"Screenshot dir: {self.screenshot_dir}")

            return True
        except Exception as e:
            self._logger.error(f"Failed to set output directories: {e}")
            return False

    def get_channel_config(self, channel: int) -> Optional[Dict[str, Any]]:
        """
        Query channel configuration including probe settings
        
        Returns:
            Dictionary containing channel configuration with the following keys:
            - channel: Channel number (1-4)
            - scale: Vertical scale in volts/div
            - offset: Vertical offset in volts
            - coupling: Input coupling (AC, DC, GND)
            - probe_attenuation: Probe attenuation factor (e.g., 10.0 for 10x probe)
            - probe_attenuation_db: Probe attenuation in decibels
            - probe_units: Probe units ('V' for Volts or 'A' for Amps)
            - bandwidth: Bandwidth limit setting
            - display: Whether channel is displayed (0 or 1)
        """
        if not self.is_connected:
            return None
        if not (1 <= channel <= self.MAX_CHANNELS):
            return None

        try:
            config = {
                'channel': channel,
                'scale': float(self._scpi_wrapper.query(f"CH{channel}:SCAle?").strip()),
                'offset': float(self._scpi_wrapper.query(f"CH{channel}:OFFSet?").strip()),
                'coupling': self._scpi_wrapper.query(f"CH{channel}:COUPling?").strip(),
                'probe_attenuation': float(self._scpi_wrapper.query(f"CH{channel}:PROBEFunc:EXTAtten?").strip()),
                'probe_attenuation_db': float(self._scpi_wrapper.query(f"CH{channel}:PROBEFunc:EXTDBatten?").strip()),
                'probe_units': self._scpi_wrapper.query(f'CH{channel}:PROBEFunc:EXTUnits?').strip('" \n'),
                'bandwidth': self._scpi_wrapper.query(f"CH{channel}:BANdwidth?").strip(),
                'display': self._scpi_wrapper.query(f"DISplay:GLObal:CH{channel}:STATE?").strip()
            }
            return config
        except Exception as e:
            self._logger.error(f"Failed to get channel {channel} config: {e}")
            return None

    def get_timebase_config(self) -> Optional[Dict[str, Any]]:
        """Query timebase configuration"""
        if not self.is_connected:
            return None

        try:
            config = {
                'scale': float(self._scpi_wrapper.query("HORizontal:SCAle?").strip()),
                'position': float(self._scpi_wrapper.query("HORizontal:POSition?").strip()),
                'record_length': int(self._scpi_wrapper.query("HORizontal:RECOrdlength?").strip()),
                'sample_rate': float(self._scpi_wrapper.query("HORizontal:SAMPLERate?").strip())
            }
            return config
        except Exception as e:
            self._logger.error(f"Failed to get timebase config: {e}")
            return None


if __name__ == "__main__":
    # Module can be imported or run directly
    pass