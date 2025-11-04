#!/usr/bin/env python3
"""
Keithley Power Supply Control Library - CONSOLIDATED FINAL
- No lock
- Robust I/O recovery
- Buffer drain and explicit write/read
- Consistent terminations and timeouts
"""

import logging
import time
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pyvisa
from pyvisa.errors import VisaIOError


class KeithleyPowerSupplyError(Exception):
    pass


class OutputState(Enum):
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"


class ProtectionState(Enum):
    NORMAL = "NORMAL"
    OVP_TRIPPED = "OVP_TRIPPED"
    OCP_TRIPPED = "OCP_TRIPPED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ChannelConfiguration:
    channel: int
    voltage: float
    current_limit: float
    ovp_level: float
    output_enabled: bool


@dataclass
class ChannelMeasurement:
    channel: int
    voltage: float
    current: float
    power: float
    output_state: OutputState
    protection_state: ProtectionState


class KeithleyPowerSupply:
    def __init__(self, visa_address: str, timeout_ms: int = 10000):
        self._visa_address = visa_address
        self._timeout_ms = timeout_ms
        self._is_connected = False
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._instrument: Any = None  # pyvisa Resource object (use Any to avoid type errors)

        self._logger = logging.getLogger(f'{self.__class__.__name__}.{id(self)}')

        self.max_channels = 3
        self.max_voltage = 30.0
        self.max_current = 3.0
        self.model = "Unknown"

        self._voltage_settling_time = 0.5
        self._current_settling_time = 0.5
        self._output_enable_time = 0.7
        self._reset_time = 3.0

        self._valid_voltage_range = (0.0, 30.0)
        self._valid_current_range = (0.001, 3.0)
        self._valid_ovp_range = (1.0, 35.0)

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._instrument is not None

    @property
    def visa_address(self) -> str:
        return self._visa_address

    def connect(self) -> bool:
        try:
            self._logger.info("Attempting to connect to Keithley power supply...")
            self._resource_manager = pyvisa.ResourceManager()
            self._logger.info("VISA resource manager created successfully")

            self._instrument = self._resource_manager.open_resource(self._visa_address)
            self._logger.info(f"Opened connection to {self._visa_address}")

            self._instrument.timeout = self._timeout_ms
            self._instrument.read_termination = '\n'
            self._instrument.write_termination = '\n'

            identification = self._instrument.query("*IDN?")
            self._logger.info(f"Instrument identification: {identification.strip()}")

            self._configure_model_parameters(identification)

            self._instrument.write("*CLS")
            time.sleep(self._reset_time)
            self._instrument.query("*OPC?")

            self._is_connected = True
            self._logger.info(f"Successfully connected to Keithley {self.model}")
            return True

        except Exception as e:
            self._logger.error(f"Connection failed: {e}")
            try:
                if self._instrument:
                    self._instrument.close()
                if self._resource_manager:
                    self._resource_manager.close()
            except Exception:
                pass
            self._instrument = None
            self._resource_manager = None
            self._is_connected = False
            return False

    def _configure_model_parameters(self, identification: str):
        parts = identification.strip().split(',')
        manufacturer = parts[0] if len(parts) > 0 else ""
        model = parts[1] if len(parts) > 1 else ""

        if "KEITHLEY" not in manufacturer.upper() and "TEKTRONIX" not in manufacturer.upper():
            self._logger.warning(f"Unexpected manufacturer: {manufacturer}")

        if "2230" in model:
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = "2230-30-3"
        elif "2231" in model:
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = "2231A-30-3"
        elif "2280S" in model:
            self.max_channels = 1
            self.max_voltage = 72.0
            self.max_current = 120.0
            self.model = "2280S"
        else:
            self.max_channels = 3
            self.max_voltage = 30.0
            self.max_current = 3.0
            self.model = model.strip()
            self._logger.warning(f"Unknown model {model}, using defaults")

        self._valid_voltage_range = (0.0, self.max_voltage)
        self._valid_current_range = (0.001, self.max_current)
        self._valid_ovp_range = (1.0, self.max_voltage + 5.0)

        self._logger.info(f"Configured model {self.model}: {self.max_channels} channels, {self.max_voltage}V/{self.max_current}A max")

    def disconnect(self):
        try:
            if self._is_connected and self._instrument:
                try:
                    self.disable_all_outputs()
                    time.sleep(0.5)
                except Exception as e:
                    self._logger.warning(f"Could not disable outputs during disconnect: {e}")
                self._instrument.close()
                self._logger.info("Instrument connection closed")
            if self._resource_manager:
                self._resource_manager.close()
                self._logger.info("VISA resource manager closed")
        except Exception as e:
            self._logger.error(f"Error during disconnection: {e}")
        finally:
            self._instrument = None
            self._resource_manager = None
            self._is_connected = False
            self._logger.info("Disconnection completed")

    def get_instrument_info(self) -> Optional[Dict[str, Any]]:
        if not self.is_connected:
            self._logger.error("Cannot get info: not connected")
            return None
        try:
            idn = self._instrument.query("*IDN?").strip()
            parts = idn.split(',')
            return {
                'manufacturer': parts[0] if len(parts) > 0 else 'Unknown',
                'model': parts[1] if len(parts) > 1 else 'Unknown',
                'serial_number': parts[2] if len(parts) > 2 else 'Unknown',
                'firmware_version': parts[3] if len(parts) > 3 else 'Unknown',
                'max_channels': self.max_channels,
                'max_voltage': self.max_voltage,
                'max_current': self.max_current,
                'visa_address': self._visa_address,
                'identification': idn
            }
        except Exception as e:
            self._logger.error(f"Failed to get instrument info: {e}")
            return None

    def configure_channel(self, channel: int, voltage: float, current_limit: float, ovp_level: float, enable_output: bool = False) -> bool:
        if not self.is_connected:
            self._logger.error("Cannot configure channel: not connected")
            return False
        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False
        if not (self._valid_voltage_range[0] <= voltage <= self._valid_voltage_range[1]):
            self._logger.error(f"Voltage {voltage}V out of range {self._valid_voltage_range}")
            return False
        if not (self._valid_current_range[0] <= current_limit <= self._valid_current_range[1]):
            self._logger.error(f"Current limit {current_limit}A out of range {self._valid_current_range}")
            return False
        if ovp_level <= voltage:
            self._logger.warning(f"OVP {ovp_level}V must be > voltage {voltage}V; adjusting to {voltage+1.0}V")
            ovp_level = voltage + 1.0
        try:
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(0.2)
            self._instrument.write(f":SOURce:VOLTage {voltage}")
            time.sleep(self._voltage_settling_time)
            self._instrument.write(f":SOURce:CURRent {current_limit}")
            time.sleep(self._current_settling_time)
            self._instrument.write(f":SOURce:VOLTage:PROTection {ovp_level}")
            time.sleep(0.3)
            if enable_output:
                self._instrument.write(":OUTPut ON")
                time.sleep(self._output_enable_time)
            actual_voltage = float(self._instrument.query(":SOURce:VOLTage?"))
            actual_current = float(self._instrument.query(":SOURce:CURRent?"))
            self._logger.info(f"CH{channel} configured: {actual_voltage:.6f}V, {actual_current:.6f}A limit, Output: {'Enabled' if enable_output else 'Disabled'}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to configure channel {channel}: {e}")
            return False

    def enable_channel_output(self, channel: int) -> bool:
        if not self.is_connected:
            self._logger.error("Cannot enable output: not connected")
            return False
        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False
        try:
            self._logger.info(f"Enabling output on CH{channel}")
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(0.2)
            self._instrument.write(":OUTPut ON")
            time.sleep(self._output_enable_time)
            state = self._instrument.query(":OUTPut?").strip().upper()
            if state in ("1", "ON"):
                self._logger.info(f"CH{channel} output enabled")
                return True
            self._logger.error(f"CH{channel} output enable failed; state='{state}'")
            return False
        except Exception as e:
            self._logger.error(f"Enable output failed on CH{channel}: {e}")
            return False

    def disable_channel_output(self, channel: int) -> bool:
        if not self.is_connected:
            self._logger.error("Cannot disable output: not connected")
            return False
        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return False
        try:
            self._logger.info(f"Disabling output on CH{channel}")
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(0.2)
            self._instrument.write(":OUTPut OFF")
            time.sleep(0.5)
            state = self._instrument.query(":OUTPut?").strip().upper()
            if state in ("0", "OFF"):
                self._logger.info(f"CH{channel} output disabled")
                return True
            self._logger.error(f"CH{channel} output disable failed; state='{state}'")
            return False
        except Exception as e:
            self._logger.error(f"Disable output failed on CH{channel}: {e}")
            return False

    def disable_all_outputs(self) -> bool:
        if not self.is_connected:
            self._logger.error("Cannot disable outputs: not connected")
            return False
        ok = True
        for ch in range(1, self.max_channels + 1):
            if not self.disable_channel_output(ch):
                ok = False
            time.sleep(0.2)
        if ok:
            self._logger.info("All outputs disabled")
        else:
            self._logger.warning("Some outputs may still be ON")
        return ok

    def measure_channel_output(self, channel: int) -> Optional[Tuple[float, float]]:
        """
        ABSOLUTE FINAL: Improved parsing and buffer management
        Returns (voltage, current) tuple or None if measurement fails
        """
        if not self.is_connected:
            self._logger.error("Cannot measure: not connected")
            return None

        if not (1 <= channel <= self.max_channels):
            self._logger.error(f"Invalid channel {channel}")
            return None

        # Capture original timeout before operations so we can always restore it
        original_timeout = self._instrument.timeout
        try:
            self._logger.info(f"Measuring channel {channel}...")

            # Set longer timeout for potentially slow measurements
            self._instrument.timeout = 15000  # 15 seconds

            # Clear the device/read buffer to remove any stale data without blocking
            try:
                self._instrument.clear()
            except Exception as clear_err:
                # Not all backends/resources support clear(); log at debug and continue
                self._logger.debug(f"Buffer clear not supported or failed: {clear_err}")

            # Select channel
            self._instrument.write(f":INSTrument:SELect CH{channel}")
            time.sleep(0.5)

            # Measure voltage
            voltage_str = self._instrument.query(":MEASure:VOLTage?").strip()
            self._logger.info(f"Raw voltage response: '{voltage_str}'")
            time.sleep(0.5)

            # Measure current
            current_str = self._instrument.query(":MEASure:CURRent?").strip()
            self._logger.info(f"Raw current response: '{current_str}'")

            # Better number parsing
            def extract_first_float(s: str) -> float:
                matches = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', s)
                if matches:
                    return float(matches[0])
                self._logger.warning(f"Could not parse number from '{s}', returning 0.0")
                return 0.0

            voltage = extract_first_float(voltage_str)
            current = extract_first_float(current_str)

            # Check output state and sanitize current if OFF
            try:
                state_str = self._instrument.query(":OUTPut?").strip()
                self._logger.debug(f"Output state: '{state_str}'")
                if state_str in ['0', 'OFF', 'off']:
                    if abs(current) > 0.001:
                        self._logger.warning(f"Output OFF but current={current}A, forcing to 0")
                        current = 0.0
            except Exception as state_err:
                self._logger.debug(f"Could not check output state: {state_err}")

            # Validate readings
            if voltage < 0 or voltage > (self.max_voltage + 5):
                self._logger.warning(f"Unrealistic voltage: {voltage}V")
            if current < 0 or current > (self.max_current + 2):
                self._logger.warning(f"Unrealistic current: {current}A")

            self._logger.info(f"Channel {channel} final: {voltage:.4f}V, {current:.4f}A")
            return (voltage, current)

        except Exception as e:
            self._logger.error(f"Measurement failed on channel {channel}: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return None
        finally:
            # Always restore the original timeout
            try:
                self._instrument.timeout = original_timeout
            except Exception as restore_err:
                self._logger.debug(f"Failed to restore timeout: {restore_err}")
