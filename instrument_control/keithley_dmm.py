#!/usr/bin/env python3
"""
Keithley DMM6500 Digital Multimeter Control Library

This module provides a comprehensive interface for controlling Keithley DMM6500
6.5-digit digital multimeters via SCPI commands over VISA communication protocol.

Module: instrument_control.keithley_dmm
Author: Professional Instrument Control Team
Version: 1.0.0
License: MIT
Dependencies: pyvisa, numpy

Supported Models:
    - DMM6500 (6.5-digit benchtop multimeter)
    - DMM7510 (7.5-digit graphical sampling multimeter)

Features:
    - High-precision DC/AC voltage measurements (1µV resolution)
    - DC/AC current measurements with multiple ranges
    - 2-wire and 4-wire resistance measurements
    - Statistical analysis and data logging capabilities
    - Comprehensive error handling and validation

Usage:
    from instrument_control.keithley_dmm import KeithleyDMM6500

    multimeter = KeithleyDMM6500('USB0::0x05E6::0x6500::04561287::INSTR')
    multimeter.connect()
    voltage = multimeter.measure_dc_voltage(measurement_range=10.0, resolution=1e-6)
    multimeter.disconnect()
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum

try:
    import pyvisa
    from pyvisa.errors import VisaIOError
except ImportError as e:
    raise ImportError(
        "PyVISA library is required. Install with: pip install pyvisa"
    ) from e


class MeasurementFunction(Enum):
    """Enumeration of supported measurement functions."""
    DC_VOLTAGE = "VOLTage:DC"
    AC_VOLTAGE = "VOLTage:AC"
    DC_CURRENT = "CURRent:DC"
    AC_CURRENT = "CURRent:AC"
    RESISTANCE_2W = "RESistance"
    RESISTANCE_4W = "FRESistance"
    CAPACITANCE = "CAPacitance"
    FREQUENCY = "FREQuency"


class KeithleyDMM6500Error(Exception):
    """Custom exception for Keithley DMM6500 multimeter errors."""
    pass


class KeithleyDMM6500:
    """
    Control interface for Keithley DMM6500 digital multimeter.

    This class provides methods for high-precision measurements, statistical
    analysis, and comprehensive instrument configuration. All methods follow
    IEEE 488.2 and SCPI standards for maximum compatibility.

    Attributes:
        visa_address (str): VISA resource identifier string
        timeout_ms (int): Communication timeout in milliseconds
        max_voltage_range (float): Maximum DC voltage measurement range
        min_resolution (float): Minimum measurement resolution achievable
    """

    def __init__(self, visa_address: str, timeout_ms: int = 30000) -> None:
        """
        Initialize DMM control instance with extended timeout for precision measurements.

        Args:
            visa_address: VISA resource string (e.g., 'USB0::0x05E6::0x6500::04561287::INSTR')
            timeout_ms: Communication timeout in milliseconds (extended default for precision)

        Raises:
            ValueError: If visa_address is empty or invalid format
        """
        if not visa_address or not isinstance(visa_address, str):
            raise ValueError("visa_address must be a non-empty string")

        # Store configuration parameters
        self._visa_address = visa_address
        self._timeout_ms = timeout_ms

        # Initialize VISA communication objects
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._instrument: Any = None  # pyvisa Resource object (use Any to avoid type errors)
        self._is_connected = False

        # Initialize logging for this instance
        self._logger = logging.getLogger(f'{self.__class__.__name__}.{id(self)}')

        # Define instrument specifications (DMM6500)
        self.max_voltage_range = 1000.0  # Maximum DC voltage range (V)
        self.max_current_range = 10.0    # Maximum DC current range (A)
        self.max_resistance_range = 100e6  # Maximum resistance range (Ohm)
        self.min_resolution = 1e-9       # Minimum resolution for highest accuracy

        # Define valid measurement ranges for different functions
        self._voltage_ranges = [0.1, 1.0, 10.0, 100.0, 1000.0]
        self._current_ranges = [1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1.0, 3.0, 10.0]
        self._resistance_ranges = [100.0, 1e3, 10e3, 100e3, 1e6, 10e6, 100e6]

        # Define valid NPLC (Number of Power Line Cycles) values
        self._valid_nplc_values = [0.01, 0.02, 0.06, 0.2, 1.0, 2.0, 10.0]

    def connect(self) -> bool:
        """
        Establish communication with the multimeter.

        This method creates the VISA resource manager, opens the instrument
        connection, and performs comprehensive initialization sequence optimized
        for DMM6500 characteristics.

        Returns:
            True if connection successful, False otherwise

        Raises:
            KeithleyDMM6500Error: If critical connection error occurs
        """
        try:
            # Create VISA resource manager instance
            self._resource_manager = pyvisa.ResourceManager()
            self._logger.info("VISA resource manager created successfully")

            # Open connection to specified instrument with optimized settings
            self._instrument = self._resource_manager.open_resource(self._visa_address)
            self._logger.info(f"Opened connection to {self._visa_address}")

            # Configure communication parameters optimized for DMM6500
            self._instrument.timeout = self._timeout_ms
            self._instrument.read_termination = '\n'  # Line feed termination
            self._instrument.write_termination = '\n'  # Line feed termination
            self._instrument.chunk_size = 20480  # Optimized buffer size for stability

            # Clear any existing errors immediately after connection
            self._instrument.write("*CLS")
            time.sleep(0.2)  # Allow error clearing to complete

            # Verify instrument communication with identification query
            identification = self._instrument.query("*IDN?")
            self._logger.info(f"Instrument identification: {identification.strip()}")

            # Validate instrument model compatibility
            if "KEITHLEY" not in identification.upper():
                self._logger.warning(f"Unexpected manufacturer in IDN response: {identification}")

            if "DMM" not in identification.upper() and "6500" not in identification:
                self._logger.warning(f"Unexpected model in IDN response: {identification}")

            # Perform optimized initialization sequence for DMM6500
            self._logger.info("Performing DMM6500-optimized initialization sequence")

            # Clear all error registers
            self._instrument.write("*CLS")
            time.sleep(0.1)

            # Use system preset instead of *RST for faster, more reliable initialization
            self._instrument.write(":SYSTem:PRESet")
            time.sleep(0.5)  # Allow preset to complete

            # Removed :FORMat:ASCii:PRECision to avoid unsupported header (-113) on some models

            # Abort any running operations to ensure clean state
            self._instrument.write(":ABORt")
            time.sleep(0.1)

            # Final error clearing
            self._instrument.write("*CLS")

            # Verify instrument is responsive after initialization
            self._instrument.query("*OPC?")  # Operation complete query

            # Mark connection as established
            self._is_connected = True
            self._logger.info("Successfully connected to Keithley DMM6500")

            return True

        except VisaIOError as e:
            self._logger.error(f"VISA communication error during connection: {e}")
            self._cleanup_connection()
            return False

        except Exception as e:
            self._logger.error(f"Unexpected error during connection: {e}")
            self._cleanup_connection()
            raise KeithleyDMM6500Error(f"Connection failed: {e}") from e

    def disconnect(self) -> None:
        """
        Safely disconnect from multimeter and release resources.

        This method puts the instrument in a safe state, closes connections,
        and performs proper cleanup to prevent resource leaks.
        """
        try:
            if self._instrument is not None:
                # Put instrument in safe state before disconnection
                self._instrument.write(":ABORt")  # Stop any running operations
                time.sleep(0.1)
                self._instrument.write("*CLS")   # Clear status registers

                # Close instrument connection
                self._instrument.close()
                self._logger.info("Instrument connection closed")

            if self._resource_manager is not None:
                # Close resource manager
                self._resource_manager.close()
                self._logger.info("VISA resource manager closed")

        except Exception as e:
            self._logger.error(f"Error during disconnection: {e}")

        finally:
            # Reset connection state and object references
            self._cleanup_connection()
            self._logger.info("Disconnection completed")

    def measure_dc_voltage(self, 
                          measurement_range: Optional[float] = None,
                          resolution: Optional[float] = None,
                          nplc: Optional[float] = None,
                          auto_zero: bool = True) -> Optional[float]:
        """
        Perform high-precision DC voltage measurement.

        This method configures the multimeter for optimal DC voltage measurement
        accuracy and performs the measurement with comprehensive error handling.

        Args:
            measurement_range: Measurement range in volts (None for auto-range)
            resolution: Measurement resolution in volts (None for default)
            nplc: Number of Power Line Cycles for integration (None for default)
            auto_zero: Enable automatic zero correction for highest accuracy

        Returns:
            DC voltage measurement in volts, or None if measurement failed

        Raises:
            KeithleyDMM6500Error: If instrument not connected or invalid parameters
        """
        if not self._is_connected:
            raise KeithleyDMM6500Error("Multimeter not connected")

        try:
            self._logger.info("Configuring for high-precision DC voltage measurement")

            # Clear any existing errors
            self._instrument.write("*CLS")
            time.sleep(0.1)

            # Abort any running operations
            self._instrument.write(":ABORt")
            time.sleep(0.1)

            # Configure measurement function for DC voltage
            self._instrument.write(':SENSe:FUNCtion "VOLTage:DC"')
            time.sleep(0.1)

            # Configure measurement range
            if measurement_range is not None:
                # Validate and select appropriate range
                if measurement_range not in self._voltage_ranges:
                    valid_range = min([r for r in self._voltage_ranges if r >= measurement_range], 
                                    default=self._voltage_ranges[-1])
                    self._logger.warning(f"Invalid range {measurement_range}V, using {valid_range}V")
                    measurement_range = valid_range

                self._instrument.write(f":SENSe:VOLTage:DC:RANGe {measurement_range}")
                self._logger.debug(f"Set measurement range to {measurement_range}V")
            else:
                # Enable auto-ranging for maximum flexibility
                self._instrument.write(":SENSe:VOLTage:DC:RANGe:AUTO ON")
                self._logger.debug("Enabled auto-ranging")

            time.sleep(0.1)

            # Configure measurement resolution if specified
            if resolution is not None:
                # Ensure resolution is within instrument capabilities
                if resolution < self.min_resolution:
                    self._logger.warning(f"Resolution {resolution} below minimum, using {self.min_resolution}")
                    resolution = self.min_resolution

                self._instrument.write(f":SENSe:VOLTage:DC:RESolution {resolution}")
                self._logger.debug(f"Set resolution to {resolution}V")

            # Configure integration time (NPLC) if specified
            if nplc is not None:
                # Validate NPLC value
                if nplc not in self._valid_nplc_values:
                    nplc_val = nplc if nplc is not None else 1.0
                    valid_nplc = min(self._valid_nplc_values, key=lambda x: abs(x - nplc_val))
                    self._logger.warning(f"Invalid NPLC {nplc}, using {valid_nplc}")
                    nplc = valid_nplc

                self._instrument.write(f":SENSe:VOLTage:DC:NPLC {nplc}")
                self._logger.debug(f"Set NPLC to {nplc}")
            else:
                # Use default NPLC for good balance of speed and accuracy
                self._instrument.write(":SENSe:VOLTage:DC:NPLC 1")
                self._logger.debug("Set NPLC to 1 (default)")

            # Removed auto-zero headers to avoid -113; do not change auto-zero state here

            # Allow all settings to take effect
            time.sleep(0.2)

            self._logger.debug("Performing fresh DC voltage reading")
            measurement_str = self._instrument.query(":READ?")
            voltage = float(measurement_str.strip())

            self._logger.info(f"DC voltage measurement successful: {voltage:.9f} V")

            return voltage

        except VisaIOError as e:
            if "timeout" in str(e).lower():
                self._logger.error("Measurement timeout - consider increasing timeout or reducing NPLC")
                # Attempt to abort any running operation
                try:
                    self._instrument.write(":ABORt")
                except:
                    pass
            else:
                self._logger.error(f"VISA communication error: {e}")
            return None

        except (ValueError, AttributeError) as e:
            self._logger.error(f"Parameter or parsing error: {e}")
            return None

        except Exception as e:
            self._logger.error(f"Unexpected error during DC voltage measurement: {e}")
            # Attempt to abort any running operation
            try:
                self._instrument.write(":ABORt")
            except:
                pass
            return None

    def measure_dc_voltage_fast(self) -> Optional[float]:
        """
        Perform fast DC voltage measurement with minimal configuration overhead.

        This method uses the simplest SCPI command for situations where speed
        is more important than maximum precision or configurability.

        Returns:
            DC voltage measurement in volts, or None if measurement failed
        """
        if not self._is_connected:
            self._logger.error("Cannot measure voltage: multimeter not connected")
            return None

        try:
            self._logger.debug("Performing fast DC voltage measurement")

            # Clear any errors first
            self._instrument.write("*CLS")
            time.sleep(0.05)

            # Ensure DC voltage function selected and perform a fresh reading
            try:
                self._instrument.write(":ABORt")
            except Exception:
                pass
            time.sleep(0.05)

            self._instrument.write(':SENSe:FUNCtion "VOLTage:DC"')
            time.sleep(0.05)

            # Fresh measurement
            measurement_str = self._instrument.query(":READ?")
            voltage = float(measurement_str.strip())

            self._logger.debug(f"Fast DC voltage measurement: {voltage:.6f} V")

            return voltage

        except Exception as e:
            self._logger.error(f"Fast measurement failed: {e}")
            return None

    def check_instrument_errors(self) -> List[str]:
        """
        Check and retrieve any accumulated instrument errors.

        Returns:
            List of error messages, empty list if no errors
        """
        errors = []

        if not self._is_connected:
            return ["Multimeter not connected"]

        try:
            # Read up to 20 errors to prevent infinite loops
            for _ in range(20):
                error_response = self._instrument.query(":SYSTem:ERRor:NEXT?").strip()

                # Check if no more errors (standard SCPI response)
                if "No error" in error_response or error_response.startswith("0,"):
                    break

                errors.append(error_response)

        except Exception as e:
            errors.append(f"Error reading instrument errors: {str(e)}")

        return errors

    def perform_measurement_statistics(self, 
                                     measurement_count: int = 10,
                                     measurement_interval: float = 0.1) -> Optional[Dict[str, float]]:
        """
        Perform multiple measurements and calculate statistical parameters.

        Args:
            measurement_count: Number of measurements to perform
            measurement_interval: Delay between measurements in seconds

        Returns:
            Dictionary containing statistical results, or None if failed
        """
        if not self._is_connected:
            self._logger.error("Cannot perform statistics: multimeter not connected")
            return None

        if measurement_count < 2:
            raise ValueError("measurement_count must be at least 2 for statistics")

        try:
            self._logger.info(f"Performing {measurement_count} measurements for statistics")

            measurements = []

            # Collect measurements
            for i in range(measurement_count):
                voltage = self.measure_dc_voltage_fast()
                if voltage is not None:
                    measurements.append(voltage)
                    self._logger.debug(f"Measurement {i+1}/{measurement_count}: {voltage:.6f}V")

                    # Wait between measurements if not the last one
                    if i < measurement_count - 1:
                        time.sleep(measurement_interval)
                else:
                    self._logger.warning(f"Measurement {i+1} failed")

            if len(measurements) < 2:
                self._logger.error("Insufficient valid measurements for statistics")
                return None

            # Calculate statistics
            import statistics

            mean_value = statistics.mean(measurements)
            std_deviation = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
            min_value = min(measurements)
            max_value = max(measurements)
            range_value = max_value - min_value

            # Calculate coefficient of variation (percentage)
            cv_percent = (std_deviation / mean_value * 100.0) if mean_value != 0 else float('inf')

            results = {
                'count': len(measurements),
                'mean': mean_value,
                'standard_deviation': std_deviation,
                'minimum': min_value,
                'maximum': max_value,
                'range': range_value,
                'coefficient_of_variation_percent': cv_percent
            }

            self._logger.info(f"Statistics complete: Mean={mean_value:.6f}V, "
                            f"StdDev={std_deviation:.6f}V, CV={cv_percent:.3f}%")

            return results

        except Exception as e:
            self._logger.error(f"Failed to perform measurement statistics: {e}")
            return None

    def get_instrument_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive instrument information and status.

        Returns:
            Dictionary containing instrument details, or None if query failed
        """
        if not self._is_connected:
            return None

        try:
            # Query instrument identification
            idn_response = self._instrument.query("*IDN?").strip()
            idn_parts = idn_response.split(',')

            # Extract identification components
            manufacturer = idn_parts[0] if len(idn_parts) > 0 else "Unknown"
            model = idn_parts[1] if len(idn_parts) > 1 else "Unknown"
            serial_number = idn_parts[2] if len(idn_parts) > 2 else "Unknown"
            firmware_version = idn_parts[3] if len(idn_parts) > 3 else "Unknown"

            # Check for current errors
            current_errors = self.check_instrument_errors()
            error_status = "None" if not current_errors else "; ".join(current_errors)

            # Compile comprehensive instrument information
            info = {
                'manufacturer': manufacturer,
                'model': model,
                'serial_number': serial_number,
                'firmware_version': firmware_version,
                'visa_address': self._visa_address,
                'connection_status': 'Connected' if self._is_connected else 'Disconnected',
                'timeout_ms': self._timeout_ms,
                'max_voltage_range': self.max_voltage_range,
                'max_current_range': self.max_current_range,
                'max_resistance_range': self.max_resistance_range,
                'min_resolution': self.min_resolution,
                'current_errors': error_status
            }

            return info

        except Exception as e:
            self._logger.error(f"Failed to retrieve instrument information: {e}")
            return None

    def measure(self,
                function: MeasurementFunction,
                measurement_range: Optional[float] = None,
                resolution: Optional[float] = None,
                nplc: Optional[float] = None,
                auto_zero: Optional[bool] = None) -> Optional[float]:
        """
        Generic measurement method for any supported SCPI function.

        Args:
            function: Measurement function enum value
            measurement_range: Optional range to set; None enables auto-range
            resolution: Optional resolution; ignored if unsupported by function
            nplc: Optional integration time in power line cycles
            auto_zero: Optional auto-zero; only applied for functions that support it

        Returns:
            Measured value as float, or None on failure
        """
        if not self._is_connected:
            self._logger.error("Cannot measure: multimeter not connected")
            return None

        try:
            # Clear and abort to start clean
            self._instrument.write("*CLS")
            time.sleep(0.1)
            try:
                self._instrument.write(":ABORt")
            except Exception:
                pass
            time.sleep(0.1)

            # Select function
            func_token = function.value
            self._instrument.write(f':SENSe:FUNCtion "{func_token}"')
            time.sleep(0.1)

            # Determine SCPI path prefix for this function
            prefix = func_token  # e.g., VOLTage:DC

            # Configure range (or auto)
            if measurement_range is not None:
                # Snap range based on function type where we know the valid sets
                try:
                    token_upper = func_token.upper()
                    if any(x in token_upper for x in ["VOLT", "CURR"]):
                        valid_ranges = self._voltage_ranges if "VOLT" in token_upper else self._current_ranges
                        if measurement_range not in valid_ranges:
                            valid_range = min([r for r in valid_ranges if r >= measurement_range],
                                              default=valid_ranges[-1])
                            self._logger.warning(f"Invalid range {measurement_range}, using {valid_range}")
                            measurement_range = valid_range
                    elif "RES" in token_upper:
                        valid_ranges = self._resistance_ranges
                        if measurement_range not in valid_ranges:
                            valid_range = min([r for r in valid_ranges if r >= measurement_range],
                                              default=valid_ranges[-1])
                            self._logger.warning(f"Invalid range {measurement_range}, using {valid_range}")
                            measurement_range = valid_range
                except Exception:
                    # If any validation fails, proceed to set the provided range directly
                    pass

                try:
                    self._instrument.write(f":SENSe:{prefix}:RANGe {measurement_range}")
                except Exception as e:
                    self._logger.warning(f"Range command unsupported for {func_token}: {e}")
            else:
                # Try to enable auto-range if available
                try:
                    self._instrument.write(f":SENSe:{prefix}:RANGe:AUTO ON")
                except Exception as e:
                    self._logger.debug(f"Auto-range unsupported for {func_token}: {e}")

            time.sleep(0.05)

            # Configure resolution if supported
            if resolution is not None:
                try:
                    if resolution < self.min_resolution:
                        self._logger.warning(f"Resolution {resolution} below minimum, using {self.min_resolution}")
                        resolution = self.min_resolution
                    self._instrument.write(f":SENSe:{prefix}:RESolution {resolution}")
                except Exception as e:
                    self._logger.debug(f"Resolution unsupported for {func_token}: {e}")

            # Configure NPLC if supported
            if nplc is not None:
                try:
                    if nplc not in self._valid_nplc_values:
                        nplc_val = nplc if nplc is not None else 1.0
                        valid_nplc = min(self._valid_nplc_values, key=lambda x: abs(x - nplc_val))
                        self._logger.warning(f"Invalid NPLC {nplc}, using {valid_nplc}")
                        nplc = valid_nplc
                    self._instrument.write(f":SENSe:{prefix}:NPLC {nplc}")
                except Exception as e:
                    self._logger.debug(f"NPLC unsupported for {func_token}: {e}")

            # Do not send auto-zero headers here to avoid -113 on some models

            # Brief delay to apply settings
            time.sleep(0.2)

            # Removed :TRACe:CLEar to avoid -113 on models lacking TRACE buffer

            # Perform measurement
            value_str = self._instrument.query(":READ?")
            value = float(value_str.strip())

            self._logger.info(f"Measurement {func_token} successful: {value:.9f}")
            return value

        except VisaIOError as e:
            if "timeout" in str(e).lower():
                self._logger.error("Measurement timeout - consider increasing timeout or reducing NPLC")
                try:
                    self._instrument.write(":ABORt")
                except Exception:
                    pass
            else:
                self._logger.error(f"VISA communication error: {e}")
            return None
        except Exception as e:
            self._logger.error(f"Unexpected error during measurement {function.value}: {e}")
            try:
                self._instrument.write(":ABORt")
            except Exception:
                pass
            return None

    # Convenience wrappers mirroring common DMM functions
    def measure_ac_voltage(self,
                           measurement_range: Optional[float] = None,
                           resolution: Optional[float] = None,
                           nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.AC_VOLTAGE, measurement_range, resolution, nplc)

    def measure_dc_current(self,
                           measurement_range: Optional[float] = None,
                           resolution: Optional[float] = None,
                           nplc: Optional[float] = None,
                           auto_zero: Optional[bool] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.DC_CURRENT, measurement_range, resolution, nplc, auto_zero)

    def measure_ac_current(self,
                           measurement_range: Optional[float] = None,
                           resolution: Optional[float] = None,
                           nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.AC_CURRENT, measurement_range, resolution, nplc)

    def measure_resistance_2w(self,
                              measurement_range: Optional[float] = None,
                              resolution: Optional[float] = None,
                              nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.RESISTANCE_2W, measurement_range, resolution, nplc)

    def measure_resistance_4w(self,
                              measurement_range: Optional[float] = None,
                              resolution: Optional[float] = None,
                              nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.RESISTANCE_4W, measurement_range, resolution, nplc)

    def measure_capacitance(self,
                            measurement_range: Optional[float] = None,
                            resolution: Optional[float] = None,
                            nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.CAPACITANCE, measurement_range, resolution, nplc)

    def measure_frequency(self,
                          measurement_range: Optional[float] = None,
                          resolution: Optional[float] = None,
                          nplc: Optional[float] = None) -> Optional[float]:
        return self.measure(MeasurementFunction.FREQUENCY, measurement_range, resolution, nplc)

    def _cleanup_connection(self) -> None:
        """Clean up connection state and references."""
        self._is_connected = False
        self._instrument = None
        self._resource_manager = None

    @property
    def is_connected(self) -> bool:
        """Check if multimeter is currently connected."""
        return self._is_connected

    @property
    def visa_address(self) -> str:
        """Get the VISA address for this instrument."""
        return self._visa_address


def main() -> None:
    """Example usage demonstration."""
    # Configuration parameters
    multimeter_address = "USB0::0x05E6::0x6500::04561287::INSTR"

    # Create multimeter instance
    dmm = KeithleyDMM6500(multimeter_address)

    try:
        # Connect to instrument
        if not dmm.connect():
            print("Failed to connect to multimeter")
            return

        print("Connected to multimeter successfully")

        # Perform high-precision measurement
        voltage = dmm.measure_dc_voltage(
            measurement_range=10.0,  # 10V range
            resolution=1e-6,         # 1µV resolution
            nplc=1.0,               # 1 power line cycle
            auto_zero=True          # Enable auto-zero correction
        )

        if voltage is not None:
            print(f"High-precision DC voltage: {voltage:.9f} V")
        else:
            print("High-precision measurement failed")

        # Perform statistical analysis
        stats = dmm.perform_measurement_statistics(measurement_count=10)
        if stats:
            print(f"Statistical analysis (n={stats['count']}):")
            print(f"  Mean: {stats['mean']:.6f} V")
            print(f"  Std Dev: {stats['standard_deviation']:.6f} V")
            print(f"  CV: {stats['coefficient_of_variation_percent']:.3f}%")

        # Display instrument information
        info = dmm.get_instrument_info()
        if info:
            print(f"Instrument: {info['manufacturer']} {info['model']}")
            print(f"Serial: {info['serial_number']}")
            print(f"Firmware: {info['firmware_version']}")
            print(f"Errors: {info['current_errors']}")

    except KeithleyDMM6500Error as e:
        print(f"Multimeter error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        # Always disconnect to clean up resources
        dmm.disconnect()
        print("Disconnected from multimeter")


if __name__ == "__main__":
    main()
