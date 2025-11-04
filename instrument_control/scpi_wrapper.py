import pyvisa
from typing import Optional, Any

class SCPIWrapper:
    def __init__(self, visa_address: str, timeout_ms: int = 10000):
        if not visa_address or not isinstance(visa_address, str):
            raise ValueError("visa_address must be a non-empty string")
        
        self._visa_address = visa_address
        self._timeout_ms = timeout_ms
        self._resource_manager: Optional[pyvisa.ResourceManager] = None
        self._instrument: Any = None  # pyvisa Resource object (use Any to avoid type errors)
        self._is_connected = False

    def connect(self) -> bool:
        try:
            self._resource_manager = pyvisa.ResourceManager()
            self._instrument = self._resource_manager.open_resource(self._visa_address)
            self._instrument.timeout = self._timeout_ms
            self._instrument.read_termination = '\n'
            self._instrument.write_termination = '\n'
            self._is_connected = True
            return True
        except pyvisa.errors.VisaIOError as e:
            print(f"VISA error connecting to {self._visa_address}: {e}")
            self._cleanup_connection()
            return False

    def disconnect(self) -> None:
        if self._instrument:
            self._instrument.close()
        if self._resource_manager:
            self._resource_manager.close()
        self._cleanup_connection()

    def _cleanup_connection(self) -> None:
        self._is_connected = False
        self._instrument = None
        self._resource_manager = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def write(self, command: str) -> None:
        if not self.is_connected or not self._instrument:
            raise ConnectionError("Instrument not connected")
        self._instrument.write(command)

    def query(self, command: str) -> str:
        if not self.is_connected or not self._instrument:
            raise ConnectionError("Instrument not connected")
        return self._instrument.query(command)

    def query_binary_values(self, command: str, datatype='B', is_big_endian=False):
        if not self.is_connected or not self._instrument:
            raise ConnectionError("Instrument not connected")
        return self._instrument.query_binary_values(command, datatype=datatype, is_big_endian=is_big_endian)

    def read_raw(self):
        if not self.is_connected or not self._instrument:
            raise ConnectionError("Instrument not connected")
        return self._instrument.read_raw()
