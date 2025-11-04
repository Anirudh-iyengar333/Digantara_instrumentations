#!/usr/bin/env python3
"""
Professional Instrument Control Library

A comprehensive, enterprise-grade Python library for controlling laboratory
and test equipment with precision and reliability.

Author: Professional Instrument Control Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Professional Instrument Control Team"
__email__ = "support@example.com"
__license__ = "MIT"
__description__ = "Professional-grade instrument control library for laboratory automation"

# Import main instrument control classes for convenient access
from .keithley_power_supply import KeithleyPowerSupply, KeithleyPowerSupplyError
from .keithley_dmm import KeithleyDMM6500, KeithleyDMM6500Error, MeasurementFunction
from .keysight_oscilloscope import KeysightDSOX6004A, KeysightDSOX6004AError

__all__ = [
    # Version information
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",

    # Keithley Power Supply classes
    "KeithleyPowerSupply",
    "KeithleyPowerSupplyError",

    # Keithley Multimeter classes
    "KeithleyDMM6500",
    "KeithleyDMM6500Error", 
    "MeasurementFunction",

    # Keysight Oscilloscope classes
    "KeysightDSOX6004A",
    "KeysightDSOX6004AError",
]

# Library information
LIBRARY_INFO = {
    "name": "Professional Instrument Control Library",
    "version": __version__,
    "author": __author__,
    "license": __license__,
    "description": __description__,
    "supported_instruments": {
        "power_supplies": [
            "Keithley 2230 Series",
            "Keithley 2231A Series", 
            "Keithley 2280S Series",
            "Keithley 2260B/2268 Series"
        ],
        "multimeters": [
            "Keithley DMM6500",
            "Keithley DMM7510"
        ],
        "oscilloscopes": [
            "Keysight DSOX6000 Series"
        ]
    }
}


def get_library_info() -> dict:
    """
    Get comprehensive library information.

    Returns:
        Dictionary containing library metadata and capabilities
    """
    return LIBRARY_INFO.copy()


def check_dependencies() -> dict:
    """
    Check availability of required dependencies.

    Returns:
        Dictionary with dependency status information
    """
    dependencies = {}

    # Check PyVISA
    try:
        import pyvisa
        dependencies['pyvisa'] = {
            'available': True,
            'version': pyvisa.__version__,
            'backends': []
        }

        # Check available VISA backends
        try:
            rm = pyvisa.ResourceManager()
            dependencies['pyvisa']['backends'].append('Default')
            rm.close()
        except:
            pass

        try:
            rm = pyvisa.ResourceManager('@py')
            dependencies['pyvisa']['backends'].append('PyVISA-py')
            rm.close()
        except:
            pass

    except ImportError:
        dependencies['pyvisa'] = {
            'available': False,
            'error': 'PyVISA not installed'
        }

    # Check NumPy
    try:
        import numpy
        dependencies['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        dependencies['numpy'] = {
            'available': False,
            'error': 'NumPy not installed'
        }

    # Check optional dependencies
    optional_deps = ['scipy', 'matplotlib', 'pandas']
    for dep in optional_deps:
        try:
            module = __import__(dep)
            dependencies[dep] = {
                'available': True,
                'version': getattr(module, '__version__', 'unknown'),
                'optional': True
            }
        except ImportError:
            dependencies[dep] = {
                'available': False,
                'error': f'{dep} not installed',
                'optional': True
            }

    return dependencies
