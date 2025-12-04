#!/usr/bin/env python3
"""
Professional Instrument Control Library

A comprehensive, enterprise-grade Python library for controlling laboratory
and test equipment with precision and reliability.

Author: Professional Instrument Control Team
Version: 1.0.1
License: MIT
"""

__version__ = "1.0.1"
__author__ = "Professional Instrument Control Team"
__email__ = "support@example.com"
__license__ = "MIT"
__description__ = "Professional-grade instrument control library for laboratory automation"

# Import main instrument control classes for convenient access
from .keithley_power_supply import KeithleyPowerSupply, KeithleyPowerSupplyError
from .keithley_dmm import KeithleyDMM6500, KeithleyDMM6500Error, MeasurementFunction
from .keysight_oscilloscope import KeysightDSOX6004A, KeysightDSOX6004AError
from .tektronix_oscilloscope import TektronixMSO24, TektronixMSO24Error

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

    # Tektronix Oscilloscope classes
    "TektronixMSO24",
    "TektronixMSO24Error",
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
            "Keysight DSOX6000 Series (DSOX6004A)",
            "Tektronix MSO2 Series (MSO24)"
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
    optional_deps = ['scipy', 'matplotlib', 'pandas', 'gradio']
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


def get_oscilloscope_comparison() -> dict:
    """
    Get comparison between supported oscilloscope models.
    
    Returns:
        Dictionary comparing oscilloscope specifications and capabilities
    """
    return {
        "keysight_dsox6004a": {
            "manufacturer": "Keysight Technologies",
            "model": "DSOX6004A",
            "bandwidth": "1 GHz",
            "channels": 4,
            "sample_rate": "20 GS/s",
            "memory_depth": "16 Mpts",
            "function_generators": 2,
            "digital_channels": 0,
            "key_features": [
                "High bandwidth (1 GHz)",
                "Dual function generators",
                "Advanced trigger modes",
                "Math functions",
                "High sample rate"
            ]
        },
        "tektronix_mso24": {
            "manufacturer": "Tektronix",
            "model": "MSO24",
            "bandwidth": "200 MHz",
            "channels": 4,
            "sample_rate": "2.5 GS/s", 
            "memory_depth": "62.5 Mpts",
            "function_generators": 1,
            "digital_channels": 16,
            "key_features": [
                "Mixed signal capability (16 digital channels)",
                "Large memory depth (62.5 Mpts)",
                "Built-in AFG",
                "Comprehensive measurement suite",
                "Professional test automation"
            ]
        }
    }


def get_recommended_usage() -> dict:
    """
    Get recommended usage scenarios for each oscilloscope.
    
    Returns:
        Dictionary with recommended applications for each model
    """
    return {
        "keysight_dsox6004a": [
            "High-frequency signal analysis (up to 1 GHz)",
            "RF and microwave applications",
            "High-speed digital signal validation",
            "Advanced signal generation requirements",
            "Applications requiring maximum bandwidth"
        ],
        "tektronix_mso24": [
            "Mixed signal debugging (analog + digital)",
            "Embedded system development", 
            "Long duration signal capture",
            "Educational and training applications",
            "Cost-effective professional testing",
            "Applications with moderate bandwidth requirements"
        ]
    }


# Convenience function for quick instrument selection
def create_oscilloscope(model: str, visa_address: str, **kwargs):
    """
    Factory function to create oscilloscope instances.
    
    Args:
        model: Oscilloscope model ("keysight_dsox6004a" or "tektronix_mso24")
        visa_address: VISA address for the instrument
        **kwargs: Additional arguments passed to the constructor
        
    Returns:
        Oscilloscope instance
        
    Raises:
        ValueError: If model is not supported
    """
    model = model.lower().replace("-", "_").replace(" ", "_")
    
    if model in ["keysight_dsox6004a", "dsox6004a", "keysight"]:
        return KeysightDSOX6004A(visa_address, **kwargs)
    elif model in ["tektronix_mso24", "mso24", "tektronix"]:
        return TektronixMSO24(visa_address, **kwargs)
    else:
        raise ValueError(f"Unsupported oscilloscope model: {model}. "
                        f"Supported models: keysight_dsox6004a, tektronix_mso24")


# Professional instrument control best practices
BEST_PRACTICES = {
    "connection_management": [
        "Always use try-catch blocks for VISA operations",
        "Implement proper timeout management for long operations",
        "Use context managers or ensure proper disconnect() calls",
        "Verify instrument identification after connection"
    ],
    "measurement_automation": [
        "Configure channels before starting measurements",
        "Use appropriate trigger settings for stable acquisition",
        "Implement error checking for measurement validity",
        "Save configuration state for reproducible results"
    ],
    "data_handling": [
        "Use professional file naming conventions with timestamps",
        "Implement proper data validation and error checking",
        "Provide comprehensive metadata with exported data",
        "Use appropriate data formats (CSV, binary) for the application"
    ],
    "thread_safety": [
        "Use locks for multi-threaded instrument access",
        "Implement proper exception handling in threads",
        "Avoid simultaneous SCPI commands to the same instrument",
        "Use queues for data sharing between threads"
    ]
}


def get_best_practices() -> dict:
    """
    Get professional instrument control best practices.
    
    Returns:
        Dictionary containing best practice guidelines
    """
    return BEST_PRACTICES.copy()