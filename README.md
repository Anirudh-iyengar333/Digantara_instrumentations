# Digantara Instrumentation Control Suite

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()

> **Enterprise-grade test automation platform for laboratory instrument control**

A comprehensive web-based automation framework providing unified control of precision laboratory test equipment through an intuitive browser interface. Designed for electronics testing, hardware validation, and research and development workflows.

**Developed by:** Anirudh Iyengar  
**Organization:** Digantara Research and Technologies Pvt. Ltd.

---

## Table of Contents

- [Overview](#overview)
- [Supported Instruments](#supported-instruments)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [User Guides](#user-guides)
- [Architecture](#architecture)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)
- [Technical Reference](#technical-reference)
- [Support](#support)

---

## Overview

The Digantara Instrumentation Control Suite eliminates the complexity of laboratory test equipment operation by providing a unified web-based interface accessible through any modern browser. Whether you're a test engineer running production validation sequences or a researcher collecting measurement data, this platform streamlines instrument control without requiring programming knowledge.

### What This Platform Provides

- **Zero-Code Operation**: Complete instrument control through intuitive web forms
- **Unified Interface**: Single application for multiple instrument types
- **Real-Time Visualization**: Live plotting of measurements and waveforms
- **Automated Testing**: Built-in waveform generation and test sequence automation
- **Data Export**: Professional-grade data logging with CSV, JSON, and Excel support
- **Remote Access**: Control instruments from any computer on your network

### Who Should Use This

- **Test Engineers**: Automated test equipment (ATE) operation and validation
- **Hardware Engineers**: Circuit characterization and power supply testing
- **Research Scientists**: Data acquisition and experimental measurement
- **Quality Assurance**: Production testing and calibration workflows
- **Lab Technicians**: Routine measurement and equipment monitoring

---

## Supported Instruments

| Category | Model | Capabilities | Interface |
|----------|-------|--------------|-----------|
| **Power Supply** | Keithley 2230-30-1 | 3-channel programmable DC PSU<br>30V/3A per channel<br>Waveform generation (Sine, Square, Triangle, Ramp) | USB/VISA |
| **Digital Multimeter** | Keithley DMM6500<br>Keithley DMM7510 | 6.5/7.5-digit precision DMM<br>DC/AC V/I, 2W/4W resistance<br>Capacitance, frequency, temperature | USB/LAN/VISA |
| **Oscilloscope** | Keysight DSOX6004A | 4-channel mixed signal scope<br>1 GHz bandwidth, 20 GSa/s<br>Advanced triggering and math | USB/LAN/VISA |

### Measurement Specifications

#### Keithley Power Supply
- **Voltage Range**: 0-30V per channel (3 independent channels)
- **Current Range**: 0-3A per channel
- **Resolution**: 1mV (voltage), 1mA (current)
- **Over-Voltage Protection**: Programmable per channel
- **Waveform Generation**: 4 waveform types with configurable amplitude and frequency

#### Keithley Digital Multimeter
- **DC Voltage**: 1µV to 1000V (6.5-digit resolution)
- **AC Voltage**: 100µV to 750V RMS
- **DC/AC Current**: 1nA to 10A
- **Resistance**: 1mΩ to 100MΩ (2-wire/4-wire)
- **Additional**: Capacitance, frequency, temperature (thermocouple/RTD)

#### Keysight Oscilloscope
- **Bandwidth**: 1 GHz (4 analog channels)
- **Sample Rate**: 20 GSa/s (single-shot), 4 GSa/s (all channels)
- **Memory Depth**: Up to 4 Mpts per channel
- **Triggering**: Edge, pulse width, pattern, serial bus
- **Analysis**: FFT, math functions, automated measurements

---

## Key Features

### Unified Control Interface

- **Multi-Instrument Dashboard**: Single tabbed interface for all connected instruments
- **Real-Time Status**: Live connection monitoring and instrument state display
- **Responsive Design**: Optimized for desktop and tablet browsers
- **Network Access**: Share instrument access across your local network

### Advanced Measurement Capabilities

- **Continuous Acquisition**: Background measurement with configurable sample rates
- **Statistical Analysis**: Real-time mean, standard deviation, min/max, and trending
- **Limit Testing**: Pass/fail evaluation with programmable thresholds
- **Data Buffering**: High-speed capture with circular buffer management

### Automated Test Workflows

- **Waveform Generation**: Create complex voltage profiles for stress testing and characterization
- **Sequence Automation**: Multi-step test procedures with conditional logic
- **Triggered Measurements**: Synchronized acquisition across multiple instruments
- **Batch Processing**: Execute repetitive measurement sequences with minimal user interaction

### Professional Data Management

- **Multiple Export Formats**: CSV (Excel-compatible), JSON (API/programming), Excel (.xlsx)
- **Timestamped Logging**: Automatic timestamp generation for measurement traceability
- **Metadata Capture**: Instrument configuration and test parameters saved with data
- **Screenshot Capture**: Save oscilloscope screen images with waveform annotations

### Safety and Reliability

- **Emergency Stop**: Hardware output disable with single-click safety shutoff
- **Over-Voltage Protection**: Configurable OVP limits prevent equipment damage
- **Connection Validation**: Pre-operation checks ensure instrument readiness
- **Error Recovery**: Automatic reconnection and graceful error handling
- **Thread-Safe Design**: Multi-threaded architecture with resource locking

---

## Quick Start

### Prerequisites

Before installation, ensure your system meets these requirements:

- **Operating System**: Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+)
- **Python Version**: 3.8 or newer (3.9-3.11 recommended)
- **VISA Backend**: Keysight IO Libraries Suite or NI-VISA runtime
- **Web Browser**: Chrome, Firefox, Edge, or Safari (latest version)
- **Hardware**: USB 2.0+ port or network connection

### 5-Minute Setup

#### Step 1: Install Python

If Python is not installed on your system:

1. Download Python from [python.org/downloads](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"** during installation
3. Verify installation by opening a terminal/command prompt:
   ```bash
   python --version
   ```
   You should see `Python 3.8.x` or higher

#### Step 2: Install VISA Drivers

VISA drivers enable communication with laboratory instruments:

**Option A: Keysight IO Libraries Suite** (Recommended for Keysight instruments)
- Download from [keysight.com/find/iosuite](https://www.keysight.com/us/en/lib/software-detail/computer-software/io-libraries-suite-downloads-2175637.html)
- Run installer and select "Full Installation"

**Option B: NI-VISA**
- Download from [ni.com/visa](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
- Install with default settings

#### Step 3: Download and Install Software

1. Navigate to the project directory:
   ```bash
   cd path\to\Digantara_instrumentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Installation takes 2-5 minutes depending on network speed*

#### Step 4: Connect Your Instrument

1. Connect instrument to computer via USB cable or network
2. Power on the instrument
3. Wait for operating system to recognize device (USB connection sound on Windows)

#### Step 5: Launch the Interface

**Unified Interface (All Instruments):**
```bash
python Unified.py
```

**Individual Instrument Control:**

Power Supply:
```bash
python scripts\keithley\keithley_PSU_gradio_automation.py
```

Digital Multimeter:
```bash
python scripts\keithley\keithley_dmm_gradio_automation.py
```

Oscilloscope:
```bash
python scripts\keysight\keysight_oscilloscope_gradio_en.py
```

#### Step 6: Access the Web Interface

After launching, you'll see console output similar to:
```
Running on local URL:  http://127.0.0.1:7860
Running on network:   http://192.168.128.175:7860
```

Open your browser and navigate to:
- **Local access**: http://localhost:7860
- **Network access**: http://[displayed-IP]:7860

The instrument control interface will appear in your browser.

---

## Installation

### System Requirements

**Minimum Configuration:**
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Storage: 500 MB free space
- Network: Optional (for remote access and instrument LAN connectivity)

**Recommended Configuration:**
- CPU: Quad-core 2.5 GHz or better
- RAM: 8 GB or more
- Storage: 2 GB free space
- Network: Gigabit Ethernet (for high-speed oscilloscope data transfer)

### Detailed Installation Steps

#### 1. Python Environment Setup

**Windows:**
```bash
# Download Python 3.10 from python.org
# During installation, check "Add Python to PATH"
# Verify installation:
python --version
pip --version
```

**macOS:**
```bash
# Install Homebrew if not present:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python:
brew install python@3.10

# Verify:
python3 --version
pip3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv

# Verify:
python3 --version
pip3 --version
```

#### 2. Clone or Download Project

**Using Git:**
```bash
git clone https://github.com/digantara/instruments.git
cd instruments
```

**Manual Download:**
- Download ZIP from repository
- Extract to desired location
- Open terminal in extracted folder

#### 3. Install Dependencies

**Standard Installation:**
```bash
pip install -r requirements.txt
```

**Virtual Environment (Recommended for isolation):**
```bash
# Create virtual environment:
python -m venv venv

# Activate virtual environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

**Development Installation (includes testing tools):**
```bash
pip install -e ".[dev]"
```

#### 4. VISA Backend Configuration

After installing VISA drivers, verify installation:

**Using NI-MAX (National Instruments):**
1. Open NI-MAX from Start menu
2. Expand "Devices and Interfaces"
3. Right-click → "Scan for Instruments"
4. Connected instruments should appear with VISA addresses

**Using Keysight Connection Expert:**
1. Launch Keysight Connection Expert
2. Click "Auto-Discover"
3. Note VISA addresses for connected instruments

**Test VISA from Python:**
```python
import pyvisa
rm = pyvisa.ResourceManager()
print(rm.list_resources())
# Should display connected instruments
```

#### 5. Verify Installation

Run the unified interface:
```bash
python Unified.py
```

Expected output:
```
Loading instrument modules...
Initializing Gradio interface...
Running on local URL: http://127.0.0.1:7860
```

If you see this output, installation is successful!

---

## User Guides

### Power Supply Operation

The Keithley 2230-30-1 power supply interface provides complete control over three independent DC channels.

#### Basic Power Supply Workflow

**1. Connect to Instrument**

- **VISA Address Field**: Enter instrument address or use auto-detect
  - Format: `USB0::0x05E6::0x2230::[SERIAL]::INSTR`
  - Example: `USB0::0x05E6::0x2230::805224014806770001::INSTR`
- Click **"Connect"** button
- Connection status will display green checkmark when successful

**2. Configure Channel Output**

For each channel (1, 2, or 3), set the following parameters:

| Parameter | Description | Range | Recommended |
|-----------|-------------|-------|-------------|
| **Voltage** | Target output voltage | 0-30V | Start with low voltage (e.g., 5V) |
| **Current Limit** | Maximum current allowed | 0-3A | Set 20% above expected load |
| **OVP (Over-Voltage Protection)** | Safety cutoff voltage | 0-33V | Set 10-20% above target voltage |

Click **"Configure"** to apply settings.

**3. Enable Output**

- Click **"Enable Output"** for desired channel
- Output status indicator turns green
- LED on physical instrument illuminates

**4. Monitor Operation**

- Click **"Measure"** to read actual voltage and current
- Click **"Measure All Channels"** for complete status overview
- Monitor values in real-time display table

**5. Disable Output**

- Click **"Disable Output"** on individual channels when finished
- Or use **"EMERGENCY STOP"** to disable all outputs immediately

#### Advanced: Automated Waveform Generation

Create complex voltage profiles for stress testing and device characterization.

**Supported Waveform Types:**

| Waveform | Description | Use Case |
|----------|-------------|----------|
| **Sine** | Smooth sinusoidal variation | AC simulation, ripple testing |
| **Square** | Step between high/low levels | Digital power cycling, ON/OFF stress |
| **Triangle** | Linear ramp up and down | Thermal cycling, gradual stress |
| **Ramp Up** | Linear increase from zero | Power-on sequencing, startup testing |
| **Ramp Down** | Linear decrease to zero | Graceful shutdown testing |

**Configuration Parameters:**

```
Target Voltage:    Maximum voltage (waveform amplitude)
Number of Cycles:  How many complete waveform repetitions
Cycle Duration:    Time for one complete cycle (seconds)
Channel:           Which output channel (1, 2, or 3)
```

**Example: 10V Sine Wave, 5 Cycles, 2-Second Period**

1. Select waveform type: **"Sine Wave"**
2. Target voltage: **10.0** V
3. Number of cycles: **5**
4. Cycle duration: **2.0** seconds
5. Channel: **1**
6. Click **"Start Ramping"**

The system will:
- Generate 5 complete sine waves
- Each cycle takes 2 seconds (0.5 Hz frequency)
- Voltage varies smoothly between 0-10V
- Total execution time: 10 seconds
- Progress bar shows real-time status

**Data Export:**
- Waveform data automatically saved to `voltage_ramp_data/` folder
- File format: `waveform_[DATE]_[TIME].csv`
- Contains: timestamp, voltage setpoint, measured voltage, current

#### Safety Features

**Emergency Stop:**
- Large red button in interface
- Immediately disables all three channels
- Use if you observe unexpected behavior, smoke, or overheating

**Over-Voltage Protection (OVP):**
- Hardware-level protection
- Automatically disables output if voltage exceeds OVP threshold
- Cannot be overridden (safety-critical feature)
- Always set OVP 10-20% above normal operating voltage

**Current Limiting:**
- Prevents damage to sensitive devices
- Supply transitions to constant-current mode if load draws excess current
- Voltage will drop to maintain current limit

---

### Digital Multimeter Operation

The Keithley DMM6500/7510 interface provides precision measurement capabilities with statistical analysis.

#### Measurement Types

| Function | Range | Resolution | Typical Use |
|----------|-------|------------|-------------|
| **DC Voltage** | 100mV - 1000V | 1µV | Power supply verification, battery voltage |
| **AC Voltage** | 100mV - 750V | 100µV | Mains power, signal amplitude |
| **DC Current** | 10µA - 10A | 10nA | Circuit current monitoring |
| **AC Current** | 10µA - 10A | 10nA | Power consumption measurement |
| **2-Wire Resistance** | 100Ω - 100MΩ | 1mΩ | General resistance measurement |
| **4-Wire Resistance** | 100Ω - 100MΩ | 1mΩ | Low resistance (<1Ω), contact resistance |
| **Capacitance** | 1nF - 10µF | 1pF | Capacitor verification |
| **Frequency** | 3Hz - 300kHz | 1mHz | Signal frequency measurement |
| **Temperature** | -200°C - +1372°C | 0.01°C | Thermocouple/RTD measurements |

#### Basic Measurement Workflow

**1. Connect to Instrument**

- Enter VISA address (format: `USB0::0x05E6::0x6500::[SERIAL]::INSTR`)
- Click **"Connect"**
- Instrument model and firmware version display upon successful connection

**2. Configure Measurement**

| Setting | Description | Typical Value |
|---------|-------------|---------------|
| **Function** | Measurement type | Select from dropdown |
| **Range** | Maximum expected value | Use "AUTO" if uncertain |
| **NPLC** | Integration time | 1.0 (balanced speed/accuracy) |
| **Auto-Zero** | Offset correction | ON (recommended) |

**Understanding NPLC (Number of Power Line Cycles):**

NPLC controls measurement integration time and noise rejection:

- **0.01 NPLC**: Fast measurements (50 readings/sec), noisy
- **0.1 NPLC**: Quick measurements (10 readings/sec), moderate noise
- **1.0 NPLC**: Standard measurements (1 reading/sec), good noise rejection
- **10 NPLC**: High-precision measurements (0.1 readings/sec), excellent noise rejection

*Use higher NPLC for stable DC measurements, lower NPLC for faster acquisition*

**3. Take Measurements**

**Single Measurement:**
- Click **"Single Measurement"**
- Result displays immediately with unit

**Continuous Measurement:**
- Click **"Start Continuous"**
- Measurements update in real-time
- Click **"Stop Continuous"** to halt
- Statistics automatically calculated (see Statistics tab)

**4. View Statistics**

Navigate to **"Statistics"** tab to view:

- **Count**: Number of measurements collected
- **Mean**: Average value
- **Std Dev**: Standard deviation (measurement consistency)
- **Min/Max**: Range of values
- **Peak-to-Peak**: Difference between max and min

**Interpreting Standard Deviation:**
- Low std dev (<0.1% of mean): Stable, consistent signal
- Medium std dev (0.1-1% of mean): Minor variations, acceptable
- High std dev (>1% of mean): Noisy signal or varying input

**5. Export Data**

Navigate to **"Data Export"** tab:

1. Choose export format:
   - **CSV**: Opens in Excel, Google Sheets (recommended for most users)
   - **JSON**: Structured format for programming/automation
   - **Excel (.xlsx)**: Native Excel format with formatting

2. Click **"Export Data"**

3. File saves to current directory with timestamp:
   - Format: `dmm_data_YYYYMMDD_HHMMSS.[format]`
   - Example: `dmm_data_20251201_143022.csv`

**CSV File Structure:**
```
Timestamp,Value,Unit,Function,Range,NPLC
2025-12-01 14:30:22.123,5.0234,V,DC_VOLTAGE,10.0,1.0
2025-12-01 14:30:23.124,5.0231,V,DC_VOLTAGE,10.0,1.0
```

#### 4-Wire Resistance Measurement (Kelvin Sensing)

For accurate low-resistance measurements (<100Ω), use 4-wire technique:

**Advantages:**
- Eliminates test lead resistance (~0.1-1Ω per lead)
- Critical for contact resistance (<1Ω)
- Accurate resistance measurement of cables, connectors, PCB traces

**Connection:**
1. Connect SENSE HI and FORCE HI to one side of resistor
2. Connect SENSE LO and FORCE LO to other side of resistor
3. Select **"4-WIRE RESISTANCE"** function
4. Take measurement as normal

---

### Oscilloscope Operation

The Keysight DSOX6004A interface provides waveform capture, analysis, and screenshot capabilities.

#### Basic Oscilloscope Workflow

**1. Connect to Instrument**

- **Connection Type**: Choose USB or LAN
  - USB: Enter VISA address (`USB0::0x0957::...::INSTR`)
  - LAN: Enter IP address (e.g., `192.168.128.175`)
- Click **"Connect"**
- Instrument ID displays upon successful connection

**2. Configure Channels**

For each channel (1-4), configure:

| Parameter | Description | Typical Settings |
|-----------|-------------|------------------|
| **Vertical Scale** | Volts per division | 1V/div for logic signals<br>100mV/div for small signals |
| **Vertical Offset** | Position on screen | 0V (center screen) |
| **Coupling** | Signal coupling | DC (most common)<br>AC (removes DC offset) |
| **Probe Attenuation** | Probe factor | 1X (direct)<br>10X (standard probe) |

**Understanding Probe Attenuation:**

Oscilloscope probes have attenuation factors marked on the probe body:

- **1X Probe**: No attenuation, direct connection
  - Use for: Low voltage (<5V), low frequency signals
  - Limitation: High capacitance, loads circuit

- **10X Probe**: Signal divided by 10 (most common)
  - Use for: General purpose (0-300V), up to 500 MHz
  - Advantage: Low capacitance, minimal circuit loading
  - **IMPORTANT**: Set software to 10X or readings will be 10× too low

Always match software setting to physical probe marking!

**3. Set Timebase**

- **Horizontal Scale**: Time per division
  - 1ms/div: Low-frequency signals (kHz range)
  - 1µs/div: Medium-frequency signals (MHz range)
  - 1ns/div: High-frequency signals (GHz range)

**4. Configure Trigger**

Triggering determines when oscilloscope captures a waveform.

| Setting | Description | Typical Value |
|---------|-------------|---------------|
| **Source** | Which channel triggers | Same as signal channel |
| **Level** | Voltage threshold | 50% of signal amplitude |
| **Slope** | Edge direction | Rising (positive edge)<br>Falling (negative edge) |
| **Mode** | Trigger mode | Edge (most common) |

**Trigger Example: Capturing 5V Digital Signal**

- Source: Channel 1
- Level: 2.5V (halfway between 0-5V)
- Slope: Rising
- Mode: Edge

Oscilloscope will trigger when Channel 1 signal crosses 2.5V in rising direction.

**5. Acquire Waveform**

1. Select channel from dropdown
2. Click **"Acquire Waveform"**
3. Waveform plot appears below
4. Click **"Save Waveform"** to export data

**Waveform Data Export:**
- File format: CSV
- Contains: time, voltage pairs
- Location: `outputs/waveform_ch[N]_[TIMESTAMP].csv`

**6. Capture Screenshot**

- Click **"Capture Screenshot"**
- Saves current oscilloscope display as PNG image
- Includes all on-screen elements: waveforms, measurements, settings
- Location: `outputs/screenshot_[TIMESTAMP].png`

**Use cases:**
- Documentation of test results
- Anomaly capture for debugging
- Report generation

#### Automated Measurements

The oscilloscope can automatically measure waveform parameters:

**Available Measurements:**
- Frequency, Period
- Peak-to-Peak voltage, Amplitude
- Rise time, Fall time
- Pulse width, Duty cycle
- RMS voltage
- Mean voltage

**Setup:**
1. Navigate to **"Measurements"** tab
2. Select measurement type
3. Select source channel
4. Click **"Add Measurement"**
5. Result displays on oscilloscope screen and in interface

---

## Architecture

### System Overview

The Digantara Instrumentation Control Suite follows a layered architecture pattern:

```
┌─────────────────────────────────────────────────────────────┐
│               Web Browser Interface                         │
│                 (Gradio UI )                                │
└──────────────────────┬──────────────────────────────────────┘
                       │ 
┌──────────────────────▼─────────────────────────────────────┐
│              Gradio Application Server (Python)            │
│  ┌────────────┬──────────────────┬──────────────────────┐  │
│  │  DMM GUI   │   PSU GUI        │  Oscilloscope GUI    │  │
│  │ Controller │   Controller     │   Controller         │  │
│  └──────┬─────┴─────┬────────────┴──────┬───────────────┘  │
└─────────┼───────────┼───────────────────┼──────────────────┘
          │           │                   │
┌─────────▼───────────▼───────────────────▼──────────────────┐
│            Instrument Driver Layer (Python Classes)        │
│  ┌──────────────┬────────────────────┬──────────────────┐  │
│  │ KeithleyDMM  │  KeithleyPSU       │  KeysightScope   │  │
│  │ Class        │  Class             │  Class           │  │
│  └──────┬───────┴──────┬─────────────┴──────┬───────────┘  │
└─────────┼──────────────┼────────────────────┼──────────────┘
          │              │                    │
┌─────────▼──────────────▼────────────────────▼──────────────┐
│              PyVISA Communication Layer                    │
│         (SCPI Command Translation & Error Handling)        │
└──────────────────────┬─────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   VISA Backend Driver                       │
│          (Keysight IO Suite / NI-VISA Runtime)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Operating System I/O Layer                     │
│         (USB, TCPIP, GPIB Hardware Abstraction)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                Physical Instruments                         │
│        (Keithley PSU, DMM, Keysight Oscilloscope)           │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Web Interface Layer:**
- User input validation and sanitization
- Real-time data visualization (Matplotlib plots)
- Responsive form controls and status indicators
- WebSocket-based live updates

**Application Controller Layer:**
- Business logic and workflow orchestration
- Multi-threaded measurement coordination
- Data aggregation and statistical processing
- Export formatting (CSV, JSON, Excel)

**Instrument Driver Layer:**
- SCPI command generation and parsing
- Instrument-specific error handling
- Connection state management
- Configuration validation

**Communication Layer:**
- Low-level VISA resource management
- Command queuing and timeout handling
- Error queue monitoring
- Connection retry logic

### Threading Model

The application uses a multi-threaded design for responsive operation:

**Main Thread:**
- Gradio event loop
- UI updates and rendering
- User interaction handling

**DMM Worker Thread:**
- Continuous measurement acquisition
- Buffer management
- Statistical calculation

**PSU Worker Thread:**
- Waveform generation execution
- Voltage ramping control
- Progress reporting

**Thread Safety:**
- All VISA I/O protected by `threading.RLock()`
- Atomic operations for shared state
- Queue-based inter-thread communication

### File Structure

```
Digantara_instrumentation/
│
├── Unified.py                          # Unified multi-instrument interface
│
├── instrument_control/                 # Core instrument driver library
│   ├── __init__.py                     # Package initialization
│   ├── keithley_dmm.py                 # DMM6500/7510 driver
│   ├── keithley_power_supply.py        # 2230-30-1 PSU driver
│   ├── keysight_oscilloscope.py        # DSOX6004A oscilloscope driver
│   └── scpi_wrapper.py                 # Base SCPI communication class
│
├── scripts/                            # Individual instrument GUIs
│   ├── keithley/
│   │   ├── keithley_PSU_gradio_automation.py     # PSU standalone GUI
│   │   ├── keithley_dmm_gradio_automation.py     # DMM standalone GUI
│   │   └── keithley_power_supply_automation.py   # Legacy PSU interface
│   └── keysight/
│       ├── keysight_oscilloscope_gradio_en.py    # Oscilloscope GUI (Enhanced)
│       ├── keysight_oscilloscope_gradio.py       # Oscilloscope GUI (minimal)
│       ├── keysight_oscilloscope_main.py         # Oscilloscope main interface
│       └── keysight_oscilloscope_main_with_livefeed.py  # With live waveform
│
├── requirements.txt                    # Python package dependencies
├── setup.py                            # Package installation configuration
│
├── README.md                           # This file (main documentation)
├── INSTALLATION.md                     # Detailed installation guide
├── QUICK_START.md                      # Quick start tutorial
├── USAGE_WORKFLOWS.md                  # Common usage scenarios
└── DOCUMENTATION_INDEX.md              # Documentation navigation
```

---

## Data Management

### Export Formats

#### CSV (Comma-Separated Values)

Best for: Excel analysis, general data sharing

**Power Supply Waveform CSV:**
```csv
Timestamp,Setpoint_Voltage_V,Measured_Voltage_V,Measured_Current_A
2025-12-01 14:30:22.123,5.000,5.012,0.234
2025-12-01 14:30:22.624,5.500,5.487,0.234
```

**DMM Measurement CSV:**
```csv
Timestamp,Value,Unit,Function,Range,NPLC
2025-12-01 14:30:22.123,5.0234,V,DC_VOLTAGE,10.0,1.0
2025-12-01 14:30:23.124,5.0231,V,DC_VOLTAGE,10.0,1.0
```

**Oscilloscope Waveform CSV:**
```csv
Time_s,Voltage_V
-0.0005,0.012
-0.0004,0.145
-0.0003,0.678
```

#### JSON (JavaScript Object Notation)

Best for: Programming, automation, API integration

**DMM Data JSON:**
```json
{
  "instrument": "Keithley DMM6500",
  "measurement_type": "DC_VOLTAGE",
  "configuration": {
    "range": 10.0,
    "nplc": 1.0,
    "auto_zero": true
  },
  "timestamp": "2025-12-01T14:30:22.123",
  "statistics": {
    "count": 100,
    "mean": 5.0234,
    "std_dev": 0.0012,
    "min": 5.0210,
    "max": 5.0258
  },
  "data": [
    {"timestamp": "2025-12-01T14:30:22.123", "value": 5.0234, "unit": "V"},
    {"timestamp": "2025-12-01T14:30:23.124", "value": 5.0231, "unit": "V"}
  ]
}
```

#### Excel (.xlsx)

Best for: Formal reports, formatted presentations

Features:
- Multiple worksheets (data, statistics, configuration)
- Formatted headers and units
- Embedded charts
- Metadata sheet with test parameters

### File Naming Conventions

All exported files follow consistent timestamp-based naming:

**Format:** `[instrument]_[datatype]_YYYYMMDD_HHMMSS.[extension]`

**Examples:**
- `dmm_measurements_20251201_143022.csv`
- `psu_waveform_sine_20251201_150145.json`
- `scope_screenshot_20251201_161530.png`
- `scope_waveform_ch1_20251201_162045.csv`

**Benefits:**
- Chronological sorting by filename
- No filename conflicts (unique timestamps)
- Easy identification of data type and source
- Compatible with Windows, macOS, Linux filesystems

### Data Storage Locations

| Instrument | Data Type | Default Location |
|------------|-----------|------------------|
| Power Supply | Waveform data | `voltage_ramp_data/` |
| DMM | Measurements | Current working directory |
| Oscilloscope | Waveforms | `outputs/` |
| Oscilloscope | Screenshots | `outputs/` |
| All | Configuration files | `.config/` |

**Customizing Storage Locations:**

Edit the respective Python script and modify the output directory constants:

```python
# In script file:
OUTPUT_DIR = "C:\\TestData\\Measurements"  # Windows
OUTPUT_DIR = "/home/user/test_data"         # Linux/macOS
```

---

## Troubleshooting

### Connection Issues

#### Problem: "Cannot find instrument" or "VISA resource not found"

**Symptoms:**
- Error message when clicking "Connect"
- VISA address not auto-detected
- Instrument not listed in resource scan

**Solutions:**

1. **Verify physical connection:**
   ```bash
   # Check USB connection:
   # Windows: Device Manager → Universal Serial Bus Controllers
   # Linux: lsusb
   # macOS: System Information → USB
   ```

2. **Verify VISA installation:**
   ```python
   import pyvisa
   rm = pyvisa.ResourceManager()
   print(rm.list_resources())
   # Should list connected instruments
   ```

3. **Use VISA utility to scan:**
   - **NI-MAX**: Start → NI-MAX → Devices and Interfaces → Scan for Instruments
   - **Keysight Connection Expert**: Launch → Auto-Discover

4. **Check instrument power and ready state:**
   - Instrument powered on
   - Boot sequence completed (no error messages on display)
   - Front panel not locked

5. **Try different USB port or cable:**
   - Use USB 2.0/3.0 port directly on computer (avoid hubs)
   - Try different USB cable (data-capable, not charge-only)

6. **Reinstall VISA drivers:**
   - Uninstall Keysight IO Libraries / NI-VISA
   - Restart computer
   - Reinstall with administrator privileges

#### Problem: "Connection timeout" or "Instrument not responding"

**Symptoms:**
- Connection succeeds but commands fail
- Long delays before error messages
- Intermittent communication

**Solutions:**

1. **Increase timeout in driver:**
   ```python
   # Edit instrument control file:
   self.instrument.timeout = 10000  # 10 seconds (default: 5000)
   ```

2. **Check instrument error queue:**
   - On instrument front panel: Utility → Error Queue
   - Clear any existing errors before reconnecting

3. **Reset instrument to default state:**
   - Front panel: Utility → Reset → Factory Reset
   - Or send SCPI command: `*RST`

4. **Verify network settings (for LAN connection):**
   ```bash
   # Ping instrument IP address:
   ping 192.168.128.175

   # Should receive replies with low latency (<10ms)
   ```

5. **Disable firewall temporarily:**
   - Windows Firewall may block VISA communication
   - Add Python and Gradio to firewall exceptions

### Measurement Issues

#### Problem: "Readings are unstable" or "High noise"

**Symptoms:**
- Measurements fluctuate significantly
- High standard deviation
- Erratic plot on trend graph

**Solutions:**

1. **Increase integration time (DMM):**
   - Increase NPLC from 1.0 to 10.0
   - Trade-off: Slower measurements, better noise rejection

2. **Enable averaging:**
   - DMM: Configure filter → Enable digital filter
   - Averaging count: 10-100 samples

3. **Check grounding:**
   - Ensure Device Under Test (DUT) and instrument share common ground
   - Use shielded cables
   - Minimize ground loops

4. **Enable auto-zero (DMM):**
   - Auto-zero setting: ON
   - Compensates for internal offset drift

5. **Verify connections:**
   - Tighten all cable connections
   - Check for damaged cables or probes
   - For low resistance (<1Ω), use 4-wire measurement

#### Problem: "Incorrect readings" or "Values don't match expected"

**Symptoms:**
- Measured value different from known source
- Consistent offset error
- Readings 10× too high/low

**Solutions:**

1. **Check probe attenuation (oscilloscope):**
   - Verify probe setting matches physical probe
   - 10X probe requires 10X setting in software

2. **Verify measurement range:**
   - Use appropriate range for signal level
   - Auto-range if uncertain

3. **Check input coupling:**
   - DC coupling: Measures total voltage including DC offset
   - AC coupling: Blocks DC, measures AC component only

4. **Verify unit scaling:**
   - Check displayed units (V, mV, µV)
   - Confirm decimal point position

5. **Calibrate instrument:**
   - Follow manufacturer calibration procedure
   - Some instruments have internal self-calibration
   - Annual professional calibration recommended

### Software Issues

#### Problem: "Module not found" or "Import error"

**Symptoms:**
```
ModuleNotFoundError: No module named 'gradio'
ImportError: cannot import name 'KeithleyDMM6500'
```

**Solutions:**

1. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

2. **Verify Python version:**
   ```bash
   python --version
   # Must be 3.8 or higher
   ```

3. **Check virtual environment activation:**
   ```bash
   # Windows:
   venv\Scripts\activate

   # macOS/Linux:
   source venv/bin/activate

   # Prompt should show (venv) prefix
   ```

4. **Install missing package individually:**
   ```bash
   pip install gradio>=4.0.0
   pip install pyvisa>=1.13.0
   ```

#### Problem: "Web interface won't load" or "Port already in use"

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Change server port:**
   - Edit Python script
   - Find: `server_port=7860`
   - Change to: `server_port=7861` (or 7862, 7863, etc.)

2. **Kill existing process on port:**
   ```bash
   # Windows:
   netstat -ano | findstr :7860
   taskkill /PID [process_id] /F

   # Linux/macOS:
   lsof -ti:7860 | xargs kill -9
   ```

3. **Use different browser:**
   - Try Chrome, Firefox, Edge
   - Clear browser cache
   - Disable browser extensions

4. **Check firewall settings:**
   - Allow Python through Windows Firewall
   - Allow connections on port 7860

---

## Technical Reference

### VISA Addressing

VISA (Virtual Instrument Software Architecture) addresses uniquely identify instruments.

**Format:** `INTERFACE::VENDOR::PRODUCT::SERIAL::PROTOCOL`

**Components:**

| Field | Description | Example |
|-------|-------------|---------|
| **INTERFACE** | Connection type | `USB0`, `TCPIP0`, `GPIB0` |
| **VENDOR** | Manufacturer ID | `0x05E6` (Keithley/Tektronix)<br>`0x0957` (Keysight) |
| **PRODUCT** | Product model ID | `0x2230` (Keithley 2230)<br>`0x6500` (DMM6500) |
| **SERIAL** | Unique serial number | `805224014806770001` |
| **PROTOCOL** | Communication protocol | `INSTR` (instrument) |

**Examples:**

```
USB Connection:
USB0::0x05E6::0x2230::805224014806770001::INSTR

LAN/Ethernet Connection:
TCPIP0::192.168.128.175::inst0::INSTR

GPIB Connection (legacy):
GPIB0::12::INSTR
```

**Finding Your Instrument's VISA Address:**

1. **NI-MAX (National Instruments):**
   - Open NI Measurement & Automation Explorer
   - Devices and Interfaces → Scan for Instruments
   - Right-click instrument → Properties → VISA Resource Name

2. **Keysight Connection Expert:**
   - Launch application
   - Auto-Discover → Instrument appears with VISA address
   - Click instrument → Copy VISA address

3. **PyVISA (programmatic):**
   ```python
   import pyvisa
   rm = pyvisa.ResourceManager()
   resources = rm.list_resources()
   for addr in resources:
       print(addr)
   ```

### SCPI Command Reference

SCPI (Standard Commands for Programmable Instruments) is the industry-standard command language.

**Common Commands (All Instruments):**

| Command | Description | Response |
|---------|-------------|----------|
| `*IDN?` | Identify instrument | Manufacturer, model, serial, firmware |
| `*RST` | Reset to default state | (None) |
| `*CLS` | Clear status and error queue | (None) |
| `SYST:ERR?` | Read error queue | Error code and message |
| `*OPC?` | Query operation complete | `1` when ready |

**Power Supply Commands:**

```scpi
# Set voltage (Channel 1 to 5.0V)
APPLy:CH1 5.0,0.5

# Set current limit (Channel 1 to 500mA)
SOURce:CH1:CURRent:LIMit 0.5

# Enable output
OUTPut:CH1 ON

# Read voltage and current
MEASure:VOLTage:DC? CH1
MEASure:CURRent:DC? CH1

# Set over-voltage protection
SOURce:CH1:VOLTage:PROTection 6.0
```

**DMM Commands:**

```scpi
# Configure DC voltage measurement
CONFigure:VOLTage:DC 10.0

# Set integration time
SENSe:VOLTage:DC:NPLCycles 1.0

# Take single measurement
READ?

# Initiate continuous measurement
INITiate:CONTinuous ON
```

**Oscilloscope Commands:**

```scpi
# Set vertical scale (Channel 1 to 1V/div)
CHANnel1:SCALe 1.0

# Set timebase (1ms/div)
TIMebase:SCALe 0.001

# Set trigger level
TRIGger:EDGE:LEVel 2.5

# Capture waveform
DIGitize CHANnel1

# Read waveform data
WAVeform:SOURce CHANnel1
WAVeform:FORMat ASCII
WAVeform:DATA?
```

### Network Configuration

**Accessing Interface from Network:**

1. **Find computer's IP address:**
   ```bash
   # Windows:
   ipconfig
   # Look for "IPv4 Address"

   # macOS/Linux:
   ifconfig
   # Look for "inet" address
   ```

2. **Configure firewall:**
   - Windows Firewall → Allow an app
   - Add Python.exe
   - Allow connections on port 7860

3. **Access from remote computer:**
   - Browser: `http://[computer-IP]:7860`
   - Example: `http://192.168.128.175:7860`

**Multiple Users:**

The Gradio interface supports concurrent connections:
- Multiple browsers can view same interface
- Only one user should operate instrument at a time (avoid command conflicts)
- Use `share=True` parameter for public internet access (Gradio tunnel)

**Custom Port Configuration:**

Edit the application launch section:

```python
# In Unified.py or individual script:
interface.launch(
    server_port=7860,      # Change to 7861, 7862, etc.
    server_name="0.0.0.0", # Allow network access
    share=False,           # Set True for internet access
    inbrowser=True         # Auto-open browser
)
```

### Performance Optimization

**Measurement Speed:**

| Instrument | Typical Rate | Maximum Rate | Limiting Factor |
|------------|--------------|--------------|-----------------|
| DMM | 1-10 readings/sec | 500 readings/sec | NPLC, auto-zero |
| Power Supply | 10 settings/sec | 50 settings/sec | VISA overhead |
| Oscilloscope | 1 waveform/sec | 10 waveforms/sec | Data transfer size |

**Tips for Faster Operation:**

1. **DMM Fast Measurements:**
   ```python
   # Reduce NPLC
   nplc=0.01  # Instead of 1.0

   # Disable auto-zero
   auto_zero=False

   # Use fixed range (no auto-ranging)
   range=10.0  # Instead of "AUTO"
   ```

2. **Oscilloscope Fast Capture:**
   ```python
   # Reduce memory depth
   # Shorter acquisition time
   # Use ASCII format instead of binary (faster parsing)
   ```

3. **Network vs USB:**
   - USB: Better for small data transfers, commands
   - LAN: Better for large waveform transfers (oscilloscope)

---

## Support

### Documentation Resources

**Manufacturer Documentation:**

- **Keithley Power Supply 2230-30-1**
  - [User Manual](https://www.tek.com/en/products/keithley/low-voltage-dc-power-supplies/series-2200)
  - [Programming Reference](https://www.tek.com/en/products/keithley/low-voltage-dc-power-supplies/series-2200)

- **Keithley DMM6500/7510**
  - [DMM6500 Reference](https://www.tek.com/en/products/keithley/digital-multimeter/dmm6500)
  - [DMM7510 Reference](https://www.tek.com/en/products/keithley/digital-multimeter/dmm7510)

- **Keysight DSOX6004A**
  - [User Guide](https://www.keysight.com/us/en/product/DSOX6004A)
  - [Programming Guide](https://www.keysight.com/us/en/product/DSOX6004A)

**Software Documentation:**

- [PyVISA Documentation](https://pyvisa.readthedocs.io/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [SCPI Standard](https://www.ivifoundation.org/specifications/)

### Getting Help

**For Issues with This Software:**

- **Developer**: Anirudh Iyengar
- **Email**: anirudh.iyengar@digantara.co.in
- **Organization**: Digantara Research and Technologies Pvt. Ltd.

**When Requesting Support, Include:**

1. Operating system and version
2. Python version (`python --version`)
3. Instrument model and firmware version
4. Complete error message (copy from terminal)
5. Steps to reproduce the issue
6. Screenshot if applicable

**Community Resources:**

- [SCPI Programming Forum](https://www.eevblog.com/forum/)
- [PyVISA GitHub Issues](https://github.com/pyvisa/pyvisa/issues)
- [Test & Measurement Community](https://www.tek.com/en/support)

### Bug Reports

If you encounter a software bug, please report:

**Bug Report Template:**

```
TITLE: Brief description of the issue

ENVIRONMENT:
- OS: Windows 10 / macOS 12 / Ubuntu 22.04
- Python Version: 3.10.5
- Software Version: 1.0.0
- Instrument: Keithley DMM6500

STEPS TO REPRODUCE:
1. Launch Unified.py
2. Connect to DMM
3. Click "Start Continuous Measurement"
4. Error appears after 10 seconds

EXPECTED BEHAVIOR:
Continuous measurements should continue indefinitely

ACTUAL BEHAVIOR:
Error message: "VISA timeout"

ERROR MESSAGE:
[Copy complete error text here]

ADDITIONAL NOTES:
Works correctly for first 10 measurements, then fails
```

### Feature Requests

Suggestions for improvements are welcome! Contact the development team with:

- Description of desired feature
- Use case and motivation
- Expected behavior
- Example workflows

---

## License and Credits

**Project**: Digantara Instrumentation Control Suite
**Version**: 1.0.0
**Release Date**: 2025-11-18
**Status**: Production Ready

**Lead Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd.
**Department**: SPD
**Contact**: anirudh.iyengar@digantara.co.in

**License**: MIT License

```
Copyright (c) 2025 Digantara Research and Technologies Pvt. Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Dependencies

This software incorporates the following open-source libraries:

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| **PyVISA** | ≥1.13.0 | MIT | Instrument communication |
| **Gradio** | ≥4.0.0 | Apache 2.0 | Web interface framework |
| **NumPy** | ≥2.0.0 | BSD | Numerical computing |
| **Pandas** | ≥2.2.0 | BSD | Data manipulation |
| **Matplotlib** | ≥3.8.0 | PSF | Visualization and plotting |
| **Pillow** | ≥10.0.0 | PIL | Image processing |
| **PySerial** | ≥3.5 | BSD | Serial communication |
| **PyUSB** | ≥1.2.1 | BSD | USB device access |

See [requirements.txt](requirements.txt) for complete dependency list.

### Acknowledgments

Special thanks to:
- Tektronix/Keithley for instrument documentation and SCPI command references
- Keysight Technologies for oscilloscope programming guides
- PyVISA development team for excellent communication library
- Gradio team for intuitive web framework

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **ATE** | Automated Test Equipment - systems for automated testing and measurement |
| **GPIB** | General Purpose Interface Bus (IEEE-488) - legacy instrument communication standard |
| **NPLC** | Number of Power Line Cycles - measurement integration time in DMM |
| **OVP** | Over-Voltage Protection - safety feature limiting maximum output voltage |
| **PSU** | Power Supply Unit - device providing regulated electrical power |
| **RTD** | Resistance Temperature Detector - temperature sensor based on resistance change |
| **SCPI** | Standard Commands for Programmable Instruments - standard instrument command language |
| **Thermocouple** | Temperature sensor using two dissimilar metals |
| **VISA** | Virtual Instrument Software Architecture - standard API for instrument communication |
| **Waveform** | Time-varying electrical signal (voltage or current vs. time) |

---

**Document Version**: 1.0
**For Software Version**: 1.0.0

For the latest documentation and updates, contact Digantara Research and Technologies.
