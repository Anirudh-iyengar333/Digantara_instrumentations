# Digantara Instrumentation Control Suite

> **Professional Laboratory Instrument Control System**
> Web-based automation for Power Supplies, Multimeters, and Oscilloscopes
>
> **Developed by: Anirudh Iyengar**
> Digantara Research and Technologies Pvt. Ltd.

---

## What is This?

This software provides an easy-to-use **web interface** (works in your browser) to control laboratory test equipment. You don't need to be an engineer or programmer to use it - if you can use a web browser, you can use this software.

### Supported Instruments

| Instrument | Model | What it Does |
|------------|-------|--------------|
| **Power Supply** | Keithley 2230-30-1 | Provides precise electrical power (voltage and current) to test circuits |
| **Digital Multimeter (DMM)** | Keithley DMM6500/DMM7510 | Measures voltage, current, resistance with high precision |
| **Oscilloscope** | Keysight DSOX6004A | Captures and displays electrical waveforms (signals over time) |

---

## Quick Start Guide

### Step 1: Installation

#### For Non-Technical Users

1. **Install Python** (if not already installed):
   - Download Python 3.8 or newer from [python.org](https://www.python.org/downloads/)
   - During installation, **check the box** that says "Add Python to PATH"
   - Click "Install Now"

2. **Open Command Prompt**:
   - Press `Windows Key + R`
   - Type `cmd` and press Enter
   - A black window will appear

3. **Navigate to the Project Folder**:
   ```bash
   cd "path\to\Digantara_instrumentation"
   ```
   Replace `path\to\Digantara_instrumentation` with the actual folder location

4. **Install Required Software**:
   ```bash
   pip install -r requirements.txt
   ```
   This will download and install all necessary components (takes 2-5 minutes)

5. **Install VISA Drivers**:
   - Download and install **NI-VISA** from [National Instruments](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
   - This allows your computer to communicate with lab instruments via USB

### Step 2: Connect Your Instrument

1. Connect your instrument to the computer using a **USB cable**
2. Turn on the instrument
3. Wait for Windows to recognize the device (you may hear a USB connection sound)

### Step 3: Launch the Web Interface

#### Option A: Unified Interface (Control All Instruments)

```bash
python Unified.py
```

#### Option B: Individual Instrument Control

**For Power Supply:**
```bash
python scripts\keithley\keithley_PSU_gradio_automation.py
```

**For Digital Multimeter:**
```bash
python scripts\keithley\keithley_dmm_gradio_automation.py
```

**For Oscilloscope:**
```bash
python scripts\keysight\keysight_oscilloscope_gradio_en.py
```

### Step 4: Open Your Web Browser

After running the command, you will see output like:
```
Running on local URL:  http://127.0.0.1:7860
Running on network:    http://192.168.1.XXX:7860
```

1. Open your web browser (Chrome, Firefox, Edge)
2. Go to: **http://localhost:7860**
3. You will see the control interface

---

## User Guides

### For Power Supply Users

The Power Supply interface lets you:

- **Set Output Voltage**: Control how much voltage (electrical pressure) to provide
- **Set Current Limit**: Limit maximum current (electrical flow) to protect your device
- **Monitor Measurements**: See actual voltage, current, and power in real-time
- **Run Automated Tests**: Create voltage waveforms (patterns) for testing

#### Basic Operation

1. **Enter VISA Address**:
   - Look like: `USB0::0x05E6::0x2230::SERIAL_NUMBER::INSTR`
   - Usually pre-filled with the correct address
   - Click "Connect"

2. **Configure a Channel**:
   - Choose Channel 1, 2, or 3
   - Set desired **Voltage** (in Volts)
   - Set **Current Limit** (in Amperes) - acts as a safety limit
   - Set **OVP** (Over-Voltage Protection) - emergency shutoff voltage
   - Click "Configure"

3. **Turn On the Output**:
   - Click "Enable Output" for your channel
   - The "Status" will show "ON"

4. **Monitor the Output**:
   - Click "Measure" to see actual voltage and current
   - Or click "Measure All Channels" to check all outputs at once

5. **Turn Off When Done**:
   - Click "Disable Output" on each channel
   - Or click "EMERGENCY STOP" to turn everything off immediately

#### Safety Features

- **Emergency Stop Button**: Turns off all outputs immediately
- **Over-Voltage Protection (OVP)**: Automatically shuts off if voltage exceeds safe limit
- **Current Limit**: Prevents excessive current flow

---

### For Digital Multimeter Users

The DMM interface provides precision measurements.

#### What Can You Measure?

- **DC Voltage**: Steady voltage (like batteries)
- **AC Voltage**: Alternating voltage (like wall outlets)
- **DC Current**: Steady current flow
- **AC Current**: Alternating current flow
- **Resistance**: How much something resists electrical flow (in Ohms)
- **Capacitance**: Ability to store electrical charge
- **Frequency**: How fast a signal alternates
- **Temperature**: Temperature measurements (with thermocouple)

#### Basic Operation

1. **Connect to Instrument**:
   - Enter VISA Address (usually starts with `USB0::0x05E6::0x6500::`)
   - Click "Connect"

2. **Choose Measurement Type**:
   - Select from dropdown: "DC_VOLTAGE", "AC_VOLTAGE", etc.
   - Set **Range**: Maximum value you expect (use auto-range if unsure)
   - Set **NPLC**: Integration time (higher = more accurate but slower)
     - 0.01 = Fast (noisy)
     - 1.0 = Standard (balanced)
     - 10.0 = Slow (very accurate)

3. **Take Measurement**:
   - Click "Single Measurement" for one reading
   - Or click "Start Continuous" for ongoing measurements

4. **View Results**:
   - Current reading shows with proper units (V, A, Ω, etc.)
   - Statistics tab shows average, min, max, standard deviation
   - Trend plot shows measurements over time

5. **Export Data**:
   - Go to "Data Export" tab
   - Choose format (CSV for Excel, JSON for programming)
   - Click "Export Data"
   - File saves to your outputs folder

---

### For Oscilloscope Users

The Oscilloscope captures and displays electrical signals.

#### Basic Operation

1. **Connect**:
   - Enter IP Address or VISA Address of oscilloscope
   - Click "Connect"

2. **Configure Channels**:
   - Enable Channel 1, 2, 3, or 4
   - Set **Vertical Scale**: Volts per division (how tall waveforms appear)
   - Set **Vertical Offset**: Move waveform up/down on screen
   - Set **Probe Attenuation**: Usually 1X or 10X (check your probe)

3. **Set Timebase**:
   - Horizontal scale: Time per division
   - Controls how "zoomed in" you are on the time axis
   - Examples: "1 ms" = 1 millisecond per division

4. **Configure Trigger**:
   - **Trigger Source**: Which channel to use for triggering
   - **Trigger Level**: Voltage level that starts capture
   - **Slope**: Rising (going up) or Falling (going down)

5. **Capture Waveform**:
   - Click "Acquire Waveform" for selected channel
   - Waveform graph appears
   - Click "Save Waveform" to save to file

6. **Take Screenshot**:
   - Click "Capture Screenshot"
   - Saves current oscilloscope screen as image

---

## Common Issues and Solutions

### Problem: "Cannot connect to instrument"

**Solutions:**
1. Check USB cable is firmly connected
2. Make sure instrument is powered on
3. Check Windows Device Manager:
   - Press `Windows Key + X` > Device Manager
   - Look under "Universal Serial Bus controllers"
   - If you see yellow warning triangle, update drivers
4. Try a different USB port
5. Restart the instrument and reconnect

### Problem: "VISA address not found"

**Solutions:**
1. Install NI-VISA software (see Installation section)
2. Use NI-MAX (National Instruments Measurement & Automation Explorer) to find connected devices:
   - Open NI-MAX from Start menu
   - Expand "Devices and Interfaces"
   - Right-click and "Scan for Instruments"
   - Copy the VISA address shown

### Problem: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```
This reinstalls all necessary components

### Problem: Web page won't load

**Solutions:**
1. Check the command prompt for errors
2. Try a different port:
   - Edit the Python file
   - Find `server_port=7860`
   - Change to `server_port=7861` (or another number)
3. Check if firewall is blocking the port

### Problem: Measurements seem incorrect

**Solutions:**
1. Check cable connections are secure
2. Verify measurement range is appropriate
3. For DMM: Increase NPLC for more stable readings
4. For oscilloscope: Adjust vertical scale and trigger level
5. Check probe attenuation matches settings

---

## Understanding the Interface

### What is a VISA Address?

A VISA address is like a "phone number" for test equipment. It uniquely identifies your instrument.

**Format:** `USB0::VendorID::ProductID::SerialNumber::INSTR`

**Example:** `USB0::0x05E6::0x2230::805224014806770001::INSTR`

- `USB0` = Connection type (USB)
- `0x05E6` = Vendor ID (Keithley/Keysight)
- `0x2230` = Product model
- `805224014806770001` = Unique serial number
- `INSTR` = Instrument type

### What is NPLC?

NPLC (Number of Power Line Cycles) controls measurement integration time:

- **Higher NPLC** = More accurate, slower (averages more samples)
- **Lower NPLC** = Less accurate, faster (takes quick snapshot)

**Recommendations:**
- Lab environment: 1-10 NPLC
- Production testing: 0.1-1 NPLC
- High precision: 10 NPLC

### What is Over-Voltage Protection (OVP)?

OVP is a safety feature that **automatically shuts off** the power supply if voltage exceeds a set threshold. This protects sensitive electronics from damage.

**Example:** If testing a 5V circuit, set OVP to 6V. If voltage accidentally goes above 6V, the supply turns off.

---

## File Organization

```
Digantara_instrumentation/
│
├── Unified.py                          # Main interface (all instruments)
│
├── instrument_control/                 # Low-level instrument drivers
│   ├── keithley_power_supply.py       # Power supply driver
│   ├── keithley_dmm.py                 # Digital multimeter driver
│   └── keysight_oscilloscope.py        # Oscilloscope driver
│
├── scripts/                            # Individual instrument GUIs
│   ├── keithley/
│   │   ├── keithley_PSU_gradio_automation.py     # Power supply GUI
│   │   └── keithley_dmm_gradio_automation.py      # DMM GUI
│   └── keysight/
│       └── keysight_oscilloscope_gradio_en.py     # Oscilloscope GUI
│
├── requirements.txt                    # Required Python packages
├── setup.py                            # Installation configuration
└── README.md                           # This file
```

---

## Data Export and Analysis

### Exporting Data

All interfaces support data export in multiple formats:

1. **CSV** (Comma-Separated Values):
   - Opens in Excel, Google Sheets
   - Easy to analyze and share
   - Best for most users

2. **JSON** (JavaScript Object Notation):
   - For programming and automation
   - Structured data format

3. **Excel** (.xlsx):
   - Native Excel format
   - Includes formatting

### Finding Exported Files

Exported files are saved to:
- Power Supply: `voltage_ramp_data/` folder
- DMM: Current working directory
- Oscilloscope: `outputs/` folder

Files are named with timestamps:
- Example: `dmm_data_20250119_143022.csv`
- Format: `instrument_data_YYYYMMDD_HHMMSS.ext`

---

## Advanced Features

### Automated Voltage Ramping (Power Supply)

Create custom voltage patterns for automated testing:

**Available Waveforms:**
- **Sine Wave**: Smooth up and down curve
- **Square Wave**: Sharp on/off pattern
- **Triangle Wave**: Linear increase and decrease
- **Ramp Up**: Gradual increase from 0 to max
- **Ramp Down**: Gradual decrease from max to 0

**Configuration:**
1. Select waveform type
2. Set target voltage (maximum voltage to reach)
3. Set number of cycles (how many repetitions)
4. Set cycle duration (time for one complete pattern)
5. Click "Start Ramping"

**Use Cases:**
- Battery charging simulations
- Temperature cycling tests
- Device stress testing
- Power-up sequence validation

### Statistical Analysis (DMM)

The Statistics tab provides:

- **Count**: Number of measurements
- **Mean**: Average value
- **Std Dev**: Standard deviation (consistency measure)
- **Min/Max**: Range of values
- **Trend Plot**: Visual representation over time

**Interpreting Results:**
- Low standard deviation = Stable, consistent measurements
- High standard deviation = Noisy or varying signal
- Trend plot shows if signal is drifting over time

---

## Safety Guidelines

### General Safety

1. **Always start with low voltage/current** and gradually increase
2. **Use appropriate safety equipment** (safety glasses if working with high voltage)
3. **Never exceed equipment ratings** listed in instrument manual
4. **Keep workspace clean and organized**
5. **Turn off outputs before connecting/disconnecting devices**

### Emergency Procedures

**If you see/smell smoke or sparks:**
1. Click "EMERGENCY STOP" immediately
2. Turn off instrument power switch
3. Unplug instrument if safe to do so
4. Do not touch equipment until it cools down
5. Report incident to supervisor

**If measurement seems wrong:**
1. Disable outputs immediately
2. Check all connections
3. Verify settings are appropriate
4. Consult instrument manual or supervisor

---

## Getting Help

### Documentation Resources

1. **Instrument Manuals**: Check manufacturer websites
   - Keithley: [www.tek.com/keithley](https://www.tek.com/keithley)
   - Keysight: [www.keysight.com](https://www.keysight.com)

2. **Python Help**: If you see Python errors
   - [Python Official Docs](https://docs.python.org/3/)
   - [Stack Overflow](https://stackoverflow.com/)

3. **VISA/Drivers**:
   - [NI-VISA Documentation](https://www.ni.com/en-us/support/documentation/supplemental/06/ni-visa-overview.html)

### Support Contacts

For issues with this software, contact:
- **Email**: anirudh.iyengar@digantara.co.in
- **Project Page**: Digantara Research and Technologies

### Reporting Bugs

If you find a problem:
1. Note what you were doing when it happened
2. Copy any error messages (from command prompt)
3. Note instrument model and connection type
4. Contact support with this information

---

## Glossary of Terms

| Term | Explanation |
|------|-------------|
| **Voltage (V)** | Electrical "pressure" or potential difference |
| **Current (A)** | Flow rate of electrical charge |
| **Resistance (Ω)** | Opposition to electrical flow |
| **Power (W)** | Energy transfer rate (Voltage × Current) |
| **VISA** | Virtual Instrument Software Architecture - standard for instrument communication |
| **GUI** | Graphical User Interface - the visual controls you interact with |
| **Waveform** | Visual representation of electrical signal over time |
| **Trigger** | Event that starts oscilloscope capture |
| **Timebase** | Horizontal scale on oscilloscope (time per division) |
| **Probe Attenuation** | Reduction factor of oscilloscope probe (1X, 10X, 100X) |
| **CSV** | Comma-Separated Values - spreadsheet file format |
| **NPLC** | Number of Power Line Cycles - measurement integration time |

---

## Frequently Asked Questions (FAQ)

### Q: Do I need to be a programmer to use this?

**A:** No! The web interface is designed for non-technical users. Just follow the step-by-step instructions above.

### Q: Can multiple people use this at the same time?

**A:** Yes, if they're on the same network. The software shows a network address (like `http://192.168.1.XXX:7860`) that others can access from their browsers. However, only one person should control the instrument at a time to avoid conflicts.

### Q: Will this work on Mac or Linux?

**A:** Yes! Python and the required libraries work on all platforms. However, you may need to adjust file paths (use `/` instead of `\` on Mac/Linux).

### Q: How do I update the software?

**A:** If you received this via Git:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

If you received files directly, ask for the latest version from your supervisor.

### Q: What if my instrument model is different?

**A:** This software is designed for specific models (Keithley 2230, DMM6500/7510, Keysight DSOX6004A). Similar models may work but aren't guaranteed. Contact support for compatibility questions.

### Q: Can I run this on multiple instruments simultaneously?

**A:** Yes! You can run multiple instances of the software on different ports:
1. Run first instrument on port 7860
2. Run second instrument - edit script to use port 7861
3. Run third instrument - edit script to use port 7862

Each opens in a different browser tab.

### Q: How long can I keep the software running?

**A:** The software can run continuously. However, it's good practice to:
- Close it when not in use
- Restart periodically (daily/weekly) to clear memory
- Disconnect instruments when finished testing

---

## Appendix: Technical Information

### System Requirements

**Minimum:**
- Windows 7/10/11, macOS 10.14+, or Linux
- Python 3.8 or newer
- 4 GB RAM
- USB 2.0 port
- Web browser (Chrome, Firefox, Edge, Safari)

**Recommended:**
- Windows 10/11
- Python 3.10 or newer
- 8 GB RAM
- USB 3.0 port
- Google Chrome browser

### Network Configuration

The software runs a local web server:
- **Default Port**: 7860
- **Access locally**: http://localhost:7860
- **Access from network**: http://[computer-ip]:7860

To find your computer's IP address:
- Windows: `ipconfig` in command prompt
- Mac/Linux: `ifconfig` in terminal

### Port Configuration

If port 7860 is in use, edit the Python file:

Find the line:
```python
server_port=7860
```

Change to an available port:
```python
server_port=7861  # or 7862, 7863, etc.
```

### Supported Python Versions

- **Python 3.8**: Minimum supported
- **Python 3.9-3.11**: Fully tested
- **Python 3.12-3.13**: Compatible
- **Python 2.x**: NOT supported

---

## License and Credits

**Project**: Digantara Instrumentation Control Suite
**Lead Developer**: Anirudh Iyengar
**Organization**: Digantara Research and Technologies Pvt. Ltd
**Contact**: anirudh.iyengar@digantara.co.in
**License**: MIT License
**Version**: 1.0.0

### Third-Party Libraries

This software uses the following open-source libraries:
- PyVISA - Instrument control
- Gradio - Web interface
- NumPy - Numerical computing
- Pandas - Data analysis
- Matplotlib - Visualization

See `requirements.txt` for complete list with versions.

---

## Document Version

**Document Version**: 1.0
**For Software Version**: 1.0.0

If you have suggestions for improving this documentation, please contact the development team.
