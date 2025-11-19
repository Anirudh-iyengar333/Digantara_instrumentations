# Usage Workflows and Scenarios
## Common Tasks and How to Accomplish Them

> Practical examples of real-world testing scenarios
>
> **Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.

---

## Table of Contents

1. [Daily Workflow Checklist](#daily-workflow-checklist)
2. [Power Supply Workflows](#power-supply-workflows)
3. [Digital Multimeter Workflows](#digital-multimeter-workflows)
4. [Oscilloscope Workflows](#oscilloscope-workflows)
5. [Combined Instrument Workflows](#combined-instrument-workflows)
6. [Data Analysis Workflows](#data-analysis-workflows)
7. [Troubleshooting Workflows](#troubleshooting-workflows)

---

## Daily Workflow Checklist

### Morning Setup (5 minutes)

```
┌─────────────────────────────────────┐
│ 1. Physical Setup                   │
│    □ Connect instruments via USB    │
│    □ Power on instruments           │
│    □ Verify front panel displays    │
│    □ Wait for initialization (30s)  │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 2. Software Launch                  │
│    □ Open Command Prompt/Terminal   │
│    □ Navigate to project folder     │
│    □ Launch: python Unified.py      │
│    □ Wait for "Running on..." msg   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 3. Web Interface                    │
│    □ Browser opens automatically    │
│    □ Or go to localhost:7860        │
│    □ Interface loads successfully   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 4. Instrument Connection            │
│    □ Enter VISA address (pre-fill)  │
│    □ Click "Connect" button         │
│    □ Verify "Connected" status      │
│    □ Click "Test Connection"        │
└─────────────────────────────────────┘
                  ↓
          ✓ Ready to Use!
```

### End of Day Shutdown (2 minutes)

```
┌─────────────────────────────────────┐
│ 1. Safety First                     │
│    □ Disable ALL outputs            │
│    □ Disconnect test devices        │
│    □ Verify outputs are OFF         │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 2. Data Management                  │
│    □ Export important data          │
│    □ Save screenshots if needed     │
│    □ Note test results in logbook   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 3. Software Shutdown                │
│    □ Close browser tab              │
│    □ Press Ctrl+C in terminal       │
│    □ Wait for program to exit       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 4. Instrument Shutdown              │
│    □ Power off instruments          │
│    □ Disconnect USB cables          │
│    □ Cover instruments (dust)       │
└─────────────────────────────────────┘
```

---

## Power Supply Workflows

### Workflow 1: Basic Device Power Test

**Scenario**: Test a microcontroller board that needs 5V @ 200mA

```
START
  │
  ├─→ [1] Select Channel (Ch1 for this example)
  │
  ├─→ [2] Configure Settings
  │     ├─ Voltage: 5.0 V
  │     ├─ Current Limit: 0.3 A (50% margin above 200mA)
  │     └─ OVP: 5.5 V (10% above voltage)
  │
  ├─→ [3] Click "Configure Ch1"
  │     └─ Wait for "Success" message
  │
  ├─→ [4] Click "Enable Output"
  │     └─ Status: OFF → ON
  │
  ├─→ [5] Verify Output (before connecting device)
  │     ├─ Click "Measure"
  │     ├─ Check: Voltage ≈ 5.0V
  │     └─ Check: Current ≈ 0.000A (no load)
  │
  ├─→ [6] Connect Your Device
  │     └─ Mind polarity: Red(+) Black(-)
  │
  ├─→ [7] Check Operation
  │     ├─ Click "Measure" again
  │     ├─ Voltage should stay ~5.0V
  │     ├─ Current should be ~0.200A
  │     └─ Power ~1.0W
  │
  ├─→ [8] Run Your Test
  │     ├─ Device operates normally
  │     └─ Monitor measurements periodically
  │
  ├─→ [9] When Done
  │     ├─ Click "Disable Output"
  │     ├─ Wait for status: ON → OFF
  │     └─ Safely disconnect device
  │
END
```

**Tips**:
- Always verify voltage WITHOUT load first
- Set current limit with safety margin
- Monitor current to detect shorts
- If current hits limit, voltage will drop (protection working!)

---

### Workflow 2: Multi-Voltage Circuit Test

**Scenario**: Circuit needs +12V, +5V, and -5V simultaneously

```
┌────────────────────────────────────────────┐
│ Channel 1: +12V Main Supply                │
│   ├─ Voltage: 12.0 V                       │
│   ├─ Current Limit: 1.0 A                  │
│   ├─ OVP: 13.0 V                           │
│   └─ Configure → Enable                    │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ Channel 2: +5V Logic Supply                │
│   ├─ Voltage: 5.0 V                        │
│   ├─ Current Limit: 0.5 A                  │
│   ├─ OVP: 5.5 V                            │
│   └─ Configure → Enable                    │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ Channel 3: Ground Reference                │
│   ├─ Use common ground from all channels   │
│   └─ No configuration needed                │
└────────────────────────────────────────────┘

**For -5V**: You'll need an external inverting circuit
or use two channels with floating outputs
(Advanced: consult with supervisor)

┌────────────────────────────────────────────┐
│ Verification Steps                         │
│   1. Click "Measure All Channels"          │
│   2. Verify all voltages correct           │
│   3. Check total current < limits          │
│   4. Monitor for stability                 │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ Connection Order (Important!)              │
│   1. Configure all channels (outputs OFF)  │
│   2. Connect GND first                     │
│   3. Connect power wires                   │
│   4. Enable Channel 2 (+5V logic) first    │
│   5. Enable Channel 1 (+12V main) second   │
│   6. Check all voltages stable             │
│   7. Power on your circuit                 │
└────────────────────────────────────────────┘
```

---

### Workflow 3: Automated Voltage Stress Test

**Scenario**: Cycle voltage 0-3V to test device under varying power

```
PREPARATION
  ├─→ Select "Voltage Ramping" section
  ├─→ Choose Waveform: "Sine" (smooth variation)
  ├─→ Target Voltage: 3.0 V
  ├─→ Cycles: 10 (10 repetitions)
  ├─→ Cycle Duration: 60 s (1 minute per cycle)
  └─→ Active Channel: 1

CONFIGURATION
  ├─→ Enable auto-measurement
  ├─→ Measurement interval: 5 seconds
  └─→ This gives you 12 readings per cycle

EXECUTION
  ├─→ Connect device to Channel 1
  ├─→ Click "Start Ramping"
  ├─→ Monitor real-time graph
  │     ├─ Blue line: Commanded voltage
  │     ├─ Red line: Actual voltage
  │     └─ Should track closely
  ├─→ Total time: 10 min (10 cycles × 1 min)
  └─→ Graph updates every 5 seconds

DURING TEST
  ├─→ Watch for anomalies
  │     ├─ Current spikes
  │     ├─ Voltage drops
  │     └─ Protection triggers
  ├─→ Emergency stop available if needed
  └─→ Take notes of observations

COMPLETION
  ├─→ Ramping finishes automatically
  ├─→ Final graph displayed
  ├─→ Data saved to CSV automatically
  ├─→ Location: voltage_ramp_data/
  │     └─ File: psu_ramping_YYYYMMDD_HHMMSS.csv
  ├─→ Graph saved to voltage_ramp_graphs/
  │     └─ File: voltage_ramp_YYYYMMDD_HHMMSS.png
  └─→ Export additional data if needed

ANALYSIS
  ├─→ Open CSV in Excel
  ├─→ Create pivot tables
  ├─→ Plot current vs voltage
  ├─→ Calculate average power
  ├─→ Identify any failures
  └─→ Document findings
```

---

## Digital Multimeter Workflows

### Workflow 1: Quick DC Voltage Check

**Scenario**: Measure battery voltage

```
SETUP (30 seconds)
  ├─→ Connect DMM via USB
  ├─→ Launch DMM software
  ├─→ Browser: localhost:7862
  ├─→ Click "Connect"
  └─→ Status: Connected

CONFIGURE (15 seconds)
  ├─→ Measurement Function: "DC_VOLTAGE"
  ├─→ Range: 100 (for 12V battery)
  ├─→ NPLC: 1.0 (standard accuracy)
  └─→ Auto Zero: ✓ Checked

MEASURE (5 seconds)
  ├─→ Connect red probe to battery +
  ├─→ Connect black probe to battery -
  ├─→ Click "Single Measurement"
  └─→ Result displays: e.g., "12.634 V"

INTERPRET
  ├─→ Fully charged 12V battery: 12.6-13.2V
  ├─→ Nominal charge: 12.2-12.6V
  ├─→ Needs charging: <12.0V
  └─→ Dead/damaged: <11.5V

OPTIONAL: Track Over Time
  ├─→ Set interval: 60 seconds (1 reading/min)
  ├─→ Click "Start Continuous"
  ├─→ Let run for desired duration
  ├─→ Click "Stop Continuous" when done
  ├─→ Go to "Statistics & Analysis"
  ├─→ Click "Calculate Statistics"
  ├─→ See: Mean, Min, Max, Std Dev
  └─→ Export data if needed
```

---

### Workflow 2: Precision Resistance Measurement

**Scenario**: Measure a precision resistor (expected ~1kΩ)

```
INSTRUMENT WARMUP
  ├─→ Turn on DMM
  ├─→ Wait 15 minutes for thermal stability
  └─→ Critical for precision measurements

CONNECTION METHOD
  ├─→ 2-Wire (for R > 1kΩ): Fast, less accurate
  │     ├─ Use front panel HI and LO terminals
  │     └─ Good for ≥1kΩ resistors
  │
  └─→ 4-Wire (for R < 1kΩ): Slower, more accurate
        ├─ Use front panel 4W terminals
        ├─ Eliminates lead resistance
        └─ Essential for mΩ measurements

CONFIGURATION (2-Wire Method)
  ├─→ Measurement Function: "RESISTANCE_2W"
  ├─→ Range: 10000 (10kΩ range for 1kΩ resistor)
  ├─→ Resolution: 0.001 (1 mΩ resolution)
  ├─→ NPLC: 10.0 (high accuracy)
  └─→ Auto Zero: ✓ Checked

MEASUREMENT PROCEDURE
  ├─→ [1] Open Circuit Test (baseline)
  │     ├─ Probes NOT connected to resistor
  │     ├─ Click "Single Measurement"
  │     └─ Should read very high (>10 MΩ) or overflow
  │
  ├─→ [2] Connect Resistor
  │     ├─ Connect one lead to HI terminal
  │     ├─ Connect other lead to LO terminal
  │     └─ Ensure good contact
  │
  ├─→ [3] Take Multiple Readings
  │     ├─ Click "Start Continuous"
  │     ├─ Interval: 1.0 second
  │     ├─ Take 100 readings (100 seconds)
  │     └─ Click "Stop Continuous"
  │
  └─→ [4] Calculate Statistics
        ├─ Go to "Statistics & Analysis" tab
        ├─ Points: 100
        ├─ Click "Calculate Statistics"
        ├─ Record:
        │     ├─ Mean: e.g., 1.0023 kΩ
        │     ├─ Std Dev: e.g., 0.0002 kΩ
        │     ├─ Min: 1.0020 kΩ
        │     └─ Max: 1.0026 kΩ
        └─ Export data for documentation

ACCURACY CHECK
  ├─→ Compare to resistor tolerance
  ├─→ Example: 1kΩ ±1% resistor
  │     ├─ Acceptable range: 990Ω - 1010Ω
  │     ├─ Measured: 1002.3Ω
  │     └─ Result: PASS (within tolerance)
  └─→ Document measurement uncertainty
```

---

### Workflow 3: Continuous Monitoring with Alerts

**Scenario**: Monitor supply voltage, alert if out of range

```
SETUP CONTINUOUS MONITORING
  ├─→ Configure for DC_VOLTAGE
  ├─→ Range: Appropriate for expected voltage
  ├─→ NPLC: 1.0 (balance speed and accuracy)
  ├─→ Interval: 5 seconds
  └─→ Click "Start Continuous"

MONITORING DASHBOARD
  ├─→ Keep "Statistics & Analysis" tab open
  ├─→ Update statistics every 30 seconds
  ├─→ Watch trend plot for patterns
  └─→ Note any sudden changes

WHAT TO WATCH FOR
  ├─→ Mean value drift
  │     └─ Indicates power supply regulation issue
  │
  ├─→ High standard deviation
  │     └─ Indicates noise or instability
  │
  ├─→ Trend direction
  │     ├─ Upward: Possible overheating
  │     └─ Downward: Possible battery discharge
  │
  └─→ Outliers (min/max far from mean)
        └─ Possible transient events or spikes

AUTOMATED DATA COLLECTION
  ├─→ Let run for extended period
  │     ├─ 1 hour for short-term stability
  │     ├─ 8 hours for work shift
  │     └─ 24 hours for long-term drift
  │
  ├─→ Periodically check
  │     ├─ Visual check of current reading
  │     ├─ Quick glance at trend plot
  │     └─ Emergency stop if anomaly seen
  │
  └─→ At completion
        ├─ Click "Stop Continuous"
        ├─ Export data (CSV format)
        ├─ Analyze in Excel/Python
        └─ Generate report

MANUAL ALERTING
  ├─→ Set expected range (e.g., 5V ±0.1V)
  ├─→ Periodically check statistics
  ├─→ If mean outside range:
  │     ├─ Take immediate reading
  │     ├─ Verify with external meter
  │     ├─ Document conditions
  │     └─ Investigate cause
  └─→ Log all out-of-spec events
```

---

## Oscilloscope Workflows

### Workflow 1: Capture Single Waveform

**Scenario**: Capture and analyze a 1kHz square wave

```
PHYSICAL SETUP
  ├─→ Connect oscilloscope via USB/LAN
  ├─→ Connect probe to Channel 1
  ├─→ Set probe attenuation switch (1X or 10X)
  ├─→ Connect probe to signal source
  └─→ Turn on oscilloscope

SOFTWARE CONNECTION
  ├─→ Launch oscilloscope interface
  ├─→ Enter VISA/IP address
  ├─→ Click "Connect"
  └─→ Verify connection status

CHANNEL CONFIGURATION
  ├─→ Select Channel 1
  ├─→ Enable: ✓ Checked
  ├─→ Vertical Scale: 1 V/div
  │     (Adjust to fit signal on screen)
  ├─→ Vertical Offset: 0 V
  ├─→ Probe Attenuation: Match probe switch
  │     ├─ If probe set to 10X → select 10X
  │     └─ If probe set to 1X → select 1X
  └─→ Coupling: DC

TIMEBASE SETUP
  ├─→ For 1kHz signal (1ms period):
  ├─→ Horizontal Scale: 200 µs/div
  │     (Shows ~5 cycles on screen)
  ├─→ This gives good detail
  └─→ Adjust as needed to see waveform clearly

TRIGGER CONFIGURATION
  ├─→ Trigger Source: Channel 1
  ├─→ Trigger Mode: Edge
  ├─→ Trigger Level: 50% of signal amplitude
  │     ├─ For 5V square wave: set to 2.5V
  │     └─ Adjust if signal not stable
  ├─→ Trigger Slope: Rising
  └─→ Click "Set Trigger"

ACQUISITION
  ├─→ Verify trigger is working
  │     └─ "TRIG'D" indicator on scope
  ├─→ Click "Acquire Waveform"
  ├─→ Wait for acquisition complete
  ├─→ Waveform displays in graph
  └─→ Check waveform looks correct

ANALYSIS
  ├─→ Use oscilloscope measurements:
  │     ├─ Frequency: Should be ~1 kHz
  │     ├─ Period: Should be ~1 ms
  │     ├─ Amplitude: Check peak-to-peak voltage
  │     ├─ Duty Cycle: Should be 50% for square wave
  │     ├─ Rise Time: Check edge speed
  │     └─ Fall Time: Check edge speed
  │
  └─→ Save waveform
        ├─ Click "Save Waveform"
        ├─ Choose format: CSV (data) or PNG (image)
        ├─ File saved to outputs folder
        └─ Can analyze in Excel/MATLAB/Python

SCREENSHOT
  ├─→ Click "Capture Screenshot"
  ├─→ Gets actual scope screen image
  ├─→ Useful for documentation
  └─→ Include in test reports
```

---

### Workflow 2: Comparing Two Signals

**Scenario**: Compare input vs output of an amplifier

```
SETUP
  ├─→ Channel 1: Amplifier Input
  │     ├─ Connect probe to input signal
  │     ├─ Vertical Scale: Appropriate for input level
  │     └─ Color: Yellow (typical)
  │
  └─→ Channel 2: Amplifier Output
        ├─ Connect probe to output signal
        ├─ Vertical Scale: Appropriate for output level
        └─ Color: Cyan (typical)

ENABLE BOTH CHANNELS
  ├─→ Channel 1: ✓ Enabled
  ├─→ Channel 2: ✓ Enabled
  └─→ Both waveforms visible

TRIGGER ON INPUT
  ├─→ Trigger Source: Channel 1 (input)
  ├─→ Trigger Level: 50% of input amplitude
  ├─→ Ensures stable display
  └─→ Output automatically synced

ACQUIRE BOTH
  ├─→ Click "Acquire All Channels"
  ├─→ Both waveforms captured
  └─→ Overlaid on same timebase

ANALYSIS
  ├─→ Visual Comparison:
  │     ├─ Check phase relationship
  │     ├─ Input and output in phase?
  │     ├─ Or 180° out of phase (inverting amp)?
  │     └─ Note any delay (propagation time)
  │
  ├─→ Amplitude Comparison:
  │     ├─ Measure input amplitude (Ch1)
  │     ├─ Measure output amplitude (Ch2)
  │     ├─ Calculate gain: Vout / Vin
  │     └─ Example: 5V out / 0.5V in = 10x gain
  │
  └─→ Distortion Check:
        ├─ Does output look like input?
        ├─ Or is it clipped/distorted?
        ├─ Check for ringing or overshoot
        └─ Note any anomalies

MATH FUNCTIONS (if available)
  ├─→ Create difference: Ch2 - Ch1
  │     └─ Shows amplifier contribution only
  ├─→ Create ratio: Ch2 / Ch1
  │     └─ Shows gain across waveform
  └─→ Export all waveforms for analysis
```

---

## Combined Instrument Workflows

### Workflow: Power Supply + DMM Verification

**Scenario**: Verify power supply accuracy using DMM

```
SETUP
  ├─→ Connect both instruments
  ├─→ Launch Unified.py (control both)
  └─→ Connect to both instruments

POWER SUPPLY SETUP
  ├─→ Configure Channel 1 for 5.000V
  ├─→ Current limit: 0.5A
  ├─→ Enable output
  └─→ PSU displays: 5.000V

DMM SETUP
  ├─→ Configure DC Voltage measurement
  ├─→ Range: 10V
  ├─→ NPLC: 10 (high accuracy)
  └─→ Auto-zero: Enabled

CONNECTION
  ├─→ DMM HI probe → PSU Channel 1 positive
  ├─→ DMM LO probe → PSU Channel 1 negative
  └─→ Create voltage measurement loop

MEASUREMENT PROCEDURE
  ├─→ [1] PSU Reading
  │     ├─ Click "Measure" on PSU
  │     └─ Record: e.g., 5.001V
  │
  ├─→ [2] DMM Reading
  │     ├─ Click "Single Measurement" on DMM
  │     └─ Record: e.g., 4.9985V
  │
  ├─→ [3] Calculate Difference
  │     ├─ Error = DMM - PSU
  │     ├─ Example: 4.9985 - 5.001 = -0.0025V
  │     └─ Percentage: (-0.0025/5.000) × 100 = -0.05%
  │
  └─→ [4] Repeat at Multiple Voltages
        ├─ Test points: 1V, 3V, 5V, 10V, 15V, 20V, 30V
        ├─ Record all readings
        ├─ Create calibration table
        └─ Document any deviations

ACCEPTANCE CRITERIA
  ├─→ PSU spec: Typically ±0.03% + 10mV
  ├─→ DMM spec: Typically ±0.01% for DC V
  ├─→ Combined uncertainty: ~±0.04%
  ├─→ For 5V: ±2mV acceptable
  └─→ If exceeded: PSU may need calibration

AUTOMATED VERSION
  ├─→ Use voltage ramping on PSU
  │     ├─ Ramp 0-30V in 100 steps
  │     └─ 10 seconds per step
  ├─→ Enable continuous DMM measurement
  │     ├─ Interval: 5 seconds
  │     └─ Captures 2 readings per voltage step
  ├─→ Export both datasets
  ├─→ Merge in Excel using timestamps
  ├─→ Plot PSU vs DMM readings
  └─→ Calculate correlation and errors
```

---

### Workflow: Power Supply + Oscilloscope Load Testing

**Scenario**: Test power supply response to load transients

```
TEST SETUP
  ├─→ Power Supply: Provides voltage
  ├─→ Oscilloscope: Monitors voltage stability
  ├─→ Electronic Load: Creates current steps
  └─→ Connection: PSU → Scope probe → Load

POWER SUPPLY CONFIG
  ├─→ Channel 1: 12V output
  ├─→ Current limit: 2.0A
  ├─→ Enable output
  └─→ Stable at 12V no-load

OSCILLOSCOPE CONFIG
  ├─→ Channel 1: Monitor voltage
  ├─→ Vertical: 500mV/div (to see ripple)
  ├─→ AC coupling (removes DC offset)
  ├─→ Timebase: 50ms/div
  ├─→ Trigger: Single shot mode
  └─→ Trigger level: 100mV deviation

TEST PROCEDURE
  ├─→ [1] Baseline Capture
  │     ├─ No load connected
  │     ├─ Capture waveform
  │     └─ Should be flat ~0V (AC coupled)
  │
  ├─→ [2] Apply Load Step
  │     ├─ Electronic load: 0A → 1A step
  │     ├─ Observe voltage transient
  │     ├─ Oscilloscope triggers automatically
  │     └─ Capture shows voltage dip
  │
  ├─→ [3] Analyze Transient Response
  │     ├─ Measure voltage dip magnitude
  │     │   └─ Good PSU: <100mV dip
  │     ├─ Measure recovery time
  │     │   └─ Good PSU: <10ms to settle
  │     ├─ Check for overshoot
  │     │   └─ Should be minimal
  │     └─ Look for ringing
  │         └─ Indicates poor compensation
  │
  ├─→ [4] Repeat with Different Load Steps
  │     ├─ 0A → 0.5A
  │     ├─ 0A → 1.0A
  │     ├─ 0A → 1.5A
  │     ├─ 0A → 2.0A (max)
  │     └─ Also test: 1A → 0A (load release)
  │
  └─→ [5] Document Results
        ├─ Save all waveforms
        ├─ Create comparison table
        ├─ Note any instability
        └─ Compare to PSU specifications

PASS/FAIL CRITERIA
  ├─→ PASS if:
  │     ├─ Voltage dip < specified limit
  │     ├─ Recovery time < specified limit
  │     ├─ No sustained oscillation
  │     └─ No overvoltage on load release
  └─→ FAIL if any criterion not met
        └─ May need to adjust PSU compensation
```

---

## Data Analysis Workflows

### Workflow: Creating a Test Report

```
DATA COLLECTION
  ├─→ Power Supply: Export ramping data (CSV)
  ├─→ DMM: Export measurement data (CSV)
  ├─→ Oscilloscope: Save waveforms (PNG + CSV)
  └─→ Save screenshots of all interfaces

ORGANIZE FILES
  ├─→ Create project folder
  │     ├─ Project: "Amplifier_Test_2025-01-19"
  │     ├─ Subfolders:
  │     │   ├─ raw_data/
  │     │   ├─ graphs/
  │     │   ├─ screenshots/
  │     │   └─ analysis/
  │     └─ Copy all exported files to raw_data/

DATA PROCESSING IN EXCEL
  ├─→ [1] Open CSV files
  │
  ├─→ [2] Clean data
  │     ├─ Remove header rows if needed
  │     ├─ Check for missing values
  │     └─ Convert timestamps to proper format
  │
  ├─→ [3] Create calculated columns
  │     ├─ Power = Voltage × Current
  │     ├─ Efficiency = Pout / Pin × 100
  │     └─ Error = Measured - Expected
  │
  ├─→ [4] Generate summary statistics
  │     ├─ Use AVERAGE(), MIN(), MAX()
  │     ├─ Use STDEV() for consistency
  │     └─ Create summary table
  │
  ├─→ [5] Create graphs
  │     ├─ Time-series plots
  │     ├─ Before/after comparisons
  │     ├─ Error distributions
  │     └─ Format professionally
  │
  └─→ [6] Save workbook
        └─ File: "Amplifier_Analysis.xlsx"

REPORT CREATION
  ├─→ Use Word or similar
  │
  ├─→ Structure:
  │     ├─ Cover Page
  │     │   ├─ Title
  │     │   ├─ Date
  │     │   ├─ Tester name
  │     │   └─ Document number
  │     │
  │     ├─ Executive Summary
  │     │   ├─ Purpose of test
  │     │   ├─ Key findings
  │     │   └─ Pass/Fail conclusion
  │     │
  │     ├─ Test Setup
  │     │   ├─ Equipment list
  │     │   ├─ Connection diagram
  │     │   ├─ Configuration settings
  │     │   └─ Screenshots of setup
  │     │
  │     ├─ Test Procedure
  │     │   ├─ Step-by-step what was done
  │     │   ├─ Reference standards used
  │     │   └─ Any deviations noted
  │     │
  │     ├─ Results
  │     │   ├─ Data tables
  │     │   ├─ Graphs from Excel
  │     │   ├─ Oscilloscope screenshots
  │     │   └─ Statistical analysis
  │     │
  │     ├─ Discussion
  │     │   ├─ Interpretation of results
  │     │   ├─ Comparison to specs
  │     │   ├─ Sources of error
  │     │   └─ Unexpected findings
  │     │
  │     ├─ Conclusions
  │     │   ├─ Did device meet specs?
  │     │   ├─ Any concerns?
  │     │   └─ Recommendations
  │     │
  │     └─ Appendices
  │         ├─ Raw data files
  │         ├─ Calibration certificates
  │         ├─ Equipment manuals
  │         └─ Additional graphs
  │
  └─→ Save as PDF for distribution
```

---

## Troubleshooting Workflows

### Workflow: Debugging Instrument Connection Issues

```
SYMPTOM: "Cannot connect to instrument"

SYSTEMATIC TROUBLESHOOTING

┌─────────────────────────────────────┐
│ STEP 1: Physical Checks (Hardware) │
└─────────────────────────────────────┘
  ├─→ Check power
  │     ├─ Is instrument powered on?
  │     ├─ Front panel lit?
  │     └─ Cooling fans running?
  │
  ├─→ Check cables
  │     ├─ USB firmly connected both ends?
  │     ├─ Try different USB cable
  │     ├─ Try different USB port on computer
  │     └─ For LAN: Check Ethernet cable + link lights
  │
  └─→ Check instrument
        ├─ Any error messages on front panel?
        ├─ Can you control manually?
        └─ Power cycle: Off 30s, then On

┌─────────────────────────────────────┐
│ STEP 2: Software Checks (OS Level) │
└─────────────────────────────────────┘
  ├─→ Windows Device Manager
  │     ├─ Press Win+X → Device Manager
  │     ├─ Look under "Universal Serial Bus"
  │     ├─ Any yellow warnings?
  │     │   └─ If yes: Update driver
  │     └─ Is device listed?
  │         └─ If no: USB hardware issue
  │
  ├─→ NI-MAX (Windows)
  │     ├─ Open NI Measurement & Automation Explorer
  │     ├─ Expand "Devices and Interfaces"
  │     ├─ Right-click → "Scan for Instruments"
  │     ├─ Does instrument appear?
  │     │   ├─ Yes: Note VISA address → use in software
  │     │   └─ No: VISA driver issue
  │     └─ Try "Test Panel" to verify communication
  │
  └─→ VISA Driver
        ├─ Is NI-VISA installed?
        ├─ Check version: Should be recent
        ├─ Try reinstalling VISA
        └─ Or use pyvisa-py backend

┌─────────────────────────────────────┐
│ STEP 3: Application Checks         │
└─────────────────────────────────────┘
  ├─→ Verify VISA address
  │     ├─ Format: USB0::0xVEND::0xPROD::SERIAL::INSTR
  │     ├─ Copy exactly from NI-MAX
  │     ├─ No extra spaces
  │     └─ Correct for your specific instrument
  │
  ├─→ Check Python environment
  │     ├─ pyvisa installed?
  │     │   └─ pip list | grep pyvisa
  │     ├─ Try import:
  │     │   └─ python -c "import pyvisa; print(pyvisa.__version__)"
  │     └─ Try listing resources:
  │         └─ python -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm.list_resources())"
  │
  └─→ Check firewall/antivirus
        ├─ Temporarily disable to test
        ├─ Add Python to allowed apps
        └─ Add VISA ports to exceptions

┌─────────────────────────────────────┐
│ STEP 4: Advanced Troubleshooting   │
└─────────────────────────────────────┘
  ├─→ Try different backend
  │     ├─ In code, use: pyvisa.ResourceManager('@py')
  │     └─ This uses pyvisa-py instead of NI-VISA
  │
  ├─→ Check instrument documentation
  │     ├─ Does it support VISA?
  │     ├─ Special configuration needed?
  │     └─ Firmware up to date?
  │
  └─→ Test with manufacturer software
        ├─ Keithley has KI-Tool
        ├─ Keysight has IO Libraries
        ├─ If those work, problem is in Python setup
        └─ If those don't work, instrument issue

IF STILL NOT WORKING
  ├─→ Check error logs
  │     └─ Look in command prompt for detailed errors
  │
  ├─→ Try on different computer
  │     └─ Isolates whether computer or instrument
  │
  ├─→ Contact support
  │     ├─ Email: info@digantara.com
  │     └─ Include:
  │         ├─ Exact error message
  │         ├─ OS and version
  │         ├─ Python version
  │         ├─ VISA driver version
  │         ├─ Instrument model
  │         └─ All steps tried
  │
  └─→ Last resort
        └─ Reinstall everything
            ├─ Uninstall VISA
            ├─ Uninstall Python packages
            ├─ Restart computer
            ├─ Reinstall VISA
            ├─ Reinstall Python packages
            └─ Test again
```

---

## Best Practices Summary

### Testing Best Practices

1. **Always start low**: Low voltage, low current, gradually increase
2. **Verify first**: Check outputs before connecting device
3. **Set limits**: Current limits and OVP before enabling
4. **Monitor continuously**: Don't set and forget
5. **Document everything**: Settings, results, observations
6. **Export data**: Regular backups of measurement data
7. **Label clearly**: Files, cables, test points
8. **Keep organized**: Folder structure for each project
9. **Safety first**: Emergency stop always accessible
10. **Ask if unsure**: Better to ask than damage equipment

### Efficiency Tips

1. **Use templates**: Save common configurations
2. **Automate repetitive**: Use ramping for repeated tests
3. **Parallel where possible**: Multiple instruments simultaneously
4. **Scripts for analysis**: Python/Excel macros
5. **Keyboard shortcuts**: Learn common shortcuts
6. **Multiple monitors**: One for GUI, one for documentation
7. **Regular calibration**: Schedule annual calibration
8. **Maintain equipment**: Clean, check cables, update firmware
9. **Share knowledge**: Document procedures for team
10. **Continuous improvement**: Note inefficiencies, improve process

---

**Document Version**: 1.0
**For**: Digantara Instrumentation Control Suite v1.0.0
