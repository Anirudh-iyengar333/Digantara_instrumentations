# Quick Start Guide for Non-Technical Users

> Get up and running in **5 minutes** or less!
>
> **Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.

---

## Before You Begin

### What You Need

- [ ] A computer with Windows, Mac, or Linux
- [ ] One of the supported instruments (Power Supply, DMM, or Oscilloscope)
- [ ] A USB cable to connect instrument to computer
- [ ] Internet connection (for initial setup only)

---

## Step-by-Step Setup (First Time Only)

### 1. Install Python

**What is Python?** Python is a programming language. This software needs Python to run, but you don't need to know how to program.

#### Windows Users:

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the big yellow "Download Python" button
3. Run the downloaded file
4. **IMPORTANT**: Check the box that says "Add Python to PATH"
5. Click "Install Now"
6. Wait for installation to complete (2-3 minutes)
7. Click "Close"

#### Mac Users:

1. Open Terminal (search "Terminal" in Spotlight)
2. Copy and paste this command:
   ```bash
   brew install python3
   ```
   (If you don't have Homebrew, visit [brew.sh](https://brew.sh) first)

#### Linux Users:

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

### 2. Download/Locate This Software

If you haven't already, you should have this folder on your computer:
```
Digantara_instrumentation/
```

**Remember this location!** You'll need it in the next step.

Example locations:
- Windows: `C:\Users\YourName\Desktop\Digantara_instrumentation`
- Mac: `/Users/YourName/Desktop/Digantara_instrumentation`
- Linux: `/home/yourname/Desktop/Digantara_instrumentation`

---

### 3. Install Required Software

#### Windows:

1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. A black window appears (Command Prompt)
4. Type this command and press Enter:
   ```bash
   cd "path\to\Digantara_instrumentation"
   ```
   **Replace** `path\to\Digantara_instrumentation` with the actual location from Step 2

   **Example**:
   ```bash
   cd "C:\Users\JohnDoe\Desktop\Digantara_instrumentation"
   ```

5. Now type this command and press Enter:
   ```bash
   pip install -r requirements.txt
   ```

6. Wait 2-5 minutes while it downloads and installs components
7. You should see "Successfully installed..." messages

#### Mac/Linux:

1. Open Terminal
2. Navigate to the project folder:
   ```bash
   cd ~/Desktop/Digantara_instrumentation
   ```
   (Adjust path if your folder is elsewhere)

3. Run installation:
   ```bash
   pip3 install -r requirements.txt
   ```

4. Wait for completion

---

### 4. Install VISA Drivers (Communication with Instruments)

**What is VISA?** It's software that lets your computer talk to lab instruments via USB.

#### All Users:

1. Go to [ni.com/visa](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
2. Click "Download" for your operating system
3. Run the downloaded installer
4. Follow the installation wizard (keep all default options)
5. Restart your computer when complete

---

## Daily Use (After Setup is Complete)

### Quick Start in 3 Steps

#### Step 1: Connect Your Instrument

1. Connect instrument to computer using USB cable
2. Turn on the instrument
3. Wait 5-10 seconds for computer to recognize it

#### Step 2: Launch the Software

**Windows:**
1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. Navigate to project folder:
   ```bash
   cd "C:\Users\YourName\Desktop\Digantara_instrumentation"
   ```
4. Launch the unified interface:
   ```bash
   python Unified.py
   ```

**Mac/Linux:**
1. Open Terminal
2. Navigate to project:
   ```bash
   cd ~/Desktop/Digantara_instrumentation
   ```
3. Launch:
   ```bash
   python3 Unified.py
   ```

#### Step 3: Open Your Browser

1. Wait for the command window to show:
   ```
   Running on local URL:  http://127.0.0.1:7860
   ```

2. Open your web browser (Chrome, Firefox, Edge, Safari)

3. Go to this address:
   ```
   http://localhost:7860
   ```

4. **You should now see the control interface!**

---

## Using the Interface - Power Supply Example

### Connecting to Your Power Supply

1. **Find the VISA Address**:
   - It should be pre-filled in the text box
   - Looks like: `USB0::0x05E6::0x2230::SERIAL::INSTR`
   - If empty, click "Scan for Instruments" (if available) or check instrument label

2. **Click "Connect"**:
   - Wait 2-3 seconds
   - Status should change to "Connected"
   - You'll see instrument information displayed

### Setting Up an Output

Let's say you want to output **5 Volts** on **Channel 1**:

1. **Go to the Channel 1 Tab**:
   - Click on "Channel 1" at the top

2. **Configure Settings**:
   - **Voltage (V)**: Move slider to `5.0`
   - **Current Limit (A)**: Set to `0.5` (safety limit - adjust as needed)
   - **OVP (V)**: Set to `6.0` (overvoltage protection - slightly above your voltage)

3. **Apply Configuration**:
   - Click "Configure Ch1" button
   - Wait for "Configuration successful" message

4. **Enable the Output**:
   - Click "Enable Output" button
   - Status changes to "ON"
   - **Your device now has power!**

5. **Check the Output**:
   - Click "Measure" button
   - You should see:
     - **Measured Voltage**: ~5.000 V
     - **Measured Current**: (depends on your load)
     - **Measured Power**: (Voltage × Current)

6. **Turn Off When Done**:
   - Click "Disable Output" button
   - Status changes to "OFF"

### Safety Tip
Always use the **Current Limit** feature! Set it to the maximum current your device should ever draw. This prevents damage if something goes wrong.

---

## Using the Interface - Digital Multimeter Example

### Measuring DC Voltage

Let's measure a DC voltage (like a battery):

1. **Connect to DMM**:
   - Enter VISA address (usually pre-filled)
   - Click "Connect"
   - Wait for "Connected" status

2. **Configure Measurement**:
   - **Measurement Function**: Select "DC_VOLTAGE" from dropdown
   - **Range**: Set to `10.0` (for voltages up to 10V)
     - If measuring 12V battery, use `100` range
   - **NPLC**: Leave at `1.0` (good balance of speed and accuracy)
   - **Auto Zero**: Keep checked ✓

3. **Connect Test Leads**:
   - Red probe to HI terminal
   - Black probe to LO terminal
   - Connect to device you're measuring

4. **Take Measurement**:
   - Click "Single Measurement"
   - Result appears in "Current Reading" box
   - Example: `5.234 V`

5. **Continuous Measurements**:
   - If you want ongoing measurements:
   - Set "Interval (seconds)" to `1.0` (one reading per second)
   - Click "Start Continuous"
   - Click "Stop Continuous" when done

6. **View Statistics**:
   - Go to "Statistics & Analysis" tab
   - Set "Number of Points" to `100`
   - Click "Calculate Statistics"
   - See average, min, max, standard deviation

7. **Export Your Data**:
   - Go to "Data Export" tab
   - Choose format: "CSV" (for Excel)
   - Click "Export Data"
   - File saves to project folder

---

## Common Questions

### Q: The web page won't open
**A:**
1. Check the command window for errors
2. Make sure you see "Running on local URL" message
3. Try http://127.0.0.1:7860 instead of http://localhost:7860
4. Check if another program is using port 7860 (close other applications)

### Q: "Cannot connect to instrument"
**A:**
1. Is the instrument turned on?
2. Is USB cable plugged in firmly?
3. Wait 10 seconds after plugging in USB
4. Try different USB port
5. Restart the instrument
6. Make sure VISA drivers are installed (Step 4 of setup)

### Q: Python command not found
**A:**
1. Did you check "Add Python to PATH" during installation?
2. Try `python3` instead of `python` (Mac/Linux)
3. Restart Command Prompt/Terminal after installing Python
4. Reinstall Python, making sure to check the PATH option

### Q: "pip is not recognized"
**A:**
1. Restart Command Prompt/Terminal
2. Try `python -m pip install -r requirements.txt` instead
3. Reinstall Python with "Add to PATH" checked

### Q: Measurements seem wrong
**A:**
1. Check that cables are connected properly
2. Verify measurement range is appropriate
3. For very precise measurements, increase NPLC to 10
4. Make sure instrument is warmed up (on for 15-30 minutes)

### Q: How do I stop the program?
**A:**
1. Close the browser tab
2. In the command window, press `Ctrl + C`
3. Wait for program to close
4. You can now unplug the instrument

### Q: Can someone on another computer use this?
**A:** Yes! Look for this line in the command window:
```
Running on network: http://192.168.1.XXX:7860
```
Other people on your network can enter that address in their browser.

**Security Note**: Only share on trusted networks (like your office network).

---

## Visual Checklist

Print this checklist and keep it at your workstation:

### Daily Startup Checklist

```
□ Instrument is connected via USB
□ Instrument is powered on
□ Command Prompt/Terminal is open
□ Navigated to Digantara_instrumentation folder
□ Launched software (python Unified.py)
□ Browser is open to http://localhost:7860
□ Clicked "Connect" button
□ Status shows "Connected"
□ Ready to use!
```

### Daily Shutdown Checklist

```
□ All instrument outputs are OFF
□ Data has been exported (if needed)
□ Browser tab is closed
□ Pressed Ctrl+C in command window
□ Program has exited
□ Instrument is turned off
□ USB cable disconnected (if moving instrument)
```

### Safety Checklist

```
□ Current limits are set appropriately
□ Over-voltage protection is configured
□ Emergency stop button location is known
□ No loose connections
□ Workspace is clear of metal objects
□ Readings make sense (not extremely high/low)
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + C` | Stop program in command window |
| `F5` | Refresh browser page |
| `Ctrl + F` | Find text on page |
| `Ctrl + +` | Zoom in (easier to read) |
| `Ctrl + -` | Zoom out |

---

## Where to Get Help

### Self-Help Resources

1. **Main README.md**: Detailed documentation in project folder
2. **Instrument Manual**: Check manufacturer website
3. **This Guide**: Re-read relevant sections

### Contact Support

**Email**: info@digantara.com
**Subject Line**: "Instrumentation Software Help - [Your Issue]"

**Include in your email**:
- What instrument you're using
- What you were trying to do
- What happened instead
- Any error messages (copy from command window)
- Screenshot if possible

---

## Tips for Success

### Best Practices

1. **Start Low**: Always start with low voltage/current and increase gradually
2. **Check Twice**: Verify settings before enabling outputs
3. **Save Often**: Export data regularly, don't wait until the end
4. **Label Clearly**: When saving files, use descriptive names
5. **Keep Notes**: Write down what test you're doing and settings used

### Time-Saving Tips

1. **Keep a log**: Write down VISA addresses of your instruments
2. **Bookmark**: Bookmark http://localhost:7860 in your browser
3. **Shortcuts**: Create desktop shortcut to project folder
4. **Templates**: Save common configurations for quick recall
5. **Batch Scripts**: Ask supervisor to create batch file to launch program

### Organization Tips

1. **Create folders** for different projects/tests
2. **Use timestamps** in filenames (automatically done by export)
3. **Keep backups** of important data
4. **Document settings** used for each test
5. **Archive old data** to keep workspace clean

---

## Next Steps

Once you're comfortable with basic operation:

1. **Explore advanced features**: Voltage ramping, automated testing
2. **Learn data analysis**: Use statistics and trend plots
3. **Customize settings**: Adjust NPLC, ranges for your specific needs
4. **Automate repetitive tasks**: Ask about scripting options
5. **Share knowledge**: Teach colleagues what you've learned

---

## Congratulations!

You're now ready to use the Digantara Instrumentation Control Suite!

**Remember**:
- When in doubt, ask for help
- Safety first - use emergency stop if needed
- This guide is always here for reference

**Pro tip**: Keep this guide open in a browser tab while working for easy reference.

---

**Document Version**: 1.0
**For**: Digantara Instrumentation Control Suite v1.0.0
