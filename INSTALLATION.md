# Complete Installation Guide
## Digantara Instrumentation Control Suite

> Step-by-step installation for Windows, Mac, and Linux
>
> **Developed by: Anirudh Iyengar** | Digantara Research and Technologies Pvt. Ltd.

---

## Table of Contents

1. [Pre-Installation Checklist](#pre-installation-checklist)
2. [Windows Installation](#windows-installation)
3. [Mac Installation](#mac-installation)
4. [Linux Installation](#linux-installation)
5. [VISA Driver Installation](#visa-driver-installation)
6. [Verification](#verification)
7. [Troubleshooting Installation](#troubleshooting-installation)
8. [Uninstallation](#uninstallation)

---

## Pre-Installation Checklist

Before you begin, make sure you have:

- [ ] **Administrator/Sudo Access**: You'll need permission to install software
- [ ] **Internet Connection**: For downloading Python packages (approximately 200-300 MB)
- [ ] **Disk Space**: At least 1 GB free space
- [ ] **Supported OS**:
  - Windows 7 SP1 or later (Windows 10/11 recommended)
  - macOS 10.14 (Mojave) or later
  - Linux: Ubuntu 18.04+, Debian 10+, Fedora 30+, or equivalent
- [ ] **USB Port**: For connecting instruments
- [ ] **Time**: Allow 15-30 minutes for complete installation

---

## Windows Installation

### Step 1: Install Python

#### Option A: Using Microsoft Store (Easiest - Windows 10/11)

1. Open **Microsoft Store** from Start menu
2. Search for **"Python 3.12"** (or latest version)
3. Click **"Get"** or **"Install"**
4. Wait for installation to complete
5. Python is now installed with PATH automatically configured

#### Option B: Using Official Installer (All Windows Versions)

1. **Download Python**:
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Click the big yellow **"Download Python"** button
   - Save the installer (e.g., `python-3.12.0-amd64.exe`)

2. **Run the Installer**:
   - Double-click the downloaded file
   - **CRITICAL**: Check the box **"Add Python to PATH"** at the bottom
   - Choose one of:
     - **Install Now** (Recommended - uses default settings)
     - **Customize Installation** (Advanced users only)
   - Click **Install Now**

3. **Wait for Installation**:
   - Progress bar shows installation
   - May take 2-5 minutes
   - May ask for Administrator permission - click **Yes**

4. **Completion**:
   - Click **"Close"** when finished
   - Do NOT skip "Disable path length limit" option if shown (click it!)

#### Verify Python Installation

1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. In the command prompt, type:
   ```bash
   python --version
   ```
4. Should show: `Python 3.x.x`
5. Also verify pip:
   ```bash
   pip --version
   ```
6. Should show: `pip 2x.x.x from ...`

**If commands not found**: Python PATH wasn't set correctly. Reinstall with "Add to PATH" checked.

---

### Step 2: Install Git (Optional but Recommended)

**Why Git?** Makes updating software easier in the future.

1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run installer with default options
3. Git is now available in Command Prompt

---

### Step 3: Get the Software

#### Option A: Download ZIP (No Git)

1. Download the `Digantara_instrumentation` folder
2. Extract to a location you'll remember
   - Example: `C:\Users\YourName\Desktop\Digantara_instrumentation`
3. Note this location - you'll need it!

#### Option B: Clone with Git (Recommended if you have Git)

1. Open Command Prompt
2. Navigate to where you want the software:
   ```bash
   cd C:\Users\YourName\Desktop
   ```
3. Clone the repository:
   ```bash
   git clone [repository-url] Digantara_instrumentation
   ```
4. Navigate into folder:
   ```bash
   cd Digantara_instrumentation
   ```

---

### Step 4: Install Python Dependencies

1. **Open Command Prompt**:
   - Press `Windows Key + R`
   - Type `cmd` and press Enter

2. **Navigate to Project Folder**:
   ```bash
   cd "C:\Users\YourName\Desktop\Digantara_instrumentation"
   ```
   **Replace** the path with your actual folder location!

3. **Upgrade pip** (recommended):
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Wait for Installation** (2-5 minutes):
   - You'll see "Collecting..." messages
   - Then "Installing..." messages
   - Finally "Successfully installed..." messages

6. **Verify Installation**:
   ```bash
   pip list
   ```
   - Should see: numpy, pandas, matplotlib, gradio, pyvisa, etc.

**If Installation Fails**: See [Troubleshooting Installation](#troubleshooting-installation) section below.

---

### Step 5: Install VISA Drivers

**What is VISA?** Software that lets your computer communicate with lab instruments.

1. **Download NI-VISA**:
   - Go to [ni.com/visa](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
   - Choose **NI-VISA** for Windows
   - Click **Download** (may need to create free account)
   - Save the installer

2. **Run the Installer**:
   - Double-click downloaded file (e.g., `ni-visa_23.0.0.exe`)
   - Click **Unzip**
   - Installation wizard opens
   - Click **Next** through all screens (keep defaults)
   - Accept license agreement
   - Choose installation type: **Typical** (recommended)
   - Click **Install**

3. **Wait for Installation** (5-10 minutes):
   - Large installation (1-2 GB)
   - Progress bar shows status
   - May restart services

4. **Restart Computer**:
   - Recommended after VISA installation
   - Ensures drivers load properly

5. **Verify VISA Installation**:
   - Open **NI-MAX** from Start menu (NI Measurement & Automation Explorer)
   - Should open without errors
   - If not found, VISA didn't install correctly

---

### Step 6: Test the Installation

1. **Connect an Instrument** (if available):
   - Plug in via USB
   - Turn on instrument
   - Wait 10 seconds

2. **Scan for Instruments**:
   - Open **NI-MAX**
   - Expand **"Devices and Interfaces"**
   - Right-click → **"Scan for Instruments"**
   - Your instrument should appear
   - Note the VISA address (e.g., `USB0::0x05E6::0x2230::...::INSTR`)

3. **Launch Software**:
   ```bash
   cd "C:\Users\YourName\Desktop\Digantara_instrumentation"
   python Unified.py
   ```

4. **Open Browser**:
   - Go to http://localhost:7860
   - Interface should load
   - Try connecting to instrument

**Success!** If you see the interface and can connect, installation is complete!

---

## Mac Installation

### Step 1: Install Homebrew (Package Manager)

Homebrew makes installing software on Mac much easier.

1. **Open Terminal**:
   - Press `Cmd + Space`
   - Type "Terminal"
   - Press Enter

2. **Install Homebrew**:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Wait for Installation** (5-10 minutes):
   - Enter your password when prompted
   - Press Enter to continue
   - Don't close Terminal during installation

4. **Verify Installation**:
   ```bash
   brew --version
   ```
   - Should show: `Homebrew 4.x.x`

---

### Step 2: Install Python

1. **Install Python via Homebrew**:
   ```bash
   brew install python@3.12
   ```

2. **Verify Installation**:
   ```bash
   python3 --version
   ```
   - Should show: `Python 3.12.x`

3. **Verify pip**:
   ```bash
   pip3 --version
   ```
   - Should show pip version

---

### Step 3: Get the Software

1. **Navigate to desired location**:
   ```bash
   cd ~/Desktop
   ```

2. **Download/Clone**:
   - If using Git:
     ```bash
     git clone [repository-url] Digantara_instrumentation
     ```
   - If using ZIP: Extract downloaded folder here

3. **Enter directory**:
   ```bash
   cd Digantara_instrumentation
   ```

---

### Step 4: Install Python Dependencies

1. **Create Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   ```

2. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```
   - Your prompt should change to show `(venv)`

3. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Wait for Installation** (2-5 minutes)

---

### Step 5: Install VISA Drivers

#### Option A: NI-VISA (Full Featured)

1. **Download**:
   - Go to [ni.com/visa](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html)
   - Choose **NI-VISA** for macOS
   - Download installer

2. **Install**:
   - Open downloaded `.dmg` file
   - Run installer
   - Follow prompts
   - Enter admin password when asked

3. **Restart Mac**

#### Option B: pyvisa-py (Pure Python, Simpler)

Already installed with requirements.txt! No additional steps needed.

**Note**: pyvisa-py has some limitations compared to NI-VISA but works for most USB instruments.

---

### Step 6: Test Installation

1. **Activate virtual environment** (if you created one):
   ```bash
   source venv/bin/activate
   ```

2. **Launch software**:
   ```bash
   python3 Unified.py
   ```

3. **Open browser** to http://localhost:7860

---

## Linux Installation

Instructions for Ubuntu/Debian. Adjust package manager for other distributions (dnf for Fedora, pacman for Arch, etc.).

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

---

### Step 2: Install Python and Dependencies

```bash
sudo apt install -y python3 python3-pip python3-venv git
```

Verify:
```bash
python3 --version
pip3 --version
```

---

### Step 3: Install System Libraries

Some Python packages need system libraries:

```bash
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y libusb-1.0-0-dev
```

---

### Step 4: Get the Software

```bash
cd ~/Desktop
git clone [repository-url] Digantara_instrumentation
cd Digantara_instrumentation
```

---

### Step 5: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 6: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 7: Install VISA Drivers

#### Option A: NI-VISA

Download from [ni.com](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html) and follow Linux installation instructions.

#### Option B: Use pyvisa-py (Recommended for Linux)

Already installed! No additional steps.

For USB instruments, add udev rules:

```bash
sudo nano /etc/udev/rules.d/99-lab-instruments.rules
```

Add:
```
# Keithley Instruments
SUBSYSTEM=="usb", ATTR{idVendor}=="05e6", MODE="0666", GROUP="plugdev"

# Keysight Instruments
SUBSYSTEM=="usb", ATTR{idVendor}=="0957", MODE="0666", GROUP="plugdev"
```

Save (Ctrl+O, Enter, Ctrl+X), then reload:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Add your user to plugdev group:
```bash
sudo usermod -a -G plugdev $USER
```

Log out and back in for changes to take effect.

---

### Step 8: Test Installation

```bash
source venv/bin/activate  # if not already activated
python3 Unified.py
```

Open browser to http://localhost:7860

---

## VISA Driver Installation

### What is VISA?

**VISA** (Virtual Instrument Software Architecture) is an industry-standard API for communicating with test and measurement instruments.

### Do I Need VISA Drivers?

**Yes, if**:
- Connecting instruments via USB
- Want full instrument compatibility
- Need advanced features

**Maybe not, if**:
- Only using network-connected instruments (LAN/LXI)
- Using pyvisa-py backend (USB instruments, limited features)

### Choosing VISA Backend

**NI-VISA** (National Instruments):
- ✅ Full-featured, most reliable
- ✅ Supports all connection types (USB, GPIB, LAN, Serial)
- ✅ Official support
- ❌ Large installation (1-2 GB)
- ❌ Requires restart
- ❌ Commercial software (free for runtime)

**pyvisa-py** (Pure Python):
- ✅ Small, no separate installation
- ✅ Open source
- ✅ Works for USB and Serial
- ✅ Already included in requirements.txt
- ❌ Doesn't support GPIB without adapters
- ❌ Some advanced features missing
- ❌ May be slower

**Recommendation**:
- **Windows**: Install NI-VISA (best compatibility)
- **Mac**: Install NI-VISA if available, pyvisa-py otherwise
- **Linux**: Use pyvisa-py (simpler)

---

## Verification

### Verify Python Installation

```bash
python --version   # or python3 --version on Mac/Linux
pip --version      # or pip3 --version on Mac/Linux
```

Expected output:
```
Python 3.8.x or higher
pip 20.x.x or higher
```

---

### Verify Package Installation

```bash
pip list
```

Check for these key packages:
- numpy (2.0.0 or higher)
- pandas (2.2.0 or higher)
- matplotlib (3.8.0 or higher)
- gradio (4.0.0 or higher)
- pyvisa (1.13.0 or higher)
- pyvisa-py (0.7.2 or higher)

---

### Verify VISA Installation

#### Using NI-MAX (Windows):

1. Open **NI-MAX** from Start menu
2. Expand **"My System"** → **"Software"**
3. Should see **"NI-VISA"** listed with version

#### Using Python:

```python
python -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm.list_resources())"
```

Expected output (with instrument connected):
```
('USB0::0x05E6::0x2230::SERIAL::INSTR',)
```

Or (with no instruments):
```
()
```

If you get an error, VISA isn't installed correctly.

---

### Test Run

1. Navigate to project folder
2. Run:
   ```bash
   python Unified.py
   ```
3. Check terminal output for errors
4. Browser should open to http://localhost:7860
5. Interface should load

If successful, installation is complete!

---

## Troubleshooting Installation

### Python Issues

**Problem**: "python is not recognized" (Windows)

**Solution**:
1. Reinstall Python
2. **Must check "Add Python to PATH"**
3. Restart Command Prompt after installation
4. Or manually add to PATH:
   - Search "Environment Variables" in Windows
   - Edit PATH variable
   - Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python312`
   - Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python312\Scripts`

---

**Problem**: "python: command not found" (Mac/Linux)

**Solution**:
- Try `python3` instead of `python`
- Try `pip3` instead of `pip`
- Make aliases:
  ```bash
  echo "alias python=python3" >> ~/.bashrc
  echo "alias pip=pip3" >> ~/.bashrc
  source ~/.bashrc
  ```

---

### Pip Installation Issues

**Problem**: "pip install fails with permission error"

**Solution (Windows)**:
```bash
pip install --user -r requirements.txt
```

**Solution (Mac/Linux)**:
```bash
pip3 install --user -r requirements.txt
```

Or use virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

**Problem**: "pip install fails: Microsoft Visual C++ required" (Windows)

**Solution**:
1. Download "Microsoft C++ Build Tools"
2. Install with Desktop Development with C++ workload
3. Or download "Visual C++ Redistributable"
4. Retry pip installation

---

**Problem**: Packages fail to compile (Mac)

**Solution**:
1. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
2. Retry installation

---

### VISA Issues

**Problem**: NI-VISA installer fails

**Solution**:
1. Temporarily disable antivirus
2. Run installer as Administrator
3. Check system requirements
4. Download latest version from NI website
5. Or use pyvisa-py instead (no separate installation)

---

**Problem**: Can't find instruments

**Solution**:
1. Check USB cable connection
2. Turn on instrument
3. Wait 10 seconds after connecting
4. Try different USB port
5. In NI-MAX (Windows): Right-click → Scan for Instruments
6. Check Windows Device Manager for unknown devices
7. Reinstall instrument USB drivers

---

**Problem**: "Backend not found" error

**Solution**:
Install pyvisa-py:
```bash
pip install pyvisa-py
```

Or specify backend in code:
```python
import pyvisa
rm = pyvisa.ResourceManager('@py')  # Use pyvisa-py backend
```

---

### Network/Firewall Issues

**Problem**: Can't access http://localhost:7860

**Solution**:
1. Check if port is in use:
   - Windows: `netstat -ano | findstr :7860`
   - Mac/Linux: `lsof -i :7860`
2. If in use, edit Python file to use different port
3. Temporarily disable firewall to test
4. Add firewall exception for Python

---

**Problem**: Can't access from other computers on network

**Solution**:
1. Check software is bound to `0.0.0.0` not `127.0.0.1`
2. Open firewall port 7860 (or your chosen port)
3. Use computer's actual IP address, not localhost
4. Ensure all computers on same network

---

### Gradio Issues

**Problem**: "gradio not found" or import errors

**Solution**:
```bash
pip install --upgrade gradio
```

If still fails:
```bash
pip uninstall gradio
pip install gradio==4.0.0  # Try specific version
```

---

**Problem**: Gradio interface looks broken

**Solution**:
1. Clear browser cache
2. Try different browser
3. Update Gradio:
   ```bash
   pip install --upgrade gradio
   ```
4. Check for JavaScript errors in browser console (F12)

---

### General Debugging

**Steps to diagnose any installation issue**:

1. **Check Python version**:
   ```bash
   python --version
   ```
   Must be 3.8 or higher

2. **Check pip works**:
   ```bash
   pip --version
   ```

3. **Try installing one package at a time**:
   ```bash
   pip install numpy
   pip install pandas
   pip install matplotlib
   # etc.
   ```

4. **Check error messages carefully**:
   - Google the exact error message
   - Check Stack Overflow
   - Look for missing system libraries

5. **Use virtual environment**:
   - Isolates from system Python
   - Prevents conflicts
   - Easier to troubleshoot

6. **Update everything**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install --upgrade -r requirements.txt
   ```

---

## Uninstallation

### Remove Python Packages

```bash
pip uninstall -r requirements.txt -y
```

Or if using virtual environment:
```bash
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows
```

---

### Remove Python (Windows)

1. Open "Add or Remove Programs"
2. Find "Python 3.x.x"
3. Click "Uninstall"
4. Follow wizard

---

### Remove Python (Mac with Homebrew)

```bash
brew uninstall python@3.12
```

---

### Remove NI-VISA

**Windows**:
1. Open "Add or Remove Programs"
2. Find "NI-VISA"
3. Click "Uninstall"

**Mac**:
1. Run NI uninstaller from Applications
2. Select VISA components
3. Uninstall

**Linux**:
```bash
sudo apt remove ni-visa
```

---

### Remove Project Files

Simply delete the `Digantara_instrumentation` folder.

---

## Next Steps

After successful installation:

1. Read [QUICK_START.md](QUICK_START.md) for basic usage
2. Read [README.md](README.md) for comprehensive documentation
3. Read instrument-specific guides for detailed operations
4. Connect your instrument and test!

---

## Getting Help

If installation problems persist:

1. **Check Documentation**:
   - README.md (main documentation)
   - QUICK_START.md (basic usage)
   - This file (installation)

2. **Online Resources**:
   - Python.org documentation
   - NI-VISA documentation
   - PyVISA documentation

3. **Contact Support**:
   - Email: info@digantara.com
   - Include:
     - Operating system and version
     - Python version (`python --version`)
     - Exact error message
     - Steps you've already tried

---

**Document Version**: 1.0
**For**: Digantara Instrumentation Control Suite v1.0.0
