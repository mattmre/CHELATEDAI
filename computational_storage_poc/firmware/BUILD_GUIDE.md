# Building the Computational Storage Firmware

This directory contains the C firmware required to turn a Raspberry Pi Pico (RP2040) into a simulated SSD controller.

## Prerequisites (Windows)
To compile this firmware into a `.uf2` file that you can drag-and-drop onto the Pico, you need the standard ARM embedded toolchain.

1. **Install CMake:** Download and install [CMake for Windows](https://cmake.org/download/). Make sure to check the box "Add CMake to the system PATH for all users".
2. **Install ARM GCC Compiler:** Download and install the [GNU Arm Embedded Toolchain](https://developer.arm.com/downloads/-/gnu-rm). Check the box "Add path to environment variable" at the very end of the installer.
3. **Install Build Tools:** Download [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) and install the "C++ build tools" workload.

Alternatively, the absolute easiest way on Windows is to run the official [Raspberry Pi Pico Windows Installer](https://github.com/raspberrypi/pico-setup-windows/releases/latest/download/pico-setup-windows-x64-standalone.exe) which installs all of the above perfectly configured in one click.

## Compiling
Once the tools are installed and your terminal is restarted:

1. Open a terminal in this `firmware/` directory.
2. Initialize the build environment:
   ```bash
   cmake -S . -B build
   ```
   *(Note: The CMakeLists.txt is configured to automatically download the Pico SDK from GitHub if you don't have it installed locally!)*
3. Compile the code:
   ```bash
   cmake --build build
   ```

## Flashing the Pico
1. Hold down the `BOOTSEL` button on your Raspberry Pi Pico and plug it into your computer's USB port.
2. It will mount as a drive called `RPI-RP2`.
3. Open the newly created `firmware/build/` directory.
4. Drag and drop the `compssd_firmware.uf2` file onto the `RPI-RP2` drive.
5. The Pico will instantly reboot and remount as a "Computational Storage Proof of Concept" Mass Storage Drive!
