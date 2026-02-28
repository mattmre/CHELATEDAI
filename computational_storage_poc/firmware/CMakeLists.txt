cmake_minimum_required(VERSION 3.13)

# Auto-fetch the Pico SDK from GitHub if you don't have PICO_SDK_PATH set locally
set(PICO_SDK_FETCH_FROM_GIT ON CACHE BOOL "Fetch the SDK from Git")

# Initialize the SDK
include(pico_sdk_import.cmake)

project(computational_storage_poc C CXX ASM)

# Initialize the Raspberry Pi Pico SDK
pico_sdk_init()

add_executable(compssd_firmware
        main.c
        usb_descriptors.c
        )

# Add tinyusb interface
target_link_libraries(compssd_firmware 
    pico_stdlib
    tinyusb_device
    tinyusb_board
)

# Enable USB, disable UART for TinyUSB to own the port
pico_enable_stdio_usb(compssd_firmware 1)
pico_enable_stdio_uart(compssd_firmware 0)

# Create map/bin/hex/uf2
pico_add_extra_outputs(compssd_firmware)
