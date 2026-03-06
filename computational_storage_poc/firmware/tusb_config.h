#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

#ifdef __cplusplus
 extern "C" {
#endif

#define CFG_TUSB_MCU            OPT_MCU_RP2040
#define CFG_TUSB_OS             OPT_OS_PICO
#define CFG_TUSB_DEBUG          0

#ifndef BOARD_TUD_RHPORT
#define BOARD_TUD_RHPORT        0
#endif

#ifndef BOARD_TUD_MAX_SPEED
#define BOARD_TUD_MAX_SPEED     OPT_MODE_FULL_SPEED
#endif

#define CFG_TUSB_RHPORT0_MODE   (OPT_MODE_DEVICE | BOARD_TUD_MAX_SPEED)
#define CFG_TUD_ENDPOINT0_SIZE  64

#define CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_ALIGN      __attribute__((aligned(4)))

// ------------- CLASS -------------//
#define CFG_TUD_CDC             0
#define CFG_TUD_MSC             1
#define CFG_TUD_HID             0
#define CFG_TUD_MIDI            0
#define CFG_TUD_VENDOR          0

// MSC Buffer size of Device Mass storage
#define CFG_TUD_MSC_EP_BUFSIZE  512

#ifdef __cplusplus
 }
#endif

#endif /* _TUSB_CONFIG_H_ */
