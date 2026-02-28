#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

#ifdef __cplusplus
 extern "C" {
#endif

#define CFG_TUSB_RHPORT0_MODE   OPT_MODE_DEVICE
#define CFG_TUD_ENDPOINT0_SIZE  64

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
