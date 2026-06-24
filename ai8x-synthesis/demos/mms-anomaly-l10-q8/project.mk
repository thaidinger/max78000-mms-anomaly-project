# This file can be used to set build configuration
# variables.  These variables are defined in a file called
# "Makefile" that is located next to this one.

# For instructions on how to use this system, see
# https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system

# **********************************************************

# Add your config here!
BOARD=FTHR_RevA

$(info Note: This project is configured for the OV7692 camera.)
override CAMERA=OV7692

# The CameraIF DMA path needs optimization to meet timing.
MXC_OPTIMIZE_CFLAGS = -O2

ifeq "$(BOARD)" "FTHR_RevA"
IPATH += TFT/fthr
VPATH += TFT/fthr
FONTS = LiberationSans16x16
endif
