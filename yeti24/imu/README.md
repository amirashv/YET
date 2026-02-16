# IMU Head Motion Prototype

Prototype project for:
- raw gyroscope acquisition
- timestamped logging
- bias calibration
- temporal filtering

Target hardware:
- Raspberry Pi Pico (Maker Pi Pico)
- LSM6DS-family IMU over I2C

This project is intentionally kept separate from YET/YETI24.


# CircuitPython Dependencies (Pico Side)

This project requires specific CircuitPython libraries to communicate with the LSM6DS-series IMU over I²C.

## Required Libraries

From the Adafruit CircuitPython Library Bundle, copy the following folders into: CIRCUITPY/lib/

Required:
	•	adafruit_lsm6ds/
	•	adafruit_bus_device/
	•	adafruit_register/

These libraries enable:
	•	LSM6DS sensor driver access
	•	I²C communication abstraction
	•	Register-level configuration handling

No additional display, NeoPixel, ADC, or MPU6050 libraries are required for the IMU-only prototype.

## Download the Library Bundle

Download the correct version of the Adafruit CircuitPython Library Bundle that matches your installed CircuitPython version:

👉 https://circuitpython.org/libraries

Steps:
	1.	Check your CircuitPython version by opening the boot_out.txt file on the Pico.
	2.	Download the matching library bundle (e.g., 9.x.x).
	3.	Extract the ZIP.
	4.	Copy only the required folders listed above into: CIRCUITPY/lib/