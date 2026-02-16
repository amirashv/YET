# IMU Module — YETI24

This module implements a gyroscope-based head motion acquisition and processing pipeline for the YETI24 system.

It provides:
	•	Embedded IMU streaming (Raspberry Pi Pico + LSM6DS)
	•	Host-side serial logging
	•	Structured CSV storage
	•	Bias estimation and compensation
	•	Temporal filtering (EMA)
	•	Numerical integration to relative orientation

The module is designed to be self-contained and reproducible.

⸻

## Architecture

imu/
├── pico/       # CircuitPython firmware (sensor streaming)
├── host/       # Serial logger (PC-side acquisition)
├── analysis/   # R-based preprocessing & integration
├── data/       # Recorded logs (Git-ignored)

Data flow:

LSM6DS → Pico (USB serial) → Host logger → CSV → R pipeline


⸻

## Hardware
	•	Raspberry Pi Pico (Maker Pi Pico compatible)
	•	LSM6DS-family IMU (tested with LSM6DS3)
	•	I²C wiring
	•	USB connection to host

⸻

## Pico Firmware

Location:

imu/pico/code.py

The firmware:
	•	Initializes the LSM6DS over I²C
	•	Streams gyroscope data (rad/s)
	•	Outputs timestamps in milliseconds
	•	Emits CSV-formatted lines via USB serial

Output format:

t_mcu_ms,gx_raw,gy_raw,gz_raw

Where:
	•	t_mcu_ms — microcontroller timestamp (ms)
	•	gx_raw, gy_raw, gz_raw — angular velocity (rad/s)

⸻

## CircuitPython Dependencies

Install CircuitPython on the Pico and copy the following libraries into:

CIRCUITPY/lib/

Required folders from the Adafruit CircuitPython Library Bundle:
	•	adafruit_lsm6ds/
	•	adafruit_bus_device/
	•	adafruit_register/

Library bundle:
https://circuitpython.org/libraries

Use the bundle version matching your installed CircuitPython firmware.

⸻

## Host Logger

Location:

imu/host/log_imu.py

The logger:
	•	Opens serial connection
	•	Reads streamed IMU data
	•	Adds host timestamp (t_pc)
	•	Writes timestamped CSV files
	•	Automatically creates imu/data/

Output format:

t_pc,t_mcu_ms,gx_raw,gy_raw,gz_raw

Install dependency:

pip install pyserial

Run:

cd imu/host
python log_imu.py

Stop with Ctrl+C.

⸻

## Analysis Pipeline

Location:

imu/analysis/imu_pipeline.Rmd

The pipeline performs:
	1.	Timestamp diagnostics (Δt stability)
	2.	Sampling rate estimation
	3.	Bias estimation (stationary recording)
	4.	Bias correction
	5.	Exponential moving average filtering
	6.	Numerical integration to roll/pitch/yaw
	7.	Visualization and validation

Units
	•	Timestamps: ms → seconds
	•	Angular velocity: rad/s
	•	Orientation: radians (converted to degrees)

⸻

## Data

Recorded logs are stored in:

imu/data/

This directory is Git-ignored.

⸻

## Limitations
	•	Gyroscope-only integration
	•	Relative orientation only
	•	Drift accumulation over time
	•	Small-angle approximation
	•	No cross-axis coupling compensation

Future extensions may include accelerometer or optical fusion.

⸻

## Integration into YETI24

This module is developed as a subsystem of YETI24 but remains independently testable.

It is intended for future head-motion compensation within the eye-tracking pipeline.
