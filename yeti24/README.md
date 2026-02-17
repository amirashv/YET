# YETI24

YETI24 is a low-cost, dual-camera eye-tracking prototype with an optional IMU module for head-motion logging.  
This repository contains everything needed to run the eye-tracker pipeline (calibration → validation → stimulus trials), the CAD files for the physical build, and the IMU logging + processing scripts.

## Repository structure

- `yeti24/et/` — **Eye-tracking module (main runtime)**
- `yeti24/cad/` — CAD models for mechanical parts (STEP/STL)
- `yeti24/imu/` — IMU logging + head-motion estimation pipeline
- `yeti24/docs/` — documentation (quickstart, build notes, etc.)

## Eye tracker (et)

The `yeti24/et/` folder contains everything required to run the dual-camera eye tracker:
- `run.py` — main entry point (UI + state machine: Detect → Calibration → Validate → Quick → Stimulus)
- `libyeti24.py` — eye-tracking library (camera handling, ROI processing, feature extraction, regression, gaze drawing, recording)
- `haarcascade_eye.xml` — OpenCV cascade used for eye detection
- `Config.csv` — experiment / participant settings (used by `run.py`)
- `Stimuli/` — stimulus images presented during trials
- `Data/` — output folder for recorded gaze data (CSV)
- `Yet.log` — runtime logs

### What it does (high level)
1. **Detect**: find eye ROIs in both camera streams  
2. **Calibration**: collect samples on a grid of targets  
3. **Validate**: show live gaze pointer on a blank screen  
4. **Quick calibration**: per-stimulus correction using an on-screen target  
5. **Stimulus**: present images and record gaze samples to CSV

 Output files are saved in `yeti24/et/Data/`.

## CAD Files

Mechanical components for the YETI24 prototype are located in:

- cad/step/  → engineering-grade STEP files
- cad/stl/   → printable STL files

All parts were designed in Onshape.

## IMU Module (Head Motion Estimation)

YETI24 includes an optional IMU subsystem for estimating relative head orientation using a Raspberry Pi Pico and an LSM6DS-series gyroscope.

The module provides:
  - 	Real-time gyroscope streaming (rad/s)
	- 	Timestamped host-side logging
	- 	Bias estimation and compensation
	- 	Temporal filtering (EMA)
	- 	Numerical integration to roll, pitch, and yaw

The IMU pipeline is implemented as a modular component and can be used independently for data acquisition and preprocessing.

Location: yeti24/imu/

The IMU is intended for future head-motion compensation within the eye-tracking pipeline.
