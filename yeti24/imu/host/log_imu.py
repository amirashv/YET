import csv
import time
import os
import serial

PORT = "/dev/cu.usbmodem1101"
BAUD = 115200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTFILE = os.path.join(DATA_DIR, f"imu_log_{timestamp}.csv")

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(1.0)
ser.reset_input_buffer()

print(f"Logging from {PORT} → {OUTFILE}")
print("Press Ctrl+C to stop.\n")

with open(OUTFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t_pc", "t_mcu_ms", "gx_raw", "gy_raw", "gz_raw"])

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("t_mcu"):
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue

            t_pc = time.time()
            t_mcu, gx, gy, gz = parts
            writer.writerow([t_pc, t_mcu, gx, gy, gz])

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
