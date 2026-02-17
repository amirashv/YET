import time
import supervisor
import board
import busio

from adafruit_lsm6ds.lsm6ds3 import LSM6DS3

i2c = busio.I2C(board.GP27, board.GP26)  # SCL, SDA (same pins you scanned)
time.sleep(0.5)

imu = LSM6DS3(i2c, address=0x6A)

print("t_mcu_ms,gx_raw,gy_raw,gz_raw")

while True:
    t = supervisor.ticks_ms()
    gx, gy, gz = imu.gyro  # rad/s typically
    print(f"{t},{gx},{gy},{gz}")
    time.sleep(0.01)