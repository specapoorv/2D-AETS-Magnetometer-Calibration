import pyzed.sl as sl
import numpy as np
import time
import math

# === Calibration parameters from your 2D-AETS fit ===
bx, by = -0.04030177317407155, 1.0210029484619534
rho = 0.012730970505513004
s_x, s_y = 2.904204630320402, 0.6236255637525812

# === Calibration function ===
def calibrate(mx, my, bx, by, rho, s_x, s_y):
    # Raw vector
    data_raw = np.array([[mx], [my]]) - np.array([[bx], [by]])

    # Rotation and scaling matrices
    R = np.array([[np.cos(-rho), np.sin(-rho)],
                  [-np.sin(-rho), np.cos(-rho)]])
    scale_mat = np.diag([1/s_x, 1/s_y])

    # Apply correction
    data_corr = R @ scale_mat @ data_raw
    return data_corr[0, 0], data_corr[1, 0]

# === Compute heading ===
def compute_heading(mx_corr, my_corr):
    # atan2 gives radians, convert to degrees
    heading = math.degrees(math.atan2(my_corr, mx_corr))
    # Normalize to 0–360 range
    if heading < 0:
        heading += 360
    return heading

# === ZED Initialization ===
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed:", repr(status))
    exit(1)

sensors_data = sl.SensorsData()
mag_data = sl.SensorsData.MagnetometerData()

print("Reading magnetometer and computing heading... (Ctrl+C to stop)\n")

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            mag_data = sensors_data.get_magnetometer_data()

            # Uncalibrated field (ZED SDK gives vector3)
            field_uncal = mag_data.get_magnetic_field_uncalibrated()
            mx, my = field_uncal[0], field_uncal[1]

            # Apply calibration
            mx_corr, my_corr = calibrate(mx, my, bx, by, rho, s_x, s_y)

            # Compute heading
            heading = compute_heading(mx_corr, my_corr)

            print(f"Raw: ({mx:.3f}, {my:.3f}) | Calibrated: ({mx_corr:.3f}, {my_corr:.3f}) | Heading: {heading:.2f}°")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user.")

zed.close()
print("ZED closed.")
