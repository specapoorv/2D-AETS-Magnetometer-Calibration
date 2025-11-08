import pyzed.sl as sl
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# === Calibration parameters from your 2D-AETS fit ===
bx, by = -0.04030177317407155, 1.0210029484619534
rho = 0.012730970505513004
s_x, s_y = 2.904204630320402, 0.6236255637525812

# === Calibration function ===
def calibrate(mx, my, bx, by, rho, s_x, s_y):
    data_raw = np.array([[mx], [my]]) - np.array([[bx], [by]])
    scale_mat = np.diag([1/s_x, 1/s_y])
    R = np.array([[np.cos(rho), np.sin(rho)],
                  [-np.sin(rho), np.cos(rho)]])
    data_corr = R @ (scale_mat @ data_raw)
    return data_corr[0, 0], data_corr[1, 0]

# === Compute heading ===
def compute_heading(mx_corr, my_corr):
    heading = math.degrees(math.atan2(my_corr, mx_corr))
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

# === Lists to accumulate readings ===
raw_x, raw_y = [], []
corr_x, corr_y = [], []
headings = []

print("Reading magnetometer and computing heading... (Ctrl+C to stop)\n")

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            mag_data = sensors_data.get_magnetometer_data()
            mx, my = mag_data.get_magnetic_field_uncalibrated()

            # Apply calibration
            mx_corr, my_corr = calibrate(mx, my, bx, by, rho, s_x, s_y)

            # Compute heading
            heading = compute_heading(mx_corr, my_corr)

            # Store data
            raw_x.append(mx)
            raw_y.append(my)
            corr_x.append(mx_corr)
            corr_y.append(my_corr)
            headings.append(heading)

            print(f"Raw: ({mx:.3f}, {my:.3f}) | Calibrated: ({mx_corr:.3f}, {my_corr:.3f}) | Heading: {heading:.2f}°")

            # Once we reach 50 points, plot
            if len(raw_x) >= 50:
                plt.figure(figsize=(6,6))
                plt.scatter(raw_x, raw_y, color='r', label='Raw')
                plt.scatter(corr_x, corr_y, color='g', label='Corrected')
                # Circle of radius 1 for reference
                circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
                plt.gca().add_artist(circle)
                plt.axis('equal')
                plt.xlabel("X (µT)")
                plt.ylabel("Y (µT)")
                plt.title("Magnetometer Calibration (50 points)")
                plt.legend()
                plt.grid(True)
                plt.show()

                # Plot headings
                plt.figure()
                plt.plot(headings, marker='o')
                plt.xlabel("Sample")
                plt.ylabel("Heading (deg)")
                plt.title("Computed Heading (50 points)")
                plt.grid(True)
                plt.show()

                # Clear lists for next batch
                raw_x.clear()
                raw_y.clear()
                corr_x.clear()
                corr_y.clear()
                headings.clear()

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user.")

zed.close()
print("ZED closed.")
