import pyzed.sl as sl
import csv
import time

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.MILLIGAUSS  # or sl.UNIT.GAUSS / TESLA depending on what you prefer
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed:", repr(status))
    exit(1)

sensors_data = sl.SensorsData()
mag_data = sl.SensorsData.MagnetometerData()

# Open CSV file
time = time.time()
with open(f"data_{time}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y"])  # header row

    print("Collecting magnetometer data... Rotate the rover slowly 360° a few times.")
    start = time.time()
    while time.time() - start < 60:  # collect for ~1 minute
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            mag_data = sensors_data.get_magnetometer_data()

            # Get uncalibrated magnetic field (3D vector)
            mx, my = mag_data.get_magnetic_field_uncalibrated()
            writer.writerow([mx, my])

        time.sleep(0.05)  # small delay to reduce load

zed.close()
print("Done! Saved magnetometer data to data.csv")
