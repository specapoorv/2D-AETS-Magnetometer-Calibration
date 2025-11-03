import pandas as pd
from io import StringIO
import json

outlier_percentage = 0.02  # 2% outliers removed; can increase to 0.1–0.2 for noisier data

def apply_offset_correction(df):
    offset_x = (df['x'].max() + df['x'].min()) / 2
    offset_y = (df['y'].max() + df['y'].min()) / 2
    corrected_df = pd.DataFrame({
        'corrected_x': df['x'] - offset_x,
        'corrected_y': df['y'] - offset_y,
    })
    return offset_x, offset_y, corrected_df


def apply_scale_correction(df):
    delta_x = (df['corrected_x'].max() - df['corrected_x'].min()) / 2
    delta_y = (df['corrected_y'].max() - df['corrected_y'].min()) / 2
    avg_delta = (delta_x + delta_y) / 2

    scale_x = avg_delta / delta_x if delta_x != 0 else 1.0
    scale_y = avg_delta / delta_y if delta_y != 0 else 1.0

    scaled_corrected_df = pd.DataFrame({
        'scaled_corrected_x': df['corrected_x'] * scale_x,
        'scaled_corrected_y': df['corrected_y'] * scale_y,
    })
    return scale_x, scale_y, scaled_corrected_df


def load_and_filter_csv(file_path, outlier_percentile=0.05):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        if len(line.split(',')) >= 2:  # Allow x,y or x,y,z
            filtered_lines.append(line)

    df = pd.read_csv(StringIO(''.join(filtered_lines)))

    # Handle cases with or without 'z' column
    columns_to_filter = [c for c in ['x', 'y'] if c in df.columns]

    for column in columns_to_filter:
        lower = df[column].quantile(outlier_percentile / 2)
        upper = df[column].quantile(1 - outlier_percentile / 2)
        df = df[(df[column] >= lower) & (df[column] <= upper)]

    return df


def main():
    df = load_and_filter_csv("/home/specapoorv/2D-AETS-Magnetometer-Calibration/mag_data_2.csv", outlier_percentage)
    print(f"Samples considered: {len(df)}, Noise removed percentage: {outlier_percentage}")

    offset_x, offset_y, df_offset = apply_offset_correction(df)
    print(f"Offset values: x={offset_x:.3f}, y={offset_y:.3f}")

    scale_x, scale_y, df_scaled = apply_scale_correction(df_offset)
    print(f"Scale values: x={scale_x:.3f}, y={scale_y:.3f}")

    calib = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scale_x": scale_x,
        "scale_y": scale_y
    }

    with open("magnetometer_calibration.json", "w") as f:
        json.dump(calib, f, indent=4)
    print("Saved calibration to magnetometer_calibration.json")


if __name__ == "__main__":
    main()
