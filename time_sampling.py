import pandas as pd
import numpy as np
import os
import json5

# Use the script's directory as the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'config.json5')

# Load the configuration file
with open(config_path, 'r') as config_file:
    config = json5.load(config_file)

def time_sample_paths(path_files, output_directory):
    sampled_path_files = []
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i, path_file in enumerate(path_files):
        data = pd.read_csv(path_file, sep='\t', header=None, names=['x', 'y'])

        x = np.array(data['x'])
        y = np.array(data['y'])

        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)

        corner_threshold = np.pi / 4  # 45 degrees
        corner_indices = np.where(np.abs(np.diff(angles)) > corner_threshold)[0] + 1

        straight_speed = config.get('Forward_speed_of_uav')
        corner_speed = config.get('Turning_speed_of_uav')
        sampling_rate = 2  # seconds
        hit_radius = config.get('Turning_radius_of_uav')

        waypoints = []
        time = 0

        def distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))

        for j in range(len(x) - 1):
            start_point = np.array([x[j], y[j]])
            end_point = np.array([x[j + 1], y[j + 1]])
            segment_distance = distance(start_point, end_point)

            is_near_corner = any(distance(start_point, [x[c], y[c]]) <= hit_radius or
                                 distance(end_point, [x[c], y[c]]) <= hit_radius
                                 for c in corner_indices)

            speed = corner_speed if is_near_corner else straight_speed

            num_samples = int(np.ceil(segment_distance / speed))

            for k in range(num_samples):
                waypoint = start_point + (end_point - start_point) * (k / num_samples)
                waypoints.append((time, waypoint[0], waypoint[1]))
                time += sampling_rate

        waypoints.append((time, x[-1], y[-1]))

        waypoints_df = pd.DataFrame(waypoints, columns=['time', 'x', 'y'])

        output_file_path = os.path.join(output_directory, f"uav_{i+1}_sampled_path.txt")
        with open(output_file_path, 'w') as f:
            for idx, (time, x, y) in enumerate(waypoints):
                f.write(f"{idx} {x:.2f} {y:.2f}\n")

        sampled_path_files.append(output_file_path)
        print(f"Waypoints generated and saved to {output_file_path}")

    return sampled_path_files
