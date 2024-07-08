import os
import fnmatch
import matplotlib.pyplot as plt
from math import sqrt

def read_path_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    coordinates = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].replace('.', '', 1).isdigit() and parts[2].replace('.', '', 1).isdigit():
            timestamp, x, y = parts
            coordinates.append((int(timestamp), float(x), float(y)))
    return coordinates

def calculate_distance(coord1, coord2):
    return sqrt((coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

def check_mesh_connectivity(robots_coordinates, distance_threshold):
    timestamps = sorted(set(ts for coords in robots_coordinates.values() for ts, _, _ in coords))
    for timestamp in timestamps:
        coords_at_timestamp = []
        for robot, coords in robots_coordinates.items():
            try:
                coord = next((ts, x, y) for ts, x, y in coords if ts == timestamp)
                coords_at_timestamp.append(coord)
            except StopIteration:
                print(f"Timestamp {timestamp} not found for {robot}. Skipping this timestamp.")
                break
        else:  # This else belongs to the for loop, not the try-except block
            for i in range(len(coords_at_timestamp)):
                for j in range(i + 1, len(coords_at_timestamp)):
                    if calculate_distance(coords_at_timestamp[i], coords_at_timestamp[j]) > distance_threshold:
                        return False, timestamp, coords_at_timestamp[i], coords_at_timestamp[j]
    return True, None, None, None

def plot_paths_from_folder(folder_path, distance_threshold):
    plt.figure(figsize=(14, 10))
    robots_coordinates = {}
    for file_name in os.listdir(folder_path):
        if fnmatch.fnmatch(file_name, 'uav_*_path_n.txt'):
            file_path = os.path.join(folder_path, file_name)
            coordinates = read_path_file(file_path)
            robots_coordinates[file_name] = coordinates
            if coordinates:  # Check if coordinates list is not empty
                timestamps = [coord[0] for coord in coordinates]
                x_vals = [coord[1] for coord in coordinates]
                y_vals = [coord[2] for coord in coordinates]
                plt.plot(x_vals, y_vals, label=file_name)
                for i, txt in enumerate(timestamps):
                    plt.annotate(txt, (x_vals[i], y_vals[i]), fontsize=8, color='red')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Paths from Multiple Files with Timestamps')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling
    plt.show()

    # Check mesh connectivity
    is_connected, timestamp, coord1, coord2 = check_mesh_connectivity(robots_coordinates, distance_threshold)
    if is_connected:
        print("Robots are connected in a mesh configuration throughout the mission.")
    else:
        print(f"Robots lost connectivity at timestamp {timestamp} between {coord1} and {coord2}.")

# Specify the folder containing the path files and the distance threshold
folder_path = '/home/samshad/Documents/multi_uav/visualizations'  # Your specified input directory
distance_threshold = 247  # Replace with your desired distance threshold

plot_paths_from_folder(folder_path, distance_threshold)
