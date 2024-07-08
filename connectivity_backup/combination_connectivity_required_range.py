import os
import fnmatch
import itertools
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import time
from multiprocessing import Pool, cpu_count

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

def shift_path(path_data, shift):
    times = [entry[0] for entry in path_data]
    coordinates = [(entry[1], entry[2]) for entry in path_data]
    shifted_coordinates = coordinates[shift:] + coordinates[:shift]
    return [(times[i], shifted_coordinates[i][0], shifted_coordinates[i][1]) for i in range(len(times))]

def reverse_shift_path(path_data, shift):
    times = [entry[0] for entry in path_data]
    coordinates = [(entry[1], entry[2]) for entry in path_data]
    shifted_coordinates = coordinates[-shift:] + coordinates[:-shift]
    return [(times[i], shifted_coordinates[i][0], shifted_coordinates[i][1]) for i in range(len(times))]

def ensure_closed_path(path_data):
    if path_data[0][1:] != path_data[-1][1:]:
        path_data.append(path_data[0])
    return path_data

def calculate_distance(coord1, coord2):
    return sqrt((coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

def calculate_required_distance(coords_at_timestamp):
    n = len(coords_at_timestamp)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            distances[i][j] = distances[j][i] = calculate_distance(coords_at_timestamp[i], coords_at_timestamp[j])
    
    max_distance = 0
    for i in range(n):
        max_distance = max(max_distance, np.min(np.sort(distances[i])[1:]))
    
    return max_distance

def process_combination(args):
    combination, original_paths, all_paths_keys, shifts = args
    robots_coordinates = {key: original_paths[key] for key in all_paths_keys}
    
    for idx, shift in enumerate(combination):
        if shift >= 0:
            robots_coordinates[all_paths_keys[idx]] = shift_path(original_paths[all_paths_keys[idx]], shift)
        else:
            robots_coordinates[all_paths_keys[idx]] = reverse_shift_path(original_paths[all_paths_keys[idx]], -shift)

    max_required_distance = 0
    timestamps = sorted(set(ts for coords in robots_coordinates.values() for ts, _, _ in coords))
    for timestamp in timestamps:
        coords_at_timestamp = []
        for robot, coords in robots_coordinates.items():
            try:
                coord = next((ts, x, y) for ts, x, y in coords if ts == timestamp)
                coords_at_timestamp.append(coord)
            except StopIteration:
                break
        else:  # This else belongs to the for loop, not the try-except block
            required_distance = calculate_required_distance(coords_at_timestamp)
            max_required_distance = max(max_required_distance, required_distance)
    
    return combination, max_required_distance

def plot_paths_from_folder(folder_path, step_size):
    plt.figure(figsize=(14, 10))
    robots_coordinates = {}
    for file_name in os.listdir(folder_path):
        if fnmatch.fnmatch(file_name, 'uav_*_path_n.txt'):
            file_path = os.path.join(folder_path, file_name)
            coordinates = read_path_file(file_path)
            robots_coordinates[file_name] = ensure_closed_path(coordinates)
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
    #plt.show()

    # Calculate required distance threshold for all combinations of shifts
    start_time = time.time()
    all_paths_keys = sorted(robots_coordinates.keys())
    original_paths = {key: robots_coordinates[key] for key in all_paths_keys}
    shifts = len(original_paths[all_paths_keys[0]])
    
    shift_options = list(range(0, shifts, step_size)) + list(range(-shifts + step_size, 0, step_size))
    all_combinations = list(itertools.product(shift_options, repeat=len(all_paths_keys)))
    
    min_combination = None
    min_required_distance = float('inf')
    
    total_combinations = len(all_combinations)
    
    with Pool(cpu_count()) as pool:
        args = [(combination, original_paths, all_paths_keys, shifts) for combination in all_combinations]
        results = pool.map(process_combination, args)
    
    for combination, max_required_distance in results:
        if max_required_distance < min_required_distance:
            min_required_distance = max_required_distance
            min_combination = combination
        
        #print(f"Combination {combination}: Maximum required distance = {max_required_distance:.2f}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Breakout of total combinations
    shift_counts = [len(shift_options)] * len(all_paths_keys)
    shift_combinations = " x ".join(map(str, shift_counts))
    
    print(f"\nMinimum required distance is {min_required_distance:.2f} with shift combination {min_combination}.")
    print(f"Total combinations tested: {total_combinations} ({shift_combinations})")
    print(f"Time taken for processing: {processing_time:.2f} seconds")

# Specify the folder containing the path files and the step size for shifting
folder_path = '/home/samshad/Documents/multi_uav/visualizations'  # Your specified input directory
step_size = 10  # Set the desired step size for shifting

plot_paths_from_folder(folder_path, step_size)
