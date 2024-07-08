import os
import fnmatch
from math import sqrt
import numpy as np
import time
import optuna

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def objective(trial):
    combination = []
    for key in all_paths_keys:
        shift = trial.suggest_int(f'shift_{key}', -shifts + step_size, shifts - step_size)
        combination.append(shift)
    
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
    
    if verbose:
        print(f"Combination {combination}: Maximum required distance = {max_required_distance:.2f}")
    
    return max_required_distance

def plot_paths_from_files(path_files, step_size, n_trials=100, verbose=False):
    global all_paths_keys, original_paths, shifts, robots_coordinates
    
    robots_coordinates = {}
    for file_path in path_files:
        coordinates = read_path_file(file_path)
        robots_coordinates[file_path] = ensure_closed_path(coordinates)
    
    num_paths = len(robots_coordinates)
    print(f"Number of UAV paths detected: {num_paths}")

    # Set up the optimization parameters
    all_paths_keys = sorted(robots_coordinates.keys())
    original_paths = {key: robots_coordinates[key] for key in all_paths_keys}
    shifts = len(original_paths[all_paths_keys[0]])
    
    start_time = time.time()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)  # You can increase the number of trials for better optimization
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print the best results
    best_combination = study.best_params
    best_distance = study.best_value
    
    print(f"\nMinimum required distance is {best_distance:.2f} with the best shift combination:")
    for key in all_paths_keys:
        print(f"{key}: shift {best_combination[f'shift_{key}']}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Time taken for processing: {processing_time:.2f} seconds")
    
    return best_distance, best_combination  # Return the best distance and shift combination

# Specify the step size for shifting
step_size = 1  # Set the desired step size for shifting

# Parameters for optimization
n_trials = 500  # Number of trials for optimization
verbose = False  # Set to True to enable logging and showing intermediate results

# Example call to plot_paths_from_files
# Replace path_files with the list of your path files
# path_files = ['path/to/uav_1_sampled_path.txt', 'path/to/uav_2_sampled_path.txt', ...]
# plot_paths_from_files(path_files, step_size, n_trials, verbose)
