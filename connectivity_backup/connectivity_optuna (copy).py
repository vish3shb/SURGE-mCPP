import pandas as pd
import numpy as np
import os
import glob
import networkx as nx
import optuna
import time
from datetime import datetime
import logging

def identify_corners_and_midpoints(data):
    corners = [0]
    midpoints = []
    current_direction = 'x' if data.iloc[1]['x'] == data.iloc[0]['x'] else 'y'
    for i in range(1, len(data) - 1):
        next_direction = 'x' if data.iloc[i + 1]['x'] == data.iloc[i]['x'] else 'y'
        if next_direction != current_direction:
            corners.append(i)
            current_direction = next_direction
    corners.append(len(data) - 1)
    for i in range(len(data) - 1):
        mid_x = (data.iloc[i]['x'] + data.iloc[i + 1]['x']) / 2
        mid_y = (data.iloc[i]['y'] + data.iloc[i + 1]['y']) / 2
        midpoints.append((i, mid_x, mid_y))
    return corners, midpoints

def check_connectivity(uav_positions, communication_range):
    G = nx.Graph()
    for uav, pos in uav_positions.items():
        G.add_node(uav, pos=pos)
    for uav1, pos1 in uav_positions.items():
        for uav2, pos2 in uav_positions.items():
            if uav1 != uav2:
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if distance <= communication_range:
                    G.add_edge(uav1, uav2)
    return nx.is_connected(G)

def calculate_minimum_required_range(uav_positions):
    max_distance = 0
    for uav1, pos1 in uav_positions.items():
        for uav2, pos2 in uav_positions.items():
            if uav1 != uav2:
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if distance > max_distance:
                    max_distance = distance
    return max_distance

def shift_path(data, shift, batch_size):
    shift = (shift * batch_size) % len(data)
    if shift == 0:
        return data.copy()
    shifted_data = data.shift(-shift).reset_index(drop=True)
    shifted_data.iloc[-shift:] = data.iloc[:shift].reset_index(drop=True)
    shifted_data['index'] = range(len(shifted_data))
    return shifted_data

def reverse_path(data):
    reversed_data = data.iloc[::-1].reset_index(drop=True)
    reversed_data['index'] = range(len(reversed_data))
    return reversed_data

def process_single_uav(path_file):
    data = pd.read_csv(path_file, delim_whitespace=True, header=None, names=['index', 'x', 'y'])
    return data, os.path.basename(path_file).split('_')[1]

def evaluate_individual(trial, uav_paths, unique_indices, batch_size, max_shift, print_intermediate):
    shifted_uav_paths = {}
    for uav in uav_paths.keys():
        shift = trial.suggest_int(f"{uav}_shift", 0, max_shift - 1)
        direction = trial.suggest_categorical(f"{uav}_direction", ['forward', 'reverse'])
        data = uav_paths[uav]
        shifted_data = shift_path(data, shift, batch_size)
        if direction == 'reverse':
            shifted_data = reverse_path(shifted_data)
        shifted_uav_paths[uav] = shifted_data
    min_required_range = 0
    for idx in unique_indices:
        if isinstance(idx, int):
            uav_positions = {uav: (data.iloc[idx]['x'], data.iloc[idx]['y']) for uav, data in shifted_uav_paths.items() if idx < len(data)}
        else:
            uav_positions = {uav: ((data.iloc[idx[0]]['x'] + data.iloc[idx[0] + 1]['x']) / 2, (data.iloc[idx[0]]['y'] + data.iloc[idx[0] + 1]['y']) / 2) for uav, data in shifted_uav_paths.items() if idx[0] < len(data) - 1}
        if not check_connectivity(uav_positions, min_required_range):
            min_required_range = calculate_minimum_required_range(uav_positions)
    if print_intermediate:
        print(f"Trial {trial.number}: Minimum required range = {min_required_range}")
    return min_required_range

def optimize_starting_points(uav_paths, unique_indices, batch_size, max_shift, n_trials, print_intermediate, n_startup_trials, n_jobs):
    # Create a unique study name
    study_name = f"UAV_Connectivity_Optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name,  # Use unique study name
        storage='sqlite:///uav_connectivity_optimization.db',
        load_if_exists=False  # Do not load if the study exists
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start_time = time.time()
    study.optimize(
        lambda trial: evaluate_individual(trial, uav_paths, unique_indices, batch_size, max_shift, print_intermediate),
        n_trials=n_trials,
        timeout=600,
        n_jobs=n_jobs  # Parallel execution with specified number of jobs
    )
    end_time = time.time()
    processing_time = end_time - start_time
    return study.best_params, study.best_value, processing_time

# Example usage
path_folder = '/home/samshad/Documents/multi_uav/visualizations'
pattern = os.path.join(path_folder, 'uav_*_sampled_path.txt')
uav_paths = {}
for path_file in glob.glob(pattern):
    data, uav_name = process_single_uav(path_file)
    uav_paths[uav_name] = data
unique_indices = set()
for data in uav_paths.values():
    corners, midpoints = identify_corners_and_midpoints(data)
    unique_indices.update(corners)
    unique_indices.update(midpoints)
num_points = len(next(iter(uav_paths.values())))
max_shift = (num_points // 10) + 1
print_intermediate = False
n_startup_trials = 5  # Set the number of initial random samples
n_jobs = -1  # Use all available cores for parallel processing
best_params, best_fitness, processing_time = optimize_starting_points(uav_paths, unique_indices, batch_size=2, max_shift=max_shift, n_trials=200, print_intermediate=print_intermediate, n_startup_trials=n_startup_trials, n_jobs=n_jobs)
print(f"Best parameters: {best_params}")
print(f"Best fitness (minimum required range): {best_fitness}")
print(f"Total processing time: {processing_time:.2f} seconds")
