import pandas as pd
import numpy as np
import os
import glob
import networkx as nx
from itertools import product

def load_paths(path_folder):
    pattern = os.path.join(path_folder, 'uav_*_path_n.txt')
    uav_paths = {}
    for path_file in glob.glob(pattern):
        data = pd.read_csv(path_file, delim_whitespace=True, header=None, names=['index', 'x', 'y'])
        uav_name = os.path.basename(path_file).split('_')[1]
        uav_paths[uav_name] = data
    return uav_paths

def identify_key_points(data):
    corners = [0]
    current_direction = 'x' if data.iloc[1]['x'] == data.iloc[0]['x'] else 'y'
    for i in range(1, len(data) - 1):
        next_direction = 'x' if data.iloc[i + 1]['x'] == data.iloc[i]['x'] else 'y'
        if next_direction != current_direction:
            corners.append(i)
            current_direction = next_direction
    corners.append(len(data) - 1)
    return corners

def predict_positions(uav_paths, time_intervals, speed):
    predicted_positions = {t: {} for t in time_intervals}
    for uav, path in uav_paths.items():
        for t in time_intervals:
            index = int((t * speed) % len(path))
            predicted_positions[t][uav] = (path.iloc[index]['x'], path.iloc[index]['y'])
    return predicted_positions

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def check_connectivity(predicted_positions, communication_range):
    for t, positions in predicted_positions.items():
        G = nx.Graph()
        for uav, pos in positions.items():
            G.add_node(uav, pos=pos)
        for uav1, pos1 in positions.items():
            for uav2, pos2 in positions.items():
                if uav1 != uav2 and manhattan_distance(pos1, pos2) <= communication_range:
                    G.add_edge(uav1, uav2)
        if not nx.is_connected(G):
            return False
    return True

def find_optimal_start(uav_paths, communication_range, speed, max_time):
    time_intervals = np.arange(0, max_time, 1)
    optimal_shift = None
    max_connected = 0

    for shifts in product(range(len(next(iter(uav_paths.values())))), repeat=len(uav_paths)):
        shifted_paths = {uav: path.shift(-shift).reset_index(drop=True) for uav, (path, shift) in zip(uav_paths.keys(), shifts)}
        predicted_positions = predict_positions(shifted_paths, time_intervals, speed)
        if check_connectivity(predicted_positions, communication_range):
            connected_count = sum(check_connectivity({t: {uav: (path.iloc[(t * speed) % len(path)]['x'], path.iloc[(t * speed) % len(path)]['y']) for uav, path in shifted_paths.items()}}, communication_range) for t in time_intervals)
            if connected_count > max_connected:
                max_connected = connected_count
                optimal_shift = shifts
    return optimal_shift

# Parameters
path_folder = '/home/samshad/Documents/multi_uav/visualizations'
communication_range = 160.0
speed = 1.0  # Assume 1 unit distance per time unit
max_time = 100  # Max time to simulate

# Load paths and find optimal start
uav_paths = load_paths(path_folder)
optimal_shifts = find_optimal_start(uav_paths, communication_range, speed, max_time)

print(f"Optimal shifts for connectivity: {optimal_shifts}")
