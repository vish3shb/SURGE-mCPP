import pandas as pd
import numpy as np
import os
import glob
import time
import networkx as nx
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime

def identify_corners_and_midpoints(data):
    corners = [0]  # Start with the first point as a corner
    midpoints = []
    current_direction = 'x' if data.iloc[1]['x'] == data.iloc[0]['x'] else 'y'

    # Iterate through the points
    for i in range(1, len(data) - 1):
        next_direction = 'x' if data.iloc[i + 1]['x'] == data.iloc[i]['x'] else 'y'
        if next_direction != current_direction:
            corners.append(i)
            current_direction = next_direction

    corners.append(len(data) - 1)  # End with the last point as a corner

    # Identify midpoints of line segments
    for i in range(len(data) - 1):
        mid_x = (data.iloc[i]['x'] + data.iloc[i + 1]['x']) / 2
        mid_y = (data.iloc[i]['y'] + data.iloc[i + 1]['y']) / 2
        midpoints.append((i, mid_x, mid_y))

    return corners, midpoints

def check_connectivity(uav_positions, communication_range):
    if not uav_positions:  # Check if uav_positions is empty
        return False
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
    return data, os.path.basename(path_file).split('_')[1]  # Extract UAV identifier from filename

def check_connectivity_combination(args):
    shift_combination, unique_indices, uav_paths, communication_range, batch_size = args
    shifted_uav_paths = {}
    for i, (uav, data) in enumerate(uav_paths.items()):
        shift, direction = shift_combination[i]
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
        
        if not check_connectivity(uav_positions, communication_range):
            min_required_range = calculate_minimum_required_range(uav_positions)
            return shift_combination, False, shifted_uav_paths, min_required_range

    return shift_combination, True, shifted_uav_paths, communication_range

def generate_shift_combinations(num_points, num_uavs, batch_size):
    max_shift = (num_points // batch_size) + 1
    directions = ['forward', 'reverse']
    for shift_combination in product(product(range(max_shift), directions), repeat=num_uavs):
        yield shift_combination

def save_shifted_paths(shifted_uav_paths, base_output_folder, suffix):
    # Create a timestamped folder within the base output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_output_folder, f"shifted_{timestamp}_{suffix}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for uav, data in shifted_uav_paths.items():
        output_file = os.path.join(output_folder, f'uav_{uav}_shifted_path.txt')
        data.to_csv(output_file, sep=' ', index=False, header=False)
    return output_folder  # Return the actual output folder used

def check_all_combinations(path_folder, communication_range, batch_size, base_output_folder):
    # Start measuring time
    start_time = time.time()

    pattern = os.path.join(path_folder, 'uav_*_path_n.txt')
    uav_paths = {}

    pool = Pool(cpu_count())

    results = pool.map(process_single_uav, glob.glob(pattern))

    for data, uav_name in results:
        uav_paths[uav_name] = data

    unique_indices = set()
    for data in uav_paths.values():
        corners, midpoints = identify_corners_and_midpoints(data)
        unique_indices.update(corners)
        unique_indices.update(midpoints)

    num_points = len(next(iter(uav_paths.values())))
    num_uavs = len(uav_paths)
    max_shift = (num_points // batch_size) + 1
    total_combinations_per_path = max_shift * 2
    total_combinations = total_combinations_per_path ** num_uavs

    print(f"Total UAVs: {num_uavs}")
    print(f"Total points per path: {num_points}")
    print(f"Batch size: {batch_size}")
    print(f"Shifts per path (considering batch size): {max_shift}")
    print(f"Directions per path: 2 (forward and reverse)")
    print(f"Total combinations per path: {total_combinations_per_path}")
    print(f"Total combinations to check: {total_combinations}")

    shift_combinations = list(generate_shift_combinations(num_points, num_uavs, batch_size))

    # Prepare arguments for parallel execution
    args = [(sc, unique_indices, uav_paths, communication_range, batch_size) for sc in shift_combinations]

    # Initialize variables for overall connectivity check
    min_ranges = []

    chunk_size = max(1, len(args) // cpu_count())
    for result in pool.imap_unordered(check_connectivity_combination, args, chunksize=chunk_size):
        shift_combination, shift_connected, shifted_uav_paths, min_range = result
        min_ranges.append((shift_combination, min_range, shifted_uav_paths))

    pool.close()
    pool.join()

    # Find the minimum required range
    min_range_value = min(min_ranges, key=lambda x: x[1])[1]

    # Ensure that only paths corresponding to the minimum required range are saved
    min_range_combinations = []
    for shift_combination, min_range, shifted_uav_paths in min_ranges:
        if min_range == min_range_value:
            actual_min_range = max(calculate_minimum_required_range({uav: (data.iloc[idx]['x'], data.iloc[idx]['y']) for uav, data in shifted_uav_paths.items()}) for idx in unique_indices)
            if actual_min_range <= min_range_value:
                min_range_combinations.append((shift_combination, min_range, shifted_uav_paths))

    # Save shifted paths for combinations with the minimum required range
    for i, (shift_combination, min_range, shifted_uav_paths) in enumerate(min_range_combinations):
        save_shifted_paths(shifted_uav_paths, base_output_folder, f"min_range_{i}")

    total_corners_checked = len(min_ranges)

    # End measuring time
    end_time = time.time()

    # Calculate the total processing time in milliseconds
    processing_time = (end_time - start_time) * 1000  # convert to milliseconds

    return min_range_combinations, min_range_value, processing_time, total_combinations, total_corners_checked

# Example usage
path_folder = '/home/samshad/Documents/multi_uav/visualizations'  # Your specified input directory
base_output_folder = '/home/samshad/Documents/multi_uav/visualizations_shifted'  # Your specified base output directory
communication_range = 100.0  # Example communication range
batch_size = 10  # Example batch size
min_range_combinations, min_range_value, processing_time, total_combinations, total_corners_checked = check_all_combinations(path_folder, communication_range, batch_size, base_output_folder)

# Print the final results
print(f"Total combinations to check: {total_combinations}")
print(f"Total combinations checked: {total_corners_checked}")
print(f"Minimum required range: {min_range_value}")
print(f"Total processing time: {processing_time:.2f} ms")
