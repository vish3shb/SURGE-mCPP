import pandas as pd
import numpy as np
import os
import glob
import time
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing
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

    for idx in unique_indices:
        if isinstance(idx, int):
            uav_positions = {uav: (data.iloc[idx]['x'], data.iloc[idx]['y']) for uav, data in shifted_uav_paths.items() if idx < len(data)}
        else:
            uav_positions = {uav: ((data.iloc[idx[0]]['x'] + data.iloc[idx[0] + 1]['x']) / 2, (data.iloc[idx[0]]['y'] + data.iloc[idx[0] + 1]['y']) / 2) for uav, data in shifted_uav_paths.items() if idx[0] < len(data) - 1}
        
        if not check_connectivity(uav_positions, communication_range):
            return shift_combination, False, shifted_uav_paths

    return shift_combination, True, shifted_uav_paths

def generate_shift_combinations(num_points, num_uavs, batch_size):
    max_shift = (num_points // batch_size) + 1
    directions = ['forward', 'reverse']
    for shift_combination in product(product(range(max_shift), directions), repeat=num_uavs):
        yield shift_combination

def save_shifted_paths(shifted_uav_paths, base_output_folder):
    # Create a timestamped folder within the base output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_output_folder, f"shifted_{timestamp}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for uav, data in shifted_uav_paths.items():
        output_file = os.path.join(output_folder, f'uav_{uav}_shifted_path.txt')
        data.to_csv(output_file, sep=' ', index=False, header=False)
    return output_folder  # Return the actual output folder used

def check_all_combinations(path_folder, communication_range, batch_size, base_output_folder):
    # Start measuring time
    start_time = time.time()

    pattern = os.path.join(path_folder, 'uav_*_path.txt')
    uav_paths = {}

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_uav, glob.glob(pattern)))

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
    shift_combinations = generate_shift_combinations(num_points, num_uavs, batch_size)
    total_combinations = max_shift ** num_uavs * 2 ** num_uavs

    # Print the details of combinations
    print(f"Total UAVs: {num_uavs}")
    print(f"Total points per path: {num_points}")
    print(f"Batch size: {batch_size}")
    print(f"Shifts per path (considering batch size): {max_shift}")
    print(f"Directions per path: 2 (forward and reverse)")
    print(f"Total combinations per path: {max_shift * 2}")
    print(f"Total combinations to check: {total_combinations}")

    # Prepare arguments for parallel execution
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, total_combinations // num_cores)
    args = [(sc, unique_indices, uav_paths, communication_range, batch_size) for sc in shift_combinations]

    # Initialize variables for overall connectivity check
    all_combinations_connected = False
    output_folder = None

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in executor.map(check_connectivity_combination, args, chunksize=chunk_size):
            shift_combination, shift_connected, shifted_uav_paths = result
            if shift_connected:
                all_combinations_connected = True
                # Save the new shifted paths
                output_folder = save_shifted_paths(shifted_uav_paths, base_output_folder)
                break  # Stop as soon as we find one combination where all corners are connected

    total_corners_checked = len(args)
    connectivity_percentage = 100.0 if all_combinations_connected else 0.0

    # End measuring time
    end_time = time.time()

    # Calculate the total processing time in milliseconds
    processing_time = (end_time - start_time) * 1000  # convert to milliseconds

    return all_combinations_connected, connectivity_percentage, processing_time, total_combinations, total_corners_checked, output_folder

# Example usage
path_folder = '/home/samshad/Documents/multi_uav/CA_DARP'  # Your specified input directory
base_output_folder = '/home/samshad/Documents/multi_uav/CA_DARP'  # Your specified base output directory
communication_range = 110.0  # Example communication range
batch_size = 10  # Example batch size
all_connected, connectivity_percentage, processing_time, total_combinations, total_corners_checked, output_folder = check_all_combinations(path_folder, communication_range, batch_size, base_output_folder)

# Print the final results
print(f"Total combinations to check: {total_combinations}")
print(f"Total combinations checked: {total_corners_checked}")
print(f"UAVs are {'connected' if all_connected else 'not connected'} at all corners for at least one combination of shifts and directions.")
print(f"Percentage of corners where UAVs are connected: {connectivity_percentage:.2f}%")
print(f"Total processing time: {processing_time:.2f} ms")
print(f"Shifted paths saved in: {output_folder}")


# connectivity check with shifting starting position and reverse direction slao considered. 
# mesh connectivity check not performing for all waypoints, only at the cornetrs and mid points of line segments. 
# combination check will stop once a feasible path found and feasible path saved to output folder 
# https://chatgpt.com/share/af47d9cf-007f-428d-8641-47ab86da8321