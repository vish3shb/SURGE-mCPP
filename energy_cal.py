import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
AIR_DENSITY = 1.225  # kg/m^3
EARTH_GRAVITY = 9.8  # m/s^2
PROPELLER_EFFICIENCY = 0.391
RANGE_POWER_CONSUMPTION_COEFF = 1.092
MOTOR_EFFICIENCY = 1.0
CRUISING_VELOCITY = 5.0  # m/s
TURNING_VELOCITY = 2.0  # m/s
UAV_MASS = 1.0  # kg, assumed constant mass for simplicity
ANGLE_THRESHOLD = 5.0  # degrees, for stitching paths

def read_path(file_path):
    """Reads a path file and returns the coordinates."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = []
    for line in lines:
        _, x, y = map(float, line.strip().split())
        coordinates.append((x, y))
    return coordinates

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points p1, p2, and p3."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    return np.abs(np.degrees(angle))

def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

def stitch_path(coordinates, angle_threshold):
    """Stitches path segments based on angle threshold."""
    if len(coordinates) < 3:
        return coordinates

    stitched_coords = [coordinates[0], coordinates[1]]
    for i in range(2, len(coordinates)):
        angle = calculate_angle(stitched_coords[-2], stitched_coords[-1], coordinates[i])
        if angle >= angle_threshold:
            stitched_coords.append(coordinates[i])
        else:
            stitched_coords[-1] = coordinates[i]
    return stitched_coords

def calculate_path_energy_with_turning_velocity(path, cruising_velocity, turning_velocity, mass):
    if len(path) < 2:
        return 0, 0, 0
    
    total_energy = 0
    total_length = 0
    num_turns = 0

    for i in range(1, len(path) - 1):
        angle = calculate_angle(path[i - 1], path[i], path[i + 1])
        
        # Energy for deceleration to turning velocity
        deceleration_energy = calculate_kinetic_energy(mass, cruising_velocity) - calculate_kinetic_energy(mass, turning_velocity)
        total_energy += deceleration_energy
        
        # Simplified energy calculation for the turn
        turn_energy = abs(angle) * turning_velocity  # This is a simplification
        total_energy += turn_energy
        
        # Energy for acceleration back to cruising velocity
        acceleration_energy = calculate_kinetic_energy(mass, cruising_velocity) - calculate_kinetic_energy(mass, turning_velocity)
        total_energy += acceleration_energy
        
        num_turns += 1
    
    for i in range(len(path) - 1):
        distance = distance_between_points(path[i], path[i + 1])
        total_length += distance
        # Simplified energy calculation for straight segments
        straight_line_energy = distance * cruising_velocity  # More realistic would be distance * power consumption rate
        total_energy += straight_line_energy
    
    return total_energy, num_turns, total_length

def analyze_paths(directory, file_pattern, angle_threshold, result_file):
    """Analyzes paths in the specified directory matching the file pattern and saves the results."""
    files = glob.glob(os.path.join(directory, file_pattern))
    
    analysis_results = {}
    
    for file in files:
        file_name = os.path.basename(file)
        iteration = file_name.split('_')[0]
        
        if iteration not in analysis_results:
            analysis_results[iteration] = {
                'energy': [],
                'turns': [],
                'lengths': [],
                'total_energy': 0,
                'total_turns': 0,
                'total_length': 0
            }
        
        coordinates = read_path(file)
        stitched_coords = stitch_path(coordinates, angle_threshold)
        
        total_energy, num_turns, total_length = calculate_path_energy_with_turning_velocity(stitched_coords, CRUISING_VELOCITY, TURNING_VELOCITY, UAV_MASS)
        
        analysis_results[iteration]['energy'].append(total_energy)
        analysis_results[iteration]['turns'].append(num_turns)
        analysis_results[iteration]['lengths'].append(total_length)
        analysis_results[iteration]['total_energy'] += total_energy
        analysis_results[iteration]['total_turns'] += num_turns
        analysis_results[iteration]['total_length'] += total_length
    
    # Sort iterations by total energy in ascending order
    sorted_iterations = sorted(analysis_results.items(), key=lambda item: item[1]['total_energy'])
    
    with open(result_file, 'w') as result:
        for iteration, data in sorted_iterations:
            result.write(f"Iteration: {iteration}\n")
            result.write(f"Energy for paths: {data['energy']} J\n")
            result.write(f"Number of turns: {data['turns']}\n")
            result.write(f"Path lengths: {data['lengths']} m\n")
            result.write(f"Total energy: {data['total_energy']:.2f} J\n")
            result.write(f"Total number of turns: {data['total_turns']}\n")
            result.write(f"Total path length: {data['total_length']:.2f} m\n\n")
    
    print(f"Analysis results saved to {result_file}")

# Base directory (current working directory)
base_directory = os.getcwd()

# Paths
folder1 = os.path.join(base_directory, 'Results', 'least_turns_results')
folder2 = os.path.join(base_directory, 'Results', 'allowed_distance_results')

# File pattern and result file
file_pattern = '*_uav_*_sampled_path.txt'
result_file1 = os.path.join(folder1, 'Energy_analysis_results.txt')
result_file2 = os.path.join(folder2, 'Energy_analysis_results.txt')

# Analyze paths and save results for both folders
analyze_paths(folder1, file_pattern, ANGLE_THRESHOLD, result_file1)
analyze_paths(folder2, file_pattern, ANGLE_THRESHOLD, result_file2)
