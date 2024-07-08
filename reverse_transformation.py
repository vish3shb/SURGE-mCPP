import os
import json
import pandas as pd
import numpy as np

def load_transformation_parameters(json_file):
    """
    Loads the transformation parameters from a JSON file.
    """
    with open(json_file, 'r') as f:
        params = json.load(f)
    bottom_left_corner = params['bottom_left_corner']
    rotation_angle = params['rotation_angle']
    return bottom_left_corner[0], bottom_left_corner[1], rotation_angle

def transform_path(file_path, shift_x, shift_y, rotation_angle):
    """
    Transforms the path coordinates by shifting them by the given values and then rotating.
    """
    # Read the path file into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['index', 'x', 'y'])
    
    # Apply the shift transformation
    df['x'] += shift_x
    df['y'] += shift_y
    
    # Apply the rotation transformation
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    df['x_rot'] = df['x'] * cos_angle - df['y'] * sin_angle
    df['y_rot'] = df['x'] * sin_angle + df['y'] * cos_angle
    
    df = df.drop(columns=['x', 'y'])
    df.rename(columns={'x_rot': 'x', 'y_rot': 'y'}, inplace=True)
    
    return df

def process_all_paths_in_folder(folder_path, output_folder, shift_x, shift_y, rotation_angle):
    """
    Processes all path files in the given folder, applies transformation, and saves them to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_path.txt'):
            file_path = os.path.join(folder_path, file_name)
            transformed_df = transform_path(file_path, shift_x, shift_y, rotation_angle)
            
            # Save the transformed path to a new file
            output_file_path = os.path.join(output_folder, file_name)
            transformed_df.to_csv(output_file_path, sep=' ', index=False, header=False)
            print(f"Processed and saved: {output_file_path}")

# Define the input folders, output folders, and JSON file path relative to the parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
input_folders = [
    os.path.join(parent_dir, 'Results', 'allowed_distance_results'),
    os.path.join(parent_dir, 'Results', 'least_turns_results')
]
output_folders = [
    os.path.join(parent_dir, 'Results', 'allowed_distance_results', 'transformed'),
    os.path.join(parent_dir, 'Results', 'least_turns_results', 'transformed')
]
json_file = os.path.join(parent_dir, 'transformation_parameters.json')  # JSON file in the same folder as the script

# Load transformation parameters from the JSON file
shift_x, shift_y, rotation_angle = load_transformation_parameters(json_file)

# Process all path files in the folders
for input_folder, output_folder in zip(input_folders, output_folders):
    process_all_paths_in_folder(input_folder, output_folder, shift_x, shift_y, rotation_angle)
