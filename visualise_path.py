import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Base folder relative to the script location
base_folder = os.path.dirname(os.path.abspath(__file__))

# Folder containing path files
path_folder = os.path.join(base_folder, 'DARP')

# Get a list of all path files in the folder
path_files = glob.glob(os.path.join(path_folder, '*_path.txt'))

# Polygon file path
polygon_file_path = os.path.join(base_folder, 'flyzon_corrected.poly')

# Initialize the plot
plt.figure(figsize=(8, 8))

# Loop through each path file and plot the data
for file_path in path_files:
    data = pd.read_csv(file_path, sep='\t', header=None, names=['x', 'y'])
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    plt.plot(x, y, marker='o', label=os.path.basename(file_path))
    
    # Annotate extreme points
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    plt.annotate(f'({min_x}, {min_y})', xy=(min_x, min_y), xytext=(min_x, min_y),
                 textcoords='offset points', ha='right', color='red')
    plt.annotate(f'({max_x}, {max_y})', xy=(max_x, max_y), xytext=(max_x, max_y),
                 textcoords='offset points', ha='right', color='red')

# Read and plot the polygon file
polygon_data = pd.read_csv(polygon_file_path, sep=' ', header=None, names=['x', 'y'])
polygon_x = polygon_data['x'].to_numpy()
polygon_y = polygon_data['y'].to_numpy()
plt.plot(marker='x', color='black', linestyle='-', linewidth=2, label='Polygon')
plt.plot(polygon_x, polygon_y, marker='x', color='black', linestyle='-', linewidth=2, label='Polygon')

# Add plot labels and title
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Robot Paths and Polygon')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
