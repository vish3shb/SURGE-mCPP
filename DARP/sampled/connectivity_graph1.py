import os
import glob
import matplotlib.pyplot as plt

# Function to read path from file
def read_path_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        path = [[float(num) for num in line.strip().split()] for line in lines]
    return path

# Function to reposition waypoints to the center of the cell
def reposition_waypoints(path):
    new_path = []
    for point in path:
        time_index, x, y = point
        new_x = (int(x // 10) * 10) + 5
        new_y = (int(y // 10) * 10) + 5
        new_path.append([time_index, new_x, new_y])
    return new_path

# Function to save the new path to file
def save_path_to_file(path, filename):
    with open(filename, 'w') as file:
        for point in path:
            file.write(f"{int(point[0])} {point[1]:.2f} {point[2]:.2f}\n")

# Function to plot paths
def plot_paths(paths, title):
    plt.figure(figsize=(10, 8))
    
    all_x_coords = []
    all_y_coords = []

    for path in paths:
        time_index = [point[0] for point in path]
        x_coords = [point[1] for point in path]
        y_coords = [point[2] for point in path]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=4)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Calculate grid range
    x_min = int(min(all_x_coords) // 10) * 10
    x_max = int(max(all_x_coords) // 10) * 10 + 10
    y_min = int(min(all_y_coords) // 10) * 10
    y_max = int(max(all_y_coords) // 10) * 10 + 10

    plt.xticks(range(x_min, x_max, 10))
    plt.yticks(range(y_min, y_max, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Plot points at the center of each cell
    for x in range(x_min + 5, x_max, 10):
        for y in range(y_min + 5, y_max, 10):
            plt.plot(x, y, 'r.', markersize=5)

    plt.legend(['UAV {}'.format(i+1) for i in range(len(paths))])
    plt.tight_layout()

# Directory where original path files are located
input_directory = '/home/samshad/Documents/multi_uav/working/CA_DARP/sampled/'
# Directory to save the new path files
output_directory = os.path.join(input_directory, 'repositioned')

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read and process each path file
path_files = glob.glob(os.path.join(input_directory, 'uav_*_sampled_path.txt'))

original_paths = []
repositioned_paths = []

for file in path_files:
    path = read_path_from_file(file)
    original_paths.append(path)
    new_path = reposition_waypoints(path)
    repositioned_paths.append(new_path)
    filename = os.path.basename(file)
    new_filename = os.path.join(output_directory, filename)
    save_path_to_file(new_path, new_filename)

# Plot original paths
plot_paths(original_paths, 'Original UAV Paths')
plt.show()

# Plot repositioned paths
plot_paths(repositioned_paths, 'Repositioned UAV Paths to Cell Centers')
plt.show()

print("Path files have been repositioned and saved to:", output_directory)
