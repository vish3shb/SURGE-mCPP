import matplotlib.pyplot as plt
import glob

# Function to read path from file
def read_path_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        path = [[float(num) for num in line.strip().split()] for line in lines]
    return path

# Function to plot paths
def plot_paths(paths):
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

    plt.title('Visualization of UAV Paths')
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
    plt.show()

# Directory where path files are located
path_files = glob.glob('/home/samshad/Documents/multi_uav/working/CA_DARP/sampled/repositioned/uav_*_sampled_path.txt')

# Read paths from files
paths = []
for file in path_files:
    path = read_path_from_file(file)
    paths.append(path)

# Plot paths
plot_paths(paths)
