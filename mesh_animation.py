import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Path to the folder containing robot path files
base_dir = os.path.dirname(os.path.abspath(__file__))
#folder_path = os.path.join(base_dir, 'Results', 'allowed_distance_results', 'transformed')

folder_path = os.path.join(base_dir, 'Results', 'least_turns_results', 'transformed')

# Function to read waypoints from a file
def read_waypoints(file_path):
    waypoints = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            index = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            waypoints.append((index, x, y))
    return waypoints

# Function to extract all waypoints from the folder
def extract_all_waypoints(folder_path):
    robot_paths = {}
    for file_name in os.listdir(folder_path):
###################################################
####################################################
        if file_name.endswith('_sampled_path.txt'):

        #if file_name.startswith('*_uav_') and file_name.endswith('_sampled_path.txt'):
#########################################
########################################
            file_path = os.path.join(folder_path, file_name)
            robot_paths[file_name] = read_waypoints(file_path)
    return robot_paths

# Function to check mesh connectivity at each index and collect the results
def check_mesh_connectivity(robot_paths, distance_threshold):
    max_index = max(len(path) for path in robot_paths.values())
    connectivity_results = []
    
    for idx in range(max_index):
        graph = nx.Graph()
        positions = {}

        # Extract positions of all robots at the current index
        for robot, path in robot_paths.items():
            if idx < len(path):
                _, x, y = path[idx]
                positions[robot] = (x, y)
                graph.add_node(robot, pos=(x, y))

        # Add edges to the graph based on connectivity criteria
        for robot1 in positions:
            for robot2 in positions:
                if robot1 != robot2:
                    dist = np.linalg.norm(np.array(positions[robot1]) - np.array(positions[robot2]))
                    if dist <= distance_threshold:
                        graph.add_edge(robot1, robot2)

        # Check if the graph is connected
        is_connected = nx.is_connected(graph)
        connectivity_results.append((idx, is_connected, graph, positions))
        print(f"At index {idx}: {'Connected' if is_connected else 'Not Connected'}")

    return connectivity_results

# Function to update the plot for each frame in the animation
def update(frame, robot_paths, connectivity_results, scatter_plots, line_plots, disconnection_points, distance_text, distance_threshold, ax):
    idx, is_connected, graph, positions = connectivity_results[frame]

    # Clear previous lines
    while len(line_plots) > 0:
        ln = line_plots.pop()
        ln.remove()

    # Plot the robots and connectivity
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]

    for i, scatter_plot in enumerate(scatter_plots):
        if i < len(x_coords):
            scatter_plot.set_offsets([[x_coords[i], y_coords[i]]])
        else:
            scatter_plot.set_offsets([])

    for (u, v) in graph.edges():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        line, = ax.plot(x, y, 'g-', alpha=0.5)
        line_plots.append(line)

    if not is_connected:
        # Highlight the disconnected points
        for x, y in zip(x_coords, y_coords):
            if (x, y) not in disconnection_points:
                disconnection_points.add((x, y))
                ax.plot(x, y, 'ro', markersize=5, alpha=0.5)

    # Update distance threshold text
    distance_text.set_text(f'Distance Threshold: {distance_threshold}')

# Function to create and display the animation
def create_animation(robot_paths, connectivity_results, distance_threshold, polygon_points):
    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Robot Paths with Connectivity Check')
    ax.axis('equal')
    ax.grid(True)

    # Plot the static polygon
    if polygon_points:
        polygon_x, polygon_y = zip(*polygon_points)
        # Close the polygon by adding the first point at the end
        polygon_x += (polygon_x[0],)
        polygon_y += (polygon_y[0],)
        ax.plot(polygon_x, polygon_y, 'b-', label='Static Polygon')
        ax.fill(polygon_x, polygon_y, 'b', alpha=0.1)  # Optional: Fill the polygon with a light color

    # Plot the entire paths of all robots on the background
    for robot, path in robot_paths.items():
        x_coords = [wp[1] for wp in path]
        y_coords = [wp[2] for wp in path]
        ax.plot(x_coords, y_coords, label=robot)

    scatter_plots = [ax.scatter([], [], s=100) for _ in robot_paths]
    line_plots = []
    disconnection_points = set()
    distance_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ani = animation.FuncAnimation(
        fig, update, frames=len(connectivity_results),
        fargs=(robot_paths, connectivity_results, scatter_plots, line_plots, disconnection_points, distance_text, distance_threshold, ax),
        interval=2, repeat=True
    )

    plt.legend()
    plt.show()

# Function to plot connectivity graph at a specific timestamp
def plot_connectivity_at_timestamp(timestamp, robot_paths, connectivity_results, distance_threshold):
    idx, is_connected, graph, positions = connectivity_results[timestamp]

    fig, ax = plt.subplots()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Connectivity at Timestamp {timestamp}')
    ax.axis('equal')
    ax.grid(True)

    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]

    # Plot the robots
    for robot, pos in positions.items():
        ax.scatter(pos[0], pos[1], s=100, label=robot)
    
    # Plot the edges
    for (u, v) in graph.edges():
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        ax.plot(x, y, 'g-', alpha=0.5)

    # Highlight disconnected points if any
    if not is_connected:
        for x, y in zip(x_coords, y_coords):
            ax.plot(x, y, 'ro', markersize=5, alpha=0.5)

    ax.legend()
    plt.show()

# Function to read the polygon file
def read_polygon(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines:
            print(f"Reading line: {line.strip()}")  # Debug statement
            if not line.startswith("#"):  # Skip comment lines
                x, y = line.split()
                points.append((float(x), float(y)))
        return points

# Main execution#########################################################################
distance_threshold = 968  # Example threshold
#######################################################################################
robot_paths = extract_all_waypoints(folder_path)
print(f"Number of paths detected: {len(robot_paths)}")

connectivity_results = check_mesh_connectivity(robot_paths, distance_threshold)
all_connected = all(is_connected for _, is_connected, _, _ in connectivity_results)
print(f"All robots are {'connected' if all_connected else 'not connected'} for all indexes")

polygon_file_path = os.path.join(base_dir, 'flyzon_corrected_reversed.poly')
polygon_points = read_polygon(polygon_file_path)

# Print the polygon points to debug
print("Polygon Points:")
for point in polygon_points:
    print(point)

create_animation(robot_paths, connectivity_results, distance_threshold, polygon_points)

# Plot the connectivity graph at a specific timestamp (e.g., timestamp 10)
plot_connectivity_at_timestamp(robot_paths, connectivity_results, distance_threshold)
