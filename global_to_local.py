import json
import numpy as np
import pyproj
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from collections import defaultdict

def read_polygon(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                lat, lon = map(float, line.strip().split())
                vertices.append((lat, lon))
    return vertices

def project_to_utm(vertices):
    proj = pyproj.Proj(proj='utm', zone=44, ellps='WGS84', preserve_units=False)
    utm_vertices = np.array([proj(lon, lat) for lat, lon in vertices])
    return utm_vertices

def calculate_centroid(utm_vertices):
    centroid = utm_vertices.mean(axis=0)
    return centroid

def translate_to_local_frame(utm_vertices, centroid):
    local_frame_vertices = utm_vertices - centroid
    return local_frame_vertices

def save_local_frame(vertices, output_path):
    with open(output_path, 'w') as file:
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]}\n")

def find_largest_side(vertices):
    max_distance = 0
    largest_side = (vertices[0], vertices[1])
    num_vertices = len(vertices)
    for i in range(num_vertices):
        start_vertex = vertices[i]
        end_vertex = vertices[(i + 1) % num_vertices]
        distance = np.linalg.norm(start_vertex - end_vertex)
        if distance > max_distance:
            max_distance = distance
            largest_side = (start_vertex, end_vertex)
    return largest_side

def calculate_rotation_angle(largest_side):
    vector = largest_side[1] - largest_side[0]
    angle = np.arctan2(vector[1], vector[0])
    return angle

def rotate_polygon(vertices, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_vertices = vertices.dot(rotation_matrix.T)
    return rotated_vertices

def shift_polygon(vertices, shift_vector):
    shifted_vertices = vertices + shift_vector
    return shifted_vertices

def create_grid(vertices, cell_size, threshold):
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    num_cols = int(np.ceil((max_x - min_x) / cell_size))
    num_rows = int(np.ceil((max_y - min_y) / cell_size))
    x_coords = np.arange(min_x, min_x + num_cols * cell_size, cell_size)
    y_coords = np.arange(min_y, min_y + num_rows * cell_size, cell_size)
    grid_cells = []
    polygon = Polygon(vertices)
    cell_count = 0
    for j, y in enumerate(y_coords):  # Changed to start from bottom
        for i, x in enumerate(x_coords):
            cell = box(x, y, x + cell_size, y + cell_size)
            intersection = polygon.intersection(cell)
            if intersection.area / cell.area >= threshold:
                grid_cells.append((cell, True, cell_count))
            else:
                grid_cells.append((cell, False, cell_count))
            cell_count += 1
    return grid_cells, cell_count

def find_best_shift(vertices, cell_size, threshold):
    min_cells = float('inf')
    best_shift = np.array([0, 0])
    for dx in np.linspace(-cell_size, cell_size, num=5):
        for dy in np.linspace(-cell_size, cell_size, num=5):
            shifted_vertices = shift_polygon(vertices, np.array([dx, dy]))
            _, cell_count = create_grid(shifted_vertices, cell_size, threshold)
            if cell_count < min_cells:
                min_cells = cell_count
                best_shift = np.array([dx, dy])
    return best_shift, min_cells

def calculate_canvas_size(vertices, cell_size):
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    num_cols = int(np.ceil((max_x - min_x) / cell_size))
    num_rows = int(np.ceil((max_y - min_y) / cell_size))
    return num_rows, num_cols

def get_outer_edges(grid_cells):
    edges = defaultdict(int)
    for cell, is_polygon_cell, _ in grid_cells:
        if is_polygon_cell:
            cell_coords = list(cell.exterior.coords)
            for i in range(len(cell_coords) - 1):
                edge = (cell_coords[i], cell_coords[i + 1])
                if edge in edges:
                    edges[edge] += 1
                else:
                    edges[edge] = 1
                reverse_edge = (cell_coords[i + 1], cell_coords[i])
                if reverse_edge in edges:
                    edges[reverse_edge] += 1
                else:
                    edges[reverse_edge] = 1
    outer_edges = [edge for edge, count in edges.items() if count == 1]
    return outer_edges

def trace_outer_polygon(outer_edges):
    vertices = []
    edge_dict = defaultdict(list)
    for edge in outer_edges:
        edge_dict[edge[0]].append(edge[1])
        edge_dict[edge[1]].append(edge[0])

    start_vertex = outer_edges[0][0]
    current_vertex = start_vertex
    prev_vertex = None

    while True:
        vertices.append(current_vertex)
        next_vertices = [v for v in edge_dict[current_vertex] if v != prev_vertex]
        if len(next_vertices) == 0:
            break
        prev_vertex = current_vertex
        current_vertex = next_vertices[0]
        if current_vertex == start_vertex:
            break

    return np.array(vertices)

def filter_corner_vertices(vertices):
    corners = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        prev_vertex = vertices[i - 1]
        current_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % num_vertices]
        vector1 = current_vertex - prev_vertex
        vector2 = next_vertex - current_vertex
        angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
        angle = np.degrees(angle)
        if angle < 0:
            angle += 360
        if not (angle == 0 or angle == 180):
            corners.append(current_vertex)
    return np.array(corners)

def plot_polygon(vertices, grid_cells, corner_vertices, title="Polygon in Local Frame with Grid"):
    plt.figure()
    polygon = Polygon(vertices)
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='blue')
    plt.fill(x, y, color='lightblue', alpha=0.5)

    for cell, is_polygon_cell, cell_index in grid_cells:
        cell_x, cell_y = cell.exterior.xy
        if cell_index == 0:
            plt.plot(cell_x, cell_y, color='yellow', linestyle='-', linewidth=3)  # Highlight cell 0
            # Get the lower-left corner coordinates
            lower_left_corner = (min(cell_x), min(cell_y))
            plt.scatter([lower_left_corner[0]], [lower_left_corner[1]], color='red', zorder=5)  # Highlight the corner
            plt.text(lower_left_corner[0], lower_left_corner[1], f"({lower_left_corner[0]:.2f}, {lower_left_corner[1]:.2f})",
                     color='red', ha='right', va='top')  # Print the coordinates
        else:
            plt.plot(cell_x, cell_y, color='green' if is_polygon_cell else 'red', linestyle='--')
        plt.text(cell.centroid.x, cell.centroid.y, str(cell_index), color='black', ha='center', va='center')

    if len(corner_vertices) > 0:
        outer_x, outer_y = zip(*corner_vertices)
        plt.plot(outer_x, outer_y, color='magenta', linestyle='-', linewidth=2)
        plt.scatter(outer_x, outer_y, color='magenta', zorder=5)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def save_corrected_polygon(vertices, output_path):
    with open(output_path, 'w') as file:
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]}\n")
        # Close the polygon by adding the first vertex to the end
        file.write(f"{vertices[0][0]} {vertices[0][1]}\n")

def inverse_rotate_polygon(vertices, angle):
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    rotated_vertices = vertices.dot(rotation_matrix.T)
    return rotated_vertices

def save_transformation_parameters(file_path, centroid, angle, lower_left_corner):
    parameters = {
        "centroid": centroid.tolist(),
        "rotation_angle": angle,
        "bottom_left_corner": lower_left_corner
    }
    with open(file_path, 'w') as file:
        json.dump(parameters, file, indent=4)

def process_polygon(input_file_path, cell_size, coverage_threshold):
    vertices = read_polygon(input_file_path)
    utm_vertices = project_to_utm(vertices)
    centroid = calculate_centroid(utm_vertices)
    local_frame_vertices = translate_to_local_frame(utm_vertices, centroid)
    largest_side = find_largest_side(local_frame_vertices)
    angle = calculate_rotation_angle(largest_side)
    rotated_vertices = rotate_polygon(local_frame_vertices, -angle)
    num_rows, num_cols = calculate_canvas_size(rotated_vertices, cell_size)
    best_shift, min_cells = find_best_shift(rotated_vertices, cell_size, coverage_threshold)
    shifted_vertices = shift_polygon(rotated_vertices, best_shift)
    grid_cells, cell_count = create_grid(shifted_vertices, cell_size, coverage_threshold)
    polygon_cells = [cell_index for cell, is_polygon_cell, cell_index in grid_cells if is_polygon_cell]
    obstacle_cells = [cell_index for cell, is_polygon_cell, cell_index in grid_cells if not is_polygon_cell]

    outer_edges = get_outer_edges(grid_cells)
    outer_polygon_vertices = trace_outer_polygon(outer_edges)
    corner_vertices = filter_corner_vertices(outer_polygon_vertices)
    
    # Save the corrected polygon vertices to a file
    save_corrected_polygon(corner_vertices, "flyzon_corrected.poly")

    plot_polygon(shifted_vertices, grid_cells, corner_vertices, "Shifted Polygon in Local Frame with Grid and Edges")

    # Reverse rotate the corrected polygon back to original orientation
    reversed_rotated_vertices = inverse_rotate_polygon(corner_vertices, -angle)
    
    # Save the reversed corrected polygon to a file
    save_corrected_polygon(reversed_rotated_vertices, "flyzon_corrected_reversed.poly")

    # Extract the bottom-left corner of grid cell 0
    cell_0 = next(cell for cell, _, cell_index in grid_cells if cell_index == 0)
    lower_left_corner = (min(cell_0.exterior.xy[0]), min(cell_0.exterior.xy[1]))

    # Save transformation parameters including the bottom-left corner
    save_transformation_parameters("transformation_parameters.json", centroid, angle, lower_left_corner)

    return {
        "canvas_size": (num_rows, num_cols),
        "polygon_cells": polygon_cells,
        "obstacle_cells": obstacle_cells,
        "cell_count": cell_count,
        "best_shift": best_shift
    }

# Helper function to save the corrected polygon vertices
def save_corrected_polygon(vertices, output_path):
    with open(output_path, 'w') as file:
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]}\n")
        # Close the polygon by adding the first vertex to the end
        file.write(f"{vertices[0][0]} {vertices[0][1]}\n")
#backup point