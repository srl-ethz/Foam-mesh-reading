import re
import numpy as np

def read_boundary(boundary_file_path):
    # Read content from the text file
    with open(boundary_file_path, 'r') as f:
        file_content = f.read()

    # Extract boundary patches and their settings using regex
    boundary_sections = re.findall(r'(\w+)\s+{([^}]*)}', file_content, re.DOTALL)

    # Initialize a dictionary to store the parsed boundary data
    boundary_data = {}

    # Process each boundary section except 'FoamFile'
    for boundary_name, boundary_content in boundary_sections:
        boundary_settings = {}
        for line in boundary_content.strip().split('\n'):
            key, value = map(str.strip, line.split(None, 1))
            if value.endswith(';'):
                value = value[:-1]  # Remove trailing semicolon
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]  # Remove surrounding double quotes
            if value.isdigit():
                value = int(value)  # Convert to integer if it's a number
            boundary_settings[key] = value
        boundary_data[boundary_name] = boundary_settings
    
    return boundary_data

def faces_to_np(face_file_path):
    with open(face_file_path, 'r') as f:
        file_content = f.read()

    # Extract data using regex
    data = re.search(r'\(\n(.*?)\n\)', file_content, re.DOTALL).group(1)

    # Process the data to convert it into a 2D numpy array
    data_list = re.findall(r'4\(([^)]*)\)', data)
    data_arrays = [list(map(int, item.split())) for item in data_list]
    data_2d_array = np.array(data_arrays)

    return data_2d_array

def points_to_np(points_file_path):
    # Read the file
    with open(points_file_path, 'r') as f:
        lines = f.readlines()

    # Initialize an empty list to store the points
    points = []

    # Flag to indicate whether points are being read
    reading_points = False

    # Iterate through the lines
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if line == '(':  # Start of points
            reading_points = True
        elif line == ')':  # End of points
            reading_points = False
        elif reading_points:
            # Split the line into x, y, z coordinates
            x, y, z = map(float, line[1:-1].split())
            points.append([x, y, z])

    # Convert the list of points to a numpy array
    points_array = np.array(points)

    return points_array

def find_boundary_faces(boundary_file_path, face_file_path, boundary_name):
    # Read boundary data
    boundary_data = read_boundary(boundary_file_path)

    # Read face data
    face_data = faces_to_np(face_file_path)

    # Find the boundary patch
    boundary_patch = boundary_data[boundary_name]

    # Find the boundary faces
    boundary_faces = face_data[boundary_patch['startFace']:boundary_patch['startFace']+boundary_patch['nFaces']]

    return boundary_faces

def find_boundary_points(boundary_file_path, face_file_path, points_file_path, boundary_name):
    # Find the boundary faces
    boundary_faces = find_boundary_faces(boundary_file_path, face_file_path, boundary_name)

    # Read points data
    points_data = points_to_np(points_file_path)


    return points_data[boundary_faces]

def quadrilateral_to_triangular(faces):
    return np.concatenate([faces[:,:3,:], faces[:, 1:, :]])

def calculate_sdf(face_points, x_grid, y_grid, z_grid):
    def point_segment_distance(p, v0, v1):
        v = v1 - v0
        w = p - v0
        v = v.reshape(1, 1, 1, -1, 3)  # Reshape v
        t = np.sum(w * v, axis=-1) / np.sum(v * v, axis=-1)

        # Create masks for different conditions
        mask_t_le_0 = t <= 0
        mask_t_ge_1 = t >= 1

        distances = np.where(mask_t_le_0, np.linalg.norm(p - v0, axis=-1),
                             np.where(mask_t_ge_1, np.linalg.norm(p - v1, axis=-1),
                                      np.linalg.norm(p - (v0 + t[..., np.newaxis] * v), axis=-1)))

        return distances
    
    def calculate_distance_direction(face_points, x_grid, y_grid, z_grid):
        grid_points = np.array(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')).T
        grid_points = grid_points.reshape(-1, 1, 3)
        
        # Calculate face normals
        face_normals = np.cross(face_points[:, 0] - face_points[:, 1], face_points[:, 1] - face_points[:, 2])
        face_normals /= np.linalg.norm(face_normals, axis=-1, keepdims=True)
        
        # Calculate vectors from grid points to face vertices
        grid_vectors = grid_points - face_points[...,0,:]
        
        # Calculate distances to face_points using dot product
        face_normal_distance = np.sum(grid_vectors * face_normals, axis=-1)

        face_normal_distance[face_normal_distance > 0] = 1
        face_normal_distance[face_normal_distance <= 0] = -1

        return face_normal_distance
    
    distance_direction = calculate_distance_direction(face_points, x_grid, y_grid, z_grid)[:, np.newaxis, np.newaxis, :]
        
    grid_points = np.array(np.meshgrid(
        x_grid, y_grid, z_grid, indexing='ij')).T
    face_points = face_points.reshape(1, 1, 1, -1, 3, 3)
    grid_points = grid_points.reshape(-1, 1, 1, 1, 3)

    x1, x2, x3 = face_points[..., 0, :], face_points[..., 1, :], face_points[..., 2, :]

    x13 = x1 - x3
    x23 = x2 - x3
    x03 = grid_points - x3
    m13 = np.sum(x13 * x13, axis=-1)
    m23 = np.sum(x23 * x23, axis=-1)
    d = np.sum(x13 * x23, axis=-1)

    invdet = 1.0 / np.maximum(m13 * m23 - d * d, 1e-30)
    a = np.sum(x13 * x03, axis=-1)
    b = np.sum(x23 * x03, axis=-1)

    w23 = invdet * (m23 * a - d * b)
    w31 = invdet * (m13 * b - d * a)
    w12 = 1 - w23 - w31

    mask = (w23 >= 0) & (w31 >= 0) & (w12 >= 0)

    # return  w23[..., np.newaxis] * x1 + w31[..., np.newaxis] * x2 + w12[..., np.newaxis] * x3

    closest_points = np.where(mask[..., np.newaxis], w23[..., np.newaxis] * x1 + w31[..., np.newaxis] * x2 + w12[..., np.newaxis] * x3,
                              np.nan)
    
    # Corrected segment_distances calculation using point_segment_distance function
    distances_1_2 = point_segment_distance(grid_points, x1, x2)
    distances_1_3 = point_segment_distance(grid_points, x1, x3)
    distances_2_3 = point_segment_distance(grid_points, x2, x3)

    # Calculate distances based on conditions
    distances_23 = np.where(w23 > 0, np.minimum(distances_1_2, distances_1_3), np.nan)
    distances_31 = np.where(w31 > 0, np.minimum(distances_1_2, distances_2_3), np.nan)
    distances_12 = np.where(w12 > 0, np.minimum(distances_1_3, distances_2_3), np.nan)

    segment_distances = np.stack([distances_23, distances_31, distances_12], axis=-1)

    min_segment_distances = np.nanmin(segment_distances, axis=-1)


    # Shape:
    # segment_distances: (num_points, 1, 1, num_faces, 3)
    points_closest_distance = np.linalg.norm(
        grid_points - closest_points, axis=-1)

    distances = np.where(mask, points_closest_distance, min_segment_distances)

    min_distance_indices = np.argmin(distances, axis=-1)
    min_distances = np.take_along_axis(distances * distance_direction, min_distance_indices[..., np.newaxis], axis=-1).squeeze(-1)

    # Shape:
    # distances: (num_points, 1, 1, num_faces)

    return min_distances.reshape(len(z_grid), len(y_grid), len(x_grid))

def calculate_sdf_from_mesh(boundary_file_path, face_file_path, points_file_path, boundary_names, 
                            x_grid, y_grid, z_grid):

    face_points = []
    for boundary_name in boundary_names:
        face_points.append(find_boundary_points(boundary_file_path, face_file_path, points_file_path, boundary_name))
    
    face_points = np.concatenate(face_points, axis=0)
    face_points = quadrilateral_to_triangular(face_points)

    sdf = calculate_sdf(face_points, x_grid, y_grid, z_grid)


    return sdf