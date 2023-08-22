# PyFoam2SDF: Convert OpenFOAM mesh to SDF

This is a simple python package to convert OpenFOAM mesh to SDF. It also facilitates the reading of OpenFOAM mesh files for python.
The calculation of SDF is based on barycentric coordinate system.

## Installation

```bash
pip install PyFoam2SDF
```

## Documentation

```python
read_boundary(boundary_file_path: str) -> dict
```

Reads boundary data from the foam mesh boundary file and returns a dictionary storing the boundary data.

**Parameters:**

- `boundary_file_path` (str): Path to the boundary file, usually at constant/polyMesh/boundary.

**Returns:**

- `dict`: A dictionary storing the boundary data.

```python
faces_to_np(face_file_path : str) -> np.ndarray
```

Reads face data from the foam mesh face file and returns a 2D numpy array of shape num_faces x 4, storing the face indices.

**Parameters:**

- `face_file_path` (str): Path to the face file, usually at constant/polyMesh/faces.

**Returns:**

- `np.ndarray`: A 2D numpy array storing the face indices. Shape: (n_faces, 4)

```python
points_to_np(points_file_path : str) -> np.ndarray
```

Reads points data from the foam mesh points file and returns a 2D numpy array of shape num_points x 3, storing the point coordinates.

**Parameters:**

- `points_file_path` (str): Path to the points file, usually at <time_step>/polyMesh/points.

**Returns:**

- `np.ndarray`: A 2D numpy array storing the point coordinates. Shape: (n_points, 3)

```python
find_boundary_faces(boundary_file_path : str, face_file_path : str, boundary_name : str) -> np.ndarray
```

Finds the faces of a specific boundary patch and returns a 2D numpy array of shape num_boundary_faces x 4, storing the face indices of the specified boundary patch.

**Parameters:**

- `boundary_file_path` (str): Path to the boundary file, usually at constant/polyMesh/boundary.
- `face_file_path` (str): Path to the face file, usually at constant/polyMesh/faces.
- `boundary_name` (str): Name of the boundary patch, e.g., 'inlet', 'outlet', 'top', 'bottom', 'front', 'back'.

**Returns:**

- `np.ndarray`: A 2D numpy array storing the face indices of the specified boundary patch. Shape: (n_boundary_faces, 4)

```python
find_boundary_points(boundary_file_path : str, face_file_path : str, points_file_path : str, boundary_name : str) -> np.ndarray
```

Finds the points of a specific boundary patch and returns a 3D numpy array of shape num_boundary_faces x 4 x 3, storing the point coordinates of the specified boundary patch.

**Parameters:**

- `boundary_file_path` (str): Path to the boundary file, usually at constant/polyMesh/boundary.
- `face_file_path` (str): Path to the face file, usually at constant/polyMesh/faces.
- `points_file_path` (str): Path to the points file, usually at <time_step>/polyMesh/points.
- `boundary_name` (str): Name of the boundary patch, e.g., 'inlet', 'outlet', 'top', 'bottom', 'front', 'back'.

**Returns:**

- `np.ndarray`: A 3D numpy array storing the point coordinates of the specified boundary patch. Shape: (n_boundary_faces, 4, 3)

```python
quadrilateral_to_triangular(faces : np.ndarray) -> np.ndarray
```

Converts the quadrilateral mesh to a triangular mesh. It cuts a quadrilateral into two triangles, so the number of faces will be doubled.

**Parameters:**

- `faces` (np.ndarray): A 3D numpy array storing the point coordinates of the specified boundary patch, in triangular mesh format. Shape: (n_triangular_boundary_faces, 3, 3)

**Returns:**

- `np.ndarray`: A 3D numpy array storing the point coordinates of the resulting triangular mesh. Shape: (n_triangular_boundary_faces, 3, 3), where n_triangular_boundary_faces = 2 * n_boundary_faces

```python
calculate_sdf(face_points : np.ndarray, x_grid : np.ndarray, y_grid : np.ndarray, z_grid : np.ndarray) -> np.ndarray
```

Calculates the signed distance function (SDF) of a boundary patch.

**Parameters:**

- `face_points` (np.ndarray): A 3D numpy array storing the point coordinates of the specified boundary patch, in triangular mesh format. Shape: (n_triangular_boundary_faces, 3, 3)
- `x_grid` (np.ndarray): A 1D numpy array storing the x coordinates of the grid points.
- `y_grid` (np.ndarray): A 1D numpy array storing the y coordinates of the grid points.
- `z_grid` (np.ndarray): A 1D numpy array storing the z coordinates of the grid points. For 2D cases, this can be set to [0.].

**Returns:**

- `np.ndarray`: A 3D numpy array storing the SDF values of the grid points. Shape: (n_x, n_y, n_z)

```python
calculate_sdf_from_mesh_files(boundary_file_path : str, face_file_path : str, points_file_path : str, boundary_names : list, x_grid : np.ndarray, y_grid : np.ndarray, z_grid : np.ndarray) -> np.ndarray
```

Calculates the signed distance function (SDF) from mesh files for multiple boundary patches.

**Parameters:**

- `boundary_file_path` (str): Path to the boundary file, usually at constant/polyMesh/boundary.
- `face_file_path` (str): Path to the face file, usually at constant/polyMesh/faces.
- `points_file_path` (str): Path to the points file, usually at <time step>/polyMesh/points.
- `boundary_names` (list): List of boundary names to calculate SDF for.
- `x_grid` (np.ndarray): A 1D numpy array storing the x coordinates of the grid points.
- `y_grid` (np.ndarray): A 1D numpy array storing the y coordinates of the grid points.
- `z_grid` (np.ndarray): A 1D numpy array storing the z coordinates of the grid points. For 2D cases, this can be set to [0.].

**Returns:**

- `np.ndarray`: A 3D numpy array storing the SDF values of the grid points. Shape: (n_x, n_y, n_z)
