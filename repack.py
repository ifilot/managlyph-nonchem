import numpy as np
from stl import mesh
from abobuilder import AboBuilder

def main():
    
    names = ['ventricle.stl', 'arteries.stl', 'atrium.stl']

    models = []    
    for name in names:
        file_path = "sources/%s" % name  # Replace with the path to your STL file
        stl_data = read_stl(file_path)
    
        positions = np.array(stl_data['positions'], dtype=np.float32)
        normals = np.array(stl_data['vertex_normals'], dtype=np.float32)
        indices = np.array(stl_data['indices'], dtype=np.uint32)
    
        models.append({
            'vertices': positions,
            'normals': normals,
            'indices': indices.flatten()
        })

    # reposition
    val = np.zeros(3)
    vertices = 0
    for model in models:
        val += np.sum(model['vertices'], axis=0)
        vertices += model['vertices'].shape[0]
    ctr = val / vertices
    for model in models:
        model['vertices'] -= ctr

    builder = AboBuilder()
    builder.build_abo_model('test.abo', models)
    
    write_ply('test.ply', stl_data)


def read_stl(file_path):
    """
    Reads an STL file and extracts positions, vertex normals, and indices.

    Parameters:
        file_path (str): The path to the STL file.

    Returns:
        dict: A dictionary containing positions, vertex normals, and indices.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)

    # Extract the positions (vertices) and flatten them
    all_positions = stl_mesh.vectors.reshape(-1, 3)

    # Find unique vertices and map old indices to new ones
    unique_positions, inverse_indices = np.unique(all_positions, axis=0, return_inverse=True)

    # Extract the face normals
    face_normals = stl_mesh.normals

    # Generate indices using the mapping
    num_faces = len(stl_mesh.vectors)
    indices = inverse_indices.reshape((num_faces, 3))

    # Compute vertex normals
    vertex_normals = np.zeros_like(unique_positions)
    vertex_contributions = np.zeros(len(unique_positions))

    for i, face in enumerate(indices):
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normals[i]
            vertex_contributions[vertex_index] += 1

    # Normalize the vertex normals
    for i in range(len(vertex_normals)):
        if vertex_contributions[i] > 0:
            vertex_normals[i] /= vertex_contributions[i]
            vertex_normals[i] = vertex_normals[i] / np.linalg.norm(vertex_normals[i])

    return {
        "positions": unique_positions,
        "vertex_normals": vertex_normals,
        "indices": indices
    }

def write_ply(output_path, data):
    """
    Writes the given data to a PLY file.

    Parameters:
        output_path (str): The path to save the PLY file.
        data (dict): A dictionary containing positions, vertex normals, and indices.
    """
    positions = data["positions"]
    vertex_normals = data["vertex_normals"]
    indices = data["indices"]

    with open(output_path, "w") as ply_file:
        # Write PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(positions)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write(f"element face {len(indices)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")

        # Write vertex data
        for position, normal in zip(positions, vertex_normals):
            ply_file.write(f"{position[0]} {position[1]} {position[2]} {normal[0]} {normal[1]} {normal[2]}\n")

        # Write face data
        for face in indices:
            ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

# Example usage
if __name__ == "__main__":
    main()