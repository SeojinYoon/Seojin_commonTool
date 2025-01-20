import numpy as np
import matplotlib.pylab as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_uv_map(uv_coordinates, 
                faces,
                highlight_face_info):
    """
    Draws a 2D UV map representation of a 3D mesh with an option to highlight specific faces.

    :param uv_coordinates(np.array - shape (#vertex, 2)): A 2D array where each row represents the UV coordinates 
        of a vertex in the mesh. Each UV coordinate is represented as (u, v).

    :param faces(np.array - shape (#face, 3)): A 2D array defining the triangular faces of the mesh. Each row contains 
        three indices into the `uv_coordinates` array, specifying the vertices that form a triangular face.

    :param highlight_face_info(dict): A dictionary specifying which faces to highlight and their colors. The keys 
        are unique names for highlight groups, and the values are dictionaries with the following structure:
        
        {
            "color": str,        # Color used for highlighting, e.g., "red", "blue".
            "data": list[int]    # A list of face indices in the `faces` array to be highlighted.
        }

        Example:
            highlight_face_info = {
                "group1": {
                    "color": "red",
                    "data": [0, 1, 2]  # Highlight faces with indices 0, 1, and 2 in red
                },
                "group2": {
                    "color": "green",
                    "data": [3, 4]     # Highlight faces with indices 3 and 4 in green
                }
            }

    :return: None
        The function displays a 2D scatter plot of UV coordinates with the option to highlight specified faces 
        on the plot.
    """
    plt.scatter(uv_coordinates[:, 0], uv_coordinates[:, 1], s = 1)
    
    for face_i, face in enumerate(faces):
        # Extract the UV coordinates for each vertex of the face
        uv_face = uv_coordinates[face]

        for name in highlight_face_info:
            info = highlight_face_info[name]
            
            component_indexes = info["data"]
            color = info["color"]

            if face_i in component_indexes:
                plt.fill(uv_face[:, 0], uv_face[:, 1], edgecolor = "black", alpha = 0.5, color = color)

def show_interactive_mesh(vertices, 
                          faces, 
                          highlight_face_info = {}):
    """
    Show interactive mesh with an option to highlight a specific face in red.

    :param vertices(np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces(np.array - shape (#face, 3)): An array defining the triangular faces of the mesh. Each row contains three indices into the vertices array, specifying the vertices that form a triangular face.
    :param highlight_face_info(dict): A dictionary that specifies the face indexes to be highlighted and their associated colors
    """

    # Default face colors: lightblue
    face_colors = ["lightblue"] * len(faces)
    
    # Apply highlighting colors
    for name in highlight_face_info:
        color = highlight_face_info[name]["color"]
        highlight_face_indexes = highlight_face_info[name]["data"]
        for face_index in highlight_face_indexes:
            face_colors[face_index] = color
            
    # Default mesh trace
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1,
        color="lightblue",
        hoverinfo="text",
        text=[f"Face {i}, vertices: {faces[i]}" for i in range(len(faces))],  # Face indices in tooltip
    )

    fig = go.Figure(data=[mesh_trace])

    # Highlight a specific face in red
    for name in highlight_face_info:
        color = highlight_face_info[name]["color"]
        highlight_face_indexes = highlight_face_info[name]["data"]
        
        for face_index in highlight_face_indexes:
            # Extract the vertices of the highlighted face
            face = faces[face_index]
            highlighted_trace = go.Mesh3d(
                x = vertices[:, 0],
                y = vertices[:, 1],
                z = vertices[:, 2],
                i = [face[0]],
                j = [face[1]],
                k = [face[2]],
                color=color,
                opacity=1.0,
                hoverinfo="text",
                text=[f"Highlighted Face {face_index}, vertices: {face}"],
            )
            fig.add_trace(highlighted_trace)

    fig.update_layout(
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
        ),
        title="Interactive 3D Mesh with Highlighted Face",
    )

    return fig

def show_non_interactive_mesh(vertices, 
                              faces, 
                              highlight_face_info={}):
    """
    Show a static 3D mesh with an option to highlight specific faces.

    :param vertices (np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces (np.array - shape (#face, 3)): An array defining the triangular faces of the mesh.
    :param highlight_face_info (dict): A dictionary specifying face indexes to be highlighted and their colors.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the main mesh
    for face in faces:
        poly = vertices[face]
        collection = Poly3DCollection([poly], color='lightblue', edgecolor='k', alpha=0.8)
        ax.add_collection3d(collection)

    # Highlight specified faces
    for name in highlight_face_info:
        color = highlight_face_info[name]["color"]
        highlight_face_indexes = highlight_face_info[name]["data"]

        for face_index in highlight_face_indexes:
            face = faces[face_index]
            poly = vertices[face]
            highlighted_collection = Poly3DCollection([poly], color=color, edgecolor='k', alpha=1.0)
            ax.add_collection3d(highlighted_collection)

    # Set axis limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    # Set axis labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.set_title("Static 3D Mesh with Highlighted Faces")
    plt.show()

    return fig, ax

def show_mesh(vertices, 
              faces, 
              vertex_index_info = {}):
    """
    Show a static 3D mesh with an option to highlight specific faces.

    The vertex's coordinate system must be RAS+ (this is correspond to MNI space)

    :param vertices (np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces (np.array - shape (#face, 3)): An array defining the triangular faces of the mesh.
    :param vertex_index_info (dictionary): A info for highligtiing faces consisted by vertex_index
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the main mesh
    for face in faces:
        poly = vertices[face]
        collection = Poly3DCollection([poly], color='lightblue', edgecolor='k', alpha=0.8)
        ax.add_collection3d(collection)

    # Highlight specified faces
    for face_index, face in enumerate(faces):
        for name in vertex_index_info:
            vertex_index_set = vertex_index_info[name]["set"]
            color = vertex_index_info[name]["color"]
            
            is_all_inSet = np.isin(element = face, test_elements = vertex_index_set)
            if np.alltrue(is_all_inSet):
                poly = vertices[face]
                highlighted_collection = Poly3DCollection([poly], color=color, edgecolor='k', alpha=1.0)
                ax.add_collection3d(highlighted_collection)
            
    # Set axis limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    # Set axis labels
    ax.set_xlabel("R+")
    ax.set_ylabel("A+")
    ax.set_zlabel("S+")

    ax.set_title("Static 3D Mesh with Highlighted Faces")
    plt.show()

    return fig, ax