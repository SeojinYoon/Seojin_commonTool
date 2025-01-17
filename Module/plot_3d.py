import matplotlib.pylab as plt
import plotly.graph_objects as go

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
    plt.scatter(uv_coordinates[:, 0], uv_coordinates[:, 1], s=1)
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
