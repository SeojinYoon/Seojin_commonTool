
import matplotlib.pylab as plt
import plotly.graph_objects as go

def show_interactive_mesh(vertices, 
                          faces, 
                          highlight_face_info = {},
                          tick_interval = 20):
    """
    Show interactive mesh with an option to highlight a specific face in red.

    The vertex's coordinate system must be RAS+ (this is correspond to MNI space)

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

    # Highlight specific faces
    for name in highlight_face_info:
        color = highlight_face_info[name]["color"]
        highlight_face_indexes = highlight_face_info[name]["data"]
        
        for face_index in highlight_face_indexes:
            face = faces[face_index]
            highlighted_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=[face[0]],
                j=[face[1]],
                k=[face[2]],
                color=color,
                opacity=1.0,
                hoverinfo="text",
                text=[f"Highlighted Face {face_index}, vertices: {face}"],
            )
            fig.add_trace(highlighted_trace)

    # Update layout to emphasize the Y-axis perspective
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title = "R+",
                dtick = tick_interval,  # Adjust tick interval for X-axis
            ),
            yaxis=dict(
                title = "A+",
                dtick = tick_interval,  # Adjust tick interval for Y-axis
            ),
            zaxis=dict(
                title = "S+",
                dtick = tick_interval,  # Adjust tick interval for Z-axis
            ),
            camera=dict(
                eye=dict(x=0, y=-2.5, z=1.5)  # Adjust camera view
            )
        ),
        title="Interactive 3D Mesh with Highlighted Face",
    )

    return fig

def show_non_interactive_mesh(vertices, 
                              faces, 
                              highlight_face_info={}):
    """
    Show a static 3D mesh with an option to highlight specific faces.

    The vertex's coordinate system must be RAS+ (this is correspond to MNI space)

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
    ax.set_xlabel("R+")
    ax.set_ylabel("A+")
    ax.set_zlabel("S+")

    ax.set_title("Static 3D Mesh with Highlighted Faces")
    plt.show()

    return fig, ax

