
# Commoon Libraries
import numpy as np
import nibabel as nb
import trimesh

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe' 

# Custom Libraries
from brain_coord import RASp_toLPSp, reference2imageCoord

# Functions
def show_interactive_mesh(vertices, 
                          faces, 
                          highlight_face_info = {},
                          tick_interval = 20):
    """
    Show interactive mesh with optimized face highlighting.

    :param vertices(np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces(np.array - shape (#face, 3)): An array defining the triangular faces of the mesh. Each row contains three indices into the vertices array, specifying the vertices that form a triangular face.
    :param highlight_face_info(dict): A dictionary that specifies the face indexes to be highlighted and their associated colors.
    """
    
    # Default face colors: lightblue
    face_colors = ["lightblue"] * len(faces)
    
    # Apply highlighting colors
    for name in highlight_face_info:
        color = highlight_face_info[name]["color"]
        highlight_face_indexes = highlight_face_info[name]["data"]
        for face_index in highlight_face_indexes:
            face_colors[face_index] = color
            
    # Mesh trace with per-face colors
    mesh_trace = go.Mesh3d(
        x = vertices[:, 0],
        y = vertices[:, 1],
        z = vertices[:, 2],
        i = faces[:, 0],
        j = faces[:, 1],
        k = faces[:, 2],
        opacity = 1,
        facecolor = face_colors,  # Assign colors to each face
        hoverinfo = "text",
        text = [f"Face {i}, vertices: {faces[i]}" for i in range(len(faces))],  # Face indices in tooltip
    )

    # Create the figure
    fig = go.Figure(data=[mesh_trace])

    # Update layout for RAS+ coordinate system
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
                eye = dict(x=0, y=-2.5, z=1.5)  # Adjust camera view
            )
        ),
        title = "Interactive 3D Mesh with Highlighted Faces",
    )

    return fig

def show_non_interactive_mesh(vertices, 
                              faces, 
                              highlight_face_info = {}):
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

def show_stat_on3d(vertices, 
                   faces, 
                   stat_path,
                   colorscale = "jet",
                   color_min = None,
                   color_max = None,
                   tick_interval = 20):
    """
    Show stat

    :param vertices(np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces(np.array - shape (#face, 3)): An array defining the triangular faces of the mesh. Each row contains three indices into the vertices array, specifying the vertices that form a triangular face.
    :param stat_path: Path to the NIfTI file containing the statistical values.
    :param colorscale(str): Name of the colormap to use.
    :param color_min(float): Minimum value for colormap normalization.
    :param color_max(float): Maximum value for colormap normalization.
    :param tick_interval(int): Interval for axis ticks.
    """
    # Load stat
    stat = nb.load(stat_path)
    affine = stat.affine
    stat_array = stat.get_fdata()

    # Stat values
    stats = []
    for vertex_i, vertex_j, vertex_k in faces:
        coord1 = reference2imageCoord(vertices[vertex_i], 
                                      affine = affine).astype(int)
        coord2 = reference2imageCoord(vertices[vertex_j], 
                                      affine = affine).astype(int)
        coord3 = reference2imageCoord(vertices[vertex_k], 
                                      affine = affine).astype(int)
        
        stat1 = stat_array[coord1[0], coord1[1], coord1[2]]
        stat2 = stat_array[coord2[0], coord2[1], coord2[2]]
        stat3 = stat_array[coord3[0], coord3[1], coord3[2]]
        mean_value = np.mean([stat1, stat2, stat3])
        stats.append(mean_value)

    # Colors
    if color_min == None:
        color_min = np.min(stats)
    if color_max == None:
        color_max = np.max(stats)
        
    # Create the mesh3d plot
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=stats,
                colorscale=colorscale,
                colorbar=dict(title="Intensity"),
                hoverinfo="text",
                text=[f"Face {i}, vertices: {faces[i]}, stat: {stats[i]:.3f}" for i in range(len(faces))],
                cmax=color_max, 
                cmin=color_min,
            )
        ]
    )
    
    # Update layout for better visualization
    fig.update_layout(scene=dict(aspectmode="data", xaxis=dict(
                title = "R+",
                dtick = tick_interval, 
            ),
            yaxis=dict(
                title = "A+",
                dtick = tick_interval, 
            ),
            zaxis=dict(
                title = "S+",
                dtick = tick_interval, 
            )))
    
    # Show the plot
    fig.show()

def show_stat_onUV(vertices, 
                   uv_coordinates,
                   faces, 
                   stat_path,
                   colorscale = "jet",
                   color_min = None,
                   color_max = None,
                   tick_interval = 20,
                   type_ = "static"):
    """
    Show stat on UV map

    :param vertices(np.array - shape (#vertex, 3)): An array of 3D coordinates for the vertices of the mesh.
    :param faces(np.array - shape (#face, 3)): An array defining the triangular faces of the mesh. Each row contains three indices into the vertices array, specifying the vertices that form a triangular face.
    :param stat_path: Path to the NIfTI file containing the statistical values.
    :param colorscale(str): Name of the colormap to use.
    :param color_min(float): Minimum value for colormap normalization.
    :param color_max(float): Maximum value for colormap normalization.
    :param tick_interval(int): Interval for axis ticks.
    """
    # Figure
    fig, axis = plt.subplots(1)
    
    # Load stat
    stat = nb.load(stat_path)
    affine = stat.affine
    stat_array = stat.get_fdata()
    
    # Stat values
    stats = []
    for vertex_i, vertex_j, vertex_k in faces:
        coord1 = reference2imageCoord(vertices[vertex_i], 
                                      affine = affine).astype(int)
        coord2 = reference2imageCoord(vertices[vertex_j], 
                                      affine = affine).astype(int)
        coord3 = reference2imageCoord(vertices[vertex_k], 
                                      affine = affine).astype(int)
        
        stat1 = stat_array[coord1[0], coord1[1], coord1[2]]
        stat2 = stat_array[coord2[0], coord2[1], coord2[2]]
        stat3 = stat_array[coord3[0], coord3[1], coord3[2]]
        mean_value = np.mean([stat1, stat2, stat3])
        stats.append(mean_value)

    # Colors
    if color_min == None:
        color_min = np.min(stats)
    if color_max == None:
        color_max = np.max(stats)

    # Apply highlighting colors
    cmap = plt.get_cmap(colorscale)
    norm = plt.Normalize(vmin=color_min, vmax=color_max)
    face_colors = []
    for stat in stats:
        normalized_value = norm(stat)
        color = cmap(normalized_value)
        face_colors.append(color)
        
    # Create the mesh3d plot
    face_i = 0
    for face, stat, color in zip(faces, stats, face_colors):
        # Extract the UV coordinates for each vertex of the face
        uv_face = uv_coordinates[face]
        axis.fill(uv_face[:, 0], uv_face[:, 1], alpha = 1, color = color)

        face_i += 1
        
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar to work
    cbar = plt.colorbar(sm)
    cbar.set_label("Statistical Value", fontsize=12)

    return fig, axis
    
def load_mesh(path, type_ = "normal"):
    """
    Load mesh

    :param path(string): path of obj file representing brain mesh

    return (dictionary): 
        -k vertex(np.array - shape: #vertex, 3): vertices
        -k face(np.array - shape: #face, 3): faces based on vertex index
        -k uv(np.array - shape: #vertex, 2): uv coordinates
    """
    
    # Load data
    mesh = trimesh.load(path)

    # Dummy
    info = {}

    # Vertex
    vertices = mesh.vertices
    info["vertex"] = np.array(vertices)

    # Face
    faces = mesh.faces
    info["face"] = np.array(faces)

    # UV
    if type_ == "uv":
        uv_coordinates = mesh.visual.uv
        info["uv"] = np.array(uv_coordinates)
        
    return info
    