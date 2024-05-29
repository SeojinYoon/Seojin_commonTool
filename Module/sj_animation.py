
import numpy as np
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xarray as xr

def animate_markers(markerDataSet,
                    line_label_pairs = [],
                    marker_colorMap = "tab20c",
                    duration = 100):
    """
    Animate marker data

    :param markerDataSet(xarray): marker dataset, which has Times, Labels, Coords(X, Y, Z)
    :param line_label_pairs(list - tuple): pairs of two label for drawing lines with two marker positions.
    :param duration(int): how much faster frame is shown - ms
    """

    # Extract numpy array
    dataset_name = list(markerDataSet.keys())[0]

    labels = list(markerDataSet["Labels"].to_numpy())

    marker_data = markerDataSet.sel(Coords = ["X", "Y", "Z"], Labels = labels)
    marker_data = marker_data.transpose("Times", "Labels", "Coords")
    marker_array = marker_data[dataset_name].to_numpy()

    times = marker_data["Times"].to_numpy()

    # Constants
    x_index = 0
    y_index = 1
    z_index = 2

    marker_size = 5

    # Range
    x_min = np.min(marker_array[:, :, x_index]) - 10
    x_max = np.max(marker_array[:, :, x_index]) + 10
    y_min = np.min(marker_array[:, :, y_index]) - 10
    y_max = np.max(marker_array[:, :, y_index]) + 10
    z_min = np.min(marker_array[:, :, z_index]) - 10
    z_max = np.max(marker_array[:, :, z_index]) + 10

    # Argument info
    n_time, n_marker, n_coord = marker_array.shape
    initial_data = marker_array[0]
    initial_time = times[0]

    # Marker colormap
    cmap = plt.get_cmap(marker_colorMap)
    colors_ = [cmap(i) for i in range(n_marker)]
    hex_colors = [colors.rgb2hex(color[:3]) for color in colors_]

    # Initial marker traces
    initial_time_i = range(0, 1)
    initial_marker_traces = []
    for marker_i in range(n_marker):
        marker_trace = go.Scatter3d(x = marker_array[initial_time_i, marker_i, x_index],
                                    y = marker_array[initial_time_i, marker_i, y_index],
                                    z = marker_array[initial_time_i, marker_i, z_index],
                                    mode = 'markers',
                                    marker = dict(size = marker_size, color = hex_colors[marker_i]),
                                    name = labels[marker_i])
        initial_marker_traces.append(marker_trace)

    for label_pair in line_label_pairs:
        label1_index = labels.index(label_pair[0])
        label2_index = labels.index(label_pair[1])

        line_trace = go.Scatter3d(
            x=[marker_array[0, label1_index, x_index], marker_array[0, label2_index, x_index]],
            y=[marker_array[0, label1_index, y_index], marker_array[0, label2_index, y_index]],
            z=[marker_array[0, label1_index, z_index], marker_array[0, label2_index, z_index]],
            mode="lines",
            showlegend=False,
            line=dict(color="black", width=2)  # Customize color and line width if needed
        )
        initial_marker_traces.append(line_trace)

    fig = go.Figure(initial_marker_traces)

    # Define frames for animation
    frames = []
    for time_i in range(n_time):
        time = times[time_i]

        traces = []

        time_range = range(time_i, time_i+1)
        # Marker
        for marker_i in range(n_marker):
            marker_trace = go.Scatter3d(x = marker_array[time_range, marker_i, x_index],
                                        y = marker_array[time_range, marker_i, y_index],
                                        z = marker_array[time_range, marker_i, z_index],
                                        mode = 'markers',
                                        marker = dict(size = marker_size, color = hex_colors[marker_i]),
                                        name = labels[marker_i])
            traces.append(marker_trace)

        # Lines
        for label_pair in line_label_pairs:
            label1_index = labels.index(label_pair[0])
            label2_index = labels.index(label_pair[1])

            line_trace = go.Scatter3d(
                x=[marker_array[time_i, label1_index, x_index], marker_array[time_i, label2_index, x_index]],
                y=[marker_array[time_i, label1_index, y_index], marker_array[time_i, label2_index, y_index]],
                z=[marker_array[time_i, label1_index, z_index], marker_array[time_i, label2_index, z_index]],
                mode="lines",
                showlegend=False,
                line=dict(color="black", width=2)
            )
            traces.append(line_trace)
        frames.append(go.Frame(data = traces,
                               layout=go.Layout(annotations=[{
                                   "text": f"Time: {time}",
                                   "xref": "paper",
                                   "yref": "paper",
                                   "x": 0.5,
                                   "xanchor": "center",
                                   "y": 1.1,
                                   "yanchor": "bottom",
                                   "showarrow": False,
                                   "font": {"size": 12}
                               }])
                              ))

    # Add frames to layout
    fig.update(frames=frames)

    # Update layout with animation settings
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    }
                ],
            },
        ],
        annotations = [{
            "text": f"Time: {initial_time}",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.1,
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 12}
        }],
        scene=dict(
            xaxis=dict(range=[x_min, x_max], autorange=False),
            yaxis=dict(range=[y_min, y_max], autorange=False),
            zaxis=dict(range=[z_min, z_max], autorange=False),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1))))

    # Show the plot
    fig.show()
