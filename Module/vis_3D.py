# Common Libraries
import copy, cv2, io
import numpy as np
import xarray
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors
from IPython.display import HTML
from PIL import Image

# Custom Libraries
from sj_array import reorient_ACS_array, get_ACS_axis_group

# Functions
class Plotter3D:
    def __init__(self,
                 visualize_coord_order = "LAS",
                 obj_info = None,
                 axis_info = None,
                 vis_info = None):
        """
        3D plot visualization manger

        :param visualize_coord_order: visualization coords ex) "LAS"
        :param obj_info: Dictionary for static objects ex) {'obj_name': {'point': [[x, y, z], ...]}}
        :param axis_info: Dictionary containing configuration for the origin axes visualization.
            * show_origin_axes: Whether to display the coordinate axes at the origin. (default: True)
            * axis_length: The length of the line for each axis. (default: 0.2)
            * cone_size: The size of the arrowhead (cone) at the end of each axis. (default: 0.1)
            * axis_origin: The (x, y, z) coordinates where the axes will be centered. (default: (0, 0, 0))
        :param vis_info: Dictionary containing configuration for visualization.
        """
        self.coord_order = visualize_coord_order
        self.obj_info = copy.deepcopy(obj_info) if obj_info else {}
        self.axis_info = axis_info if axis_info else {}
        self.vis_info = vis_info if vis_info else {}
        
        self.x_index, self.y_index, self.z_index = 0, 1, 2
        self.axis_titles = self._build_axis_titles()
        
        # Preprocessing obj_info and corners only once at initialization to prevent duplicate reorientation
        self._preprocess_obj_info()

    # Helper functions
    def _build_axis_titles(self):
        """
        Set axis title from pre-defined coordinate system
        """
        if not self.coord_order:
            return ["X", "Y", "Z"]
            
        opposite_order = []
        for vis_coord in self.coord_order:
            axis_group = get_ACS_axis_group(vis_coord)
            opposite_order.append(axis_group.replace(vis_coord, ""))
        return [f"{self.coord_order[i]}(-), {opposite_order[i]}(+)" for i in range(3)]

    def _preprocess_dataset(self, dataset_3d):
        """
        Align dataset coordinate system

        :param dataset_3d: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'
        """
        dataset_3d = copy.deepcopy(dataset_3d)
        if not self.coord_order:
            return dataset_3d
            
        ds_coord_order = [coord[0] for coord in dataset_3d.Coord.to_numpy()]
        dataset_3d["3D"].data = reorient_ACS_array(dataset_3d["3D"].data, ds_coord_order, self.coord_order)
        return dataset_3d

    def _preprocess_obj_info(self):
        """
        Align objects coordinate system
        """
        if not self.coord_order or not self.obj_info:
            return
            
        for obj_name in self.obj_info:
            obj_data = self.obj_info[obj_name]
            if "coord" not in obj_data:
                continue
                
            obj_coord_order = [e[0] for e in obj_data["coord"]]
            
            if "point" in obj_data:
                for kind in obj_data["point"]:
                    pts = np.array(obj_data["point"][kind])
                    pts = reorient_ACS_array(pts[None, :, :], obj_coord_order, self.coord_order)
                    obj_data["point"][kind] = pts[0]
                    
            if "corner" in obj_data:
                for kind in obj_data["corner"]:
                    for corner_name in obj_data["corner"][kind]:
                        corner_pt = np.array(obj_data["corner"][kind][corner_name])
                        reoriented_pt = reorient_ACS_array(corner_pt[None, None, :], obj_coord_order, self.coord_order)
                        obj_data["corner"][kind][corner_name] = reoriented_pt[0, 0]

    def _create_axis_traces(self):
        """
        Create axis on origin
        """
        if not self.axis_info.get("show_origin_axes", False):
            return []
            
        length = self.axis_info.get("axis_length", 0.2)
        cone_size = self.axis_info.get("cone_size", 0.1)
        orig = self.axis_info.get("axis_origin", (0, 0, 0))
        
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]
        dirs = [([orig[0], orig[0]+length], [orig[1], orig[1]], [orig[2], orig[2]]),
                ([orig[0], orig[0]], [orig[1], orig[1]+length], [orig[2], orig[2]]),
                ([orig[0], orig[0]], [orig[1], orig[1]], [orig[2], orig[2]+length])]
        uvw = [(length * 0.3, 0, 0), (0, length * 0.3, 0), (0, 0, length * 0.3)]
        
        traces = []
        for i in range(3):
            traces.append(go.Scatter3d(
                x=dirs[i][0], y=dirs[i][1], z=dirs[i][2],
                mode="lines+text", line=dict(color=colors[i], width=8),
                text=["", labels[i]], textposition="top center", showlegend=False
            ))
            traces.append(go.Cone(
                x=[dirs[i][0][1]], y=[dirs[i][1][1]], z=[dirs[i][2][1]],
                u=[uvw[i][0]], v=[uvw[i][1]], w=[uvw[i][2]],
                colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False,
                sizemode="absolute", sizeref=cone_size
            ))
        return traces

    def _create_obj_traces(self, use_qualitative_colors = False):
        """
        Create objects
        """
        traces = []
        plotly_colors = plotly.colors.qualitative.Plotly
        
        for idx, obj_name in enumerate(self.obj_info):
            kinds = self.obj_info[obj_name].get("point", {})
            color = plotly_colors[idx % len(plotly_colors)] if use_qualitative_colors else "black"
            
            for j, kind in enumerate(kinds):
                pts = np.array(kinds[kind])
                traces.append(go.Scatter3d(
                    x=pts[:, self.x_index], y=pts[:, self.y_index], z=pts[:, self.z_index],
                    mode="lines", line=dict(color=color, width=2),
                    visible=True, showlegend=(j == 0 if use_qualitative_colors else True),
                    name=obj_name if use_qualitative_colors else kind, hoverinfo="name"
                ))
        return traces

    def _create_skeleton_traces(self,
                                marker_coordinates: np.ndarray,
                                labels: list,
                                skeletons: list):
        """
        Create skeletons

        :param marker_coordinates: marker positions (#marker, xyz)
        :param labels: label of marker
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        """
        traces = []
        for p1, p2 in skeletons:
            if p1 in labels and p2 in labels:
                i1, i2 = labels.index(p1), labels.index(p2)
                traces.append(go.Scatter3d(
                    x=[marker_coordinates[i1, self.x_index], marker_coordinates[i2, self.x_index]],
                    y=[marker_coordinates[i1, self.y_index], marker_coordinates[i2, self.y_index]],
                    z=[marker_coordinates[i1, self.z_index], marker_coordinates[i2, self.z_index]],
                    mode="lines", line=dict(color="gray", width=4),
                    visible=True, showlegend=False, hoverinfo="skip"
                ))
        return traces

    def _calculate_scene_layout(self, dataset_3d_list: list[xarray.Dataset]):
        """
        Optimize scene ranges

        :param dataset_3d_list: List of datasets containing the '3D' variable with 'Time', 'Label', 'Coord' dimensions
        """
        candidates = []
        
        if self.axis_info.get("show_origin_axes", False):
            orig = self.axis_info.get("axis_origin", (0, 0, 0))
            length = self.axis_info.get("axis_length", 0.2)
            candidates.append(np.array(orig) - length)
            candidates.append(np.array(orig) + length)
            
        if self.obj_info:
            all_corners = []
            for obj in self.obj_info:
                obj_data = self.obj_info[obj]
                if "corner" in obj_data:
                    for kind in obj_data["corner"]:
                        all_corners.extend([obj_data["corner"][kind][c] for c in obj_data["corner"][kind]])
            if all_corners:
                candidates.append(np.min(all_corners, axis=0))
                candidates.append(np.max(all_corners, axis=0))
                
        for ds in dataset_3d_list:
            mins = ds["3D"].min(dim=[d for d in ds["3D"].dims if d != "Coord"], skipna=True).to_numpy()
            maxs = ds["3D"].max(dim=[d for d in ds["3D"].dims if d != "Coord"], skipna=True).to_numpy()
            candidates.append(mins)
            candidates.append(maxs)
            
        candidates = np.array(candidates)
        scene_min = np.min(candidates, axis=0)
        scene_max = np.max(candidates, axis=0)
        widths = scene_max - scene_min
        
        vis_ranges = [(scene_min[i] - widths[i] * 0.2, scene_max[i] + widths[i] * 0.2) for i in range(3)]
        axis_bg = self.vis_info.get("axis_bg", "rgba(230,230,230,30)")

        camera_config = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.8, y=-1.8, z=0.6)
        )

        return go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
            scene=dict(
                xaxis=dict(title=self.axis_titles[0], range=vis_ranges[0], backgroundcolor=axis_bg, autorange=False),
                yaxis=dict(title=self.axis_titles[1], range=vis_ranges[1], backgroundcolor=axis_bg, autorange=False),
                zaxis=dict(title=self.axis_titles[2], range=vis_ranges[2], backgroundcolor=axis_bg, autorange=False),
                aspectmode="data",
                camera=camera_config,
            ),
            showlegend=True,
        )

    # API
    def plot_time_series(self, dataset_3d: xarray.Dataset, skeletons: list = []):
        """
        Plot time series data
        
        :param dataset_3d: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        """
        dataset_3d = self._preprocess_dataset(dataset_3d)
        
        # 1. Initialization
        times = dataset_3d["Time"].to_numpy()
        n_frame = len(times)
        labels = list(dataset_3d["Label"].to_numpy())
        
        # Separate static object traces to keep them persistent during slider steps
        static_obj_traces = self._create_obj_traces()
        n_static = len(static_obj_traces)

        # Helper function
        def make_dynamic_traces(step_i: int):
            sel_times = times[:step_i + 1]
            marker_coordinates = dataset_3d.sel(Time = sel_times, Label = labels)["3D"].to_numpy()
                
            # Visualize - data markers
            cmap = plt.get_cmap("tab10")
            colors = cmap(np.linspace(0, 1, len(labels)))
            
            marker_traces = []
            for target, color in zip(labels, colors):
                target_index = labels.index(target)
                
                color_str = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
                trace = go.Scatter3d(
                    x=marker_coordinates[:, target_index, self.x_index],
                    y=marker_coordinates[:, target_index, self.y_index],
                    z=marker_coordinates[:, target_index, self.z_index],
                    mode="markers",
                    marker=dict(size=2, opacity=0.6, color=color_str),
                    visible=False,
                    name=target
                )
                marker_traces.append(trace)

            # Visualize - skeleton
            skeleton_traces = []
            for p1, p2 in skeletons:
                if p1 in labels and p2 in labels:
                    i1 = labels.index(p1)
                    i2 = labels.index(p2)
                
                    trace = go.Scatter3d(
                        x=[marker_coordinates[-1, i1, self.x_index], marker_coordinates[-1, i2, self.x_index]],
                        y=[marker_coordinates[-1, i1, self.y_index], marker_coordinates[-1, i2, self.y_index]],
                        z=[marker_coordinates[-1, i1, self.z_index], marker_coordinates[-1, i2, self.z_index]],
                        mode="lines",
                        line=dict(color="gray", width=4),
                        visible=False,
                        showlegend=False,
                        hoverinfo="skip"
                    )
                    skeleton_traces.append(trace)
            return skeleton_traces + marker_traces

        # 2. Create all traced over all frames
        all_traces = list(static_obj_traces)
        dynamic_blocks = []
        for step_i in range(n_frame):
            block = make_dynamic_traces(step_i)
            dynamic_blocks.append(block)
            all_traces.extend(block)
        
        n_dynamic_per_frame = len(dynamic_blocks[0]) if dynamic_blocks else 0
        total_traces_count = len(all_traces)

        # Make slider step
        steps = []
        for frame_i in range(n_frame):
            # Static objects remain visible (True), dynamic ones default to False
            visibility_mask = [True] * n_static + [False] * (total_traces_count - n_static)
            
            # Change related trace per frame
            start_idx = n_static + (frame_i * n_dynamic_per_frame)
            for j in range(n_dynamic_per_frame):
                visibility_mask[start_idx + j] = True  
            step = dict(
                method="restyle",
                args=["visible", visibility_mask],
                label=f"Frame {frame_i}"
            )
            steps.append(step)

        # Active first frame
        for j in range(n_dynamic_per_frame):
            all_traces[n_static + j].visible = True

        # Make figure and layout
        layout = self._calculate_scene_layout([dataset_3d])
        layout.update(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Time Step: "},
                pad={"t": 50},
                steps=steps
            )],
            height=800
        )

        fig = go.Figure(data=all_traces, layout=layout)
        return HTML(fig.to_html(include_plotlyjs="cdn"))

    def plot_single_dataset(self,
                            position_ds: xarray.Dataset,
                            targets: list = [], 
                            skeletons: list = []):
        """
        Plot single dataset

        :param position_ds: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'.
        :param targets: List of marker labels (Targets) to visualize.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        """
        position_ds = self._preprocess_dataset(position_ds)
        
        times = position_ds["Time"].to_numpy()
        targets = list(position_ds.Label.to_numpy()) if len(targets) == 0 else list(targets)
        
        """
        1. Visualize - Static Objects (e.g., table, environment boundaries)
        """
        obj_traces = self._create_obj_traces()

        """
        2. Visualize - Coordinate Axes (Origin)
        """
        axis_traces = self._create_axis_traces()
            
        """
        3. Visualize - Markers (Dynamic time-series data)
        """
        marker_traces = []
        
        # Extract coordinates for the selected times and targets
        selected_position_ds = position_ds.sel(Time=times, Label=targets)
        selected_position_array = selected_position_ds["3D"].to_numpy()
        
        # Define color gradient based on time progression (Coolwarm colormap)
        num_colors = len(times)
        my_cmap = plt.get_cmap("coolwarm")
        
        # Convert colormap to Plotly-compatible RGBA strings and handle ZeroDivisionError
        if num_colors > 1:
            colors = [f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})" 
                      for i in range(num_colors) for c in [my_cmap(i / (num_colors - 1))]]
        else:
            colors = ["rgba(0, 0, 255, 1)"] # Default blue if only one time step exists

        mode = self.vis_info.get("marker_mode", "markers")
        for target in targets:
            target_index = targets.index(target)
            trace = go.Scatter3d(
                x = selected_position_array[:, target_index, self.x_index],
                y = selected_position_array[:, target_index, self.y_index],
                z = selected_position_array[:, target_index, self.z_index],
                showlegend=False,
                mode = mode,
                marker = dict(
                    size = 4,
                    opacity = 0.8,
                    color = colors, # Apply time-based color gradient
                ),
                name = target,
                text = [f"{target} {t}" for t in times],
            )
            marker_traces.append(trace)
        
        """
        4. Skeleton
        """
        labels = list(selected_position_ds.Label.to_numpy())
        skeleton_traces = self._create_skeleton_traces(selected_position_array[-1], labels, skeletons)
                
        """
        5. Layout Configuration
        """
        layout = self._calculate_scene_layout([position_ds])
        layout.title = f"3D Estimation Traces ({len(times)} frames)"
        
        """
        5. Construct Figure and Render to HTML
        """
        data = axis_traces + obj_traces + marker_traces + skeleton_traces
        fig = go.Figure(data=data, layout=layout)
        return HTML(fig.to_html(include_plotlyjs="cdn"))

    def plot_multiple_datasets(self,
                               position_ds_list: list[xarray.Dataset],
                               targets: list = [],
                               skeletons_list: list = [],
                               dataset_names: list = []):
        """
        Plot multiple dataset

        :param position_ds: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'.
        :param targets: List of marker labels (Targets) to visualize.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        """
        processed_ds_list = [self._preprocess_dataset(ds) for ds in position_ds_list]
            
        # Validation check
        n_ds = len(processed_ds_list)
        dataset_names = [f"{i}" for i in range(n_ds)] if len(dataset_names) == 0 else dataset_names
        assert len(dataset_names) == n_ds, "dataset_names and position_ds_list must have the same length"
        
        """
        1. Static objects
        """
        obj_traces = self._create_obj_traces(use_qualitative_colors=True)

        """
        2. Marker traces for multiple datasets
        """
        marker_traces = []

        # dataset-level colors
        cmap_dataset = plt.get_cmap("tab10")
        dataset_rgbs = [cmap_dataset(i % 10)[:3] for i in range(n_ds)]
        
        mode = self.vis_info.get("marker_mode", "markers")
        for ds_idx, position_ds in enumerate(processed_ds_list):
            ds_name = dataset_names[ds_idx]
            rgb = dataset_rgbs[ds_idx]
            r, g, b = [int(v * 255) for v in rgb]

            times = position_ds["Time"].to_numpy()
            alphas = np.linspace(1.0, 0.15, len(times))
            point_colors = [f"rgba({r},{g},{b},{a})" for a in alphas]
            sel_t = list(position_ds.Label.to_numpy()) if len(targets) == 0 else targets
            marker_coordinates = position_ds.sel(Time=times, Label=sel_t)["3D"].to_numpy()

            for target_idx, target in enumerate(sel_t):
                trace = go.Scatter3d(
                    x=marker_coordinates[:, target_idx, self.x_index],
                    y=marker_coordinates[:, target_idx, self.y_index],
                    z=marker_coordinates[:, target_idx, self.z_index],
                    mode=mode,
                    marker=dict(
                        size=4,
                        opacity=0.8,
                        color=point_colors,
                    ),
                    name=ds_name,
                    legendgroup=ds_name,
                    showlegend=(target_idx == 0),
                    text=[f"{target} {t}" for t in times],
                )
                marker_traces.append(trace)

        """
        3. Skeleton
        """
        skeleton_traces = []
        for ds_idx, position_ds in enumerate(processed_ds_list):
            times = position_ds["Time"].to_numpy()
            marker_coordinates = position_ds.sel(Time=times)["3D"].to_numpy()
            labels = list(position_ds.Label.to_numpy())

            if ds_idx < len(skeletons_list):
                skeletons = skeletons_list[ds_idx]
            else:
                continue
                
            skeleton_traces.extend(self._create_skeleton_traces(marker_coordinates[-1], labels, skeletons))
        
        """
        4. Axis
        """
        axis_traces = self._create_axis_traces()
            
        """
        5. Layout configuration
        """
        layout = self._calculate_scene_layout(processed_ds_list)
        layout.title = "3D Estimation Traces (Multiple Datasets)"

        data = axis_traces + obj_traces + marker_traces + skeleton_traces
        fig = go.Figure(data=data, layout=layout)
        return HTML(fig.to_html(include_plotlyjs="cdn"))

    def export_to_video(self,
                        dataset_3d: xarray.Dataset,
                        skeletons: list = [],
                        file_path: str = "plotly_animation.mp4",
                        fps: int = 30,
                        width = 480,
                        height = 640):
        """
        Export the 3D time series animation to an MP4 video file frame by frame.
        Requires kaleido and opencv-python libraries.
        
        :param dataset_3d: xarray Dataset containing the 3D marker data.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        :param file_path: Output file path for the video.
        :param fps: Frames per second for the video output.
        """
        dataset_3d = self._preprocess_dataset(dataset_3d)
        times = dataset_3d["Time"].to_numpy()
        labels = list(dataset_3d["Label"].to_numpy())
        
        obj_traces = self._create_obj_traces()
        axis_traces = self._create_axis_traces()
        layout = self._calculate_scene_layout([dataset_3d])
        
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(labels)))
        mode = self.vis_info.get("marker_mode", "markers")
        
        video_writer = None
        print("Rendering video frames via Plotly + Kaleido. Please wait...")
        
        for frame_i in range(len(times)):
            sel_times = times[:frame_i + 1]
            marker_coordinates = dataset_3d.sel(Time=sel_times, Label=labels)["3D"].to_numpy()
            
            marker_traces = []
            for idx, (target, color) in enumerate(zip(labels, colors)):
                color_str = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
                marker_traces.append(go.Scatter3d(
                    x=marker_coordinates[:, idx, self.x_index],
                    y=marker_coordinates[:, idx, self.y_index],
                    z=marker_coordinates[:, idx, self.z_index],
                    mode=mode, marker=dict(size=4, opacity=0.8, color=color_str),
                    name=target, showlegend=False
                ))
            
            skeleton_traces = self._create_skeleton_traces(marker_coordinates[-1], labels, skeletons)
            
            frame_data = axis_traces + obj_traces + marker_traces + skeleton_traces
            frame_fig = go.Figure(data=frame_data, layout=layout)
            
            img_bytes = frame_fig.to_image(format="png", width=width, height=height)
            
            image = Image.open(io.BytesIO(img_bytes))
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if video_writer is None:
                height, width, _ = frame_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
            video_writer.write(frame_bgr)
            
        if video_writer is not None:
            video_writer.release()
        print(f"Video export complete! Saved as: {file_path}")
    