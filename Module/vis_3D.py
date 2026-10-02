# Common Libraries
import copy, cv2, io
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors
from IPython.display import HTML
from PIL import Image
from typing import Any, Mapping, Optional, Sequence, Union

# Custom Libraries
from sj_array import (
    reorient_ACS_array, 
    get_ACS_axis_group, 
    get_ACS_explicit_orientation,
)

# Functions
class Plotter3D:
    def __init__(self,
                 visualize_coord_order = None,
                 line_ds_list: list[xr.Dataset] = None,
                 mesh_ds_list: list[xr.Dataset] = None,
                 axis_info = None,
                 vis_info = None):
        """
        3D plot visualization manger

        :param visualize_coord_order: visualization coords ex) "LAS"
        :param line_ds_list: List of line xarray Datasets created via make_line_ds
        :param mesh_ds_list: List of mesh xarray Datasets (e.g., planes) created via make_mesh_ds or make_plane_mesh_ds
        :param axis_info: Dictionary containing configuration for the origin axes visualization.
            * show_origin_axes: Whether to display the coordinate axes at the origin. (default: True)
            * axis_length: The length of the line for each axis. (default: 0.2)
            * cone_size: The size of the arrowhead (cone) at the end of each axis. (default: 0.1)
            * axis_origin: The (x, y, z) coordinates where the axes will be centered. (default: (0, 0, 0))
        :param vis_info: Dictionary containing configuration for visualization.
        """
        self.coord_order = visualize_coord_order
        self.axis_info = axis_info if axis_info else {}
        self.vis_info = vis_info if vis_info else {}
        
        self.x_index, self.y_index, self.z_index = 0, 1, 2
        self.axis_titles = self._build_axis_titles()
        if line_ds_list is not None:
            if isinstance(line_ds_list, xr.Dataset):
                line_ds_list = tablet_ds_to_line_ds_list(line_ds_list)
            self.line_ds_list = [self._preprocess_line(l) for l in line_ds_list]
        else:
            self.line_ds_list = []

        if mesh_ds_list is not None:
            if isinstance(mesh_ds_list, xr.Dataset):
                mesh_ds_list = [mesh_ds_list]
            self.mesh_ds_list = [self._preprocess_mesh(m) for m in mesh_ds_list]
        else:
            self.mesh_ds_list = []

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
            if axis_group:
                opposite_order.append(axis_group.replace(vis_coord, ""))
            else:
                opposite_order.append("")
        return [
            f"{self.coord_order[i]}(-), {opposite_order[i]}(+)"
            if opposite_order[i]
            else str(self.coord_order[i])
            for i in range(3)
        ]

    def _preprocess_dataset(self, dataset_3d):
        """
        Align dataset coordinate system

        :param dataset_3d: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'
        """
        dataset_3d = copy.deepcopy(dataset_3d)
        if not self.coord_order:
            return dataset_3d
            
        ds_coord_order = [coord[0] for coord in dataset_3d.Coord.to_numpy()]
        acs_chars = {"L", "R", "A", "P", "S", "I"}
        if not all(c in acs_chars for c in ds_coord_order) or not all(c in acs_chars for c in self.coord_order):
            return dataset_3d
        dataset_3d["3D"].data = reorient_ACS_array(dataset_3d["3D"].data, ds_coord_order, self.coord_order)
        return dataset_3d

    def _preprocess_mesh(self, mesh_ds: xr.Dataset) -> xr.Dataset:
        mesh_ds = copy.deepcopy(mesh_ds)
        if not self.coord_order:
            return mesh_ds

        ds_coord_order = [coord[0] for coord in mesh_ds.Coord.to_numpy()]
        acs_chars = {"L", "R", "A", "P", "S", "I"}
        if not all(c in acs_chars for c in ds_coord_order) or not all(c in acs_chars for c in self.coord_order):
            return mesh_ds
        verts = np.asarray(mesh_ds["vertices"].data)
        
        aligned_verts = reorient_ACS_array(verts, ds_coord_order, self.coord_order)
        mesh_ds["vertices"].data = aligned_verts
        mesh_ds["Coord"] = list(self.coord_order)
        return mesh_ds

    def _preprocess_line(self, line_ds: xr.Dataset) -> xr.Dataset:
        line_ds = copy.deepcopy(line_ds)
        if not self.coord_order:
            return line_ds

        ds_coord_order = [coord[0] for coord in line_ds.Coord.to_numpy()]
        acs_chars = {"L", "R", "A", "P", "S", "I"}
        if not all(c in acs_chars for c in ds_coord_order) or not all(c in acs_chars for c in self.coord_order):
            return line_ds
        pts = np.asarray(line_ds["points"].data)
        
        aligned_pts = reorient_ACS_array(pts, ds_coord_order, self.coord_order)
        line_ds["points"].data = aligned_pts
        line_ds["Coord"] = list(self.coord_order)
        return line_ds

    def _create_line_trace(self,
                           points: np.ndarray,
                           name: str = "Line",
                           color: str = "black",
                           width: float = 3.0,
                           opacity: float = 1.0,
                           dash: str = "solid",
                           visible: bool = True,
                           showlegend: bool = True):
        return go.Scatter3d(
            x=points[:, self.x_index],
            y=points[:, self.y_index],
            z=points[:, self.z_index],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            name=name,
            visible=visible,
            showlegend=showlegend,
            hoverinfo="name",
        )
        
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

    def _calculate_scene_layout(self,
                                dataset_3d_list: list[xr.Dataset],
                                mesh_ds_list: list[xr.Dataset] = None,
                                line_ds_list: list[xr.Dataset] = None):
        """
        Optimize scene ranges

        :param dataset_3d_list: List of datasets containing the '3D' variable with 'Time', 'Label', 'Coord' dimensions
        :param mesh_ds_list: List of datasets containing 'vertices'
        :param line_ds_list: List of datasets containing 'points'
        """
        candidates = []
        
        if self.axis_info.get("show_origin_axes", False):
            orig = self.axis_info.get("axis_origin", (0, 0, 0))
            length = self.axis_info.get("axis_length", 0.2)
            candidates.append(np.array(orig) - length)
            candidates.append(np.array(orig) + length)

        # Dataset markers range
        for ds in dataset_3d_list:
            if "3D" in ds:
                mins = ds["3D"].min(dim=[d for d in ds["3D"].dims if d != "Coord"], skipna=True).to_numpy()
                maxs = ds["3D"].max(dim=[d for d in ds["3D"].dims if d != "Coord"], skipna=True).to_numpy()
                candidates.append(mins)
                candidates.append(maxs)

        # Mesh datasets range
        if mesh_ds_list:
            for m_ds in mesh_ds_list:
                if "vertices" in m_ds:
                    verts = m_ds["vertices"].to_numpy()
                    candidates.append(np.nanmin(verts, axis=(0, 1)))
                    candidates.append(np.nanmax(verts, axis=(0, 1)))

        # Line datasets range
        combined_line_list = []
        if getattr(self, "line_ds_list", None):
            combined_line_list.extend(self.line_ds_list)
        if line_ds_list:
            combined_line_list.extend(line_ds_list)
        for l_ds in combined_line_list:
            if "points" in l_ds:
                pts = l_ds["points"].to_numpy()
                candidates.append(np.nanmin(pts, axis=(0, 1)))
                candidates.append(np.nanmax(pts, axis=(0, 1)))
                    
        candidates = np.array(candidates)
        scene_min = np.min(candidates, axis=0)
        scene_max = np.max(candidates, axis=0)
        widths = scene_max - scene_min
        
        max_width = np.max(widths) if len(widths) > 0 else 1.0
        default_pad = max_width * 0.2 if max_width > 0 else 0.5
        vis_ranges = []
        for i in range(3):
            if widths[i] < 1e-4:
                vis_ranges.append((float(scene_min[i] - default_pad), float(scene_max[i] + default_pad)))
            else:
                vis_ranges.append((float(scene_min[i] - widths[i] * 0.2), float(scene_max[i] + widths[i] * 0.2)))
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

    def _create_mesh_trace(self,
                           vertices: np.ndarray,
                           faces: np.ndarray,
                           name: str = "Mesh",
                           color: str = "lightblue",
                           opacity: float = 0.6,
                           visible: bool = True,
                           showlegend: bool = True) -> go.Mesh3d:
        """
        Create a 3D mesh trace using go.Mesh3d.
        Expects already preprocessed vertices.

        :param vertices: (V, 3) preprocessed vertex coordinates
        :param faces: (F, 3) triangle face indices
        :param name: Display name in legend/hover
        :param color: Mesh surface color
        :param opacity: Surface transparency (0.0 ~ 1.0)
        :param visible: Initial visibility
        :param showlegend: Whether to show in the legend
        """
        return go.Mesh3d(x=vertices[:, self.x_index],
                         y=vertices[:, self.y_index],
                         z=vertices[:, self.z_index],
                         i=faces[:, 0],
                         j=faces[:, 1],
                         k=faces[:, 2],
                         color=color,
                         opacity=opacity,
                         name=name,
                         visible=visible,
                         showlegend=showlegend,
                         flatshading=False,
                         lighting=dict(ambient=1.0,
                                       diffuse=0.0,
                                       specular=0.0,
                                       roughness=0.0,
                                       fresnel=0.0),
                         hoverinfo="none",
                         hovertemplate=None)

    # Dummy
    def _create_empty_position_ds(self, coords = ["X", "Y", "Z"]) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "3D": (("Time", "Label", "Coord"), np.full((1, 1, 3), np.nan))
            },
            coords={
                "Time": [0],
                "Label": ["__dummy__"],
                "Coord": coords,
            },
        )
        
    # API
    def plot_time_series(self,
                         dataset_3d: xr.Dataset,
                         skeletons: list = [],
                         mesh_ds_list: list[xr.Dataset] = [],
                         line_ds_list: list[xr.Dataset] = []):
        """
        Plot time series data
        
        :param dataset_3d: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        :param mesh_ds_list: List of mesh datasets containing 'vertices' and 'faces'.
        :param line_ds_list: List of line datasets containing 'points'.
        """
        dataset_3d = self._preprocess_dataset(dataset_3d)
        processed_meshes = list(self.mesh_ds_list) + [self._preprocess_mesh(m) for m in mesh_ds_list]
        all_lines = list(self.line_ds_list) + [self._preprocess_line(l) for l in line_ds_list]
        
        # 1. Initialization
        times = dataset_3d["Time"].to_numpy()
        n_frame = len(times)
        labels = list(dataset_3d["Label"].to_numpy())
        
        # Static line traces to keep them persistent during slider steps
        static_line_traces = []
        dynamic_lines = []
        for l_ds in all_lines:
            if l_ds.sizes["Time"] == 1:
                pts = l_ds["points"].isel(Time=0).to_numpy()
                static_line_traces.append(
                    self._create_line_trace(
                        points=pts,
                        name=l_ds.attrs.get("name", "Line"),
                        color=l_ds.attrs.get("color", "black"),
                        width=l_ds.attrs.get("width", 3.0),
                        opacity=l_ds.attrs.get("opacity", 1.0),
                        dash=l_ds.attrs.get("dash", "solid"),
                        visible=True,
                        showlegend=True,
                    )
                )
            else:
                dynamic_lines.append(l_ds)

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

            # Meshes per step
            step_mesh_traces = []
            for m_ds in processed_meshes:
                if "vertices" in m_ds and "faces" in m_ds.attrs:
                    frame_idx = step_i if m_ds.sizes["Time"] > 1 else 0
                    m_verts = m_ds["vertices"].isel(Time=frame_idx).to_numpy()
                    faces = np.array(m_ds.attrs["faces"])
                    step_mesh_traces.append(
                        self._create_mesh_trace(
                            vertices=m_verts,
                            faces=faces,
                            name=m_ds.attrs.get("name", "Mesh"),
                            color=m_ds.attrs.get("color", "lightblue"),
                            opacity=m_ds.attrs.get("opacity", 0.6),
                            visible=False,
                            showlegend=(step_i == 0),
                        )
                    )

            # Lines per step (dynamic)
            step_line_traces = []
            for l_ds in dynamic_lines:
                frame_idx = min(step_i, l_ds.sizes["Time"] - 1)
                pts = l_ds["points"].isel(Time=frame_idx).to_numpy()
                step_line_traces.append(
                    self._create_line_trace(
                        points=pts,
                        name=l_ds.attrs.get("name", "Line"),
                        color=l_ds.attrs.get("color", "black"),
                        width=l_ds.attrs.get("width", 3.0),
                        opacity=l_ds.attrs.get("opacity", 1.0),
                        dash=l_ds.attrs.get("dash", "solid"),
                        visible=False,
                        showlegend=(step_i == 0),
                    )
                )
                    
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
            return step_mesh_traces + step_line_traces + skeleton_traces + marker_traces

        # 2. Create all traced over all frames
        all_traces = list(static_line_traces)
        n_static = len(all_traces)
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
        layout = self._calculate_scene_layout([dataset_3d], mesh_ds_list=processed_meshes, line_ds_list=all_lines)
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
        return HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_single_dataset(self,
                            position_ds: xr.Dataset = None,
                            targets: list = [], 
                            skeletons: list = [],
                            mesh_ds_list: list[xr.Dataset] = [],
                            line_ds_list: list[xr.Dataset] = []):
        """
        Plot single dataset

        :param position_ds: Dataset containing '3D' variable with 'Time', 'Label', 'Coord'.
        :param targets: List of marker labels (Targets) to visualize.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        :param mesh_ds_list: List of mesh datasets containing 'vertices' and 'faces'.
        :param line_ds_list: List of line datasets containing 'points'.
        """
        if position_ds is None:
            coords = list(mesh_ds_list[0].Coord.to_numpy()) if mesh_ds_list else ["X", "Y", "Z"]
            position_ds = self._create_empty_position_ds(coords=coords)
        
        position_ds = self._preprocess_dataset(position_ds)
        processed_meshes = list(self.mesh_ds_list) + [self._preprocess_mesh(m) for m in mesh_ds_list]
        all_lines = list(self.line_ds_list) + [self._preprocess_line(l) for l in line_ds_list]
            
        times = position_ds["Time"].to_numpy()
        targets = list(position_ds.Label.to_numpy()) if len(targets) == 0 else list(targets)
        
        """
        1. Visualize - Coordinate Axes (Origin)
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
        5. Mesh
        """
        mesh_traces = []
        for m_ds in processed_meshes:
            if "vertices" in m_ds and "faces" in m_ds.attrs:
                proc_mesh_verts = m_ds["vertices"].isel(Time=-1).to_numpy()
                faces = np.array(m_ds.attrs["faces"])
                mesh_traces.append(
                    self._create_mesh_trace(
                        vertices=proc_mesh_verts,
                        faces=faces,
                        name=m_ds.attrs.get("name", "Mesh"),
                        color=m_ds.attrs.get("color", "lightblue"),
                        opacity=m_ds.attrs.get("opacity", 0.6),
                        visible=True,
                        showlegend=True,
                    )
                )

        """
        6. Line
        """
        line_traces = []
        for l_ds in all_lines:
            if "points" in l_ds:
                proc_pts = l_ds["points"].isel(Time=-1).to_numpy()
                line_traces.append(
                    self._create_line_trace(
                        points=proc_pts,
                        name=l_ds.attrs.get("name", "Line"),
                        color=l_ds.attrs.get("color", "black"),
                        width=l_ds.attrs.get("width", 3.0),
                        opacity=l_ds.attrs.get("opacity", 1.0),
                        dash=l_ds.attrs.get("dash", "solid"),
                        visible=True,
                        showlegend=True,
                    )
                )
            
        """
        7. Layout Configuration
        """
        layout = self._calculate_scene_layout([position_ds], mesh_ds_list=processed_meshes, line_ds_list=all_lines)
        layout.update(title = f"3D Estimation Traces ({len(times)} frames)", height = 800)
        
        """
        8. Construct Figure and Render to HTML
        """
        data = mesh_traces + axis_traces + line_traces + skeleton_traces + marker_traces
        fig = go.Figure(data=data, layout=layout)
        return HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_multiple_datasets(
        self,
        position_ds_list: list[xr.Dataset] = None,
        targets: list = [],
        skeletons_list: list = [],
        dataset_names: list = [],
        mesh_ds_list: list[xr.Dataset] = [],
        line_ds_list: list[xr.Dataset] = [],
    ):
        """Plot multiple datasets with optional 3D meshes and lines.

        :param position_ds_list: List of datasets containing '3D' variable
          with 'Time', 'Label', 'Coord'.
        :param targets: List of marker labels (Targets) to visualize.
        :param skeletons_list: List of skeleton configurations per dataset.
        :param dataset_names: Custom names for each dataset legend group.
        :param mesh_ds_list: List of xarray Datasets containing 'vertices' and
          'faces' as attrs.
        :param line_ds_list: List of line xarray Datasets containing 'points'.
        """
        if position_ds_list is None:
            coords = (
                list(mesh_ds_list[0].Coord.to_numpy())
                if mesh_ds_list
                else ["X", "Y", "Z"]
            )
            position_ds_list = [self._create_empty_position_ds(coords=coords)]

        processed_ds_list = [
            self._preprocess_dataset(ds) for ds in position_ds_list
        ]
        processed_meshes = list(self.mesh_ds_list) + [self._preprocess_mesh(m) for m in mesh_ds_list]
        all_lines = list(self.line_ds_list) + [self._preprocess_line(l) for l in line_ds_list]

        if not skeletons_list:
            skeletons_list = [[] for _ in processed_ds_list]

        """
        1. Visualize - Coordinate axes (Origin)
        """
        axis_traces = self._create_axis_traces()

        """
        3. Visualize - Markers
        """
        marker_traces = []
        skeleton_traces = []

        # Color palettes for different datasets
        colors = plotly.colors.qualitative.Plotly
        symbols = [
            "circle",
            "diamond",
            "square",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
        ]

        mode = self.vis_info.get("marker_mode", "markers")

        for ds_idx, ds in enumerate(processed_ds_list):
            current_targets = (
                list(ds.Label.to_numpy()) if len(targets) == 0 else list(targets)
            )
            times = ds["Time"].to_numpy()

            # Extract coordinates for the selected targets
            selected_ds = ds.sel(Time=times, Label=current_targets)
            selected_array = selected_ds["3D"].to_numpy()

            ds_name = (
                dataset_names[ds_idx]
                if ds_idx < len(dataset_names)
                else f"Dataset {ds_idx + 1}"
            )
            ds_color = colors[ds_idx % len(colors)]
            ds_symbol = symbols[ds_idx % len(symbols)]

            # A. Markers for this dataset
            for target in current_targets:
                target_idx = current_targets.index(target)
                marker_traces.append(
                    go.Scatter3d(
                        x=selected_array[:, target_idx, self.x_index],
                        y=selected_array[:, target_idx, self.y_index],
                        z=selected_array[:, target_idx, self.z_index],
                        showlegend=(target == current_targets[0]),
                        legendgroup=ds_name,
                        mode=mode,
                        marker=dict(
                            size=4,
                            opacity=0.8,
                            color=ds_color,
                            symbol=ds_symbol,
                        ),
                        name=ds_name,
                        text=[f"[{ds_name}] {target} {t}" for t in times],
                    )
                )

            # B. Skeletons for this dataset
            labels = list(selected_ds.Label.to_numpy())
            skeletons = (
                skeletons_list[ds_idx] if ds_idx < len(skeletons_list) else []
            )

            for p1, p2 in skeletons:
                if p1 in labels and p2 in labels:
                    i1, i2 = labels.index(p1), labels.index(p2)
                    skeleton_traces.append(
                        go.Scatter3d(
                            x=[
                                selected_array[-1, i1, self.x_index],
                                selected_array[-1, i2, self.x_index],
                            ],
                            y=[
                                selected_array[-1, i1, self.y_index],
                                selected_array[-1, i2, self.y_index],
                            ],
                            z=[
                                selected_array[-1, i1, self.z_index],
                                selected_array[-1, i2, self.z_index],
                            ],
                            mode="lines",
                            line=dict(color=ds_color, width=3),
                            visible=True,
                            showlegend=False,
                            legendgroup=ds_name,
                            hoverinfo="skip",
                        )
                    )

        """
        4. Visualize - Meshes
        """
        mesh_traces = []
        for m_ds in processed_meshes:
            if "vertices" in m_ds and "faces" in m_ds.attrs:
                proc_mesh_verts = m_ds["vertices"].isel(Time=-1).to_numpy()
                faces = np.array(m_ds.attrs["faces"])
                mesh_traces.append(
                    self._create_mesh_trace(
                        vertices=proc_mesh_verts,
                        faces=faces,
                        name=m_ds.attrs.get("name", "Mesh"),
                        color=m_ds.attrs.get("color", "lightblue"),
                        opacity=m_ds.attrs.get("opacity", 0.6),
                        visible=True,
                        showlegend=True,
                    )
                )

        """
        5. Visualize - Lines
        """
        line_traces = []
        for l_ds in all_lines:
            if "points" in l_ds:
                proc_pts = l_ds["points"].isel(Time=-1).to_numpy()
                line_traces.append(
                    self._create_line_trace(
                        points=proc_pts,
                        name=l_ds.attrs.get("name", "Line"),
                        color=l_ds.attrs.get("color", "black"),
                        width=l_ds.attrs.get("width", 3.0),
                        opacity=l_ds.attrs.get("opacity", 1.0),
                        dash=l_ds.attrs.get("dash", "solid"),
                        visible=True,
                        showlegend=True,
                    )
                )

        """
        6. Layout configuration
        """
        layout = self._calculate_scene_layout(
            processed_ds_list, mesh_ds_list=processed_meshes, line_ds_list=all_lines
        )

        title_frame_count = (
            len(processed_ds_list[0]["Time"]) if processed_ds_list else 1
        )
        layout.update(
            title=f"3D Estimation Traces ({title_frame_count} frames)",
            height=800,
        )

        """
        7. Construct Figure and Render to HTML
        """
        data = (
            mesh_traces
            + axis_traces
            + line_traces
            + skeleton_traces
            + marker_traces
        )
        fig = go.Figure(data=data, layout=layout)
        return HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def export_to_video(self,
                        dataset_3d: xr.Dataset,
                        skeletons: list = [],
                        mesh_ds_list: list[xr.Dataset] = [],
                        line_ds_list: list[xr.Dataset] = [],
                        file_path: str = "plotly_animation.mp4",
                        fps: int = 30,
                        width = 480,
                        height = 640):
        """
        Export the 3D time series animation to an MP4 video file frame by frame.
        Requires kaleido and opencv-python libraries.
        
        :param dataset_3d: xarray Dataset containing the 3D marker data.
        :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
        :param mesh_ds_list: List of mesh datasets containing 'vertices' and 'faces'.
        :param line_ds_list: List of line datasets containing 'points'.
        :param file_path: Output file path for the video.
        :param fps: Frames per second for the video output.
        """
        dataset_3d = self._preprocess_dataset(dataset_3d)
        times = dataset_3d["Time"].to_numpy()
        labels = list(dataset_3d["Label"].to_numpy())
        
        processed_meshes = list(self.mesh_ds_list) + [self._preprocess_mesh(m) for m in mesh_ds_list]
        all_lines = list(self.line_ds_list) + [self._preprocess_line(l) for l in line_ds_list]
        
        axis_traces = self._create_axis_traces()
        layout = self._calculate_scene_layout([dataset_3d], mesh_ds_list=processed_meshes, line_ds_list=all_lines)
        
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
            
            mesh_traces = []
            for m_ds in processed_meshes:
                if "vertices" in m_ds and "faces" in m_ds.attrs:
                    frame_idx = frame_i if m_ds.sizes["Time"] > 1 else 0
                    m_verts = m_ds["vertices"].isel(Time=frame_idx).to_numpy()
                    faces = np.array(m_ds.attrs["faces"])
                    mesh_traces.append(self._create_mesh_trace(
                        vertices=m_verts,
                        faces=faces,
                        name=m_ds.attrs.get("name", "Mesh"),
                        color=m_ds.attrs.get("color", "lightblue"),
                        opacity=m_ds.attrs.get("opacity", 0.6),
                        visible=True,
                        showlegend=False,
                    ))

            line_traces = []
            for l_ds in all_lines:
                if "points" in l_ds:
                    frame_idx = min(frame_i, l_ds.sizes["Time"] - 1)
                    pts = l_ds["points"].isel(Time=frame_idx).to_numpy()
                    line_traces.append(self._create_line_trace(
                        points=pts,
                        name=l_ds.attrs.get("name", "Line"),
                        color=l_ds.attrs.get("color", "black"),
                        width=l_ds.attrs.get("width", 3.0),
                        opacity=l_ds.attrs.get("opacity", 1.0),
                        dash=l_ds.attrs.get("dash", "solid"),
                        visible=True,
                        showlegend=False,
                    ))

            frame_data = mesh_traces + line_traces + axis_traces + marker_traces + skeleton_traces
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

def load_obj(file_path: str) -> dict:
    """
    Load .obj file to read vertex and face informations

    :param file_path: obj file path
    
    return vertexes, faces
    """
    verts = []
    triangles = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                v_idx = [int(p.split('/')[0]) - 1 for p in parts]
                for t in range(1, len(v_idx) - 1):
                    triangles.append([v_idx[0], v_idx[t], v_idx[t + 1]])
                    
    verts = np.array(verts)
    triangles = np.array(triangles)

    result = {}
    result["vertex"] = verts
    result["face"] = triangles

    return result

def draw_obj(vertices: np.ndarray,
             faces: np.ndarray,
             width = 900,
             height = 750):
    """
    Draw obj file

    :param vertices(shape: #vertex, xyz): position of vertex
    :param faces(shape: #face, #vertex): faces
    :param width: horizontal length of graph
    :param height: vertical length of graph
    """
    # Draw mesh
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=vertices[:, 2],
        colorscale='Viridis',
        showscale=True,
    )

    # Draw figure
    fig = go.Figure(data=[mesh_trace])
    fig.update_layout(
        title=dict(text="Interactive Plotly 3D Arm Marker Mesh", font=dict(size=18)),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        width=width,
        height=height
    )
    
    return HTML(fig.to_html(include_plotlyjs="cdn"))

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _normalize_3d_timeseries(arr: np.ndarray,
                             times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
                             name: str = "array") -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D (N, 3) or 3D (T, N, 3) input to (T, N, 3) and validate time coordinates.

    :param arr: Input coordinate array with shape (N, 3) or (T, N, 3)
    :param times: Optional array, sequence, or scalar representing time coordinates
    :param name: Variable name used for descriptive error messages
    
    :return: Tuple of normalized 3D array (T, N, 3) and 1D time coordinates
    """
    arr = np.asarray(arr)

    # Align to 3D tensor: (T, N, 3)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            f"Expected {name} to have 2 or 3 dimensions, got shape {arr.shape}"
        )

    n_times = arr.shape[0]

    # Resolve time coordinates
    if times is None:
        time_coords = np.arange(n_times)
    else:
        if np.isscalar(times):
            time_coords = np.array([times])
        else:
            time_coords = np.asarray(times)

        if len(time_coords) != n_times:
            raise ValueError(
                f"Length of times ({len(time_coords)}) does not match "
                f"{name} time dimension ({n_times})"
            )

    return arr, time_coords


def build_3d_dataset(var_name: str,
                     data: np.ndarray,
                     dims: tuple[str, str, str],
                     coords: Mapping[str, Sequence],
                     attrs: Optional[dict[str, Any]] = None) -> xr.Dataset:
    """
    Build a standard 3D single-variable xarray.Dataset.

    :param var_name: Key name for the primary data variable
    :param data: 3D data array
    :param dims: Dimension names tuple, e.g., ('Time', 'Vertex', 'Coord')
    :param coords: Mapping of coordinate names to values
    :param attrs: Optional dataset metadata dictionary
    
    :return: Constructed xarray.Dataset
    """
    return xr.Dataset(
        data_vars={var_name: (dims, data)},
        coords=coords,
        attrs=attrs or {},
    )


# ---------------------------------------------------------
# Public Dataset Constructors
# ---------------------------------------------------------
def make_mesh_ds(vertices: np.ndarray,
                 faces: np.ndarray,
                 coord_order: str = "XYZ",
                 times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
                 name: str = "Mesh",
                 color: str = "lightblue",
                 opacity: float = 0.6) -> xr.Dataset:
    """
    Create a 3D Mesh xarray.Dataset compatible with Plotter3D.

    :param vertices: (V, 3) for single frame or (T, V, 3) for time series
    :param faces: (F, 3) triangle face indices
    :param coord_order: Source coordinate system string (e.g., "XYZ", "RDF", "RAS")
    :param times: Optional array or sequence of time indices/timestamps
    :param name: Display name for the mesh
    :param color: Default surface color (e.g., "lightblue", "lightpink")
    :param opacity: Surface opacity (0.0 to 1.0)
    
    :return: xr.Dataset with ('Time', 'Vertex', 'Coord') dims
    """
    vertices, time_coords = _normalize_3d_timeseries(
        vertices, times, name="vertices"
    )
    faces = np.asarray(faces)

    return build_3d_dataset(
        var_name="vertices",
        data=vertices,
        dims=("Time", "Vertex", "Coord"),
        coords={
            "Time": time_coords,
            "Vertex": np.arange(vertices.shape[1]),
            "Coord": list(coord_order),
        },
        attrs={
            "faces": faces,
            "name": name,
            "color": color,
            "opacity": opacity,
        },
    )

def make_plane_mesh_ds(
    corners: Optional[Union[np.ndarray, Sequence]] = None,
    *,
    center: Optional[Sequence[float]] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    plane: str = "xy",
    normal: Optional[Sequence[float]] = None,
    name: str = "Plane",
    color: str = "lightblue",
    opacity: float = 0.4,
    coord_order: str = "XYZ",
    times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
    double_sided: bool = True,
) -> xr.Dataset:
    """
    Create a 3D rectangular/quad plane mesh xarray.Dataset compatible with Plotter3D.

    The plane can be defined in either of two ways:
    1. By passing 4 corner points (corners):
       A sequence or (4, 3) array of 4 vertices ordered along the perimeter
       (e.g., [p0, p1, p2, p3] where edges connect p0-p1, p1-p2, p2-p3, p3-p0).
    2. By passing center, width, height, and plane orientation:
       - plane: "xy", "xz", or "yz" (default: "xy")
       - or normal: a 3D normal vector to orient the plane perpendicular to it.

    :param corners: (4, 3) coordinates of the 4 corners in perimeter order.
    :param center: (3,) center of the plane when specifying by dimensions.
    :param width: Width of the plane (along primary local axis).
    :param height: Height of the plane (along secondary local axis).
    :param plane: Preset plane orientation ("xy", "xz", "yz").
    :param normal: Optional 3D normal vector for arbitrary orientation.
    :param name: Display name for the plane.
    :param color: Surface color (e.g., 'lightblue', 'lightgray', '#1f77b4', 'rgba(...)').
    :param opacity: Surface opacity (0.0 = fully transparent, 1.0 = opaque).
    :param coord_order: Coordinate system string (e.g., 'XYZ', 'RAS', 'LPS').
    :param times: Optional time index or array of time coordinates.
    :param double_sided: If True, includes front and back face triangles so the plane
                         is visible from both sides.
    :return: xr.Dataset with ('Time', 'Vertex', 'Coord') dims and 'faces' in attrs.
    """
    if corners is None:
        if center is None or width is None or height is None:
            raise ValueError("Either 'corners' (4 points) or ('center', 'width', 'height') must be provided.")
        center = np.asarray(center, dtype=float).reshape(3)
        w2 = float(width) / 2.0
        h2 = float(height) / 2.0

        if normal is not None:
            n = np.asarray(normal, dtype=float)
            norm = np.linalg.norm(n)
            if norm == 0:
                raise ValueError("Normal vector must have non-zero length.")
            n = n / norm
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(n, ref)
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            corners = np.array([
                center - w2 * u - h2 * v,
                center + w2 * u - h2 * v,
                center + w2 * u + h2 * v,
                center - w2 * u + h2 * v,
            ])
        else:
            plane_lower = plane.lower()
            if plane_lower == "xy":
                corners = np.array([
                    [center[0] - w2, center[1] - h2, center[2]],
                    [center[0] + w2, center[1] - h2, center[2]],
                    [center[0] + w2, center[1] + h2, center[2]],
                    [center[0] - w2, center[1] + h2, center[2]],
                ])
            elif plane_lower == "xz":
                corners = np.array([
                    [center[0] - w2, center[1], center[2] - h2],
                    [center[0] + w2, center[1], center[2] - h2],
                    [center[0] + w2, center[1], center[2] + h2],
                    [center[0] - w2, center[1], center[2] + h2],
                ])
            elif plane_lower == "yz":
                corners = np.array([
                    [center[0], center[1] - w2, center[2] - h2],
                    [center[0], center[1] + w2, center[2] - h2],
                    [center[0], center[1] + w2, center[2] + h2],
                    [center[0], center[1] - w2, center[2] + h2],
                ])
            else:
                raise ValueError(f"Unknown plane orientation '{plane}'. Choose from 'xy', 'xz', 'yz', or provide 'normal'.")
    else:
        corners = np.asarray(corners, dtype=float)
        if corners.ndim == 2:
            if corners.shape != (4, 3):
                raise ValueError(f"Expected corners of shape (4, 3), but got {corners.shape}.")
        elif corners.ndim == 3:
            if corners.shape[1:] != (4, 3):
                raise ValueError(f"Expected corners of shape (T, 4, 3), but got {corners.shape}.")
        else:
            raise ValueError(f"Invalid corners array dimensions: {corners.ndim}")

    if double_sided:
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 2, 1],
            [0, 3, 2],
        ])
    else:
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])

    return make_mesh_ds(
        vertices=corners,
        faces=faces,
        coord_order=coord_order,
        times=times,
        name=name,
        color=color,
        opacity=opacity,
    )


# Alias
make_plane_ds = make_plane_mesh_ds


def make_plane_wireframe_ds(
    corners: Optional[Union[np.ndarray, Sequence]] = None,
    *,
    center: Optional[Sequence[float]] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    plane: str = "xy",
    normal: Optional[Sequence[float]] = None,
    name: str = "Plane_Wireframe",
    color: str = "black",
    width_pixels: float = 2.0,
    opacity: float = 1.0,
    dash: str = "solid",
    coord_order: str = "XYZ",
    times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
) -> xr.Dataset:
    """
    Create a closed 3D wireframe outline (boundary loop) dataset for a rectangular/quad plane.

    :param corners: (4, 3) coordinates of the 4 corners in perimeter order.
    :param center: (3,) center of the plane when specifying by dimensions.
    :param width: Width of the plane.
    :param height: Height of the plane.
    :param plane: Preset plane orientation ("xy", "xz", "yz").
    :param normal: Optional 3D normal vector.
    :param name: Display name for the wireframe.
    :param color: Line color.
    :param width_pixels: Line width in pixels.
    :param opacity: Line opacity (0.0 to 1.0).
    :param dash: Line dash style.
    :param coord_order: Coordinate system string.
    :param times: Optional time index.
    :return: xr.Dataset compatible with Plotter3D line_ds_list.
    """
    plane_ds = make_plane_mesh_ds(
        corners=corners,
        center=center,
        width=width,
        height=height,
        plane=plane,
        normal=normal,
        coord_order=coord_order,
        times=times,
    )
    verts = plane_ds["vertices"].to_numpy()
    loop_pts = np.concatenate([verts, verts[:, :1, :]], axis=1)
    if loop_pts.shape[0] == 1 and times is None:
        loop_pts = loop_pts[0]

    return make_line_ds(
        points=loop_pts,
        coord_order=list(plane_ds.Coord.to_numpy()),
        times=plane_ds.Time.to_numpy(),
        name=name,
        color=color,
        width=width_pixels,
        opacity=opacity,
        dash=dash,
    )



def make_point_ds(positions: np.ndarray,
                  times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
                  coord_order: str = "XYZ",
                  labels: Optional[Sequence[str]] = None) -> xr.Dataset:
    """
    Create a 3D point/marker xarray.Dataset.

    :param positions: (P, 3) for single frame or (T, P, 3) for time series
    :param times: Optional array or sequence of time indices/timestamps
    :param coord_order: Coordinate ordering string (e.g., "XYZ")
    :param labels: Optional list of names for each point
    
    :return: xr.Dataset with ('Time', 'Label', 'Coord') dims
    """
    positions, time_coords = _normalize_3d_timeseries(
        positions, times, name="positions"
    )
    n_pts = positions.shape[1]

    # Generate default point labels if none are supplied
    if labels is None:
        label_coords = [f"point_{i}" for i in range(n_pts)]
    else:
        label_coords = list(labels)
        if len(label_coords) != n_pts:
            raise ValueError(
                f"Length of labels ({len(label_coords)}) does not match "
                f"point dimension ({n_pts})"
            )

    return build_3d_dataset(
        var_name="3D",
        data=positions,
        dims=("Time", "Label", "Coord"),
        coords={
            "Time": time_coords,
            "Label": label_coords,
            "Coord": list(coord_order),
        },
    )


def make_ACS_timeseries(data: np.ndarray,
                        labels: Sequence[str],
                        coord_system: Any,
                        times: Optional[Union[float, int, Sequence, np.ndarray]] = None) -> xr.Dataset:
    """
    Create a time series dataset in Anatomical Coordinate System (ACS).

    :param data: (L, 3) for single frame or (T, L, 3) for time series
    :param labels: Label sequence for each marker/body landmark
    :param coord_system: Source coordinate system passed to get_ACS_explicit_orientation
    :param times: Optional array or sequence of time indices/timestamps
    
    :return: xr.Dataset with ('Time', 'Label', 'Coord') dims
    """
    data, time_coords = _normalize_3d_timeseries(data, times, name="ACS data")
    orientation = get_ACS_explicit_orientation(coord_system)

    if len(labels) != data.shape[1]:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match "
            f"marker dimension ({data.shape[1]})"
        )

    return build_3d_dataset(
        var_name="3D",
        data=data,
        dims=("Time", "Label", "Coord"),
        coords={
            "Time": time_coords,
            "Label": list(labels),
            "Coord": list(orientation),
        },
    )

def make_labeled_ds(data: np.ndarray,
                    labels: Sequence[str],
                    times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
                    coords: Sequence[str] = ("X", "Y", "Z"),
                    var_name: str = "3D",
                    attrs: Optional[dict[str, Any]] = None) -> xr.Dataset:
    """
    Create an xarray.Dataset with ('Time', 'Label', 'Coord') dimensions.

    :param data: Input array of shape (N, 3) or (T, N, 3)
    :param labels: Names for each entity along the 'Label' axis
    :param times: Optional timestamp sequence or scalar (defaults to 0..T-1)
    :param coords: Coordinate axis names (default: ("X", "Y", "Z"))
    :param var_name: Dataset variable name (default: "3D")
    :param attrs: Optional metadata dictionary
    
    :return: xr.Dataset
    """
    # 1. Normalize 2D/3D shape and resolve time coordinates
    data_3d, time_coords = _normalize_3d_timeseries(data, times, name="data")

    # 2. Validate label count against the entity axis
    label_list = list(labels)
    if len(label_list) != data_3d.shape[1]:
        raise ValueError(
            f"Length of labels ({len(label_list)}) does not match "
            f"entity dimension ({data_3d.shape[1]})"
        )

    # 3. Construct dataset via standard builder
    return build_3d_dataset(
        var_name=var_name,
        data=data_3d,
        dims=("Time", "Label", "Coord"),
        coords={
            "Time": time_coords,
            "Label": label_list,
            "Coord": list(coords),
        },
        attrs=attrs,
    )

def make_line_ds(points: np.ndarray,
                 coord_order: Sequence[str] = ("X", "Y", "Z"),
                 times: Optional[Union[float, int, Sequence, np.ndarray]] = None,
                 name: str = "Line",
                 color: str = "black",
                 width: float = 3.0,
                 opacity: float = 1.0,
                 dash: str = "solid") -> xr.Dataset:
    """
    Create a line dataset with ('Time', 'Point', 'Coord') dimensions.

    :param points: Coordinates of line vertices. Shape can be:
                   - (N, 3): Single-frame line path with N points.
                   - (T, N, 3): Time-series line path with T frames and N points.
    :param coord_order: Coordinate order of the input points (e.g. ("X", "Y", "Z") or "XYZ").
    :param times: Optional timestamp sequence or scalar (defaults to 0..T-1).
    :param name: Name of the line / object.
    :param color: CSS or hex color string (e.g. "red", "#E74C3C").
    :param width: Line width in pixels.
    :param opacity: Line opacity (0.0 to 1.0).
    :param dash: Dash style ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot").
    :return: xr.Dataset with ('Time', 'Point', 'Coord') dims and 'points' data variable.
    """
    points_3d, time_coords = _normalize_3d_timeseries(np.asarray(points), times, name="points")
    n_pts = points_3d.shape[1]

    return build_3d_dataset(
        var_name="points",
        data=points_3d,
        dims=("Time", "Point", "Coord"),
        coords={
            "Time": time_coords,
            "Point": np.arange(n_pts),
            "Coord": list(coord_order),
        },
        attrs={
            "name": name,
            "color": color,
            "width": float(width),
            "opacity": float(opacity),
            "dash": dash,
        },
    )

def make_tablet_ds(outer_corners: np.ndarray | dict[str, Sequence[float]],
                   inner_corners: np.ndarray | dict[str, Sequence[float]] = None,
                   angle: float = 0.0,
                   coords: Sequence[str] = ("X", "Y", "Z"),
                   times: Sequence[Any] = (0,),
                   attrs: dict = None) -> xr.Dataset:
    """
    Create a standardized tablet xarray.Dataset containing outer and inner corner keypoints.

    Labels standard order:
    - outer: 'outer_ur', 'outer_ul', 'outer_dl', 'outer_dr'
    - inner: 'inner_ur', 'inner_ul', 'inner_dl', 'inner_dr'

    :param outer_corners: Either a dict with keys ('ur'/'up right', 'ul'/'up left', 'dl'/'down left', 'dr'/'down right')
                          or an array of shape (4, 3) or (T, 4, 3) in [ur, ul, dl, dr] order.
    :param inner_corners: Optional dict or array for drawing area corners in the same format.
    :param angle: Tablet angle in degrees.
    :param coords: Coordinate labels (e.g. ("X", "Y", "Z") or "LAS").
    :param times: Optional timestamp sequence or scalar (defaults to (0,)).
    :param attrs: Optional additional metadata attributes.
    :return: xr.Dataset created via make_labeled_ds with '3D' variable and ('Time', 'Label', 'Coord') dims.
    """
    def _extract_4_corners(c):
        if isinstance(c, dict):
            def _get_val(d, keys):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
                return None

            ur = _get_val(c, ["ur", "up right", "outer_ur", "inner_ur"])
            ul = _get_val(c, ["ul", "up left", "outer_ul", "inner_ul"])
            dl = _get_val(c, ["dl", "down left", "outer_dl", "inner_dl"])
            dr = _get_val(c, ["dr", "down right", "outer_dr", "inner_dr"])
            pts = [ur, ul, dl, dr]
            assert all(p is not None for p in pts), f"Missing corner in {list(c.keys())}"
            return np.array(pts)
        pts = np.asarray(c)
        assert pts.shape[-2:] == (4, 3), f"Expected shape (*, 4, 3), got {pts.shape}"
        return pts

    outer_arr = _extract_4_corners(outer_corners)
    if outer_arr.ndim == 2:
        outer_arr = outer_arr[None, :, :]

    labels = ["outer_ur", "outer_ul", "outer_dl", "outer_dr"]
    data_list = [outer_arr]

    if inner_corners is not None:
        inner_arr = _extract_4_corners(inner_corners)
        if inner_arr.ndim == 2:
            inner_arr = inner_arr[None, :, :]
        labels.extend(["inner_ur", "inner_ul", "inner_dl", "inner_dr"])
        data_list.append(inner_arr)

    combined_data = np.concatenate(data_list, axis=1)

    all_attrs = {"angle": float(angle), "data_type": "tablet"}
    if attrs:
        all_attrs.update(attrs)

    return make_labeled_ds(
        data=combined_data,
        labels=labels,
        times=times,
        coords=coords,
        attrs=all_attrs,
    )

def tablet_ds_to_line_ds_list(tablet_ds: xr.Dataset,
                              outer_color: str = "black",
                              inner_color: str = "blue",
                              width: float = 2.0) -> list[xr.Dataset]:
    """
    Convert a tablet xr.Dataset into a list of line datasets (make_line_ds) for outer and inner boundaries.

    :param tablet_ds: Tablet xr.Dataset containing outer and optional inner corner labels.
    :param outer_color: Line color for outer tablet boundary.
    :param inner_color: Line color for inner drawing area boundary.
    :param width: Line width.
    :return: List of line xarray datasets ready for Plotter3D(line_ds_list=...).
    """
    coord_order = list(tablet_ds.Coord.to_numpy())
    outer_loop = ["outer_ur", "outer_ul", "outer_dl", "outer_dr", "outer_ur"]
    inner_loop = ["inner_ur", "inner_ul", "inner_dl", "inner_dr", "inner_ur"]

    outer_pts = tablet_ds["3D"].sel(Label=outer_loop).to_numpy()
    if tablet_ds.sizes.get("Time", 1) == 1:
        outer_pts = outer_pts[0]

    lines = [
        make_line_ds(
            points=outer_pts,
            coord_order=coord_order,
            name="Tablet_outer",
            color=outer_color,
            width=width,
        )
    ]

    has_inner = all(k in tablet_ds.Label.values for k in ["inner_ur", "inner_ul", "inner_dl", "inner_dr"])
    if has_inner:
        inner_pts = tablet_ds["3D"].sel(Label=inner_loop).to_numpy()
        if tablet_ds.sizes.get("Time", 1) == 1:
            inner_pts = inner_pts[0]
        lines.append(
            make_line_ds(
                points=inner_pts,
                coord_order=coord_order,
                name="Tablet_inner",
                color=inner_color,
                width=width,
            )
        )

    return lines

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example 1: Generic 3D Dataset (Financial Stock Prices)
    # ---------------------------------------------------------
    companies = ["AAPL", "GOOGL", "MSFT"]
    dates = ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"]
    price_types = ["Open", "High", "Low", "Close"]
    
    # Shape: (3 companies, 4 dates, 4 price metrics)
    np.random.seed(42)
    stock_prices = np.random.uniform(150.0, 300.0, size=(len(companies), len(dates), len(price_types)))
    
    stock_ds = build_3d_dataset(
        var_name="Stock Prices",
        data=stock_prices,
        dims=("Company", "Dates", "Prices"),
        coords={
            "Company": companies,
            "Dates": dates,
            "Prices": price_types,
        },
    )
    
    print("=== 1. Stock Dataset ===")
    print(stock_ds)
    print()
    
    
    # ---------------------------------------------------------
    # Example 2: 3D Point Dataset (Marker Tracking)
    # ---------------------------------------------------------
    n_t = 10
    n_marker = 3
    n_coord = 3
    
    # Shape: (10 frames, 3 markers, 3 coords)
    dummy_marker_pos = np.random.random((n_t, n_marker, n_coord))
    marker_labels = ["Wrist", "Elbow", "Shoulder"]
    timestamps = np.arange(n_t) * 0.02  # Recorded at 50 Hz
    
    point_ds = make_point_ds(
        positions=dummy_marker_pos,
        times=timestamps,
        coord_order="XYZ",
        labels=marker_labels,
    )
    
    print("=== 2. Point Dataset ===")
    print(point_ds)
    print()
    
    
    # ---------------------------------------------------------
    # Example 3: 3D Mesh Dataset (Single-frame Geometry)
    # ---------------------------------------------------------
    # Single-frame triangular pyramid: 4 vertices in 3D
    pyramid_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])
    
    pyramid_faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ])
    
    mesh_ds = make_mesh_ds(
        vertices=pyramid_vertices,
        faces=pyramid_faces,
        coord_order="XYZ",
        name="Pyramid",
        color="steelblue",
        opacity=0.7,
    )
    
    print("=== 3. Mesh Dataset ===")
    print(mesh_ds)
    print()
    
    
    # ---------------------------------------------------------
    # Example 4: ACS Timeseries Dataset (Multi-frame Pelvis Markers)
    # ---------------------------------------------------------
    n_frames = 5
    pelvis_landmarks = ["ASIS_R", "ASIS_L", "PSIS"]
    dummy_pelvis_data = np.random.uniform(-100.0, 100.0, size=(n_frames, len(pelvis_landmarks), 3))
    
    acs_ds = make_ACS_timeseries(
        data=dummy_pelvis_data,
        labels=pelvis_landmarks,
        coord_system="RAS",
        times=np.arange(n_frames),
    )
    
    print("=== 4. ACS Timeseries Dataset ===")
    print(acs_ds)

    
    # ---------------------------------------------------------
    # Example 5: 3D Line Dataset (Static & Dynamic Lines)
    # ---------------------------------------------------------
    # Static boundary/tablet line
    table_corners = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    static_line_ds = make_line_ds(
        points=table_corners,
        coord_order="XYZ",
        name="TableBoundary",
        color="crimson",
        width=3.0,
    )
    print("=== 5. Static Line Dataset ===")
    print(static_line_ds)
    print()

    # Dynamic time-series muscle / trajectory line
    n_frames = 5
    n_points_per_line = 6
    dynamic_line_data = np.random.uniform(-10.0, 10.0, size=(n_frames, n_points_per_line, 3))
    dynamic_line_ds = make_line_ds(
        points=dynamic_line_data,
        coord_order="XYZ",
        times=np.arange(n_frames),
        name="MusclePath",
        color="#E74C3C",
        width=4.0,
    )
    print("=== 6. Dynamic Line Dataset ===")
    print(dynamic_line_ds)
