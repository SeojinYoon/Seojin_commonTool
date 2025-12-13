
# Common Libraries
import os
import sys
import json
import numpy as np
import nibabel as nb
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from nilearn.plotting import plot_surf_roi
from SUITPy.flatmap import _map_color
import plotly.graph_objects as go
from IPython.display import HTML

# Custom Libraries
from surface_data import surf_paths, map_2d_to3d
from surface_roi import show_sulcus
from surface_util import mean_datas

sys.path.append("/home/seojin")
import surfAnalysisPy as surf # Dierdrichsen lab's library

sys.path.append("/home/seojin/Seojin_commonTool/Module")
from sj_matplotlib import make_colorbar
from brain_coord import image2referenceCoord

# Functions
def draw_surf_roi(roi_value_array, roi_info, surf_hemisphere, resolution = 32, alpha = 0.3):
    """
    Draw ROI on surface map

    :param roi_value_array(np.array - shape: #vertex): roi value array
    :param roi_info(dictionary -k: roi_name, -v: location(xy)): roi information dictionary
    :param surf_hemisphere(string): orientation of hemisphere ex) "L", "R"
    """
    ax = surf.plot.plotmap(data = roi_value_array, 
                           surf = f"fs{resolution}k_{surf_hemisphere}",
                           threshold = 0.01,
                           alpha = alpha)
    for i, roi_name in enumerate(roi_info):
        loc = roi_info[roi_name]
        ax.text(x = loc[0], y = loc[1], s = roi_name)
    return (ax.get_figure(), ax) 

def draw_surf_selectedROI(surf_roi_labels, roi_name, surf_hemisphere, resolution = 32, alpha = 0.3):
    """
    Draw surface roi

    :param surf_roi_labels(np.array - shape: #vertex): roi label array
    :param roi_name(string): roi name
    :param surf_hemisphere(string): orientation of hemisphere ex) "L", "R"
    """
    roi_value_array = np.where(surf_roi_labels == roi_name, 1, 0)
    ax = surf.plot.plotmap(data = roi_value_array, 
                           surf = f"fs{resolution}k_{surf_hemisphere}",
                           threshold = 0.01,
                           alpha = alpha)
    return (ax.get_figure(), ax) 

def show_surf_withGrid(surf_vis_ax, x_count = 30, y_count = 30):
    """
    Show surface with grid

    :param surf_vis_ax(axis)
    :param x_count: #count for dividing x
    :param y_count: #count for dividing y

    return figure
    """
    copy_ax = copy(surf_vis_ax)
    
    copy_ax.grid(True)
    copy_ax.axis("on")
    x_min, x_max = int(copy_ax.get_xlim()[0]), int(copy_ax.get_xlim()[1])
    y_min, y_max = int(copy_ax.get_ylim()[0]), int(copy_ax.get_ylim()[1])
    
    x_interval = (x_max - x_min) / x_count
    y_interval = (y_max - y_min) / y_count
    copy_ax.set_xticks(np.arange(x_min, x_max, x_interval).astype(int))
    copy_ax.set_xticklabels(np.arange(x_min, x_max, x_interval).astype(int), rotation = 90)
    
    copy_ax.set_yticks(np.arange(y_min, y_max, y_interval).astype(int))
    copy_ax.set_yticklabels(np.arange(y_min, y_max, y_interval).astype(int), rotation = 0)
    
    return copy_ax

def show_both_hemi_sampling_coverage(l_sampling_coverage: np.array, 
                                     r_sampling_coverage: np.array,
                                     save_dir_path: str,
                                     surf_resolution: int = 32,
                                     left_bounding_box: dict = None,
                                     right_bounding_box: dict = None,
                                     dpi: int = 300,
                                     is_sulcus_label: bool = False,
                                     sulcus_dummy_name: str = "sulcus"):
    """
    Show sampling coverage on both hemispheres

    :param l_sampling_coverage(shape: (#sampling, #vertex)): coverage per sampling for left hemi
    :param r_sampling_coverage(shape: (#sampling, #vertex)): coverage per sampling for right hemi
    :param save_dir_path: directory path for saving images
    :param surf_resolution: surface resolution
    :param left_bounding_box: data for drawing bounding box of left hemi
    :param right_bounding_box: data for drawing bounding box of right hemi
    :param dpi: dpi for saving image
    :param is_sulcus_label: flag for representing sulcus label
    :param sulcus_dummy_name: sulcus dummy file name ex) "sulcus", "sulcus_sensorimotor"
    """
    # Left
    plt.clf()
    l_sampling_coverages_sum = np.array([np.where(e != 0, i/10, 0) for i, e in enumerate(l_sampling_coverage)]).T
    l_sampling_coverages_sum = np.sum(l_sampling_coverages_sum, axis = 1)
    l_coverage_ax = surf.plot.plotmap(data = l_sampling_coverages_sum, 
                                      surf = f"fs{surf_resolution}k_L", 
                                      colorbar = False, 
                                      threshold = 0.001,
                                      alpha = 0.5)
    show_sulcus(surf_ax = l_coverage_ax, 
                hemisphere = "L", 
                isLabel = is_sulcus_label,
                sulcus_dummy_name = sulcus_dummy_name)

    if left_bounding_box is not None:
        rect = Rectangle(xy = left_bounding_box["left_bottom"], 
                         width = left_bounding_box["width"], 
                         height = left_bounding_box["height"], 
                         linewidth = 1, 
                         edgecolor = "r",
                         facecolor = "none")
        l_coverage_ax.add_patch(rect)
    
    l_surf_path = os.path.join(save_dir_path, f"L_hemi_coverage.png")
    l_coverage_ax.get_figure().savefig(l_surf_path, dpi = dpi, transparent = True)
    print(f"save: {l_surf_path}")

    # Right
    plt.clf()
    r_sampling_coverages_sum = np.array([np.where(e != 0, i/10, 0) for i, e in enumerate(r_sampling_coverage)]).T
    r_sampling_coverages_sum = np.sum(r_sampling_coverages_sum, axis = 1)
    r_coverage_ax = surf.plot.plotmap(data = r_sampling_coverages_sum, 
                                      surf = f"fs{surf_resolution}k_R",
                                      colorbar = False, 
                                      threshold = 0.001,
                                      alpha = 0.5)
    show_sulcus(surf_ax = r_coverage_ax, 
                hemisphere = "R",
                isLabel = is_sulcus_label,
                sulcus_dummy_name = sulcus_dummy_name)

    if right_bounding_box is not None:
        rect = Rectangle(xy = right_bounding_box["left_bottom"], 
                         width = right_bounding_box["width"], 
                         height = right_bounding_box["height"], 
                         linewidth = 1, 
                         edgecolor = "r",
                         facecolor = "none")
        r_coverage_ax.add_patch(rect)
        
    r_surf_path = os.path.join(save_dir_path, f"R_hemi_coverage.png")
    r_coverage_ax.get_figure().savefig(r_surf_path, dpi = dpi, transparent = True, bbox_inches = "tight")
    print(f"save: {r_surf_path}")

    # Both
    plt.clf()
    both_surf_img_path = os.path.join(save_dir_path, f"both_hemi_coverage")
    show_both_hemi_images(l_surf_img_path = l_surf_path, 
                          r_surf_img_path = r_surf_path, 
                          both_surf_img_path = both_surf_img_path)

def show_both_hemi_images(l_surf_img_path, 
                          r_surf_img_path, 
                          both_surf_img_path,
                          colorbar_path = None,
                          zoom = 0.2,
                          dpi = 300):
    """
    Show both surf hemi images

    :param l_surf_img_path(string): left hemisphere image path 
    :param r_surf_img_path(string): right hemisphere image path
    :param both_surf_img_path(string): save image path

    return fig, axis
    """
    fig, ax = plt.subplots()
    
    # Left    
    img = mpimg.imread(l_surf_img_path)
    imagebox = OffsetImage(img, zoom = zoom)  # Adjust zoom for size
    ab = AnnotationBbox(imagebox, (0, 0.5), frameon=False)
    ax.add_artist(ab)

    # Right
    img = mpimg.imread(r_surf_img_path)
    imagebox = OffsetImage(img, zoom = zoom)  # Adjust zoom for size
    ab = AnnotationBbox(imagebox, (0.9, 0.5), frameon=False)
    ax.add_artist(ab)

    # Colorbar
    if colorbar_path != None:
        colorbar_img = mpimg.imread(colorbar_path)
        colorbar_box = OffsetImage(colorbar_img, zoom = zoom)  # Adjust zoom for size

        ab = AnnotationBbox(colorbar_box, (0.5, 1.0), frameon=False)
        ax.add_artist(ab)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(both_surf_img_path, dpi = dpi, transparent = True, bbox_inches = "tight")
    print(f"save: {both_surf_img_path}.png")
    
    return fig, ax

def show_both_hemi_stats(l_stat, 
                         r_stat,
                         threshold,
                         cscale,
                         save_dir_path,
                         n_inner_ticks = 3,
                         surf_resolution = 32,
                         left_bounding_box = None,
                         right_bounding_box = None,
                         is_focusing_bounding_box = False,
                         zoom = 0.2,
                         dpi = 300,
                         is_sulcus_label = False,
                         sulcus_dummy_name: str = "sulcus",
                         colorbar_decimal = 4,
                         is_show_colorbar = True):
    """
    Show stats on both surf hemispheres

    :param l_stat(np.array - #vertex): left hemisphere stat
    :param r_stat(np.array - #vertex): right hemisphere stat
    :param threshold(int): threshold
    :param cscale(tuple - (vmin, vmax)): color bar scale
    :param n_inner_ticks(int): the number of colorbar ticks without min and max value
    :param save_dir_path(string): directory path for saving images
    :param surf_resolution(int): surface resolution
    :param left_bounding_box(dictionary): bounding box for left hemi
    :param right_bounding_box(dictionary): bounding box for right hemi
    :param zoom(float): zoom to load image
    :param colorbar_decimal(int): decimal value of colorbar
    :param dpi(int): dpi for saving image
    :param is_sulcus_label(boolean): is showing sulcus label on the flatmap
    :param sulcus_dummy_name: sulcus dummy file name ex) "sulcus", "sulcus_sensorimotor"
    
    return fig, axis
    """
    
    rect_linewidth = 1
    rect_edgecolor = "r"
    
    # Left
    plt.clf()
    l_ax = surf.plot.plotmap(data = l_stat, 
                           surf = f"fs{surf_resolution}k_L", 
                           colorbar = False, 
                           threshold = threshold,
                           cscale = cscale)
    show_sulcus(surf_ax = l_ax, 
                hemisphere = "L",
                isLabel = is_sulcus_label,
                sulcus_dummy_name = sulcus_dummy_name)
    
    if is_focusing_bounding_box:
        if left_bounding_box is not None:
            min_x, min_y = left_bounding_box["left_bottom"]
            max_x, max_y = min_x + left_bounding_box["width"], min_y + left_bounding_box["height"]
            l_ax.set_xlim(min_x, max_x)
            l_ax.set_ylim(min_y, max_y)
    else:
        if left_bounding_box is not None:
            l_rect = Rectangle(xy = left_bounding_box["left_bottom"], 
                               width = left_bounding_box["width"], 
                               height = left_bounding_box["height"], 
                               linewidth = rect_linewidth, 
                               edgecolor = rect_edgecolor,
                               facecolor = "none")
            l_ax.add_patch(l_rect)
        
    l_surf_img_path = os.path.join(save_dir_path, f"L_hemi_stat.png")
    l_ax.get_figure().savefig(l_surf_img_path, dpi = dpi, transparent = True, bbox_inches = "tight")
    print(f"save: {l_surf_img_path}")
    
    # Right
    plt.clf()
    r_ax = surf.plot.plotmap(data = r_stat, 
                           surf = f"fs{surf_resolution}k_R", 
                           colorbar = False, 
                           threshold = threshold,
                           cscale = cscale)
    show_sulcus(surf_ax = r_ax, 
                hemisphere = "R",
                isLabel = is_sulcus_label,
                sulcus_dummy_name = sulcus_dummy_name)

    if is_focusing_bounding_box:
        if right_bounding_box is not None:
            min_x, min_y = right_bounding_box["left_bottom"]
            max_x, max_y = min_x + right_bounding_box["width"], min_y + right_bounding_box["height"]
            r_ax.set_xlim(min_x, max_x)
            r_ax.set_ylim(min_y, max_y)
    else:
        if right_bounding_box is not None:
            r_rect = Rectangle(xy = right_bounding_box["left_bottom"], 
                               width = right_bounding_box["width"], 
                               height = right_bounding_box["height"], 
                               linewidth = rect_linewidth, 
                               edgecolor = rect_edgecolor,
                               facecolor = "none")
            r_ax.add_patch(r_rect)
        
    r_surf_img_path = os.path.join(save_dir_path, f"R_hemi_stat.png")
    r_ax.get_figure().savefig(r_surf_img_path, dpi = dpi, transparent = True, bbox_inches = "tight")
    print(f"save: {r_surf_img_path}")

    # Colorbar
    if is_show_colorbar:
        plt.clf()
        colorbar_path = os.path.join(save_dir_path, "colorbar.png")
        
        figsize = (10, 1)
        fig, axis, ticks = make_colorbar(cscale[0], 
                                         cscale[1], 
                                         figsize = figsize, 
                                         n_inner_ticks = n_inner_ticks, 
                                         orientation = "horizontal",
                                         tick_precision = colorbar_decimal)
        fig.savefig(colorbar_path, dpi = dpi, transparent = True, bbox_inches = "tight")
        print(f"save: {colorbar_path}")
    
    # Both
    plt.clf()
    both_surf_img_path = os.path.join(save_dir_path, f"both_hemi_stat")
    fig, ax = show_both_hemi_images(l_surf_img_path = l_surf_img_path, 
                                    r_surf_img_path = r_surf_img_path, 
                                    both_surf_img_path = both_surf_img_path,
                                    colorbar_path = colorbar_path if is_show_colorbar else None,
                                    zoom = zoom)
    return fig, ax

def plot_virtualStrip_on3D_surf(virtual_stip_mask, 
                                save_dir_path, 
                                vmax,
                                hemisphere = "L",
                                view = "lateral",
                                cmap = "Purples",
                                darkness = 1,
                                dpi = 300):
    """
    Plot a virtual strip on a 3D brain surface and save the result as a PNG image.

    :param virtual_stip_mask(numpy array):  Binary mask indicating vertices that form the virtual strip.
    :param save_dir_path(string):  Path to the directory where the output image will be saved.
    :param vmax(float):  Maximum value for color mapping.
    :param hemisphere(string):  Hemisphere to plot ("L" for left, "R" for right). Default is "L".
    :param view(string):  View angle for plotting the brain surface (e.g., "lateral", "medial"). Default is "lateral".
    :param cmap(string):  Colormap used to visualize the strip on the surface. Default is "Purples".

    :return: The generated figure.
    """
    
    path_info = surf_paths(hemisphere)
    template_path = surf_paths(hemisphere)[f"{hemisphere}_template_surface_path"]
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]

    rect_vertexes = vertex_locs[np.where(virtual_stip_mask == 1, True, False)]
    min_rect_x, max_rect_x = np.min(rect_vertexes[:, 0]), np.max(rect_vertexes[:, 0])
    min_rect_y, max_rect_y = np.min(rect_vertexes[:, 1]), np.max(rect_vertexes[:, 1])
    within_x = (vertex_locs[:, 0] >= min_rect_x) & (vertex_locs[:, 0] <= max_rect_x)
    within_y = (vertex_locs[:, 1] >= min_rect_y) & (vertex_locs[:, 1] <= max_rect_y)
    is_within_rectangle = np.logical_and(within_x, within_y)

    fig = plot_surf_roi(surf_mesh = path_info[f"{hemisphere}_inflated_brain_path"],
                        roi_map = np.where(virtual_stip_mask, 0.7, np.where(is_within_rectangle, 1, 0)),
                        bg_map = path_info[f"{hemisphere}_shape_gii_path"],
                        hemi = "left" if hemisphere == "L" else "right",
                        cmap = cmap,
                        alpha = 2, 
                        vmax = vmax,
                        bg_on_data = True,
                        darkness = darkness,
                        view = view,
    )
    path = os.path.join(save_dir_path, f"{hemisphere}_virtual_strip.png")
    fig.savefig(path, dpi = dpi, transparent = True, bbox_inches = "tight")
    print(f"save: {path}")
    
    return fig

def show_interactive_brain(data_info: dict,
                           reference_data_path: str,
                           threshold: float,
                           cscale: tuple,
                           cmap: str = "jet",
                           is_do_smoothing = False,
                           underscale: list = [-1.5, 1],
                           alpha: float = 1.0,
                           depths: list = [0,0.2,0.4,0.6,0.8,1.0],
                           query_port: int = 5000,
                           surf_dir_path: str = "/home/seojin/Seojin_commonTool/Module/Brain_Surface/Datas",
                           color_bar_info = {
                               "n_inner_ticks" : 3,
                               "tick_precision" : 4,
                           }):
    """
    Show interactive brain

    References
    - https://github.com/DiedrichsenLab/surfAnalysisPy
    - https://www.humanconnectome.org/software/workbench-command/-volume-to-surface-mapping

    :param data_info: input data info
        -k volume_data_paths(list): nifti file paths
        -k l_surf_data(np.array - shape: (#vertices))
        -k r_surf_data(np.array - shape: (#vertices))
    :param reference_data_path: reference nifti brain
    :param threshold: threshold to cut overlay 
    :param cmap: overlay color map
    :param cscale: overlay color scale
    :param underscale: underlay  color scale
    :param alpha: alpha value to overlay
    :param depths: depths of points along line at which to map (0=white/gray, 1=pial)
    :param query_port: server port for calling whereami
    :param surf_dir_path: directory containing surface-related data
    """
    # Paths
    l_path_info = surf_paths(surf_hemisphere = "L", surf_dir_path = surf_dir_path)
    r_path_info = surf_paths(surf_hemisphere = "R", surf_dir_path = surf_dir_path)
    l_flat_surf_path = l_path_info["L_template_surface_path"]
    l_underlay_path = l_path_info["L_shape_gii_path"]
    l_pial_surf_path = l_path_info["L_pial_surf_path"]
    l_white_surf_path = l_path_info["L_white_surf_path"]
    l_sulcus_path = l_path_info["L_sulcus_path"]
    
    r_flat_surf_path = r_path_info["R_template_surface_path"]
    r_underlay_path = r_path_info["R_shape_gii_path"]
    r_pial_surf_path = r_path_info["R_pial_surf_path"]
    r_white_surf_path = r_path_info["R_white_surf_path"]
    r_sulcus_path = r_path_info["R_sulcus_path"]

    # Load basic coordinate informations
    l_flat_surf = nb.load(l_flat_surf_path)
    r_flat_surf = nb.load(r_flat_surf_path)

    affine = nb.load(reference_data_path).affine

    # Load data
    volume_data_paths = data_info.get("volume_data_paths", [])
    l_surf_data = data_info.get("l_surf_data", None)
    r_surf_data = data_info.get("r_surf_data", None)
    if l_surf_data is None:
        l_surf_data = mean_datas(volume_data_paths, hemisphere = "L", is_do_smoothing = is_do_smoothing)
    if r_surf_data is None:
        r_surf_data = mean_datas(volume_data_paths, hemisphere = "R", is_do_smoothing = is_do_smoothing)

    # Map 2D coord to 3D MNI coord
    l_img_coords = map_2d_to3d(reference_data_path,
                               pial_surf_path = l_pial_surf_path,
                               white_surf_path = l_white_surf_path,
                               depths = depths)
    r_img_coords = map_2d_to3d(reference_data_path,
                               pial_surf_path = r_pial_surf_path,
                               white_surf_path = r_white_surf_path,
                               depths = depths)
    l_aggregated_img_coords = np.mean(l_img_coords, axis = 0).astype(np.int32)
    r_aggregated_img_coords = np.mean(r_img_coords, axis = 0).astype(np.int32)

    l_mni_coords = np.array([image2referenceCoord(ijk, affine) for ijk in l_aggregated_img_coords])
    r_mni_coords = np.array([image2referenceCoord(ijk, affine) for ijk in r_aggregated_img_coords])

    # Build vertices
    l_vertices = l_flat_surf.darrays[0].data
    r_vertices = r_flat_surf.darrays[0].data
    
    x_addition_4_rHemi = 550
    y_addition_4_rHemi = 20
    r_vertices[:, 0] = r_vertices[:, 0] + x_addition_4_rHemi # Adjust x position of right hemisphere for plotting both hemi
    r_vertices[:, 1] = r_vertices[:, 1] + y_addition_4_rHemi # Adjust x position of right hemisphere for plotting both hemi

    # Build faces
    l_faces = l_flat_surf.darrays[1].data
    r_faces = r_flat_surf.darrays[1].data


    # Build overlay colors
    l_overlay_color, l_cmap, l_cscale = _map_color(data = l_surf_data,
                                                   faces = l_faces, 
                                                   cscale = cscale,
                                                   cmap = cmap, 
                                                   threshold = threshold)
    
    r_overlay_color, r_cmap, r_cscale = _map_color(data = r_surf_data,
                                                   faces = r_faces, 
                                                   cscale = cscale,
                                                   cmap = cmap, 
                                                   threshold = threshold)

    # Build underlay colors
    l_flat_underlay = nb.load(l_underlay_path).darrays[0].data
    r_flat_underlay = nb.load(r_underlay_path).darrays[0].data
    
    l_underlay_color, _, _ = _map_color(data = l_flat_underlay,
                                        faces = l_faces, 
                                        cscale = underscale,
                                        cmap = "gray")
    
    r_underlay_color, _, _ = _map_color(data = r_flat_underlay,
                                        faces = r_faces, 
                                        cscale = underscale,
                                        cmap = "gray")

    # Combine overlay color and underlay color
    l_color = l_underlay_color * (1-alpha) + l_overlay_color * alpha
    l_isnan_i = np.isnan(l_color.sum(axis=1))
    l_color[l_isnan_i,:] = l_underlay_color[l_isnan_i,:]
    l_color[l_isnan_i,3] = 1.0
    
    r_color = r_underlay_color * (1-alpha) + r_overlay_color * alpha
    r_isnan_i = np.isnan(r_color.sum(axis=1))
    r_color[r_isnan_i,:] = r_underlay_color[r_isnan_i,:]
    r_color[r_isnan_i,3] = 1.0

    # load marked sulcus info
    with open(l_sulcus_path, "r") as file:
        l_sulcus_info = json.load(file)
    
    with open(r_sulcus_path, "r") as file:
        r_sulcus_info = json.load(file)
    
    """
    Rendering
    """
    traces = []
    
    # render both sided hemishpere
    traces.append(go.Mesh3d(x = l_vertices[:, 0], y = l_vertices[:, 1], z = l_vertices[:, 2],
                            i = l_faces[:, 0], j = l_faces[:, 1], k = l_faces[:, 2],
                            facecolor = l_color,
                            vertexcolor = None,
                            lightposition = dict(x = 0, y = 0, z = 2.5),
                            hoverinfo = "skip"))
    traces.append(go.Mesh3d(x = r_vertices[:, 0], y = r_vertices[:, 1], z = r_vertices[:, 2],
                            i = r_faces[:, 0], j = r_faces[:, 1], k = r_faces[:, 2],
                            facecolor = r_color,
                            vertexcolor = None,
                            lightposition = dict(x = 0, y = 0, z = 2.5),
                            hoverinfo = "skip"))
    
    # load both sided vertices
    traces.append(go.Scatter3d(x = l_vertices[:, 0],
                               y = l_vertices[:, 1],
                               z = l_vertices[:, 2],
                               mode = "markers",
                               marker = dict(size = 1, color = "black", opacity = 0.0),
                               hoverinfo = "text",
                               text = [f"L({x:.2f}, {y:.2f})" for x, y, z in l_vertices],
                               name = "L_nodes"))
    
    traces.append(go.Scatter3d(x = r_vertices[:, 0],
                               y = r_vertices[:, 1],
                               z = r_vertices[:, 2],
                               mode = "markers",
                               marker = dict(size = 1, color = "black", opacity = 0.0),
                               hoverinfo = "text",
                               text = [f"R({(x - x_addition_4_rHemi):.2f}, {(y - y_addition_4_rHemi):.2f})" for x, y, z in r_vertices],
                               name = "R_nodes"))

    # Sulcus
    for sulcus_name in l_sulcus_info:
        pts = np.array(l_sulcus_info[sulcus_name])
        n_point = len(pts)
        z_lift = 1.0
        
        traces.append(go.Scatter3d(
            x = pts[:,0],
            y = pts[:,1],
            z = np.zeros(n_point) + z_lift,
            mode = "lines",
            line=dict(color = "white", width = 3, dash = "dash"),
            hoverinfo = "skip",
        ))
    
    for sulcus_name in r_sulcus_info:
        pts = np.array(r_sulcus_info[sulcus_name])
        n_point = len(pts)
    
        traces.append(go.Scatter3d(
            x = pts[:,0] + x_addition_4_rHemi,
            y = pts[:,1] + y_addition_4_rHemi,
            z = np.zeros(n_point) + z_lift,
            mode = "lines",
            line = dict(color = "white", width = 3, dash = "dash"),
            hoverinfo = "skip",
        ))

    # Color bar
    tick_vals = np.linspace(cscale[0], cscale[1], n_inner_ticks + 2)
    tick_text = [f"{v:.{tick_precision}f}" for v in tick_vals]

    dummy_colorbar_trace = go.Scatter3d(
        x = [0], y = [0], z = [0],
        mode = "markers",
        marker = dict(
            size = 0,
            opacity = 0,
            cmin = cscale[0],
            cmax = cscale[1],
            colorscale = cmap.capitalize(), # cmap 변수가 'jet'이라면 Plotly에서는 'Jet' (대소문자 주의)
            showscale = True,
            colorbar = dict(
                orientation = "h",
                thickness = 20,
                len = 0.6,
                x = 0.5,
                y = 0.8,
                tickmode = "array",
                tickvals = tick_vals,
                ticktext = tick_text,
                tickfont = dict(size = 12, color = "black"),
                outlinewidth = 0
            )
        ),
        hoverinfo = "skip",
        name = "Colorbar_Dummy",
    )
    traces.append(dummy_colorbar_trace)
    
    # Camera
    camera = dict(up = dict(x = 0, y = 1, z = 0),
              center = dict(x = 0, y = 0, z = 0),
              eye = dict(x = 0, y = 0, z = 11.5))
    
    xaxis_dict= dict(visible = False, 
                     showbackground = False,
                     showline = False,
                     showgrid = False,
                     showspikes = False,
                     showticklabels = False,
                     title = None)
    yaxis_dict = xaxis_dict.copy()
    zaxis_dict = xaxis_dict.copy()
    scene = dict(xaxis = xaxis_dict,
                 yaxis = yaxis_dict,
                 zaxis = zaxis_dict,
                 aspectmode = 'data')
    
    # Make figure
    fig = go.Figure(data = traces)
    fig.update_layout(scene_camera = camera,
                      dragmode = False,
                      margin = dict(r = 0, l = 0, b = 0, t = 0),
                      scene = scene,
                      width = 700,
                      height = 500,
                      paper_bgcolor = "#ffffff",
                      showlegend = False)
    
    # Make javascript codes for interaction process
    div_id = "brain-surface-plot"
    
    l_mni_coords_js = json.dumps(l_mni_coords.tolist())
    r_mni_coords_js = json.dumps(r_mni_coords.tolist())

    post_script = f"""
    var myPlot = document.getElementById('{div_id}');
    var infoDiv = document.getElementById('click-info');
    
    myPlot.on('plotly_click', function(data) {{
        var pt = data.points[0];
        var l_mni_coords = {l_mni_coords_js};
        var r_mni_coords = {r_mni_coords_js};
    
        const serverPort = {query_port};
        const serverURL = `http://${{window.location.hostname}}:${{serverPort}}/click`;
        
        // // Skip events that are not scatter3d; mesh clicks should be ignored
        if (pt.data.type !== 'scatter3d') {{
            return;
        }}

        var hemi = (pt.data.name === 'L_nodes') ? 'L' : 
                   (pt.data.name === 'R_nodes') ? 'R' : 'unknown';
    
        var adjX = pt.x;
        var adjY = pt.y;
    
        // Restore the right hemisphere to its original coordinates
        if (hemi === 'R') {{
            adjX = pt.x - 550;   // x_addition_4_rHemi
            adjY = pt.y - 20;    // y_addition_4_rHemi
            var mni_coord = r_mni_coords[pt.pointNumber];
        }} else {{
            var mni_coord = l_mni_coords[pt.pointNumber];
        }}
    
        // 🔥 Send a POST request to the Python server here
        fetch(serverURL, {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
                hemi: hemi,
                idx: pt.pointNumber,
                mni: mni_coord
            }})
        }})
        .then(response => response.json())
        .then(result => {{
            const atlas_data = result.atlas_data;
            console.log("Server response:", atlas_data);
    
            if (infoDiv) {{
                const lines = atlas_data.map(d =>
                    `${{d.atlas}} | ${{d.info}} | ${{d.name}}`
                );
    
                infoDiv.innerHTML =
                    '<b>Clicked vertex</b><br>' +
                    'Hemisphere: ' + hemi + '<br>' +
                    'Index: ' + pt.pointNumber + '<br>' +
                    'vertex (x, y): (' + adjX.toFixed(2) + ', ' + adjY.toFixed(2) + ')<br>' + 
                    'MNI (x,y,z): ' + mni_coord + '<br>' +
                    lines.join('<br>');
            }}
        }})
        .catch(error => {{
            console.error("Error calling Python server:", error);
        }});
    }});
    """

    # Convert fig into html
    fig_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        post_script=post_script
    )
    
    # Add layout - textbox
    html = f"""
    <div style="display:flex; align-items:flex-start;">
        <div>
            {fig_html}
        </div>
        <div id="click-info"
             style="
                margin-left:20px;
                min-width:220px;
                font-family:monospace;
                font-size:13px;
                border:1px solid #ccc;
                padding:8px;
                border-radius:4px;
                background-color:#f9f9f9;
                color:black;
             ">
            vertex information will represent here
        </div>
    </div>
    """
    return HTML(html)
    
# Examples
if __name__ == "__main__":
    hemisphere = "L"
    roi_values = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_roi_values.npy")
    with open(os.path.join(surface_mask_dir_path, f"{hemisphere}_roi_vertex_info.json"), 'rb') as f:
        loaded_info = json.load(f)
    draw_surf_roi(roi_values, loaded_info, "L")
    

    volume_data_dir_path = "/mnt/ext1/seojin/HR/exp_blueprint_0324v4/fMRI_data/output/Group/rsa/rdm/set/WholeSet/Searchlight/base_seq_move_rest_separetely/None/react_mean_dissim_withBaseline"
    volume_data_data_paths = sorted(glob(volume_data_dir_path + "/??.nii.gz"))
    show_interactive_brain(volume_data_data_paths, 0.001, cscale = (0.001, 0.005))
    