
# Common Libraries
import os
import sys
import numpy as np
import nibabel as nb
import nitools as nt
import warnings
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from scipy.spatial import KDTree
import matplotlib.pylab as plt
from copy import copy
from cv2 import minAreaRect, boxPoints, pointPolygonTest
import json
from collections import Counter
from matplotlib.patches import Rectangle
from nilearn.plotting import plot_surf_roi
from scipy.stats import sem
from scipy.stats import ttest_1samp

# Custom Libraries
sys.path.append("/home/seojin")
import surfAnalysisPy as surf # Dierdrichsen lab's library

sys.path.append("/home/seojin/Seojin_commonTool/Module")
from sj_matplotlib import make_colorbar, draw_ticks, draw_spine, draw_label
from sj_math import projection, round_down
from brain_coord import reference2imageCoord

# Functions
def surf_paths(surf_hemisphere, 
               surf_dir_path = "/mnt/sda2/Common_dir/Atlas/Surface", 
               surf_resolution = 32,
               sulcus_dummy_name = "sulcus",
               atlas = "Brodmann"):
    surf_dir_path = os.path.join(surf_dir_path, f"fs_LR_{surf_resolution}")
    
    # Template
    pial_surf_path = os.path.join(surf_dir_path, f"fs_LR.{surf_resolution}k.{surf_hemisphere}.pial.surf.gii")
    white_surf_path = os.path.join(surf_dir_path, f"fs_LR.{surf_resolution}k.{surf_hemisphere}.white.surf.gii")
    template_surface_path = os.path.join(surf_dir_path, f"fs_LR.{surf_resolution}k.{surf_hemisphere}.flat.surf.gii")
    inflated_brain_path = os.path.join(surf_dir_path, f"fs_LR.{surf_resolution}k.{surf_hemisphere}.inflated.surf.gii")
    shape_gii_path = os.path.join(surf_dir_path, f"fs_LR.32k.{surf_hemisphere}.shape.gii")
    
    # Sulcus
    sulcus_path = os.path.join(surf_dir_path, "borders", f"{surf_hemisphere}_{sulcus_dummy_name}.json")
    
    # ROI
    roi_label_path = os.path.join(surf_dir_path, atlas, f"{surf_hemisphere}_rois.npy")

    return {
        f"{surf_hemisphere}_pial_surf_path" : pial_surf_path,
        f"{surf_hemisphere}_white_surf_path" : white_surf_path,
        f"{surf_hemisphere}_template_surface_path" : template_surface_path,
        f"{surf_hemisphere}_inflated_brain_path" : inflated_brain_path,
        f"{surf_hemisphere}_shape_gii_path" : shape_gii_path,
        f"{surf_hemisphere}_sulcus_path" : sulcus_path,
        f"{surf_hemisphere}_roi_label_path" : roi_label_path,
    }
    
def surface_profile(template_surface_path, 
                    surface_data, 
                    from_point, 
                    to_point, 
                    width,
                    n_sampling = None):
    """
    Do profile analysis based on virtual strip axis

    :param template_surface_path(string): template gii file ex) '/home/seojin/single-finger-planning/data/surf/fs_LR.164k.L.flat.surf.gii'
    :param surface_data(string or np.array - shape: (#vertex, #data): data gii file path or data array ex) '/home/seojin/single-finger-planning/data/surf/group.psc.L.Planning.func.gii'
    :param from_point(list): location of start virtual strip - xy coord ex) [-43, 86]
    :param to_point(list): location of end virtual strip - xy coord ex) [87, 58]
    :param width(int): width of virtual strip ex) 20
    :param n_sampling(int): the number of sampling across virtual strip

    return 
        -k virtual_stip_mask(np.array - #vertex): mask
        -k sampling_datas(np.array - #sampling, #data): sampling datas based on virtual strip
        -k sampling_coverages(np.array - #sampling, #vertex): spatial coverage per sampling point
        -k sampling_center_coords(np.array - #sampling, #coord): sampling center coordinates
    """
    if n_sampling == None:
        n_sampling = abs(from_point[0] - to_point[0])
    
    # Load data metric file
    surface_gii = nb.load(template_surface_path)
    flat_coord = surface_gii.darrays[0].data
    
    if type(surface_data) == str:
        data_gii = nb.load(surface_data_path)
    
        # Check - all data has same vertex shape
        darrays = data_gii.darrays
        is_valid = np.all([darray.dims[0] == darrays[0].dims[0] for darray in darrays])
        assert is_valid, "Please check data shape"
    
        data_arrays = np.array([e.data for e in darrays]).T
    else:
        data_arrays = surface_data
        
    # Data information
    n_vertex = data_arrays.shape[0]
    n_data = data_arrays.shape[1]
    
    # Check - surface and data have same vertex
    assert flat_coord.shape[0] == n_vertex, "Data vertex must be matched with surface"
    
    # Extract vertices (x, y)
    vertex_2d = flat_coord[:, :2]
    
    # Move vertex origin
    points = vertex_2d - from_point
    
    # Set virtual vector(orientation of virtual strip)
    virtual_vec = to_point - from_point
    
    # Values for explaining vertex relative to virtual vector
    project = (np.dot(points, virtual_vec)) / np.dot(virtual_vec, virtual_vec)
    
    # Difference between vertex and projection vector
    residual = points - np.outer(project, virtual_vec)
    
    # Distance between vertex and virtual vector
    distance = np.sqrt(np.sum(residual**2, axis=1))

    ## Dummy for sampling result
    sampling_datas = np.zeros((n_sampling, n_data))
    virtual_stip_mask = np.zeros(n_vertex)
    sampling_center_coords = np.zeros((n_sampling, flat_coord.shape[1]))
    sampling_coverages = np.zeros((n_sampling, n_vertex))

    # Find points on the strip
    graduation_onVirtualVec = np.linspace(0, 1, n_sampling + 1)
    for i in range(n_sampling):
        # Filter only the vertices that are inside the virtual strip from all vertices
        start_grad = graduation_onVirtualVec[i]
        next_grad = graduation_onVirtualVec[i + 1]
    
        within_distance = distance < width
        upper_start = (project >= start_grad)
        lower_end = (project <= next_grad)
        no_origin = (np.sum(vertex_2d ** 2, axis=1) > 0)
        
        is_virtual_strip = within_distance & upper_start & lower_end & no_origin
        indx = np.where(is_virtual_strip)[0]

        sampling_coverages[i, indx] = 1
        
        # Perform cross-section
        sampling_datas[i, :] = np.nanmean(data_arrays[indx, :], axis=0) if len(indx) > 0 else 0
        virtual_stip_mask[indx] = 1
        sampling_center_coords[i, :] = np.nanmean(flat_coord[indx, :], axis=0) if len(indx) > 0 else 0

    result_info = {}
    result_info["sampling_datas"] = sampling_datas
    result_info["virtual_stip_mask"] = virtual_stip_mask
    result_info["sampling_center_coords"] = sampling_center_coords
    result_info["sampling_coverages"] = sampling_coverages
    return result_info

def vol_to_surf(volume_data_path, 
                pial_surf_path, 
                white_surf_path,
                ignoreZeros = False,
                depths = [0,0.2,0.4,0.6,0.8,1.0],
                stats = "nanmean"):
    """
    Adapted from https://github.com/DiedrichsenLab/surfAnalysisPy
    
    Maps volume data onto a surface, defined by white and pial surface.
    Function enables mapping of volume-based data onto the vertices of a
    surface. For each vertex, the function samples the volume along the line
    connecting the white and gray matter surfaces. The points along the line
    are specified in the variable 'depths'. default is to sample at 5
    locations between white an gray matter surface. Set 'depths' to 0 to
    sample only along the white matter surface, and to 0.5 to sample along
    the mid-gray surface.

    The averaging across the sampled points for each vertex is dictated by
    the variable 'stats'. For functional activation, use 'mean' or
    'nanmean'. For discrete label data, use 'mode'.

    If 'exclude_thres' is set to a value >0, the function will exclude voxels that
    touch the surface at multiple locations - i.e. voxels within a sulcus
    that touch both banks. Set this option, if you strongly want to prevent
    spill-over of activation across sulci. Not recommended for voxels sizes
    larger than 3mm, as it leads to exclusion of much data.

    For alternative functionality see wb_command volumne-to-surface-mapping
    https://www.humanconnectome.org/software/workbench-command/-volume-to-surface-mapping

    @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        volume_data_path (string): nifti image path
        whiteSurfGifti (string or nibabel.GiftiImage): White surface, filename or loaded gifti object
        pialSurfGifti (string or nibabel.GiftiImage): Pial surface, filename or loaded gifti object
    OPTIONAL:
        ignoreZeros (bool):
            Should zeros be ignored in mapping? DEFAULT:  False
        depths (array-like):
            Depths of points along line at which to map (0=white/gray, 1=pial).
            DEFAULT: [0.0,0.2,0.4,0.6,0.8,1.0]
        stats (str or lambda function):
            function that calculates the Statistics to be evaluated.
            lambda X: np.nanmean(X,axis=0) default and used for activation data
            lambda X: scipy.stats.mode(X,axis=0) used when discrete labels are sampled. The most frequent label is assigned.
    OUTPUT:
        mapped_data (numpy.array):
            A Data array for the mapped data
    """
    # Stack datas
    depths = np.array(depths)
    
    # Load datas
    volume_img = nb.load(volume_data_path)
    whiteSurfGiftiImage = nb.load(white_surf_path)
    pialSurfGiftiImage = nb.load(pial_surf_path)
    
    whiteSurf_vertices = whiteSurfGiftiImage.darrays[0].data
    pialSurf_vertices = pialSurfGiftiImage.darrays[0].data
    
    assert whiteSurf_vertices.shape[0] == pialSurf_vertices.shape[0], "White and pial surfaces should have same number of vertices"
    
    # Informations
    n_vertex = whiteSurf_vertices.shape[0]
    n_point = len(depths)
    
    # 2D vertex location -> 3D voxel index with considering depth of graymatter
    voxel_indices = np.zeros((n_point, n_vertex, 3), dtype=int)
    for i in range(n_point):
        coeff_whiteMatter = 1 - depths[i]
        coeff_grayMatter = depths[i]
    
        weight_sum_vertex_2d = coeff_whiteMatter * whiteSurf_vertices.T + coeff_grayMatter * pialSurf_vertices.T
        voxel_indices[i] = nt.coords_to_voxelidxs(weight_sum_vertex_2d, volume_img).T
    
    # Read the data and map it
    data_consideringGraymatterDepth = np.zeros((n_point, n_vertex))
    
    ## Load volume array
    volume_array = volume_img.get_fdata()
    if ignoreZeros == True:
        volume_array[volume_array==0] = np.nan
    
    ## volume data without outside
    for i in range(n_point):
        data_consideringGraymatterDepth[i,:] = volume_array[voxel_indices[i,:,0], voxel_indices[i,:,1], voxel_indices[i,:,2]]
        outside = (voxel_indices[i,:,:]<0).any(axis=1) # These are vertices outside the volume
        data_consideringGraymatterDepth[i, outside] = np.nan
    
    # Determine the right statistics - if function - call it
    if stats == "nanmean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mapped_data = np.nanmean(data_consideringGraymatterDepth,axis=0)
    elif callable(stats):
        mapped_data  = stats(data_consideringGraymatterDepth)
        
    return mapped_data

def draw_surf_roi(roi_value_array, roi_info, surf_hemisphere, resolution = 32, alpha = 0.3):
    """
    Draw surface roi

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

def load_surfData_fromVolume(volume_data_paths, hemisphere, depths = [0,0.2,0.4,0.6,0.8,1.0]):
    """
    Load surface data from volume data

    :param volume_data_paths(list - string): volume data path(.nii)
    :param hemisphere(string): "L" or "R"
    :param depths(list): Depths of points along line at which to map (0=white/gray, 1=pial). ex) [0.0,0.2,0.4,0.6,0.8,1.0]
    """
    surf_info = surf_paths(hemisphere)
    
    surface_datas = []
    for path in volume_data_paths:
        surface_data = vol_to_surf(volume_data_path = path,
                                   pial_surf_path = surf_info[f"{hemisphere}_pial_surf_path"],
                                   white_surf_path = surf_info[f"{hemisphere}_white_surf_path"],
                                   depths = depths)
        surface_datas.append(surface_data)
    surface_datas = np.array(surface_datas).T

    return surface_datas
    
def gaussian_weighted_smoothing(coords, values, sigma=1.0):
    """
    Apply Gaussian smoothing to scattered data without using a grid.
    
    Args:
    - coords: (N, 2) array of x, y coordinates.
    - values: (N,) array of corresponding values.
    - sigma: Standard deviation for Gaussian weighting.
    
    Returns:
    - smoothed_values: Smoothed values at each original coordinate.
    """
    tree = KDTree(coords)
    smoothed_values = np.zeros_like(values)
    for i, point in enumerate(coords):
        distances, indices = tree.query(point, k=50)  # Consider 50 nearest neighbors
        weights = np.exp(-distances**2 / (2 * sigma**2))
        smoothed_values[i] = np.sum(values[indices] * weights) / np.sum(weights)
    return smoothed_values

def surface_profile_nifti(volume_data_paths, 
                          surf_hemisphere, 
                          from_point, 
                          to_point, 
                          width,
                          n_sampling = None):
    """
    Do profile analysis based on virtual strip axis

    :param template_surface_path(string): template gii file ex) '/home/seojin/single-finger-planning/data/surf/fs_LR.164k.L.flat.surf.gii'
    :param surface_data(string or np.array - shape: (#vertex, #data): data gii file path or data array ex) '/home/seojin/single-finger-planning/data/surf/group.psc.L.Planning.func.gii'
    :param from_point(list): location of start virtual strip - xy coord ex) [-43, 86]
    :param to_point(list): location of end virtual strip - xy coord ex) [87, 58]
    :param width(int): width of virtual strip ex) 20
    :param n_sampling(int): the number of sampling across virtual strip

    return 
        -k virtual_stip_mask(np.array - #vertex): mask
        -k sampling_datas(np.array - #sampling, #data): sampling datas based on virtual strip
        -k sampling_coverages(np.array - #sampling, #vertex): spatial coverage per sampling point
        -k sampling_center_coords(np.array - #sampling, #coord): sampling center coordinates
    """
    template_surface_path = surf_paths(surf_hemisphere)[f"{surf_hemisphere}_template_surface_path"]
    surface_datas = load_surfData_fromVolume(volume_data_paths, surf_hemisphere)
    
    if n_sampling == None:
        n_sampling = abs(from_point[0] - to_point[0])
    
    # Load data metric file
    surface_gii = nb.load(template_surface_path)
    flat_coord = surface_gii.darrays[0].data
    
    if type(surface_datas) == str:
        data_gii = nb.load(surface_data_path)
    
        # Check - all data has same vertex shape
        darrays = data_gii.darrays
        is_valid = np.all([darray.dims[0] == darrays[0].dims[0] for darray in darrays])
        assert is_valid, "Please check data shape"
    
        data_arrays = np.array([e.data for e in darrays]).T
    else:
        data_arrays = surface_datas
        
    # Data information
    n_vertex = data_arrays.shape[0]
    n_data = data_arrays.shape[1]
    
    # Check - surface and data have same vertex
    assert flat_coord.shape[0] == n_vertex, "Data vertex must be matched with surface"
    
    # Extract vertices (x, y)
    vertex_2d = flat_coord[:, :2]
    
    # Move vertex origin
    points = vertex_2d - from_point
    
    # Set virtual vector(orientation of virtual strip)
    virtual_vec = to_point - from_point
    
    # Values for explaining vertex relative to virtual vector
    project = (np.dot(points, virtual_vec)) / np.dot(virtual_vec, virtual_vec)
    
    # Difference between vertex and projection vector
    residual = points - np.outer(project, virtual_vec)
    
    # Distance between vertex and virtual vector
    distance = np.sqrt(np.sum(residual**2, axis=1))

    ## Dummy for sampling result
    sampling_datas = np.zeros((n_sampling, n_data))
    virtual_stip_mask = np.zeros(n_vertex)
    sampling_center_coords = np.zeros((n_sampling, flat_coord.shape[1]))
    sampling_coverages = np.zeros((n_sampling, n_vertex))

    # Find points on the strip
    graduation_onVirtualVec = np.linspace(0, 1, n_sampling + 1)
    for i in range(n_sampling):
        # Filter only the vertices that are inside the virtual strip from all vertices
        start_grad = graduation_onVirtualVec[i]
        next_grad = graduation_onVirtualVec[i + 1]
    
        within_distance = distance < width
        upper_start = (project >= start_grad)
        lower_end = (project <= next_grad)
        no_origin = (np.sum(vertex_2d ** 2, axis=1) > 0)
        
        is_virtual_strip = within_distance & upper_start & lower_end & no_origin
        indx = np.where(is_virtual_strip)[0]

        sampling_coverages[i, indx] = 1
        
        # Perform cross-section
        sampling_datas[i, :] = np.nanmean(data_arrays[indx, :], axis=0) if len(indx) > 0 else 0
        virtual_stip_mask[indx] = 1
        sampling_center_coords[i, :] = np.nanmean(flat_coord[indx, :], axis=0) if len(indx) > 0 else 0

    result_info = {}
    result_info["sampling_datas"] = sampling_datas
    result_info["virtual_stip_mask"] = virtual_stip_mask
    result_info["sampling_center_coords"] = sampling_center_coords
    result_info["sampling_coverages"] = sampling_coverages
    return result_info

def get_bounding_box(hemisphere, virtual_strip_mask):
    """
    Get bounding box from virtual strip mask

    :param hemisphere(string): "L" or "R"
    :param virtual_strip_mask(np.array): strip mask

    return rect
    """
    template_path = surf_paths(hemisphere)[f"{hemisphere}_template_surface_path"]
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]
    
    rect_vertexes = vertex_locs[np.where(virtual_strip_mask == 1, True, False)]
    min_rect_x, max_rect_x = np.min(rect_vertexes[:, 0]), np.max(rect_vertexes[:, 0])
    min_rect_y, max_rect_y = np.min(rect_vertexes[:, 1]), np.max(rect_vertexes[:, 1])

    left_bottom = (min_rect_x, min_rect_y)
    width = max_rect_x - min_rect_x
    height = max_rect_y - min_rect_y

    return {
        "left_bottom" : left_bottom,
        "width" : width,
        "height" : height,
    }

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

def show_sulcus(surf_ax, 
                hemisphere, 
                color = "white", 
                linestyle = "dashed",
                isLabel = False,
                sulcus_dummy_name = "sulcus"):
    """
    Show sulcus base on surf axis

    :param surf_ax(axis)
    :param hemisphere(string): "L" or "R"

    return axis
    """
    
    sulcus_path = surf_paths(hemisphere, sulcus_dummy_name = sulcus_dummy_name)[f"{hemisphere}_sulcus_path"]
    with open(sulcus_path, "r") as file:
        marking_data_info = json.load(file)
    
    copy_ax = copy(surf_ax)
    for sulcus_name in marking_data_info:
        copy_ax.plot(np.array(marking_data_info[sulcus_name])[:, 0], 
                     np.array(marking_data_info[sulcus_name])[:, 1], 
                     color = color,  
                     linestyle = linestyle)

        if isLabel:
            x = np.mean(np.array(marking_data_info[sulcus_name])[:, 0])
            y = np.max(np.array(marking_data_info[sulcus_name])[:, 1]) + 5
            surf_ax.text(x = x, 
                         y = y, 
                         s = sulcus_abbreviation_name(sulcus_name), 
                         color = "white", 
                         horizontalalignment = "center", 
                         verticalalignment = "center",
                         size = 10)

    return copy_ax

def detect_sulcus(hemisphere, 
                  sampling_coverages, 
                  is_first_index = False,
                  sulcus_dummy_name = "sulcus"):
    """
    Detect sulcus based on surface map
    
    :param hemisphere(string): "L" or "R"
    :param sampling_coverages(np.array - shape: (#vertex)): cross-section area coverages
    :param is_first_index: select sulcus name if the sulcus name appears firstly when same sulcus name appears sequentially
    :param sulcus_dummy_name: sulcus dummy file name ex) "sulcus", "sulcus_sensorimotor"
    """
    surf_info = surf_paths(hemisphere, sulcus_dummy_name = sulcus_dummy_name)
    
    # Sulcus marking data
    sulcus_path = surf_info[f"{hemisphere}_sulcus_path"]
    with open(sulcus_path, "r") as file:
        marking_data_info = json.load(file)

    # Template
    template_path = surf_info[f"{hemisphere}_template_surface_path"]
    temploate_surface_data = nb.load(template_path)
    vertex_locs = temploate_surface_data.darrays[0].data[:, :2]

    # Sulcus prob
    having_sulcus_prob_info = {}
    for sulcus_name in marking_data_info:
        sulcus_pts = marking_data_info[sulcus_name]
    
        having_sulcus_probs = []
        for coverage in sampling_coverages:
            coverage_vertexes = vertex_locs[np.where(coverage == 0, False, True)]
            rect = minAreaRect(coverage_vertexes)
            box = boxPoints(rect)
        
            is_having_sulcus = np.array([pointPolygonTest(box, pts, False) for pts in sulcus_pts])
            having_sulcus_prob = np.sum(is_having_sulcus == 1) / len(sulcus_pts)
            having_sulcus_probs.append(having_sulcus_prob)
        having_sulcus_prob_info[sulcus_name] = np.array(having_sulcus_probs)

    # Sulcus names
    sulcus_names = ["" for _ in range(sampling_coverages.shape[0])]
    for sulcus_name in having_sulcus_prob_info:    
        if is_first_index:
            searches = np.where(having_sulcus_prob_info[sulcus_name] != 0)[0]
    
            if len(searches) > 0:
                first_index = searches[0]
                sulcus_names[first_index] = sulcus_name
        else:
            max_prob = max(having_sulcus_prob_info[sulcus_name])
        
            if max_prob != 0:
                max_prob_index = np.argmax(having_sulcus_prob_info[sulcus_name])
                sulcus_names[max_prob_index] = sulcus_name
    sulcus_names = np.array(sulcus_names)

    return sulcus_names

def detect_roi_names(sampling_coverages, hemisphere = "L", atlas = "Brodmann"):
    """
    Detect sampling coverage's roi name

    :param sampling_coverages(np.array - (#sampling, #vertex)): sampling coverage array
    :param hemisphere(string): brain hemisphere ex) "L" or "R"
    :param atlas(string): atlas name ex) "Brodmann"
    """
    # ROIs
    n_sampling = sampling_coverages.shape[0]
    roi_labels = np.load(surf_paths(surf_hemisphere = hemisphere, 
                                    atlas = atlas)[f"{hemisphere}_roi_label_path"])

    # Calculate ROI probs
    sampling_coverage_roi_probs = []
    for sampling_i in range(n_sampling):
        # Cover labels
        is_covering = sampling_coverages[sampling_i] == 1
        cover_labels = roi_labels[np.where(is_covering, True, False)]

        # Prob
        n_convering = np.sum(is_covering)
        counter = Counter(cover_labels)
        rois = np.array(list(counter.keys()))
        probs = np.array(list(counter.values())) / n_convering
        
        # Decending order
        sorted_prob_indexes = np.argsort(probs)[::-1]
        
        prob_info = {}
        for prob_index in sorted_prob_indexes:
            prob_info[rois[prob_index]] = probs[prob_index]
        
        sampling_coverage_roi_probs.append(prob_info)

    # Allocate roi using maximum prob
    rois = [max(roi_prob, key = roi_prob.get) for roi_prob in sampling_coverage_roi_probs]

    return rois

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
    
    if type(left_bounding_box) != type(None):
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

    if type(right_bounding_box) != type(None):
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
                         n_middle_tick = 3,
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
    :param n_middle_tick(int): the number of colorbar ticks without min and max value
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
        if type(left_bounding_box) != type(None):
            min_x, min_y = left_bounding_box["left_bottom"]
            max_x, max_y = min_x + left_bounding_box["width"], min_y + left_bounding_box["height"]
            l_ax.set_xlim(min_x, max_x)
            l_ax.set_ylim(min_y, max_y)
    else:
        if type(left_bounding_box) != type(None):
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
        if type(right_bounding_box) != type(None):
            min_x, min_y = right_bounding_box["left_bottom"]
            max_x, max_y = min_x + right_bounding_box["width"], min_y + right_bounding_box["height"]
            r_ax.set_xlim(min_x, max_x)
            r_ax.set_ylim(min_y, max_y)
    else:
        if type(right_bounding_box) != type(None):
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
                                         n_middle_tick = n_middle_tick, 
                                         orientation = "horizontal",
                                         tick_decimal = colorbar_decimal)
        fig.savefig(colorbar_path, dpi = dpi, transparent = True, bbox_inches = "tight")
        print(f"save: {colorbar_path}")
    
    # Both
    plt.clf()
    both_surf_img_path = os.path.join(save_dir_path, f"Both_hemi_stat")
    fig, ax = show_both_hemi_images(l_surf_img_path, 
                                    r_surf_img_path, 
                                    both_surf_img_path,
                                    colorbar_path if is_show_colorbar else None,
                                    zoom)
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

def sulcus_abbreviation_name(sulcus_name):
    if sulcus_name == "Precentral sulcus":
        return "prCS"
    elif sulcus_name == "Central sulcus":
        return "CS"
    elif sulcus_name == "Post central sulcus":
        return "poCS"
    elif sulcus_name == "Intra parietal sulcus":
        return "IPS"
    elif sulcus_name == "Parieto occipital sulcus":
        return "POS"
    elif sulcus_name == "Superior frontal sulcus":
        return "SFS"
    elif sulcus_name == "Inferior frontal sulcus":
        return "IFS"
    elif sulcus_name == "Superior temporal sulcus":
        return "STS"
    elif sulcus_name == "Middle temporal sulcus":
        return "MTS"
    elif sulcus_name == "Collateral sulcus":
        return "CLS"
    elif sulcus_name == "Cingulate sulcus":
        return "Cing"
    
def draw_cross_section_1dPlot(ax: plt.Axes, 
                              sampling_datas: np.array, 
                              sulcus_names: np.array, 
                              roi_names: np.array,
                              p_threshold: float = 0.05,
                              n_MCT: int = 1,
                              y_range: tuple = None,
                              tick_size: float = 18,
                              sulcus_text_size: int = 10,
                              y_tick_round: int = 4,
                              n_middle_yTick: int = 1,
                              cmap: str = "tab10",
                              xlabel: str = "Brodmann area",
                              ylabel: str = "Distance (a.u.)"):
    """
    Draw 1d plot for cross-section coverage analysis
    
    :param ax: Matplotlib Axes object where the plot will be drawn
    :param sampling_datas(shape - (n_condition, n_sampling_coverage, n_data)): 3D array of shape  with data to be plotted
    :param sulcus_names: 1D array containing sulcus names for each condition (can be empty strings or None)
    :param roi_names: 1D array containing ROI (Region of Interest) names for each condition
    :param n_MCT: the number of multiple comparison for correcting p-value using Bonferroni
    :param p_threshold: P-value threshold for marking significant areas (default is 0.05)
    :param y_range: specifying y-axis limits (e.g., (y_min, y_max)). If None, limits are calculated automatically
    :param tick_size: size of x and y axis' tick
    :param sulcus_text_size: text size of sulcus
    :param y_tick_round: tick round location
    :param n_middle_yTick: the number of y-tick without y_min and y_max
    :param cmap: colormap ex) "tab10"
    :param xlabel: text for x-axis label
    :param ylabel: text for y-axis label
    """

    n_cond, n_coverage, n_samples = sampling_datas.shape
    
    y_min_padding = 0
    y_max_padding = 0

    cmap = plt.get_cmap(cmap)
    cmap_colors = cmap.colors
    
    # Plot
    is_set_minMax = False
    if type(y_range) != type(None):
        y_min_, y_max_ = y_range
        is_set_minMax = True
    else:
        y_min_ = None
        y_max_ = None
    
    for cond_i, sampling_data in enumerate(sampling_datas):
        color = cmap_colors[cond_i]
        
        xs = np.arange(sampling_data.shape[0]).astype(str)
        mean_values = np.mean(sampling_data, axis = 1)
        errors = sem(sampling_data, axis = 1)
        ax.plot(xs, mean_values, color = color)
        ax.fill_between(xs,
                        mean_values - errors, mean_values + errors, 
                        alpha = 0.2,
                        color = color)

        if is_set_minMax == False:
            if y_min_ == None:
                y_min_ = np.min(mean_values - errors)
            if y_max_ == None:
                y_max_ = np.max(mean_values + errors)
    
    # Set ticks
    n_div = n_middle_yTick + 2
    interval = (y_max_ - y_min_) / n_div
    y_data = np.linspace(y_min_, y_max_, n_div)
    
    unique_rois = np.unique(roi_names)
    roi_names = copy(roi_names)
    roi_start_indexes = np.array(sorted([list(roi_names).index(roi) for roi in unique_rois])) # Select start index of ROI
    roi_names[roi_start_indexes] = ""
    
    tick_info = {}
    tick_info["x_data"] = np.arange(len(roi_names))
    tick_info["x_names"] = roi_names
    tick_info["x_tick_rotation"] = 0
    tick_info["x_tick_size"] = tick_size
    tick_info["y_data"] = y_data
    tick_info["y_names"] = y_data
    tick_info["y_tick_size"] = tick_size
    draw_ticks(ax, tick_info)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.{y_tick_round}f}"))
    
    # Draw spines
    draw_spine(ax)

    # Draw labels
    label_info = {}
    label_info["x_label"] = xlabel
    label_info["y_label"] = ylabel
    label_info["x_size"] = tick_size
    label_info["y_size"] = tick_size
    draw_label(ax, label_info)

    # Sulcus
    sulcus_indexes = np.where(sulcus_names != None)[0]
    if (len(sulcus_indexes) > 0) and (len(sulcus_names) > 0):
        y_max_padding += (interval / 3)
            
        sulcuses = sulcus_names[sulcus_indexes]
        sulcus_indexes = np.where(sulcus_names != "")[0]
        for sulcus_i in sulcus_indexes:
            sulcus_name = sulcus_names[sulcus_i]
            sulcus_name = sulcus_abbreviation_name(sulcus_name)
            
            ax.text(x = sulcus_i, 
                    y = y_max_ + (y_max_padding * 1.5), 
                    s = sulcus_name,  
                    va = "center", 
                    ha = "center",
                    size = sulcus_text_size,
                    rotation = 30)
            
            ax.text(x = sulcus_i, 
                    y = y_max_ + (y_max_padding / 2), 
                    s = "▼",  
                    va = "center", 
                    ha = "center",
                    size = 11,
                    rotation = 0)

    # Show significant areas
    y_min_padding += interval
    rect_height = interval / 10

    max_height_forSig = n_cond * rect_height

    for cond_i, sampling_data in enumerate(sampling_datas):
        color = cmap_colors[cond_i]
        
        stat_result = ttest_1samp(sampling_data, popmean = 0, axis = 1)
        significant_indexes = np.where(stat_result.pvalue * n_MCT < p_threshold)[0]
        
        cond_number = cond_i + 1
        y = y_min_ - y_min_padding + max_height_forSig - (rect_height * cond_number)

        for sig_i in significant_indexes:
            ax.add_patch(Rectangle(xy = (sig_i - 0.5, y), 
                                   width = 1, 
                                   height = rect_height, 
                                   color = color))

    # Draw roi
    for roi_start_i in list(roi_start_indexes) + [len(roi_names) - 1]:
        ax.axvline(x = roi_start_i, 
                   color = "black", 
                   linestyle = "dashed", 
                   alpha = 0.3,
                   ymin = 0,
                   ymax = (y_max_ - y_min_ + y_min_padding) / (y_max_ - y_min_ + y_min_padding + y_max_padding))

    ax.set_xlim(0, n_coverage - 1)

    if y_range != None:
        # ax.set_ylim(y_range[0], y_range[1])
        ax.set_ylim(min(y_range[0], y_min_ - y_min_padding), max(y_range[1], y_max_ - y_max_padding))
    else:
        ax.set_ylim(y_min_ - y_min_padding, y_max_ + y_max_padding)
    
def surface_profile_onUV(data_paths,
                         vertices, 
                         uv_coordinates, 
                         from_point, 
                         to_point, 
                         n_sampling, 
                         width):
    """
    Surface profile on UV coordinates.

    This function calculates surface profile data over UV coordinates by sampling 
    along a defined path and width. It projects UV coordinates onto a specified line, 
    samples data from given image files, and evaluates whether vertices fall within 
    defined coverage areas.

    :param data_paths(list of str): List of file paths to the image data (e.g., NIfTI files).
                       Each file represents an image from which sampling will be performed.
    :param vertices(np.ndarray of shape (n_vertices, 3)): Array of vertices in 3D space representing the surface geometry.
    :param uv_coordinates(np.ndarray of shape (n_vertices, 2)): UV coordinate array of shape (n_vertices, 2).
                           Each row represents the (u, v) coordinate of a vertex.
                           
    :param from_point(np.ndarray of shape (2,)): Starting point of the projection line in UV space.
                       
    :param to_point(np.ndarray of shape (2,)) Ending point of the projection line in UV space.
                     
    :param n_sampling(int): Number of sampling intervals along the projection line.
                       This determines the granularity of sampling.
                       
    :param width(float): Width of the sampling region around the projection line.
                  Vertices within this distance from the line are considered covered.

    :return: A dictionary with the following keys:
             - "sampling_datas": A 2D array where each row corresponds to a data file in 
                                `data_paths` and each column corresponds to a sampling interval.
                                (np.ndarray of shape (len(data_paths), n_sampling))
             - "virtual_strip_mask": A boolean mask array indicating whether each vertex 
                                     falls within the coverage area of any sampling interval.
                                     (np.ndarray of shape (n_vertices,))
             - "sampling_coverages": A boolean 2D array indicating for each sampling interval 
                                     and vertex whether the vertex is covered by that interval.
                                     (np.ndarray of shape (n_sampling, n_vertices))
    """
    total_distance = np.sqrt(np.sum(to_point - from_point) ** 2)

    n_vertex = vertices.shape[0]
    
    # Coverage units
    coverage_interval = 1 / n_sampling
    sampling_units = np.arange(0, 1 + coverage_interval, coverage_interval)
    coverage_units = np.array([(sampling_units[i], sampling_units[i+1]) for i in range(len(sampling_units)) if i < len(sampling_units) - 1])

    # Projection
    on_vector = to_point - from_point
    projection_info = projection(uv_coordinates, 
                                 from_point, 
                                 to_point, 
                                 type_ = "origin_correction")
    scalars = projection_info["scalar"]
    projected_data = projection_info["projected_data"]
    residual_data = projection_info["residual_data"]

    # Sampling coverages
    distance = np.sqrt(np.sum(residual_data ** 2, axis = 1))
    sampling_coverages = np.full((n_sampling, n_vertex), False, dtype=bool)
    for i, (start_unit, end_unit) in enumerate(coverage_units):
        is_within_units = (scalars >= start_unit) & (scalars < end_unit)
        within_distance = distance < width
        is_coverage = is_within_units & within_distance

        sampling_coverages[i, :] = is_coverage

    # Sampling datas
    sampling_datas = np.zeros((len(data_paths), n_sampling))
    for i, path in enumerate(data_paths):
        img = nb.load(path)
        affine = img.affine
        img_array = img.get_fdata()
    
        sampling_data = []
        for is_coverage in sampling_coverages:
            target_image_indices = np.apply_along_axis(lambda coord: reference2imageCoord(coord, affine = affine), 
                                                       axis = 1, 
                                                       arr = vertices[is_coverage])
            value = np.mean(img_array[target_image_indices[:, 0], target_image_indices[:, 1], target_image_indices[:, 2]])
            sampling_data.append(value)
        sampling_datas[i, :] = sampling_data

    # Virtual_stip_mask
    virtual_stip_mask = np.any(sampling_coverages, axis = 0)
    
    result_info = {}
    result_info["sampling_datas"] = sampling_datas
    result_info["virtual_strip_mask"] = virtual_stip_mask
    result_info["sampling_coverages"] = sampling_coverages
    
    return result_info
    
if __name__ == "__main__":
    # Parameters
    template_surface_path = '/home/seojin/single-finger-planning/data/surf/fs_LR.164k.L.flat.surf.gii'
    surface_data_path = '/home/seojin/single-finger-planning/data/surf/group.psc.L.Planning.func.gii'
    from_point = np.array([-43, 86])  # x_start, y_start
    to_point = np.array([87, 58])    # x_end, y_end
    width = 20
    
    cross_section_result_info = surface_profile(template_surface_path = template_surface_path, 
                                                 urface_data_path = surface_data_path, 
                                                 from_point = from_point, 
                                                 to_point = to_point, 
                                                 width = width)
    virtual_stip_mask = cross_section_result_info["virtual_stip_mask"]
    
    vol_to_surf(volume_data_path = "/mnt/ext1/seojin/temp/stat.nii",
                pial_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.pial.surf.gii",
                white_surf_path = "/mnt/sda2/Common_dir/Atlas/Surface/fs_LR_32/fs_LR.32k.L.white.surf.gii")

    hemisphere = "L"
    roi_values = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_roi_values.npy")
    with open(os.path.join(surface_mask_dir_path, f"{hemisphere}_roi_vertex_info.json"), 'rb') as f:
        loaded_info = json.load(f)
    draw_surf_roi(roi_values, loaded_info, "L")

    surf_roi_labels = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_rois.npy")
    draw_surf_selectedROI(surf_roi_labels, "1+2+3", f"{hemisphere}")

    surf_paths("L")
    