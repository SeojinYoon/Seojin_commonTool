
# Common Libraries
import os
import sys
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Rectangle
from scipy.stats import sem
from scipy.stats import ttest_1samp
from matplotlib.lines import Line2D

# Custom Libraries
from surface_data import surf_paths, load_surfData_fromVolume

sys.path.append("/home/seojin")
import surfAnalysisPy as surf # Dierdrichsen lab's library

sys.path.append("/home/seojin/Seojin_commonTool/Module")
from sj_matplotlib import draw_ticks, draw_spine, draw_label

# Functions
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
                              y_tick_precision: int = 4,
                              n_inner_yTick: int = 1,
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
    :param y_tick_precision: Number of decimal places to display on y-axis ticks
    :param n_inner_yTick: the number of y-tick without y_min and y_max
    :param cmap: colormap ex) "tab10"
    :param xlabel: text for x-axis label
    :param ylabel: text for y-axis label
    """

    n_cond, n_coverage, n_samples = sampling_datas.shape
    
    y_min_padding = 0
    y_max_padding = 0

    cmap = plt.get_cmap(cmap)
    cmap_colors = cmap.colors

    # Lists to store calculated stats for later range calculation
    all_means = []
    all_errors = []

    # Plot
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

        # Store for range calculation
        all_means.append(mean_values)
        all_errors.append(errors)
    all_means = np.array(all_means)
    all_errors = np.array(all_errors)
    
    # Determine y-Axis limits
    if y_range is not None:
        y_min, y_max = y_range
    else:
        # Calculate global min/max based on all plotted data
        y_min = np.min(all_means - all_errors)
        y_max = np.max(all_means + all_errors)
        
    # Set ticks
    n_div = n_inner_yTick + 2
    interval = (y_max - y_min) / n_div
    y_data = np.linspace(y_min, y_max, n_div)
    
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
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.{y_tick_precision}f}"))
    
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
                    y = y_max + (y_max_padding * 1.5), 
                    s = sulcus_name,  
                    va = "center", 
                    ha = "center",
                    size = sulcus_text_size,
                    rotation = 30)
            
            ax.text(x = sulcus_i, 
                    y = y_max + (y_max_padding / 2), 
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

        # t-test
        stat_result = ttest_1samp(sampling_data, popmean = 0, axis = 1)

        # bonferroni correction
        significant_indexes = np.where(stat_result.pvalue * n_MCT < p_threshold)[0]
        
        cond_number = cond_i + 1
        y = y_min - y_min_padding + max_height_forSig - (rect_height * cond_number)

        # draw significant area
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
                   ymax = (y_max - y_min + y_min_padding) / (y_max - y_min + y_min_padding + y_max_padding))

    ax.set_xlim(0, n_coverage - 1)

    if y_range != None:
        ax.set_ylim(min(y_range[0], y_min - y_min_padding), max(y_range[1], y_max - y_max_padding))
    else:
        ax.set_ylim(y_min - y_min_padding, y_max + y_max_padding)

def draw_both_hemi_cross_section_1dPlot(y_range: tuple, 
                                        l_sampling_datas: np.ndarray,
                                        l_sulcus_names: np.ndarray,
                                        l_roi_names: np.ndarray,
                                        r_sampling_datas: np.ndarray,
                                        r_sulcus_names: np.ndarray,
                                        r_roi_names: np.ndarray,
                                        y_tick_precision: int = 4,
                                        n_inner_yTick = 1,
                                        p_threshold = 0.05,
                                        n_MCT = 1,
                                        cmap = "tab10"):
    """
    Draw side-by-side 1d cross-section plots for both Left and Right hemispheres.

    :param y_range: tuple specifying global y-axis limits (y_min, y_max) applied to both plots.
    :param l_sampling_datas(shape: [#cond, #roi, #data]): data array for Left Hemisphere .
    :param l_sulcus_names(shape: #roi): array of sulcus names for Left Hemisphere.
    :param l_roi_names(shape: #roi): array of ROI names for Left Hemisphere.
    :param r_sampling_datas(shape: [#cond, #roi, #data]): data array for Right Hemisphere.
    :param r_sulcus_names(shape: #roi): array of sulcus names for Right Hemisphere.
    :param r_roi_names(shape: #roi): array of ROI names for Right Hemisphere.
    :param y_tick_precision: number of decimal places to display on y-axis ticks (Precision).
    :param n_inner_yTick: number of intermediate y-ticks between y_min and y_max.
    :param p_threshold: p-value threshold for statistical significance (default: 0.05).
    :param n_MCT: number of multiple comparisons for Bonferroni correction.
    :param cmap: matplotlib colormap name (e.g., "tab10").
    
    :return: (fig, axes) - The created Matplotlib Figure and Axes objects.
    """

    # Draw cross-section data of left hemisphere
    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.set_figwidth(15)
    draw_cross_section_1dPlot(ax = axes[0], 
                              sampling_datas = l_sampling_datas, 
                              sulcus_names = l_sulcus_names, 
                              roi_names = l_roi_names,
                              p_threshold = p_threshold,
                              n_MCT = n_MCT,
                              y_range = y_range,
                              tick_size = 24,
                              sulcus_text_size = 16,
                              y_tick_precision = y_tick_precision,
                              n_inner_yTick = n_inner_yTick,
                              cmap = cmap)

    # Draw cross-section data of right hemisphere
    draw_cross_section_1dPlot(ax = axes[1], 
                              sampling_datas = r_sampling_datas, 
                              sulcus_names = r_sulcus_names, 
                              roi_names = r_roi_names,
                              p_threshold = p_threshold,
                              n_MCT = n_MCT,
                              y_range = y_range,
                              tick_size = 24,
                              sulcus_text_size = 16,
                              y_tick_precision = y_tick_precision,
                              n_inner_yTick = n_inner_yTick,
                              cmap = cmap)

    # others
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[1].get_yaxis().set_visible(False)
    axes[1].spines['left'].set_visible(False)

    return fig, axes
    
if __name__ == "__main__":
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
    