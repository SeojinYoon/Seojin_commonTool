
# Common Libraries
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from nilearn.plotting import plot_surf_roi

# Custom Libraries
sys.path.append("/home/seojin")
import surfAnalysisPy as surf # Dierdrichsen lab's library
from surface_roi import show_sulcus

sys.path.append("/home/seojin/Seojin_commonTool/Module")
from sj_matplotlib import make_colorbar

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

# Examples
if __name__ == "__main__":
    hemisphere = "L"
    roi_values = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_roi_values.npy")
    with open(os.path.join(surface_mask_dir_path, f"{hemisphere}_roi_vertex_info.json"), 'rb') as f:
        loaded_info = json.load(f)
    draw_surf_roi(roi_values, loaded_info, "L")
    

