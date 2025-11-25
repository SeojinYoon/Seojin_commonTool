
# Common Libraries
import json
import numpy as np
import nibabel as nb
from copy import copy
from collections import Counter
from cv2 import minAreaRect, boxPoints, pointPolygonTest

# Custom Libraries
from surface_data import surf_paths

# Detection
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

# Rendering
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
    
# Examples
if __name__ == "__main__":
    roi_values = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_roi_values.npy")
    with open(os.path.join(surface_mask_dir_path, f"{hemisphere}_roi_vertex_info.json"), 'rb') as f:
        loaded_info = json.load(f)
    draw_surf_roi(roi_values, loaded_info, "L")

    surf_roi_labels = np.load(f"/mnt/ext1/seojin/dierdrichsen_surface_mask/Brodmann/{hemisphere}_rois.npy")
    draw_surf_selectedROI(surf_roi_labels, "1+2+3", f"{hemisphere}")
    