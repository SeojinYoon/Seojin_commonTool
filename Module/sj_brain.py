
# Common Libraries
import os
import numpy as np
import pandas as pd
import copy
import datetime
import nltools

# Preprocessing
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from scipy.stats import zscore
from multiprocessing.pool import ThreadPool

# RSAtoolbox
import rsatoolbox
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from rsatoolbox.data.dataset import Dataset
from tqdm import tqdm
from rsatoolbox.rdm import calc_rdm, RDMs
from rsatoolbox.data.noise import prec_from_residuals

# Visualize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Brain
import nilearn
import nibabel as nb
from nilearn import image
from nilearn.plotting import plot_design_matrix

from nltools.data import Brain_Data

# Multiprocessing
from tqdm.contrib.concurrent import process_map

# Custom Libraries
from sj_visualization import plot_timeseries
from sj_sequence import get_multiple_elements_in_list, slice_list_usingDiff, check_duplication
from sj_file_system import load, save
from sj_higher_function import recursive_mapWithDepth
from sj_preprocessing import change_df
from sj_datastructure import quick_sort
from sj_string import make_pad_fromInt, str_join
from sj_brain_mask import apply_mask_change_shape
from brain_coord import image3d_to_1d

# Sources
class Shrinkage_type:
    eye = "shrinkage_eye"
    diag = "shrinkage_diag"
    
    @staticmethod
    def name(shrinkage_type):
        if shrinkage_type == Shrinkage_type.eye:
            return Shrinkage_type.eye
        elif shrinkage_type != Shrinkage_type.diag:
            return Shrinkage_type.diag
        
class Similarity_type:
    # https://rsatoolbox.readthedocs.io/en/latest/_modules/rsatoolbox/rdm/calc.html#calc_rdm_mahalanobis
    
    # For calculating dissmilarity
    euclidean = "euclid"
    pearson = "correlation"
    poisson = "poisson"
    mahalanobis = "mahalanobis" # Require - noise
    crossnobis = "crossnobis" # Require - noise
    
    # For comparison between model and data comparison
    spearman = "spearman"
    kendall_tau_b = "kendall"
    kendall_tau_a = "tau-a"


    @staticmethod
    def name(sim_type):
        if sim_type == Similarity_type.pearson:
            return "" # Default...
        elif sim_type != Similarity_type.pearson:
            return sim_type
        
def load_behaviors(behavior_paths):
    """
    load behaviors
    
    :param behavior_paths: behavior data paths(string)
    
    return behavior data list(element - dataframe)
    """
    behaviors = []
    for i in range(0, len(behavior_paths)):
        behaviors.append(pd.read_csv(behavior_paths[i]))
    return behaviors

def load_head_motion_datas(head_motion_paths):
    """
    load head motion datas from paths
    
    :param head_motion_paths: head motion data paths(string)
    
    return headmotion data list(element - dataframe)
    """
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]

    head_motion_datas = []
    for data_path in head_motion_paths:
        head_motion_data = pd.read_csv(data_path, delimiter=" ", names=add_reg_names)
        head_motion_datas.append(head_motion_data)

    return head_motion_datas

def type_transform(brain_data, tranform_type):
    """
    Transform fmri data type 
    
    :param brain_data: fmri_data (Nifti1Image, nltools Brain_data)
    :param tranform_type: nltools, array, nibabel
    return transformed data
    """
    if type(brain_data) == nb.Nifti1Image:
        if tranform_type == "nltools":
            result = Brain_Data(brain_data)
        elif tranform_type == "array":
            result = brain_data.get_fdata()
    elif type(brain_data) == Brain_Data:
        if tranform_type == "array":
            result = brain_data.to_nifti().get_fdata()
        elif tranform_type == "nibabel":
            result = brain_data.to_nifti()
            
    return result

def nifiti_4d_to_1d(img):
    """
    transform image shape from 4d to 1d
    
    slice image per time and flatten

    :param img: nifti image
    
    return: array(element: 1d array) / shape (n x p) / n: measure_timing, p: voxels
    """
    results = []

    time = img.shape[-1]
    for t in range(0, time):
        sliced_img = img.slicer[..., t]
        array1d = sliced_img.get_fdata().reshape(-1)
        results.append(array1d)
    return np.array(results)

def concat_fMRI_datas(interest_fMRI_data_indexes = None, fMRI_datas = []):
    """
    concatenate fMRI datas by flattening

    :params interest_fMRI_data_indexes: fMRI_datas's indexes ex) [1,2,3]
    :params fMRI_datas: array of Nifti1Image

    return: flatten fMRI data(3d numpy array)
    """

    if interest_fMRI_data_indexes == None:
        array_fMRI_datas = []
        for interest in range(0, len(fMRI_datas)):
            fMRI_data = fMRI_datas[interest]
            array_fMRI_datas.append(nifiti_4d_to_1d(fMRI_data))
        interest_fMRI_data = np.concatenate(array_fMRI_datas, axis=0)
    else:
        array_fMRI_datas = []
        for interest in interest_fMRI_data_indexes:
            fMRI_data = fMRI_datas[interest]
            array_fMRI_datas.append(nifiti_4d_to_1d(fMRI_data))
        interest_fMRI_data = np.concatenate(array_fMRI_datas, axis=0)
    return interest_fMRI_data

def flatten(fMRI_datas):
    """
    flatten voxel shape

    :param fMRI_datas: array of Nifti1Image
    
    return flattened fMRI datas(list - numpy array)
    """
    flatten_datas = []
    for data in fMRI_datas:
        count_of_timing = data.shape[-1]
        flatten_datas.append(data.get_fdata().reshape(count_of_timing, -1))
    return flatten_datas

def split_data_pairs(datas, behaviors, train_indexes, test_indexes):
    """
    Split fMRI datas and behavior datas to obtain train and test dataset
    
    :param datas: fmri_datas(list - nifti_image) - Each image is separted by run probably
    :param behaviors: behaviors(list - dataframe) - Each behavior is separted by run probably
    :param train_indexes: Train data set indexes, Each index probably represents a run
    :param test_indexes: Test data set indexes, Each index probably represents a run
    
    return train_datas, train_behavior, test_datas(4d numpy array), test_behavior
    """
    train_datas = concat_fMRI_datas(train_indexes, datas)
    train_behavior = get_multiple_elements_in_list(in_list=behaviors,
                                                           in_indices=train_indexes)
    train_behavior = concat_pandas_datas(train_behavior)

    test_datas = concat_fMRI_datas(test_indexes, datas)
    test_behavior = get_multiple_elements_in_list(in_list=behaviors,
                                                          in_indices=test_indexes)
    test_behavior = concat_pandas_datas(test_behavior)

    return train_datas, train_behavior, test_datas, test_behavior

def get_specific_images(img, mask_condition):
    result = nilearn.image.index_img(img, list(map(lambda x: x[0], np.argwhere(mask_condition))))

    return result


def highlight_stat(roi_array, stat_array, stat_threshold):
    """
    highlight roi area's statistics in stat map,

    :param roi_array: array
    :param stat_array: array
    :param stat_threshold: threshold
    """
    highlight_array = roi_array.copy()
    highlight_array[:] = 1

    non_highlight_array = highlight_array.copy()
    non_highlight_array[:] = -1

    zero_array = non_highlight_array.copy()
    zero_array[:] = 0

    conditions = [np.logical_and(roi_array > 0, stat_array > stat_threshold), stat_array > stat_threshold, True]

    from matplotlib import cm
    color_map = cm.get_cmap('viridis', 2)

    return_obj = {
        "data": np.select(conditions, [highlight_array, non_highlight_array, zero_array]),
        "color_map": color_map
    }

    return return_obj


def colored_roi_with_stat(roi_arrays, stat_map, stat_threshold):
    """
    show colorred roi and represent stat

    :param roi_arrays: array of roi
    :param stat_map: array of statistics
    :param stat_threshold: threshold
    """
    # preprocessing
    roi_arrays = [roi.astype(np.int16) for roi in roi_arrays]

    # pre-data
    shape = roi_arrays[0].shape

    # make roi array
    zero_array = np.repeat(0, shape[0] * shape[1] * shape[2]).reshape(shape)
    zero_array[:] = 0

    color_data = 1
    colored_roi_arrays = [roi.copy() for roi in roi_arrays]
    for roi in colored_roi_arrays:
        roi[roi == True] = color_data
        color_data += 1

    # make roi_stat hightlight
    color_data += 1
    roi_stat_highlight = []
    color_values = []
    conditions = []
    for roi in roi_arrays:
        conditions += [np.logical_and(stat_map > stat_threshold, roi == True)]

        color_values.append(color_data)
        roi_stat_highlight.append(np.repeat(color_data, shape[0] * shape[1] * shape[2]).reshape(shape))

    # conditions
    conditions += [stat_map > stat_threshold]  # highlight
    conditions += [roi > 0 for roi in roi_arrays]
    conditions += [True]

    color_data += 1
    stat_array = np.repeat(color_data, shape[0] * shape[1] * shape[2]).reshape(shape)

    result = np.select(conditions, roi_stat_highlight + [stat_array] + colored_roi_arrays + [zero_array])

    from matplotlib import cm
    color_map = cm.get_cmap('viridis', len(result) - 1)  # -1: remove zero

    return_obj = {
        "data": result,
        "color_map": color_map
    }

    return return_obj

def mean_img(imgs, threshold=None):
    """
    mean nilearn images
    
    :param imgs: target images to mean(nilearn image 4d)
    :papram threshold: if this value is set, mean_img is subtracted by this value
    """
    result = nilearn.image.mean_img(imgs)
    
    if threshold != None:
        result = nb.Nifti1Image(result.get_fdata() - threshold, affine = result.affine)
    return result

def mean_img_within_diff(fMRI_data, lower_diff, upper_diff):
    """
    mean image from fMRI_data within target_lower_bound <= ~ <= target_upper_bound

    :param fMRI_data: Niimg-like obj
    :param lower_diff: lower bound about each index, unsigned integer
    :param upper_diff: upper bound about each index, unsigned integer
    """
    fmri_data_count = fMRI_data.shape[-1]

    mean_datas = []
    for target_index in range(0, fmri_data_count):
        # get data from fMRI_data within target_lower_bound <= ~ <= target_upper_bound
        target_lower_bound = target_index - lower_diff
        target_upper_bound = target_index + upper_diff + 1

        if target_lower_bound < 0:
            target_lower_bound = 0
        if target_upper_bound > fmri_data_count:
            target_upper_bound = fmri_data_count

        # slice data
        sliced_fMRI_data = fMRI_data.slicer[..., target_lower_bound: target_upper_bound]

        # mean
        mean_data = image.mean_img(sliced_fMRI_data)
        mean_datas.append(mean_data)

    return image.concat_imgs(mean_datas)

def mean_img_within_diff_with_targetIndex(fMRI_data, lower_diff, upper_diff, target_indexes):
    """
    mean image from fMRI_data within target_lower_bound <= ~ <= target_upper_bound

    :param fMRI_data: Niimg-like obj
    :param lower_diff: lower bound about each index, unsigned integer
    :param upper_diff: upper bound about each index, unsigned integer
    :param target_indexes: interest indexes ex) [1,2,3]
    """
    fmri_data_count = fMRI_data.shape[-1]

    mean_datas = []
    for target_index in target_indexes:
        # get data from fMRI_data within target_lower_bound <= ~ <= target_upper_bound
        target_lower_bound = target_index - lower_diff
        target_upper_bound = target_index + upper_diff + 1

        if target_lower_bound < 0:
            target_lower_bound = 0
        if target_upper_bound > fmri_data_count:
            target_upper_bound = fmri_data_count

        # slice data
        sliced_fMRI_data = fMRI_data.slicer[..., target_lower_bound: target_upper_bound]

        # mean
        mean_data = image.mean_img(sliced_fMRI_data)
        mean_datas.append(mean_data)

    return image.concat_imgs(mean_datas)

def upper_tri(RDM):
    """
    upper_tri returns the upper triangular index of an RDM
    
    :param RDM: squareform RDM(numpy array)
    
    return upper triangular vector of the RDM(1D array) 
    """
    
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def total_RDM_dissimilarity(RDM):
    """
    total dissimilarity 
    
    :param RDM: RDM(numpy 2d array)
    
    return: scalar value
    """
    
    up_RDM = upper_tri(RDM)
    return np.mean(up_RDM)

def make_rdm_from_1dTriangle(triangle_1d, mat_dim):
    """
    make rdm from triangle values
    
    :param triangle_1d: lower triangle values or upper triangle values(list - 1d)
    :param mat_dim: dimension of square matrix(int)
    
    return array(2d)
    """
    
    matrix = np.zeros((mat_dim, mat_dim))
    
    # upper triangle
    upper_triangle_indexes = np.triu_indices(mat_dim, 1)
    r_indexes = upper_triangle_indexes[0]
    c_indexes = upper_triangle_indexes[1]
    for e, r_i, c_i in zip(triangle_1d, r_indexes, c_indexes):
        matrix[r_i, c_i] = e

    # lower triangle
    lower_triangle_indexes = np.tril_indices(mat_dim, -1)
    lower_triangle_indexes = [(r_i,c_i) for r_i, c_i in zip(lower_triangle_indexes[0], lower_triangle_indexes[1])]
    
    def sort_func(e1, e2):
        r1_i, c1_i = e1[0], e1[1]
        r2_i, c2_i = e2[0], e2[1]

        if c1_i == c2_i:
            return r1_i < r2_i
        else:
            return c1_i < c2_i
        
    lower_triangle_indexes = quick_sort(lower_triangle_indexes, sort_func)
    for e, index in zip(triangle_1d, lower_triangle_indexes):
        r_i = index[0]
        c_i = index[1]
        matrix[r_i, c_i] = e
    
    return matrix

def searchlight_RDM(betas, 
                    mask, 
                    conditions, 
                    save_rdm_path = None, 
                    save_region_path = None,
                    radius=2, 
                    threshold=1,
                    method=Similarity_type.pearson,
                    load_rdm = True):
    """
    Searches through the non-zero voxels of the mask, selects centers where
    proportion of sphere voxels >= self.threshold
    
    threshold (float, optional): Threshold of the proportion of voxels that need to
        be inside the brain mask in order for it to be
        considered a good searchlight center.
        Values go between 0.0 - 1.0 where 1.0 means that
        100% of the voxels need to be inside
        the brain mask.
        Defaults to 1.0.
    
    Each beta value's element must match with corresponding condition
    
    if you want to use crossvalidation RDM then many beta values need to be insulted.
    Crossvalidation RDM works using default cv descriptor.
    
    Pattern descriptor's index is allocated by the order of conditions excluding conditions already mapped to index.
    
    :param betas: beta values(nifti array or nltools Brain data)
    :param mask: must be binary data
    :param conditions: conditions(list of string)
    :param save_region_path: if save_rdm_path is not none, save region data
    :param save_rdm_path: if save_rdm_path is not none, save rdm data
    :param radius: searchlight radius
    :param threshold: threshold(float)
    :param method: distance method(Similarity_type)
    
    return RDMs(rsatoolbox)
    """
    
    if save_rdm_path != None:
        try:
            print("load rdm: ", save_rdm_path)
            
            if load_rdm:
                SL_RDM = load(save_rdm_path)
                return SL_RDM
            else:
                pass
        except:
            print("load rdm fail: ", save_rdm_path)
            
    print("RDM Calculate Start")    
    if type(betas) == list and type(betas[0]) == nb.Nifti1Image:
        # checking shape is same
        assert all(map(lambda data: data.shape == betas[0].shape, betas)), "nifti_datas element shape is not same"

        array_betas = np.array([betas[data_i].get_fdata() for data_i in range(0, len(betas))])
    elif type(betas) == nb.Nifti1Image:
        array_betas = []
        for i in range(len(conditions)):
            beta = betas.slicer[..., i]
            array_betas.append(beta.get_fdata())
        array_betas = np.array(array_betas)
    elif type(betas) == Brain_Data:
        array_betas = []
        for condition_i in range(0, len(conditions)):
            array_betas.append(betas[condition_i].to_nifti().get_fdata())    

        array_betas = np.array(array_betas)
    elif type(betas) == list and type(betas[0]) == Brain_Data:
        array_betas = np.array([beta.to_nifti().get_fdata() for beta in betas])
    
    if save_region_path != None:        
        try:
            centers, neighbors = load(save_region_path)
        except:
            print("load region fail: ", save_region_path)
            """
            Searches through the non-zero voxels of the mask, 
            selects centers where proportion of sphere voxels >= self.threshold
            
            This process searches neighbors matched within radius using euclidean distance.
            
            reference: 
            https://rsatoolbox.readthedocs.io/en/latest/_modules/rsatoolbox/util/searchlight.html#get_volume_searchlight
            
            note!!!!!!!!
            RDM searchlight uses an index and calculates the Euclidean distance to apply a mask.(not mm)
            
            """
            centers, neighbors = get_volume_searchlight(mask.get_fdata(), 
                                                        radius=radius, 
                                                        threshold=threshold)
            save((centers, neighbors), save_region_path)
            print("save region: ", save_region_path)
        
    # reshape data so we have n_observastions x n_voxels
    n_conditions, nx, ny, nz = array_betas.shape
    print(n_conditions, nx, ny, nz)
    
    data_2d = array_betas.reshape([n_conditions, -1])
    print(data_2d.shape)
    print(len(conditions))
    data_2d = np.nan_to_num(data_2d)
    
    """
    reference: 
    """
    SL_RDM = get_searchlight_RDMs(data_2d=data_2d, 
                                  centers=centers, 
                                  neighbors=neighbors, 
                                  events=conditions, 
                                  method=method)
    
    if save_rdm_path != None:
        print("save RDM: ", save_rdm_path)
        save(SL_RDM, save_rdm_path)
        
    return SL_RDM

class RDM_model:
    """
    This class's purpose is managing RDM model
    """
    
    def __init__(self, model_2d_array, model_name, conditions):
        """
        :param model_2d_array: model(2d numpy array)
        :param model_name: model name(str)
        :param conditions: conditions(list of string)
        """
        self.model = model_2d_array
        self.name = model_name
        self.conditions = conditions
        
    def draw(self, 
             fig = None, 
             axis = None, 
             style_info = {}):
        """
        Draw rdm matrix
        
        :param fig: figure
        :param axis: axis
        :param style_info: see draw_rdm style_info parameter
        """
        if fig is None and axis is None:
            fig, axis = plt.subplots(1,1)
        
        RDM_model.draw_rdm(rdm = self.model, 
                           conditions = self.conditions, 
                           fig = fig,
                           axis = axis,
                           style_info = style_info)
    
    @staticmethod
    def draw_rdm(rdm, 
                 conditions, 
                 fig,
                 axis,
                 style_info = {}):
        """
        :param rdm: numpy 2d array
        :param conditions: list of condition
        :param fig: matplotlib figure
        :param axis: axis
        :param style_info: style information
            -k, cmap(str): color map ex) seismic
            -k, title(str): title of RDM ex) "abc"
            -k, title_wight(str): title weight ex) "bold"
            -k, title_size(float): title font size ex) 10
            -k, x_tick_rotation(int): rotation of x_tick ex) 90
            -k, tick_weight(str): tick weight ex) "bold"
            -k, tick_size(int): tick size ex) 20
            -k, color_range(tuple): visualization range ex) (-0.1, 0.1)
            -k, legend_padding(float): spacing between rdm and legend ex) 0.1
            -k, legend_label(str): legend label ex) "label"
            -k, legend_size(float): legend font size ex) 10
            -k, legend_weight(float): legend font weight ex) 10
            -k, legend_tick_size(float): legend font size ex) 10
            -k, legend_tick_weight(str): legend tick weight ex) "bold"
            -k, legend_ticks(list): ticks ex) [1,2,3]
            -k, legend_labels(list): tick label ex) ["1","2","3"]
            -k, decimal_digit(int): decimal digit for visualization
        """
        cmap = style_info.get("cmap", "coolwarm")
        
        # Title constants
        title = style_info.get("title", "")
        title_wight = style_info.get("title_wight", "bold")
        title_size = style_info.get("title_size", 20)
        
        # Tick constants
        x_tick_rotation = style_info.get("x_tick_rotation", 45)
        tick_weight = style_info.get("tick_weight", "bold")
        tick_size = style_info.get("tick_size", 20)
        ticks_range = np.arange(0, len(conditions))
        
        # range
        decimal_digit = style_info.get("decimal_digit", None)
        if decimal_digit != None:
            rdm = np.round(rdm, decimal_digit)
        
        v_min = np.min(rdm)
        v_max = np.max(rdm)
        color_range = style_info.get("color_range", (v_min, v_max))
        
        # legend constants
        is_legend = style_info.get("is_legend", True)
        legend_padding = style_info.get("legend_padding", 0.1)
        legend_label = style_info.get("legend_label", "Dissimilarity")
        legend_size = style_info.get("legend_size", 20)
        legend_weight = style_info.get("legend_weight", "bold")
        
        legend_tick_size = style_info.get("legend_tick_size", 20)
        legend_tick_weight = style_info.get("legend_tick_weight", "bold")
        legend_ticks = style_info.get("legend_ticks", [color_range[0], color_range[1]])
        legend_tick_labels = style_info.get("legend_labels", [str(e) for e in legend_ticks])
        legend_font_properties = {'size': legend_tick_size, 'weight': legend_tick_weight}
        
        # Matrix
        im = axis.imshow(rdm, cmap = cmap, vmin = color_range[0], vmax = color_range[1])
        
        # Legend
        if is_legend:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size = '5%', pad = legend_padding)
            colorbar = fig.colorbar(im, cax = cax, orientation = 'vertical')
            colorbar.set_label(legend_label, weight = legend_weight, size = legend_size)

            # Set custom ticks on the color bar
            colorbar.set_ticks(legend_ticks)

            # You can also set custom tick labels if desired
            colorbar.set_ticklabels(legend_tick_labels, weight = legend_tick_weight, size = legend_tick_size)
        
        # Matrix Ticks
        slicing_indexes = slice_list_usingDiff(conditions)
        rdm_conditions = [conditions[start_i] for start_i, end_i in slicing_indexes]
        
        axis.set_xticks([(start_i + end_i)/2 for start_i, end_i in slice_list_usingDiff(conditions)], 
                        rdm_conditions,
                        size = tick_size,
                        weight = tick_weight,
                        rotation = x_tick_rotation)
        axis.set_yticks([(start_i + end_i)/2 for start_i, end_i in slice_list_usingDiff(conditions)], 
                        rdm_conditions,
                        size = tick_size,
                        weight = tick_weight)
       
        # Spine
        for spine in axis.spines.values():
            spine.set_visible(False)
    
        # Title
        axis.set_title(title, weight = title_wight, size = title_size)
        
        # Grid
        axis.grid(False)
        
def RSA(models, 
        mask,
        rdms = None,
        save_rdm_path = None,
        save_corr_brain_dir_path = None,
        datas = None, 
        conditions = None,
        region_path = None,
        radius = 2,
        threshold = 1,
        n_jobs = 1,
        eval_method = Similarity_type.spearman,
        rdm_distance_method = Similarity_type.pearson,
        return_type = "rdm_brain",
        load_rdm = False):
    """
    Do representational Similarity Analysis
    
    Searches through the non-zero voxels of the mask, selects centers where
    proportion of sphere voxels >= self.threshold
    
    threshold (float, optional): Threshold of the proportion of voxels that need to
        be inside the brain mask in order for it to be
        considered a good searchlight center.
        Values go between 0.0 - 1.0 where 1.0 means that
        100% of the voxels need to be inside
        the brain mask.
        Defaults to 1.0.
    
    :param rdms: searchlight rdms(rsatoolbox.RDMs)
    :param models: RDM Model list, made by RDM_model or ModelFixed
    :param mask: must be binary data
    :param save_rdm_path: (string) if save_rdm_path is not none, save rdm data
    :param save_corr_brain_dir_path: (string) correlation brain with model
    :param datas: nifti array or nltools Brain data
    :param conditions: data conditions(1d list)
    :param region_path: (string) if region_path is not none, save region data
    :param radius: searchlight radius
    :param threshold: threshold(float)
    :param rdm_distance_method: distance method (Similarity_type)
    :param return_type: 'rdm_brain', 'eval_result'
    
    return: RDM_brains
    """
    # input validation
    assert ((type(datas) != None and type(conditions) != None) or load_rdm_path != None), "Please input nifti data or load_rdm_path"
    
    # model validation
    for model in models:
        if type(model) == ModelFixed:
            pass
        else:
            # check all model's condition is same
            is_same_cond = np.all(models[0].conditions == models[0].conditions)
            assert is_same_cond, "all models condition is not matched!!"
            
            if eval_method == Similarity_type.spearman:
                # check model degree of freedom for computing correlation
                assert len(np.unique(upper_tri(model.model))) != 1, "degree of freedom is 0!"

    # Calculate RDM
    if type(rdms) == type(None):
        SL_RDM = searchlight_RDM(betas=datas, 
                                 conditions=conditions,
                                 mask=mask,
                                 save_rdm_path=save_rdm_path,
                                 save_region_path = region_path,
                                 radius=radius,
                                 threshold=threshold,
                                 method=rdm_distance_method,
                                 load_rdm = load_rdm)
    else:
        SL_RDM = rdms


    # Make models
    cp_models = []
    fixed_models = []
    for i, model in enumerate(models):
        # Filter if the rsa result exists already.
        save_rsa_path = os.path.join(save_corr_brain_dir_path, 
                                     str_join([eval_method, model.name]) + ".nii.gz")
        
        if os.path.exists(save_rsa_path):
            print("RSA result exists aready!", save_rsa_path)
            continue
        
        # Stack fixed model
        if str(type(model)) == str(RDM_model):
            t_model = ModelFixed(model.name, upper_tri(model.model))
        elif str(type(model)) == str(ModelFixed):
            t_model = model
        
        fixed_models.append(t_model)
        cp_models.append(model)
        
    # Get rsa result between RDM and model
    eval_results = evaluate_models_searchlight(sl_RDM = SL_RDM,
                                               models = fixed_models,
                                               eval_function = eval_fixed,
                                               method = eval_method,
                                               n_jobs = n_jobs)
    
    eval_results = [e.evaluations for e in eval_results]
    
    eval_score = np.array(list(map(lambda x: x.reshape(-1), eval_results)))
    scores = np.array(list(map(lambda score: score.reshape(-1), eval_score)))
    
    # Make RDM Brains
    corr_brains = []
    for model_index in range(0, len(cp_models)):
        # Create an 3D array, with the size of mask, and 
        x, y, z = mask.shape
        corr_brain = np.zeros([x*y*z])
        corr_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = scores[:,model_index]
        corr_brain = corr_brain.reshape([x, y, z])
        corr_brains.append(corr_brain)
    
    # save rsa result which contains brain RDM 
    if save_corr_brain_dir_path != None:
        os.makedirs(save_corr_brain_dir_path, exist_ok=True)
        
        for model_i in range(0, len(cp_models)):
            model = cp_models[model_i]  
            
            save_corr_path = os.path.join(save_corr_brain_dir_path, str_join([eval_method, model.name]) + ".nii.gz")
            
            corr_img = nb.Nifti1Image(corr_brains[model_i], affine = mask.affine)
            print(save_corr_path)
            nb.save(corr_img, save_corr_path)  
    
    # result
    result = {}
    for model_i in range(0, len(cp_models)):
        model = cp_models[model_i]
        result[model.name] = corr_brains[model_i]
        
    if return_type == "rdm_brain":
        return result
    else:
        return eval_results

def make_3darray_from_Indexes(data, voxel_indexes, shape_3d):
    """
    Create a 3D array from 1D data array using voxel indexes to position the elements in the 3D array.
    
    :param data(np.array - shape: 1d): data of elements
    :param voxel_indexes(np.array - shape: 1d): index in corresponding with data
    :param shape_3d(tuple): 3d shape
    
    return (np.array - 3d shape)
    """
    array_3d = np.zeros(shape_3d)
    index_3ds = np.unravel_index(voxel_indexes, shape_3d)
    array_3d[index_3ds] = data

    return array_3d

def make_RDM_brain(brain_shape, RDMs, conditions, is_return_1d=False):
    """
    Make RDM brain (nx, ny, nz, n_condition x n_condition)
    
    :param brain_shape: x,y,z(tuple)
    :param RDMs: rsatoolbox.rdm.RDMs
    :param conditions: unique conditions
    
    return 5d array(x, y, z, condition, condition)
    """
    assert type(RDMs[0]) == rsatoolbox.rdm.RDMs, "Please input rsatoolbox RDMs"
    
    condition_length = len(conditions)
    
    x, y, z = brain_shape
    brain_1d = list(np.zeros([x * y * z]))
    brain_1d_RDM = list(map(lambda _: np.repeat(0, condition_length * condition_length).reshape(condition_length, 
                                                                                                condition_length).tolist(), 
                        brain_1d))
    
    for RDM in RDMs:
        voxel_index = RDM.rdm_descriptors["voxel_index"]
        rdm_mat = RDM.get_matrices()
    
        assert len(voxel_index) == 1 and len(rdm_mat) == 1, "multi voxel index is occured"
        
        voxel_index = voxel_index[0]
        rdm_mat = rdm_mat[0]
        
        brain_1d_RDM[voxel_index] = rdm_mat.tolist()
    
    if is_return_1d:
        return brain_1d_RDM
    else:
        return np.array(brain_1d_RDM).reshape([x, y, z, condition_length, condition_length])
    
    return brain_1d_RDM

def brain_total_dissimilarity(rdm_brain):
    """
    Get total dissimilarity from rdm_brain
    
    :param rdm_brain: 5d array(array) made by make_RDM_brain function
    
    return np.array(nx, ny, nz)
    """
    nx, ny, nz = rdm_brain.shape[0], rdm_brain.shape[1], rdm_brain.shape[2]

    result = np.zeros([nx,ny,nz])
    for i in range(nx):
        for j in range(ny):
            for z in range(nz):
                result[i][j][z] = total_RDM_dissimilarity(rdm_brain[i][j][z])
    return result

def masked_rdm_brain(rdm_brain, nifti_mask, debug=None):
    """
    Apply mask to RDM brain
    
    :param rdm_brain: 5d array(array)
    :param nifti_mask: nifti
    
    return masked_data_only
    """
    rdm_shape = rdm_brain.shape
    rdm_brain_1d = rdm_brain.reshape(-1, rdm_shape[3], rdm_shape[4])
    mask_data_1d = nifti_mask.get_fdata().reshape(-1)
    
    if debug != None:
        # rdm_brain과 nifti mask를 1차원으로 축약해서 mask를 씌워도
        # 같다는 공간이라는 것을 보이기 위함
        test = np.sum(np.sum(rdm_brain_1d, axis=1), axis = 1)
        
        for i in range(0, len(mask_data_1d)):
            if mask_data_1d[i] == True:
                test[i] = np.sum(test[i])
            else:
                test[i] = 0
        return test
    
    masked_data_only = rdm_brain_1d[mask_data_1d > 0, :, :]
    
    return masked_data_only

def make_rdm(conditions, pair_values):
    """
    Make rdm using dataframe
    
    :param conditions: conditions(list)
    :param pair_values: (cond1, cond2, value)
    
    return dataframe
    """
    rdm = pd.DataFrame(index = conditions,
                       columns = conditions)
    
    for cond1, cond2, value in pair_values:
        rdm[cond1][cond2] = value
        rdm[cond2][cond1] = value
    
    for cond in conditions:
        rdm[cond][cond] = 0
        
    return rdm

def construct_contrast(design_matrix_columns, contrast_info):
    """
    construct contrast array
    
    :param design_matrix_columns: columns(list)
    :param contrast_info: dictionary
    
    return contrast array
    
    example)
    construct_contrast(["['1', '4', '2', '3', '1', '2', '4', '3']"], {"['1', '4', '2', '3', '1', '2', '4', '3']" : 1})
    
    """
    
    assert check_duplication(design_matrix_columns) != True, "column is duplicated!"

    candidates = np.zeros(len(design_matrix_columns))
    for condition in contrast_info:
        i_condition = design_matrix_columns.index(condition)
        candidates[i_condition] = contrast_info[condition]

    return candidates

def get_statsWithDM(fMRI_datas, 
                    dsg_mats, 
                    full_mask):
    """
    get stat values from dsg_mats
    
    :param fMRI_datas: fMRI_datas(list of nilearn)
    :param dsg_mats: design matrixes
    :param full_mask: full_mask
    
    return: stats, dsg_mats
    """
    # convert nilearn -> nltools
    brain_datas = [Brain_Data(data, mask=full_mask) for data in fMRI_datas]
    for run_index in range(0, len(brain_datas)):
        brain_datas[run_index].X = dsg_mats[run_index]
    
    # calculate Beta Values per run
    stats = []
    for run_index in range(0, len(brain_datas)):
        stats.append(brain_datas[run_index].regress())
        
    return stats, dsg_mats

def change_event_typeForLSA(events_):    
    """
    convert event_type for doing lsa
    
    :param events: list of event
    
    return dataframe
    """
    # events
    events = copy.deepcopy(events_)
    
    # info data
    info_datas = []
    for run_index in range(0, len(events)):
        info_data = {}
        for stimulus_type in np.unique(events[run_index]["trial_type"]):
            info_data[stimulus_type] = 0            
        info_datas.append(info_data)
    
    # Iterate all runs
    for run_index in range(0, len(events)):
        stimulus_column = events[run_index]["trial_type"]
        info_data = info_datas[run_index]

        temp_stimulus_column = [] # Saving event type converted from all event
        
        # Iterate all stimulus
        for stimulus_i in range(0, len(stimulus_column)):
            stimulus = stimulus_column[stimulus_i]
            
            if stimulus == "+":
                temp_stimulus_column.append(stimulus + "_" + str(info_data[stimulus]))
                info_data[stimulus] = info_data[stimulus] + 1
            else:
                # Convert event stimulus condition type
                temp_stimulus_column.append(stimulus + "_" + str(info_data[stimulus]))
                info_data[stimulus] = info_data[stimulus] + 1
        
        # Assign converted event type
        events[run_index]["trial_type"] = temp_stimulus_column
            
    return events

def change_event_typeForLSS(lsa_event, rest_condition, target_condition, trial_index):
    """
    Change event type to do LSS
    
    :param lsa_event: event(dataframe)
    :param rest_condition: rest condition
    :param target_condition: condition
    :param trial_index: index
    
    return event
    """
    event = lsa_event.copy()
    target_conditionWithTrial = str_join([target_condition, str(trial_index)])
    
    # Rest to Nuisance Rest
    df = change_df(event, "trial_type", lambda t_type: "Nuisance_Rest" if t_type != target_conditionWithTrial and rest_condition in t_type else t_type)
    
    # 
    df = change_df(df, "trial_type", lambda t_type: "Nuisance_Move" if t_type != target_conditionWithTrial and t_type != "Nuisance_Rest" else t_type)
    
    return df

def compare_design_mats(design_mats1, design_mats2, mat1_description="", mat2_description=""):
    """
    Compare design matrices between design_mats1 and design_mats2 to use drawing matrix.
    
    :param design_mats1: design matricies(list - df)
    :param design_mats2: design matricies(list - df)
    :param mat1_description: (string)
    :param mat1_description: (string)
    """
    assert len(design_mats1) == len(design_mats2), "Please match list size"
    
    run_length = len(design_mats1)
    
    fig, axes = plt.subplots(nrows=2, ncols=run_length) # nrow=2: (origin, parametric modulation)
    fig.set_size_inches(30, 30)

    dm1_axis_index = 0
    dm2_axis_index = 1
    
    for run_index in range(run_length):
        dm1_axis = axes[dm1_axis_index][run_index]
        plot_design_matrix(design_mats1[run_index], ax = dm1_axis)
        dm1_axis.set_title(str_join([mat1_description, str(run_index + 1)]))

    for run_index in range(run_length):
        dm2_axis = axes[dm2_axis_index][run_index]
        plot_design_matrix(design_mats2[run_index], ax = dm2_axis)
        dm2_axis.set_title(str_join([mat2_description, str(run_index + 1)]))

def compare_design_mats_hemodynamic(design_mats1, 
                        design_mats2, 
                        conditions,
                        mat1_description = "design1", 
                        mat2_description = "design2",
                        ylim = (-0.5, 5)):
    """
    Compare design matrices between design_mats1 and design_mats2 to use drawing hemodynamic response.
    
    :param design_mats1: design matricies(list - df)
    :param design_mats2: design matricies(list - df)
    :param conditions: conditions(list)
    :param mat1_description: (string)
    :param mat1_description: (string)
    :param ylim: limination of y-axis(tuple)
    """
    assert len(design_mats1) == len(design_mats2), "Please match list size"
    
    run_length = len(design_mats1)
    
    fig, axes = plt.subplots(nrows=run_length, ncols=2)
    fig.set_size_inches(30, 20)

    for run_index in range(run_length):
        axis_index = 0
        """
        Design Matrix1
        """
        dsg_mat1 = design_mats1[run_index]
        move_conditions_data = np.array(list(map(lambda condition: dsg_mat1[str(condition)].to_numpy(), conditions))).T
        move_legends = list(map(lambda condition: str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = move_conditions_data)
        axis_index += 1

        """
        Design Matrix2
        """
        dsg_mat2 = design_mats2[run_index]
        move_conditions_data = np.array(list(map(lambda condition: dsg_mat2[str(condition)].to_numpy(), conditions))).T
        move_legends = list(map(lambda condition: str(condition), conditions))
        plot_timeseries(axis = axes[run_index][axis_index], 
                        data = move_conditions_data)
        axis_index += 1

    plt.tight_layout()

def searchlight_with_beta(Xs, 
                          Ys, 
                          full_mask, 
                          subj_name, 
                          searchlight_dir_path, 
                          n_jobs = 1, 
                          radius=6, 
                          estimator = "svc",
                          prefix = ""):
    """
    Do searchlight Decoding analysis using beta values
    
    :param Xs: list of nifti image(list - shape (#x, #y, #z, #conditions)) seperated by run, 
    :param Ys: list of label(list) seperated by run ex) [ [condition1, condition1, condition2, condition2], [condition1, condition1, condition2, condition2] ]
    :param full_mask: full_mask(nifti image)
    :param searchlight_dir_path: directory path where the result is located.
    :param n_jobs: n_jobs
    :param radius: radius
    :param estimator: ‘svr’, ‘svc’, or an estimator object implementing ‘fit’
    :param prefix: save file prefix(string)
    
    return searchlight_obj
    """
    
    start = datetime.datetime.now()
    print(start)

    cv = GroupKFold(len(Xs))
    
    groups = []
    for run_i in range(0, len(Xs)):
        for _ in range(0, Xs[run_i].shape[-1]):
            groups.append(run_i)
    groups = np.array(groups)

    Xs = image.concat_imgs(Xs)
    Ys = np.concatenate(Ys)
    
    preproc_np_data = np.nan_to_num(zscore(Xs.get_fdata(), axis=-1)) 
    Xs = nb.Nifti1Image(preproc_np_data, 
                        full_mask.affine, 
                        full_mask.header)
    
    if estimator == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        estimator_clf = LinearDiscriminantAnalysis()
    else:
        estimator_clf = estimator
        
    # Make Model
    searchlight = nilearn.decoding.SearchLight(
        full_mask,
        radius=radius, 
        n_jobs=n_jobs,
        verbose=False,
        cv=cv,
        estimator = estimator_clf,
        scoring="balanced_accuracy")

    # Fitting Model
    searchlight.fit(imgs=Xs, y=Ys, groups=groups)
    
    end = datetime.datetime.now()
    
    # Save
    save_file_name = str_join([prefix, subj_name, estimator, "searchlight_clf"], deliminator = "_")   
    save(searchlight, os.path.join(searchlight_dir_path, save_file_name))
                 
    score_img = image.new_img_like(ref_niimg = full_mask, data = searchlight.scores_)
    nb.save(score_img, os.path.join(searchlight_dir_path, save_file_name + ".nii"))
    
    print(start, end)
    
    return searchlight

def apply_func_rdmBrain(rdm_brain, func):
    """
    Apply function to rdm brain
    
    :param rdm_brain: rdm_brain(list) - shape (nx, ny, nz, n_cond, n_cond)
    :param func: function to apply rdm
    
    return list(matched with brain shape)
    """
    nx, ny, nz = rdm_brain.shape[0], rdm_brain.shape[1], rdm_brain.shape[2]

    result = np.zeros([nx,ny,nz]).tolist()
    for i in range(nx):
        for j in range(ny):
            for z in range(nz):
                result[i][j][z] = func({"rdm" : rdm_brain[i][j][z]})
                
    return result

def get_uniquePattern(conds):
    """
    Get unique pattern from conditions
    
    :param conds: conditions(list)
    
    return unique pattern of conditions(keep ordering)
    """
    
    return list(dict.fromkeys(conds))

def sort_rdm(rdm_array, origin_conditions, reordered_conditions):
    """
    Sort rdm by reordered_conditions
    
    :param rdm_array: rdm array(2d array)
    :param origin_conditions: list of condition(1d list)
    :param reordered_conditions: list of condition(1d list)
    
    retrun sorted rdm(2d array)
    """
    cond_length = len(origin_conditions)
    
    pattern_info = dict(zip(origin_conditions, np.arange(0, cond_length)))
    
    re_order_indexes = [pattern_info[cond] for cond in reordered_conditions]
    
    origin_axis1, origin_axis2 = np.meshgrid(origin_conditions, origin_conditions)
    convert_axis1, convert_axis2 = np.meshgrid(reordered_conditions, reordered_conditions)
    
    orgin_grid = np.zeros((cond_length, cond_length)).tolist()
    sorted_grid = np.zeros((cond_length, cond_length)).tolist()
    result_grid = np.zeros((cond_length, cond_length)).tolist()
    
    for i in range(cond_length):
        for j in range(cond_length):
            orgin_grid[i][j] = (origin_axis2[i][j], origin_axis1[i][j])
            sorted_grid[i][j] = (convert_axis2[i][j], convert_axis1[i][j])

    for i in range(cond_length):
        for j in range(cond_length):
            target_pair = sorted_grid[i][j]
            target_array = np.array(recursive_mapWithDepth(orgin_grid, 
                                                           lambda x: x == target_pair, 
                                                           1))

            x_indexes, y_indexes = np.where(target_array == True)
            assert len(x_indexes) == 1 and len(y_indexes) == 1, "Please check duplicate"
            x_i = x_indexes[0]
            y_i = y_indexes[0]

            result_grid[i][j] = rdm_array[x_i][y_i]
            
    return np.array(result_grid)

def filter_rdm(rdm, cond_origin, cond_target):
    """
    filter rdm from target condition
    
    :param rdm (np.array)
    :param cond_origin (list)
    :param cond_target (list)

    return rdm(np.array)
    """
    cp_cond = cond_origin.copy()
    for e in cond_target:
        cp_cond.remove(e)
    
    sorted_rdm = sort_rdm(rdm, cond_origin, cond_target + cp_cond)
    
    n_target_cond = len(cond_target)
    return sorted_rdm[:n_target_cond, :n_target_cond]
    
"""
Related to Multivariate Noise Normalization
"""
def calc_precision_mat_basic(run_length, residuals, mask, residual_cov_method = "shrinkage_diag"):
    """
    calculate precision matrix from residual per each (run or session)
    
    :param run_length: run lengh of mri
    :param residuals: residuals(list of Brain_Data(run or session))
    :param mask: nifti mask
    :param residual_cov_method: (string) ex) diag, full, shrinkage_eye, shrinkage_diag
    
    return noise preicision matrix (voxel x voxel)
    """
    select_data = 0
    select_first = 0

    # Get Precision matrixes
    noise_precision_mats = []
    for r_i in range(run_length):
        target_residual = residuals[r_i]

        # masking data to reduce data size, if voxel count is too high, covariance matrix will not calculated.
        if type(target_residual) != nb.Nifti1Image:
            target_residual = target_residual.to_nifti()
        masked_data = apply_mask_change_shape([target_residual], mask)[select_data][select_first]
        
        print(masked_data.shape)
        
        # noise covariance matrix
        noise_precision_mat = prec_from_residuals(masked_data, method = residual_cov_method)

        noise_precision_mats.append(noise_precision_mat)
    return noise_precision_mats

def construct_dataset(subj_number, beta_values, run_length, conditions):
    """
    construct rsatoolbox dataset 
    
    :param subj_number: subject number
    :param beta_values: beta values(list, element(beta array): per run or session, shape is like (n_conds, n_voxels))
    :param run_length: run length
    :param conditions: condition per trial corresponding to each element of beta_values (list), This value needs to match with all other runs.(1d list)
    
    Conditions
        ex)
           [
               cond1,
               cond2,
               cond3,
               ...
            ]
            
    return: dataset
    """
    measurements = np.concatenate(beta_values)

    # measurement에 대응하는 피험자 정보
    des = {'subj': subj_number}

    # 각 measurement의 row에 대응하는 정보
    sessions = np.repeat([i+1 for i in range(run_length)], len(conditions))
    conds = np.tile(conditions, run_length)
    obs_des = {'conds': conds, 'sessions': sessions}

    # voxel description
    nVox = beta_values[0].shape[1]
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}

    # make dataset
    dataset = rsatoolbox.data.Dataset(measurements=measurements,
                       descriptors=des,
                       obs_descriptors=obs_des,
                       channel_descriptors=chn_des)
    
    return dataset

def calc_crossnobis(subj_number,
                    masked_betas, 
                    masked_residuals, 
                    conditions, 
                    residual_cov_method = "shrinkage_diag"):
    """
    Calculate crossnobis
    
    :param subj_number(int): subject number ex) 1
    :param masked_betas(list - (Brain_Data - shape: (#cond, #voxels): beta values 
    :param masked_residuals(list - (Brain_Data - shape: (#data, #voxels)): beta residual
    :param conditions(np.array - 1d): condition of betas ex) ["a", "b", "c", "d", "a", "b", "c", "d"]
    :param residual_cov_method(string): residual covariance method ex) shrinkage_eye
    """
        
    """
    Step 1: Make dataset
    """
    # Session
    sessions = []
    for sess_i in np.arange(len(masked_betas)):
        sess_number = sess_i + 1
        n_cond = masked_betas[sess_i].data.shape[0]
    
        session_numbers = np.repeat(sess_number, n_cond)
        sessions.append(session_numbers)
    sessions = np.array(sessions).reshape(-1)

    # measurment information
    masked_betas = nltools.utils.concatenate(masked_betas)
    measurements = masked_betas.data

    residual_measurements = []
    for r_i in range(len(masked_residuals)):
        residual_measurements.append(masked_residuals[r_i].data)
    
    des = {'subj': subj_number}
    obs_des = {'conds': conditions, 'sessions': sessions}
    nVox = measurements.shape[-1]
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}

    dataset = Dataset(measurements = measurements,
                      descriptors = des,
                      obs_descriptors = obs_des,
                      channel_descriptors = chn_des)
    """
    Step 2: Calculate precision matrix
    """
    noise_precision_mats = []
    for r_i in range(len(residual_measurements)):
        noise_precision_mat = prec_from_residuals(residuals = residual_measurements[r_i],
                                                  method = residual_cov_method)
        noise_precision_mats.append(noise_precision_mat)

    
    """
    Step 3: Calculate crossnobis distance
    """
    rdm_crossnobis = calc_rdm(dataset = dataset, 
                              descriptor = 'conds', 
                              method = 'crossnobis',
                              noise = noise_precision_mats,
                              cv_descriptor = 'sessions')
    
    return rdm_crossnobis
    
def calc_roi_crossnobis(subj_number,
                        betas, 
                        conditions, 
                        sessions,
                        run_length,
                        residuals, 
                        roi_mask, 
                        roi_name, 
                        full_mask, 
                        residual_cov_method = "shrinkage_diag"):
    """
    Calculate crossnobis based on roi img
    
    :param subj_number(int): subject number ex) 1
    :param betas(Brain_Data - shape: (#cond, #voxels): beta values 
    :param conditions(np.array - 1d): condition of betas ex) ["a", "b", "c", "d", "a", "b", "c", "d"]
    :param sessions(np.array - 1d): session of beta ex) [1,1,1,1, 2,2,2,2, 3,3,3,3]
    :param run_length(int): the number of run
    :param residuals(list - Brain_Data): beta residual
    :param roi_mask(nifti - shape: (nx, ny, nz)): mask img 
    :param roi_name(string): roi_name ex) "cuneate nucleus"
    :param full_mask(nifti - shape: (nx, ny, nz)): full mask img
    :param residual_cov_method(string): residual covariance method ex) shrinkage_eye
    """
    
    """
    Step 1: Apply roi mask
    """
    mask = Brain_Data(roi_mask, mask = full_mask)
    measurements = betas.apply_mask(mask).data
    
    masked_residuals = []
    for r_i in range(run_length):
        target_residual = residuals[r_i].apply_mask(mask)
        masked_residuals.append(target_residual.data)
        
    """
    Step 2: Make dataset
    """
    # measurment information
    des = {'subj': subj_number}
    obs_des = {'conds': conditions, 'sessions': sessions}
    nVox = measurements.shape[-1]
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}

    dataset = Dataset(measurements = measurements,
                      descriptors = des,
                      obs_descriptors = obs_des,
                      channel_descriptors = chn_des)
    """
    Step 3: Calculate precision matrix
    """
    noise_precision_mats = []
    for r_i in range(run_length):
        noise_precision_mat = prec_from_residuals(residuals = masked_residuals[r_i],
                                                  method = residual_cov_method)
        noise_precision_mats.append(noise_precision_mat)

    
    """
    Step 4: Calculate crossnobis distance
    """
    rdm_crossnobis = calc_rdm(dataset = dataset, 
                              descriptor = 'conds', 
                              method = 'crossnobis',
                              noise = noise_precision_mats,
                              cv_descriptor = 'sessions')
    
    return rdm_crossnobis

    
def calc_crossnobis_with_prewhitening(subj_number, 
                                      beta_values, 
                                      residuals, 
                                      mask, 
                                      run_length, 
                                      conditions,
                                      residual_cov_method = "shrinkage_diag"):
    """
    calculate crossnobis distance with prewhitening
    
    :param beta_values: beta values of subject(list of Brain_Data(run or session))
    :param residuals: residuals(list of Brain_Data(run or session))
    :param mask: nifti mask
    :param run_length: run length(int)
    :param conditions: condition(list)
    :param residual_cov_method: (string) ex) diag, full, shrinkage_eye, shrinkage_diag
    
    return rdm matrix (condition x condition)
    """
    select_data = 0
    
    # Apply ROI
    masked_beta = apply_mask_change_shape([beta.to_nifti() for beta in beta_values], mask)[select_data]

    # Calc precision matrix
    noise_precision_mats = calc_precision_mat_basic(run_length, residuals, mask, residual_cov_method)
    
    # Construct dataset
    dataset = construct_dataset(subj_number, masked_beta, run_length, conditions)
    
    # calculate crossnobis distance
    rdm_crossnobis = rsatoolbox.rdm.calc_rdm(dataset, 
                                             descriptor='conds', 
                                             method='crossnobis',
                                             noise=noise_precision_mats,
                                             cv_descriptor='sessions')
    
    return rdm_crossnobis

def check_condition_withinRun(run_conditions, population_conds):
    """
    Check there is no trial per condition within a run
    
    Zero trial per condition can happen when trials are dropped multiple times within run due to condition filtering.
    
    :param conditions: conditions per Run(list)
    ex)
    [array(['a', 'd', 'e', 'g', 'm', 'o', 'p', 's', 'u', 'z'], dtype=object), 
    array(['a', 'd', 'e', 'g', 'm', 'o', 's', 'u', 'z'], dtype=object),
    array(['a', 'd', 'e', 'g', 'm', 'o', 'p', 's', 'u', 'z'], dtype=object)]
    
    :param population_conds: condition's population(list) 
    ex) ["a", "b" ,"c", "d", "e", "f", "g"]
        
    return no trial condition(list)
    """
    
    no_trial_cond = []
    for run_index in range(len(run_conditions)):
        uq_cond_perRun = np.unique(run_conditions[run_index])

        for cond in population_conds:
            if len(uq_cond_perRun[cond == uq_cond_perRun]) == 0:
                no_trial_cond.append(cond)

    return no_trial_cond

def convert_data_type(data, full_mask, convert_type = "nifti"):
    """
    Convert brain data type
    
    :param data: target data to be converted
    :param convert_type: to_type (nifti, brain_data, full_array, masked_array)
    :param full_mask: (nifti1Image)
    
    return converted brain data(convert_type)
    """
    nx, ny, nz = full_mask.shape
    
    if type(data) == nb.Nifti1Image:
        if convert_type == "nifti":
            return data
        elif convert_type == "brain_data":
            return Brain_Data(data, mask = full_mask)
        elif convert_type == "masked_array":
            masked_array = apply_mask_change_shape([data], full_mask)[0][0]
            return masked_array
    elif type(data) == Brain_Data:
        if convert_type == "nifti":
            return data.to_nifti()
        elif convert_type == "brain_data":
            return data
        elif convert_type == "masked_array":
            return data.data
    elif type(data) == np.ndarray:
        if convert_type == "nifti":
            if len(data.shape) > 1:
                n_point = data.shape[0]
            else:
                n_point = 1
                
            mask_3d_indexes = np.where(full_mask.get_fdata() == 1)
            
            if n_point == 1:
                mask_array = np.repeat(0, nx * ny * nz).reshape((nx, ny, nz)).astype(np.float64)
                mask_array[mask_3d_indexes] = data
                image = nb.Nifti1Image(mask_array, full_mask.affine)
            else:
                mask_array = np.repeat(0, n_point * nx * ny * nz).reshape((nx, ny, nz, n_point)).astype(np.float64)
                for i in range(n_point):
                    n_v = len(mask_3d_indexes[0])
                    mask_array[mask_3d_indexes + (np.repeat(i, n_v),)] = data[i]
                image = nb.Nifti1Image(mask_array, full_mask.affine)
            return image
        elif convert_type == "brain_data":
            nifti_img = convert_data_type(data, full_mask, convert_type = "nifti")
            return Brain_Data(nifti_img, mask = full_mask)
        elif convert_type == "masked_array":
            return data

def untangle_nii(nii_path, destination_dir_path = None, prefix = None, n_digit = None):
    """
    Untangle 4d nifti file to 3d nifti files
    
    :param nii_path: to be untangled(string) 
        -shape: (#x, #y, #z, #point)
    :param destination_dir_path: directory path to be saved(string)
    :param prefix: file name prefix to be saved(string)
    :param n_digit: indexing form(int) ex) if 2: 1 -> 01
    
    Save untangled files as file_name + d{index}
    """
    
    # path information
    p = pathlib.Path(nii_path)
    dir_path = p.parent
    file_name_stem = p.stem
    file_extension = p.suffix
    
    if file_extension == "":
        full_file_name = file_name_stem + ".nii"
    else:
        full_file_name = file_name_stem + file_extension
    
    # load nifti img
    nii_path = os.path.join(dir_path, full_file_name) 
    brain_imgs = nb.load(nii_path)
    
    # save format
    saving_file_name_format = "{file_name}_i{index}"
    
    # untangle nifti img (4d -> 3d)
    save_paths = []
    n_p = brain_imgs.shape[-1]
    
    for i in range(n_p):
        img = brain_imgs.slicer[..., i]
        
        if n_digit == None:
            i_str = str(i)
        else:
            i_str = sj_string.make_pad_fromInt(integer = i, n_digit = n_digit)
        
        if prefix != None:
            save_file_name = saving_file_name_format.format(file_name = prefix,
                                                            index = i_str)
        else:
            save_file_name = saving_file_name_format.format(file_name = file_name_stem,
                                                            index = i_str)
        
        if destination_dir_path != None:
            dir_path = destination_dir_path
        
        saving_file_path = os.path.join(dir_path, save_file_name)
        nb.save(img, saving_file_path)
        save_paths.append(saving_file_path)
        
        print(f"save: {saving_file_path}")
        
    return save_paths

def get_r2(fmri_signal, brain_beta, design_mat):
    """
    Get r2 about GLM model
    
    nx: #x
    ny: #y
    nz: #z
    nt: #time
    nr: #regressor
    
    :param fmri_signal: fmri signals(np.ndarray - shape: nx, ny, nz, nt)
    :param brain_beta: beta values(np.ndarray - shape: nx, ny, nz, nr)
    :param design_mat: (pd.DataFrame)
    
    return r2(np.array)
    """
    nx, ny, nz, nt = fmri_signal.shape

    # mean(Y)
    mean_signal = np.mean(fmri_signal, axis = 3)
    
    # y^
    n_regressor = design_mat.shape[-1]
    y_hat = brain_beta.reshape(-1, n_regressor) @ design_mat.to_numpy().T
    y_hat = y_hat.reshape(nx, ny, nz, -1)
    
    # R^2
    SS_residual = np.sum((fmri_signal - y_hat) ** 2, axis = 3)
    SS_total = np.sum((fmri_signal - np.expand_dims(mean_signal, axis = 3)) ** 2, axis = 3)
    r_squared = 1 - (SS_residual / SS_total)
    
    return r_squared

def get_roi_3d_indexes(roi_img, roi_value):
    """
    Get roi 3d indexes
    
    :param roi_img(nb.Nifti1Image): roi image where the value of roi is [roi_value]
    :param roi_value: value of roi
    
    return roi_3d_indexes(np.array - #voxel, (x_index, y_index, z_index))
    """
    roi_xs, roi_ys, roi_zs = np.where(roi_img.get_fdata() == roi_value)
    
    roi_xs = np.expand_dims(roi_xs, 1)
    roi_ys = np.expand_dims(roi_ys, 1)
    roi_zs = np.expand_dims(roi_zs, 1)
    
    roi_3d_indexes = np.concatenate([roi_xs, roi_ys, roi_zs], axis = 1)
    return roi_3d_indexes

def get_roi_1d_indexes(roi_img, roi_value):
    """
    Get roi 1d indexes
    
    :param roi_img(nb.Nifti1Image): roi image where the value of roi is [roi_value]
    :param roi_value: value of roi
    
    return roi_1d_indexes(np.array)
    """
    
    roi_3d_indexes = get_roi_3d_indexes(roi_img, roi_value)
    roi_1d_indexes = np.apply_along_axis(lambda index_3d: image3d_to_1d(index_3d, roi_img.shape), 1, roi_3d_indexes)
    return roi_1d_indexes

def make_1d_voxel_indexes(index_3ds, shape_3d):
    """
    Make 1d indexes from 3d indexes
    
    :param index_3ds(np.array - shape: (nx, ny, nz)): 3d index array
    :param shape(tuple): 3d shape
    
    return 1d indexes (np.array - 1d array)
    """
    index_1d = np.ravel_multi_index(index_3ds, mask.shape)
    return index_1d

if __name__ == "__main__":
    # highlight_stat
    result = sj_brain.highlight_stat(roi_array=motor_left_mask.get_data(),
                                     stat_array=np.load(
                                         "/Users/yoonseojin/statistics_sj2/CLMN/Replay_Exp/experiment/20210407_blueprint_0324v2/HR01/searchlight/preprocessed_2mm/HR01_searchlight_interest_10.npy"),
                                     stat_threshold=0.6)
    plotting.view_img(nb.Nifti1Image(result["data"], full_mask.affine, full_mask.header),
                      anat,
                      cmap=result["color_map"])
    
    # colored_roi_with_stat
    result = sj_brain.colored_roi_with_stat(
        roi_arrays=[mask_left_precentral_gyrus.get_fdata(), mask_occipital_cortex.get_fdata(),
                    mask_all_hippocampus.get_fdata()],
        stat_map=np.load(
            "/Users/yoonseojin/statistics_sj2/CLMN/Replay_Exp/experiment/20210407_blueprint_0324v2/HR01/searchlight/preprocessed_2mm/HR01_searchlight_interest_2.npy"),
        stat_threshold=0.60)

    plotting.view_img(nb.Nifti1Image(result["data"], full_mask.affine, full_mask.header),
                      anat,
                      cmap=result["color_map"])
    
    # upper_tri
    upper_tri(np.repeat(3, 9).reshape(3,3))
    
    # RSA
    a = RDM_model(np.array([1,0,1,0]).reshape(2,2), "transition", ["1","2"])
    RDM_brains = RSA(models=[a],
                     conditions=["!", "@", "#"],
                     full_mask=full_mask,
                     datas = beta_values,
                     save_rdm_path = os.path.join(output_dir_path, "rdm"),
                     save_corr_brain_path=os.path.join(output_dir_path, "corr_brain"),
                     n_jobs=3
                    )
    
    # make_RDM_brain
    make_RDM_brain(brain_shape, rdm)
    
    # sort_rdm
    rdm = np.array(
        [
            [0,2,3],
            [4,0,6],
            [7,8,0],
        ]
    )
    sort_rdm(rdm, ["A", "B", "C"], ["B", "A", "C"])
    
    # make_rdm
    make_rdm(conditions = ["a", "b", "c", "d"],
             pair_values = [(trial_cond1, trial_cond2, 1) for trial_cond1, trial_cond2 in list(itertools.combinations(["a","b","c","d"], 2))])
    
    # check_condition_withinRun
    check_condition_withinRun(["a", "b"], alphabet)
    
    # convert_data_type
    full_mask = nb.load("/mnt/sdb2/seojin/tutorial/preprocessed/full_mask.HP01.nii.gz")
    data = np.repeat(3, np.sum(full_mask.get_fdata() == 1))
    s = convert_data_type(data, full_mask, convert_type = "brain_data").data
    
    
    a = np.array([
        [0,4,7],
        [4,0,8],
        [7,8,0]
    ])
    filter_rdm(rdm = a, cond_origin = [1,2,3], cond_target = [2,3])
    
    untangle_nii("nifti_path")
    
    rdm_model = RDM_model(a, 
                          model_name = "abc",
                          conditions = ["1", "2", "3"])
    rdm_model.draw()

    # Roi index
    roi_img = nb.load("/mnt/sdb2/DeepDraw/mri_mask/targetROIs/Lt_BA6_ventrolateral.nii.gz")
    roi_3d_indexes = get_roi_3d_indexes(roi_img, 1)
    roi_1d_indexes = get_roi_1d_indexes(roi_img, 1)
    
    # Indexing
    roi_img = nb.load("/mnt/sdb2/DeepDraw/mri_mask/targetROIs/Lt_BA6_ventrolateral.nii.gz")
    index_3d = make_3darray_from_Indexes(data = [1], voxel_1d_indexes = [10, 20], shape_3d = roi_img.shape)
    
    index_2d = make_1d_voxel_indexes(index_3d, mask.shape)
    