
import os
import sys
import cupy as cp
import numpy as np
import matplotlib.pylab as plt
from rsatoolbox.rdm import concat, RDMs
from rsatoolbox.data.noise import prec_from_residuals
from itertools import product, combinations, permutations
from mpl_toolkits.axes_grid1 import make_axes_locatable

def slice_list_usingDiff(data):
    """
    Slice list when a different value happens
    
    :param data: (list)
    
    return [(start index, stop index)]
    """
    if len(data) == 0:
        return []
    elif len(data) == 1:
        return [(data[0],1)]
    
    check_elements = []
    lengths = []
    
    length = 1
    for i, element in enumerate(data):
        if len(check_elements) == 0:
            length = 1
            check_elements.append(element)
        elif i == len(data) - 1:
            if element == check_elements[-1]:
                length += 1
                lengths.append(length)
            else:
                lengths.append(length)
                
                check_elements.append(element)
                length = 1
                lengths.append(length)
            
        elif element != check_elements[-1]:
            lengths.append(length)

            length = 1
            check_elements.append(element)
        else:
            length += 1
        
    start_indexes = []
    stop_indexes = []
    
    start_index = 0
    for element, length in list(zip(check_elements, lengths)):
        start_indexes.append(start_index)
        
        stop_index = start_index + length - 1
        stop_indexes.append(stop_index)
        
        start_index = stop_index + 1
    return list(zip(start_indexes, stop_indexes))
    
def make_1dRDM(RDM):
    """
    upper_tri returns the upper triangular index of an RDM
    
    :param RDM: squareform RDM(numpy array)
    
    return upper triangular vector of the RDM(1D array) 
    """
    
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

def pattern_separation(RDM):
    """
    total dissimilarity 
    
    :param RDM: RDM(numpy 2d array)
    
    return: scalar value
    """
    
    up_RDM = upper_tri(RDM)
    return np.mean(up_RDM)

def make_2dRDM_from_1dRDM(triangle_1d, mat_dim):
    """
    make rdm from triangle values
    
    :param triangle_1d: upper triangle values(list - 1d)
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
        matrix[c_i, r_i] = e
    
    return matrix

class RDM_model:
    """
    This class's purpose is managing RDM model
    """
    
    def __init__(self, 
                 dissimilarities: np.array, 
                 model_name: str, 
                 conditions: list):
        """
        :param dissimilarities: dissimilarity matrix (2d or 1d) 
        :param model_name: model name
        :param conditions: conditions
        """
        n_shape = len(dissimilarities.shape)
        if n_shape == 2:
            rdm_2d = dissimilarities
        elif n_shape == 1:
            rdm_2d = make_2dRDM_from_1dRDM(dissimilarities, len(conditions))

        self.dissimilarities = rdm_2d
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
        
        RDM_model.draw_rdm(rdm = self.dissimilarities, 
                           conditions = self.conditions, 
                           fig = fig,
                           axis = axis,
                           style_info = style_info)
        return fig, axis
    
    @staticmethod
    def draw_rdm(rdm: np.array, 
                 conditions: list, 
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
        
        xlocs = [(start_i + end_i)/2 for start_i, end_i in slicing_indexes]
        ylocs = [(start_i + end_i)/2 for start_i, end_i in slicing_indexes]
        axis.set_xticks(xlocs, 
                        rdm_conditions,
                        size = tick_size,
                        weight = tick_weight,
                        rotation = x_tick_rotation,
                        minor = False)
        axis.set_yticks(ylocs, 
                        rdm_conditions,
                        size = tick_size,
                        weight = tick_weight,
                        minor = False)
       
        # Spine
        for spine in axis.spines.values():
            spine.set_visible(False)
    
        # Title
        axis.set_title(title, weight = title_wight, size = title_size)
        return fig, axis

def selection_RDM(conditions, selection_func):
    """
    Make selection RDM to filter dissimilarities

    :param conditions(list): condition of RDM ex) ["a", "b", "c", "a"]
    :param selection_func(function): selection function ex) lamba x: x in ["a"]

    return RDM_model
    """
    rdm = np.tile(False, (len(conditions), len(conditions)))
    indices = np.where([selection_func(cond) for cond in conditions])[0]
    for r, c in list(permutations(indices, 2)):
        rdm[r,c] = True
    rdm_model = RDM_model(dissimilarities = rdm, model_name = "selection", conditions = conditions)
    return rdm_model

def sort_rdm(rdm_array, origin_conditions, reordered_conditions):
    """
    Sort RDM according to reordered_conditions.
    
    - :param rdm_array(np.array - 2D): 2D numpy array (square RDM matrix)
    - :param origin_conditions(list): list of original condition labels
    - :param reordered_conditions(list): list of condition labels in new order
    
    - return sorted RDM: 2D numpy array with rows/columns reordered
    """
    origin_conditions = np.array(origin_conditions)
    reordered_conditions = np.array(reordered_conditions)
    assert np.alltrue(np.array([sum(origin_conditions == cond) for cond in np.unique(origin_conditions)]) == 1), "Condition is duplicated"
    assert np.alltrue(np.array([sum(reordered_conditions == cond) for cond in np.unique(reordered_conditions)]) == 1), "Condition is duplicated"
    
    # Map from condition name to index
    condition_to_index = {cond: idx for idx, cond in enumerate(origin_conditions)}
    
    # Get new order of indices
    new_indices = [condition_to_index[cond] for cond in reordered_conditions]
    
    # Reorder rows and columns
    return rdm_array[np.ix_(new_indices, new_indices)]

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

def make_RDM(conditions, dissim_mapping_func):
    """
    Make RDM using dissimilarity mapping function

    :param conditions(list - 1d): condition of RDM
    :param dissim_mapping_func(function): dissimilarity function

    return RDM_model
    """
    row_conditions = list(conditions)
    column_conditions = list(conditions)
    rdm = np.tile(0, (len(conditions), len(conditions)))
    for r_cond, c_cond in product(row_conditions, column_conditions):
        if r_cond == c_cond:
            continue
        rdm[row_conditions.index(r_cond), column_conditions.index(c_cond)] = dissim_mapping_func(r_cond, c_cond)
    rdm_model = RDM_model(dissimilarities = rdm, model_name = "temp", conditions = conditions)
    return rdm_model

def upper_tri_index_pairs(dim: int, k: int = 1):
    """
    Get index mapping for upper-triangular pairs of a square matrix.

    This function returns a dictionary mapping each (row, column) pair
    in the upper triangle (above the main diagonal, determined by k)
    to a unique sequential index (0, 1, 2, ...).

    Parameters
    ----------
    :param dim: The dimension of the square matrix (e.g., 4 for a 4x4 matrix).
    :param k: optional
        Diagonal offset:
        - k = 0 → include the main diagonal and above
        - k = 1 → exclude the main diagonal (use only above-diagonal elements)
        - k > 1 → start further above the diagonal
        (Default: 1)

    Example
    -------
    For dim = 4, k = 1:
        (0,1), (0,2), (0,3)  -> 0, 1, 2
        (1,2), (1,3)         -> 3, 4
        (2,3)                -> 5

    return: Mapping from (row, column) pair to its sequential index.
    """
    r, c = np.triu_indices(dim, k=k)
    pair_to_idx = {(ri, ci): i for i, (ri, ci) in enumerate(zip(r, c))}
    return pair_to_idx

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

def make_rdm_withPairs(conditions, pair_values):
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

def calc_rdm_crossnobis_subSession(betas: np.ndarray, 
                                   residuals: np.ndarray, 
                                   residual_types: np.ndarray,
                                   conditions: np.ndarray, 
                                   sessions: np.ndarray, 
                                   subSessions: np.ndarray, 
                                   masks: np.ndarray,
                                   shrinkage_method) -> RDMs:
    """
    Calculate rdm crossnobis manually

    :param betas(shape: (#cond, #whole_brain_channel)): beta values
    :param residuals(shape: (#source, #time, #whole_brain_channel)): beta values
    :param residual_types(shape: #source): source information per residual
    :param conditions(shape: #cond): corresponding condition per beta
    :param sessions(shape: #cond): session per cond
    :param subSessions(shape: #cond): sub session per cond
    :param masks(shape: (#mask, #whole_brain_channel)): masks
    :param shrinkage_method: shrinkage method to calculate covariance
    
    return: RDMs
    """
    uq_conditions = np.array(list(dict.fromkeys(conditions)))
    uq_sessions = np.array(list(dict.fromkeys(sessions)))
    uq_subSessions = np.array(list(dict.fromkeys(subSessions)))

    rdm_crossnobis_manually = []
    for i, mask in enumerate(masks):
        # Masking
        masked_betas = betas[:, mask]
        n_cond, n_mask_channel = masked_betas.shape
        masked_residuals = residuals[:, :, mask]
    
        # Precision matrix
        noise_precision_mats = np.array(prec_from_residuals(masked_residuals, method = shrinkage_method))
        noise_precision_mats = cp.array(noise_precision_mats)
    
        # Sqrt - precision matrix
        prec_mat_sqrts = []
        for prec_mat in noise_precision_mats:
            w, Q = cp.linalg.eigh(prec_mat)
            w_sqrt = cp.sqrt(cp.clip(w, 0, None))
            prec_mat_sqrt = (Q * w_sqrt) @ Q.T
            prec_mat_sqrts.append(prec_mat_sqrt)
        prec_mat_sqrts = np.array([mat.get() for mat in prec_mat_sqrts])
        
        # Denoising
        denoised_masked_betas = []
        denoised_masked_conditions = []
        denoised_masked_sessions = []
        for session, subSession in product(uq_sessions, uq_subSessions):
            is_session = (sessions == session)
            is_subSession = (subSessions == subSession)
            is_target = np.logical_and(is_session, is_subSession)
        
            target_betas = masked_betas[is_target]
            target_conditions = conditions[is_target]
            
            source = f"{session}-{subSession}"
            source_precision = prec_mat_sqrts[residual_types == source][0]
            
            denoised_masked_beta = (target_betas @ source_precision)
            denoised_masked_betas.append(denoised_masked_beta)
            denoised_masked_conditions.append(target_conditions)
            denoised_masked_sessions.append(sessions[is_target]) 
        denoised_masked_betas = np.concatenate(denoised_masked_betas, axis = 0)
        denoised_masked_conditions = np.concatenate(denoised_masked_conditions)
        denoised_masked_sessions = np.concatenate(denoised_masked_sessions)
    
        # b_{i} - b{j}
        n_session = len(uq_sessions)
        n_dissim = len(list(combinations(uq_conditions, 2)))
        
        diff_betas = np.zeros((n_session, n_dissim, n_mask_channel))
        diff_conds = np.zeros((n_session, n_dissim)).astype(np.str_)
        for session_i, session in enumerate(uq_sessions):
            dissim_i = 0
            for cond1, cond2 in combinations(uq_conditions, 2):
                is_session = (denoised_masked_sessions == session)
                is_condition1 = (denoised_masked_conditions == cond1)
                is_condition2 = (denoised_masked_conditions == cond2)
        
                cond1_beta = np.mean(denoised_masked_betas[np.logical_and(is_session, is_condition1)], axis = 0)
                cond2_beta = np.mean(denoised_masked_betas[np.logical_and(is_session, is_condition2)], axis = 0)
                diff_beta = (cond1_beta - cond2_beta)
                
                diff_betas[session_i, dissim_i] = diff_beta
                diff_conds[session_i, dissim_i] = f"{cond1}&{cond2}"
                dissim_i += 1
    
        # cross-validation
        n_cv = len(list(combinations(range(len(uq_sessions)), 2)))
        cv_distance = np.zeros((n_cv, n_dissim))
        cv_conds = np.zeros((n_cv, n_dissim)).astype(np.str_)
        
        cv_i = 0
        for session1_i, session2_i in combinations(range(len(uq_sessions)), 2):
            dissim_i = 0
            for cond1, cond2 in combinations(uq_conditions, 2):
                diff_beta1 = diff_betas[session1_i][diff_conds[session1_i] == f"{cond1}&{cond2}"]
                diff_beta2 = diff_betas[session2_i][diff_conds[session2_i] == f"{cond1}&{cond2}"]
        
                assert len(diff_beta1), "Check"
                distance = diff_beta1[0] @ diff_beta2[0].T
        
                cv_distance[cv_i][dissim_i] = distance
                cv_conds[cv_i][dissim_i] = f"{cond1}&{cond2}"
                dissim_i += 1
            cv_i += 1
        cv_distance = cv_distance / n_mask_channel
    
        # crossnobis
        crossnobis_distance = np.mean(cv_distance, axis = 0)
        rdm = RDMs(crossnobis_distance, 
                   rdm_descriptors = { "index" : [i] },
                   pattern_descriptors = { "index" : np.arange(len(uq_conditions)), "cond" : uq_conditions })
        rdm_crossnobis_manually.append(rdm)
    rdm_crossnobis_manually = concat(rdm_crossnobis_manually)
    return rdm_crossnobis_manually
    
if __name__ == "__main__":
    rdm_model = RDM_model(a, 
                          model_name = "abc",
                          conditions = ["1", "2", "3"])
    rdm_model.draw()

    # sort_rdm
    rdm = np.array(
        [
            [0,2,3],
            [4,0,6],
            [7,8,0],
        ]
    )
    sort_rdm(rdm, ["A", "B", "C"], ["B", "A", "C"])
    filter_rdm(rdm = rdm, cond_origin = ["A","B","C"], cond_target = ["A","B"])
    make_RDM([1,2,3], lambda x, y: 1 if x == 1 else 0).draw()

    upper_tri_index_pairs(8)

    # make_RDM_brain
    make_RDM_brain(brain_shape, rdm)

    # make_rdm
    make_rdm_withPairs(conditions = ["a", "b", "c", "d"],
                       pair_values = [(trial_cond1, trial_cond2, 1) for trial_cond1, trial_cond2 in list(itertools.combinations(["a","b","c","d"], 2))])
    