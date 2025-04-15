
import numpy as np
import matplotlib.pylab as plt
from itertools import permutations
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
    
    def __init__(self, dissimilarities, model_name, conditions):
        """
        :param dissimilarities(np.array): dissimilarity matrix (2d or 1d) 
        :param model_name: model name(str)
        :param conditions: conditions(list of string)
        """
        n_shape = len(dissimilarities.shape)
        if n_shape == 2:
            rdm_2d = dissimilarities
        elif n_shape == 1:
            rdm_2d = make_2dRDM_from_1dRDM(dissimilarities)

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
    