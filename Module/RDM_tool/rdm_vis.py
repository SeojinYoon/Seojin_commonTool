
import os
import sys
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sj_sequence import slice_list_usingDiff

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
        return fig, axis
    
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
        return fig, axis

if __name__ == "__main__":
    rdm_model = RDM_model(a, 
                          model_name = "abc",
                          conditions = ["1", "2", "3"])
    rdm_model.draw()
    