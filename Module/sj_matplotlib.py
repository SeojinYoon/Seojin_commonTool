
"""
This file contains the basic source code to visualize graph using matplotlib
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import cumfreq

from sj_sequence import slice_list_usingDiff
from sj_string import search_stringAcrossTarget, str_join

def multi_font_strings(texts, sep = " ", info = {}):    
    """
    Get multi font string
    
    :param texts(list): list of string 
    :param sep(string): appendix to each string
    :param info(dictionary):
        -k, fonts(list): font style of string
        
    return (string): 
    """
    # medium $\mathtt{text}$
    # bold $\mathbf{text}$
    # italic $\mathit{text}$
    
    # low \\regular_{sub_text}
    styles = info.get("fonts", ["bold" for _ in texts])

    comp_texts = []
    for text, style in zip(texts, styles):
        if style == "bold":
            s = f"$\mathbf" + "{" + text + "}" + "$"
        elif style == "medium":
            s = f"$\mathtt$" + "{" + text + "}" + "$"
        elif style == "italic":
            s = f"$\mathit$" + "{" + text + "}" + "$"
        elif style == "low":
            s = f"$\\regular_" + "{" + text + "}" + "$"
        
        comp_texts.append(s)
        
    comp_text = str_join(comp_texts, sep)
        
    return comp_text

def draw_title(axis, title_info = {}):
    """
    Draw title in the axis
    
    :param title_info(dictionary): 
         -k, title(string): title
         -k, title_weight(string): weight of font
         -k, title_size(string): size of title
         -k, title_y_pos(string): y pos of title
    """
    title = title_info.get("title", "")
    title_weight = title_info.get("title_weight", "bold")
    title_size = title_info.get("title_size", 20)
    title_y_pos = title_info.get("title_y_pos", 1.0)
    
    axis.set_title(title, 
                   fontweight = title_weight,
                   size = title_size, 
                   y = title_y_pos)
    
def draw_grid(axis, grid_info = {}):
    """
    Draw grid in the axis
    
    :param grid_info(dictionary): 
        -k, is_draw_x_grid(boolean): flags to decide whether it draws x grid
        -k, is_draw_y_grid(boolean): flags to decide whether it draws y grid
        -k, x_grid_alpha(float): alpha value of x grid
        -k, y_grid_alpha(float): alpha value of y grid
        -k, x_grid_linewidth(float): line width of x grid
        -k, y_grid_linewidth(float): line width of y grid
        -k, x_grid_line_style(float): line style of x grid
        -k, y_grid_line_style(float): line style of y grid
    """
    # X
    is_draw_x_grid = grid_info.get("is_draw_x_grid", False)
    x_grid_alpha = grid_info.get("x_grid_alpha", 0.5)
    x_grid_linewidth = grid_info.get("x_grid_linewidth", 0.5)
    x_grid_line_style = grid_info.get("x_grid_line_style", "--")
    
    # Y
    is_draw_y_grid = grid_info.get("is_draw_y_grid", False)
    y_grid_alpha = grid_info.get("y_grid_alpha", 0.5)
    y_grid_linewidth = grid_info.get("y_grid_linewidth", 2)
    y_grid_line_style = grid_info.get("y_grid_line_style", "--")
    
    if is_draw_x_grid:
        axis.grid(True, 
                  axis = "x",
                  alpha = x_grid_alpha, 
                  linestyle = x_grid_line_style, 
                  linewidth = x_grid_linewidth)
    if is_draw_y_grid:
        axis.grid(True, 
                  axis = "y", 
                  alpha = y_grid_alpha, 
                  linestyle = y_grid_line_style, 
                  linewidth = y_grid_linewidth)
        
def draw_label(axis, label_info = {}):
    """
    Draw label in the axis
    
    :param label_info(dictionary): 
        -k, color(string): label color
        -k, x_label(string): x label text
        -k, y_label(string): y label text
        -k, x_weight(string): x label weight
        -k, y_weight(string): y label weight
        -k, x_size(float): x label size
        -k, y_size(float): y label size
    """
    color = label_info.get("color", "black")
    
    x_label = label_info.get("x_label", "")
    y_label = label_info.get("y_label", "")
    
    x_weight = dict(weight = label_info.get("x_weight", "bold"))
    y_weight = dict(weight = label_info.get("y_weight", "bold"))
    
    x_label_size = label_info.get("x_size", 16)
    y_label_size = label_info.get("y_size", 16)
    
    x_label_pad = label_info.get("x_label_pad", 10)
    y_label_pad = label_info.get("y_label_pad", 10)
    axis.set_xlabel(x_label, 
                    color = color, 
                    fontdict = x_weight, 
                    size = x_label_size,
                    labelpad = x_label_pad)
    
    axis.set_ylabel(y_label, 
                    color = color, 
                    fontdict = y_weight, 
                    size = y_label_size,
                    labelpad = y_label_pad)

def draw_threshold(axis, threshold_info = {}):
    """
    Draw threshold in the axis
    
    :param threshold_info(dictionary): threshold information
        -k, is_draw(boolean): Check whether threshold plot and text are showing
        
        # Regarding visualization constant
        -k, weight(string): font style, ex) bold
        -k, size(float): threshold annotation text size, ex) 10
        -k, color(string): threshold text color, ex) "black"
        
        # Regarding plots
        -k, x_margin(float): threshold plot margin size ex) 10
        -k, linewidth(float: line width of threshold plot, ex) 1
        
        # Regarding datas
        -k, threshold(list): threshold value, ex) [0, 1]
        -k, text(list): string to annotate threshold, ex) ["0.05", "0.01"]
                
        # Regarding text position
        x_min, x_max, y_min, y_max are used to calculate appropriate position for drawing threshold text.
        
        -k, x_min(float): minimum of x_values, ex) 0 
        -k, x_max(float): maximum of x_values, ex) 10
        -k, y_min(float): minimum of y_values, ex) 0
        -k, y_max(float): maximum of y_values, ex) 8
        -k, x_spacing(float): spacing between threshold line and annotation text, ex) 4
        -k, y_spacing(float): spacing between threshold line and annotation text, ex) 4
    """
    xmargin = threshold_info.get("x_margin", 0)
    xmin = threshold_info.get("x_min", 0)
    xmax = threshold_info.get("x_max", 0)
    color = threshold_info.get("color", "black")
    ymin = threshold_info.get("y_min", 0)
    ymax = threshold_info.get("y_max", 0)
    
    is_draw_threshold = threshold_info.get("is_draw", False)
    threshold = threshold_info.get("threshold", [])
    linestyle = threshold_info.get("linestyle", ":")
    threshold_annotate = threshold_info.get('text', [])
    threshold_annotate_size = threshold_info.get('size', 9)
    threshold_linewidth = threshold_info.get('linewidth', 2)

    threshold_weight = threshold_info.get('weight', "bold")        
    x_spacing = threshold_info.get('x_spacing', (xmax - xmin) / 30)
    y_spacing = threshold_info.get('y_spacing', (ymax - ymin) / 30)

    if is_draw_threshold:
        for i in range(len(threshold)):
            t = threshold[i]
            axis.plot([xmin - xmargin, xmax + xmargin], 
                      [t, t], 
                      linestyle = linestyle, 
                      linewidth = threshold_linewidth, 
                      color = color)
        
            if threshold_annotate != []:
                text = threshold_annotate[i]
            else:
                text = "{:.2f}".format(t)
                
            axis.annotate(text = text, 
                          xy = (xmax - x_spacing, t + y_spacing), 
                          weight = threshold_weight, 
                          size = threshold_annotate_size)
            
def draw_ticks(axis, tick_info = {}):
    """
    Draw ticks in the axis
    
    :param tick_info(dictionary): tick style configuration
        -k, x_data(list): x tick positions ex) [1,2,3]
        -k, x_names(list): x tick text ex) ["a", "b", "c"]
        -k, x_tick_weight(string): x tick weight
        -k, x_tick_size(float): x tick size
        -k, x_tick_rotation(int): x tick rotation
        
        -k, y_data(list): y tick positions ex) [1,2,3]
        -k, y_names(list): y tick text ex) ["a", "b", "c"] 
        -k, y_tick_round(int): round method to show y-tick appropriately
        -k, y_tick_weight(string): y tick weight
        -k, y_tick_size(float): y tick size
        -k, y_tick_rotation(int): y tick rotation
        -k, y_divided(int): the number of division for showing y-tick
        -k, y_need_tick(list): the y-tick which must to show
    """
    x_data = tick_info.get("x_data", [])
    x_names = tick_info.get("x_names", [])
    
    x_tick_weight = tick_info.get("x_tick_weight", "normal")
    x_tick_size = tick_info.get("x_tick_size", 14)
    x_tick_rotation = tick_info.get("x_tick_rotation", 90)
    
    y_data = tick_info.get("y_data", [])
    y_names = tick_info.get("y_names", [])
    
    y_tick_weight = tick_info.get("y_tick_weight", "normal")
    y_tick_size = tick_info.get("y_tick_size", 14)
    y_tick_rotation = tick_info.get("y_tick_rotation", 0)
    y_tick_round = tick_info.get("y_tick_round", None)
    
    # X
    x_data = np.array(x_data)
    x_names = np.array(x_names)
    x_tick_duplication = np.array([int((start + end) / 2) for start, end in slice_list_usingDiff(x_names)], dtype = int)

    if len(x_tick_duplication) > 0:
        x_names = x_names[x_tick_duplication]
        x_pos_dup = x_data[x_tick_duplication]
        x_data = x_pos_dup
        
    axis.set_xticks(x_data, 
                    x_names, 
                    rotation = x_tick_rotation, 
                    weight = x_tick_weight, 
                    size = x_tick_size)
    
    # Y
    axis.set_yticks(y_data, 
                    y_names,
                    rotation = y_tick_rotation,
                    weight = y_tick_weight,
                    size = y_tick_size)
    
def draw_spine(axis, spine_info = {}):
    """
    Draw spine in the axis
    
    :param spine_info(dictionary): 
        -k, spine_linewidth(float): spine line width ex) 2
        -k, spine_color(string): spine line color ex) "black"
        -k, invisibles(list): invisible informations ex) ["right", "top"]
    """
    spine_line_width = spine_info.get("spine_linewidth", 2)
    spine_color = spine_info.get("spine_color", "black")
    invisibles = spine_info.get("invisibles", ["right", "top"])
    
    all_spines = ["bottom", "top", "left", "right"]
    for invisible in invisibles:
        axis.spines[invisible].set_visible(False)
        all_spines.remove(invisible)
        
    for ax_name in all_spines:
        axis.spines[ax_name].set_linewidth(spine_line_width)
        axis.spines[ax_name].set_linewidth(spine_line_width)
        axis.spines[ax_name].set_color(spine_color)
        
def draw_legend(axis, legend_info = {}):
    """
    Draw legend in the axis
    
    :param legend_info(dictionary): 
        -k, is_draw_legend(float): Check whether legend is showing
        -k, names(list): legend names, ex) ["a"]
        -k, loc(tuple): legend location, ex) (1, 0.8)
        -k, font_size(float): legend font size, ex) 24
        -k, weight(string): legend font weight, ex) "bold"
        -k, frame_alpha(float): legend frame alpha, ex) 0
        -k, rect_width(float): each legend rect width, ex) 1
        -k, rect_height(float): each legend rect height, ex) 1
    """
    is_draw_legend = legend_info.get("is_draw_legend", False)
    colors = legend_info.get("colors", [])
    names = legend_info.get("names", [])
    loc = legend_info.get("loc", (1, 0.8))
    font_size = legend_info.get("font_size", 24)
    weight = legend_info.get("weight", "bold")
    frame_alpha = legend_info.get("frame_alpha", 0)
    rect_width = legend_info.get("rect_width", 1)
    rect_height = legend_info.get("rect_height", 1)
    legends = legend_info.get("legends", None)
    
    if is_draw_legend:
        if legends == None:
            handles = [plt.Rectangle((0,0), rect_width, rect_height, color = color) for color in colors]
            axis.legend(handles, 
                        names, 
                        loc = loc, 
                        prop = {
                            "weight" : weight,
                            "size" : font_size,
                        },
                        framealpha = frame_alpha)
        else:
            axis.legend(legends, 
                        names, 
                        loc = loc, 
                        prop = {
                            "weight" : weight,
                            "size" : font_size,
                        },
                        framealpha = frame_alpha)
        
def draw_vlines(axis, search_names, draw_names, draw_info = {}):
    """
    Draw each vline in the axis
    
    :param axis(AxesSubplot): axis
    :param search_names(list): search text in xaxis, ex) [10]
    :param draw_names(list): drawing name to correspond search_name, ex) ["10"]
    :param draw_info: (dictionary)
        -k, line_styles(list): line styles of vline, ex) [":"]
        -k, line_widths(list): line widths of vline, ex) [2]
        -k, label_sizes(list): label sizes of vline, ex) [10]
        -k, label_weights(list): label weights of vline, ex) ["bold"]
        -k, label_has(list): label horizontal align, ex) ["center"]
        -k, line_colors(list): label color, ex) ["black"]
        -k, label_rotations(list): label color, ex) [0]
    """
    line_styles = draw_info.get("line_styles", [":" for _ in search_names])
    line_widths = draw_info.get("line_widths", [2 for _ in search_names])
    label_sizes = draw_info.get("label_sizes", [10 for _ in search_names])
    label_weights = draw_info.get("label_weights", ["bold" for _ in search_names])
    label_has = draw_info.get("label_has", ["center" for _ in search_names])
    line_colors = draw_info.get("line_colors", ["black" for _ in search_names])
    label_rotations = draw_info.get("label_rotations", [0 for _ in search_names])
    
    vline_info = vline_pos(axis, search_names)
    for i in vline_info:
        x_pos = vline_info[i]["x_pos"]
        y_pos = vline_info[i]["y_pos"]
        label_y_pos = vline_info[i]["label_y_pos"]
        search_name = vline_info[i]["search_name"]
        
        search_name_i = search_names.index(search_name)
        line_style = line_styles[search_name_i]
        draw_name = draw_names[search_name_i]
        line_width = line_widths[search_name_i]
        label_size = label_sizes[search_name_i]
        label_weight = label_weights[search_name_i]
        label_ha = label_has[search_name_i]
        line_color = line_colors[search_name_i]
        label_rotation = label_rotations[search_name_i]
        
        axis.plot(x_pos,
                  y_pos,
                  linestyle = line_style,
                  linewidth = line_width,
                  color = line_color)

        axis.text(x_pos[0], 
                  label_y_pos, 
                  draw_name,
                  rotation = label_rotation,
                  ha = label_ha, 
                  weight = label_weight, 
                  size = label_size)
        
        
def vline_pos(axis, search_xs, pos_info = {}):
    """
    Get vline positions of searched text
    
    :param axis(AxesSubplot): axis
    :param search_xs(list): search text in xaxis, ex) [10]
                    
    return (dictionary):
        -k, x_pos(float): x position of v line
        -k, y_pos(float): y position of v line
        -k, label_y_pos(float): y text position of v line 
        -k, search_name(string): searched name
    """
    n_divid = pos_info.get("n_divid", 20)
    x_round = pos_info.get("x_round", 2)
    y_round = pos_info.get("y_round", 2)
    x_pos_align = pos_info.get("x_pos_align", "center")
    
    x_tick_labels = axis.xaxis.get_ticklabels()
    
    y_min, y_max = axis.get_ylim()
    
    unit_diff = (y_max - y_min) / n_divid
    
    x_tick_xs = []
    x_tick_ys = []
    x_tick_texts = []
    for x_tick_text in x_tick_labels:
        pos = x_tick_text.get_position()
        x = round(pos[0], x_round)
        y = round(pos[1], y_round)

        text = x_tick_text.get_text()

        x_tick_xs.append(x)
        x_tick_ys.append(y)
        x_tick_texts.append(text)
    
    key_i = 0
    annot_pos = {}
    for search_i, search_x in enumerate(search_xs):
        search_name = search_xs[search_i]
        searched_x_indexes = search_stringAcrossTarget(x_tick_texts, [search_x], return_type = "index")

        for searched_x_index in searched_x_indexes:
            if searched_x_index != 0:
                previous_searched_x_index = searched_x_index - 1
                diff_x = x_tick_xs[searched_x_index] - x_tick_xs[previous_searched_x_index]
                mean_diff_roi = np.round(diff_x / 2, 2)
                
                if x_pos_align == "center":
                    x_tick_x = x_tick_xs[previous_searched_x_index] + mean_diff_roi
                elif x_pos_align == "left":
                    x_tick_x = x_tick_xs[previous_searched_x_index]
                elif x_pos_align == "right":
                    x_tick_x = x_tick_xs[searched_x_index]
                
                annot_pos[key_i] = {
                    "x_pos" : [x_tick_x, x_tick_x],
                    "y_pos" : [y_min, y_max],
                    "label_y_pos" : y_max + unit_diff,
                    "search_name" : search_name
                }
                key_i += 1
                
    return annot_pos

def draw_text(axis, style_info = {}):
    """
    Draw text
    
    :param axis(AxesSubplot): axis
    :param style_info: (dictionary)
        -k, text(string): text
        -k, x(float): x pos of text
        -k, y(float): y pos of text
        -k, h_align(string): holizontal align
        -k, v_align(string): vertical align
        -k, weight(string): font
        -k, size(float): text size
    """
    text = style_info.get("text", "")
    x = style_info.get("x", 0)
    y = style_info.get("y", 0)
    
    h_align = style_info.get("h_align", "center")
    v_align = style_info.get("v_align", "center")
    
    weight = style_info.get("weight", "normal")
    size = style_info.get("size", 15)
    axis.text(x = x, 
              y = y, 
              s = text,
              horizontalalignment = h_align,
              verticalalignment = v_align,
              weight = weight,
              size = size)

def plot_histogram_and_cdf(data, 
                           axis=None, 
                           num_bins=30,
                           hist_label = "Histogram",
                           cdf_label = "CDF"):
    """
    Plot histogram and cumulative distribution function (CDF) of the given data.
    
    :param data(1D array or list): input data
    :param axis(Matplotlib axis): Matplotlib axis, the axis to plot on (default is None, which creates a new figure)
    :param num_bins(int): number of bins for the histogram (default is 30)
    """
    if axis is None:
        fig, axis = plt.subplots()

    # Plot histogram
    axis.hist(data, bins=num_bins, density=True, alpha=0.6, color='g', label=hist_label)

    # Calculate cumulative distribution function (CDF) using cumfreq
    cf = cumfreq(data, numbins=num_bins)
    x_values = cf.lowerlimit + np.linspace(0, cf.binsize * cf.cumcount.size, cf.cumcount.size)

    # Normalize the CDF values
    cdf_values = cf.cumcount / len(data)

    # Plot the CDF
    axis.plot(x_values, cdf_values, 'r--', linewidth=2, label=cdf_label)

    # Add labels and legend
    axis.set_xlabel('Value')
    axis.set_ylabel('Frequency / Cumulative Probability')
    axis.legend()

def make_colorbar(vmin, 
                  vmax, 
                  figsize = (2, 6), 
                  n_div = 4, 
                  cmap = "jet", 
                  tick_decimal = 4, 
                  orientation = "horizontal"):
    """
    Creates a customizable colorbar using matplotlib.


    :param vmin (float): The minimum value for the colorbar.
    :param vmax (float): The maximum value for the colorbar.
    :param figsize (tuple): The size of the figure (width, height). Defaults to (2, 6).
    :param n_div (int): Number of divisions (ticks) on the colorbar. Defaults to 4.
    :param cmap (str): Colormap to use for the colorbar. Defaults to "jet".
    :param tick_decimal (int): Number of decimal places to display on the tick labels. Defaults to 4.
    :param orientation (str): Orientation of the colorbar, either "horizontal" or "vertical". Defaults to "horizontal".

    return (tuple): A tuple containing the matplotlib figure and axis objects.
    """
    interval = (vmax - vmin) / n_div
    ticks = np.arange(vmin, vmax + interval, interval)

    # Create the figure and axis for the colorbar
    fig = plt.figure(figsize = figsize)

    if orientation == "vertical":
        axis = fig.add_axes([0.05, 0.05, 0.15, 0.9])  # [left, bottom, width, height]
    else:
        axis = fig.add_axes([0.1, 0.5, 0.8, 0.3])  # [left, bottom, width, height]
    
    # Create the colorbar
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, orientation=orientation)
    
    # Set the ticks and labels
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.{tick_decimal}f}" for tick in ticks])

    return fig, axis, ticks
    
if __name__=="__main__":
    multi_font_strings(["a", "b"])
    
    # Example usage with random data
    random_data = np.random.normal(loc=0, scale=1, size=1000)

    # Create a new figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Call the function with the first axis
    plot_histogram_and_cdf(random_data, axis=axes[0])

    # Color bar
    fig, axis = make_colorbar(0.0007, 0.0014, figsize = (2, 4), n_div = 4, orientation = "vertical")
    plt.show()
