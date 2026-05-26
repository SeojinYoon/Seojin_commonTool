# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:12:48 2019

@author: STU24
"""

# Visualize 관련

# Common Libraries
import math, plotly, xarray, copy
import numpy as np
import pandas as pd
import seaborn as sns
from enum import Enum
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.image as mpimg
import plotly.graph_objects as go
from collections.abc import Iterable
from matplotlib.patches import Rectangle
from scipy.stats import sem, ttest_1samp
from IPython.display import display, clear_output, HTML
from ipywidgets.widgets import Button, IntSlider, interact, HBox, VBox, Output

# Custom Libraries
from sj_enum import Visualizing
from sj_array import reorient_array
from sj_string import search_string
from sj_matplotlib import draw_title, draw_grid, draw_threshold, draw_ticks, draw_spine, draw_text, draw_legend, draw_label

# Sources
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def draw_bar_plot(dataframe, axis, label_info = {}, title_info = {}, tick_info = {}, spine_info = {}):
    axis = sns.barplot(dataframe, color = "black")
    
    draw_label(axis, label_info)
    draw_title(axis, title_info)
    
    cp_tick_info = tick_info.copy()
    cp_tick_info["x_data"] = cp_tick_info.get("x_data", axis.get_xticks())
    cp_tick_info["x_names"] = cp_tick_info.get("x_names", dataframe.columns)
    
    cp_tick_info["y_tick_round"] = 0
    cp_tick_info["y_data"] = axis.get_yticks()
    cp_tick_info["y_names"] = axis.get_yticks()
    
    draw_ticks(axis, cp_tick_info)
    draw_spine(axis, spine_info)

def draw_scatter_plot(x_list, y_list, title = "Title", xlabel = "xlabel", ylabel = "ylabel"):
    plt.scatter(x_list, y_list)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def draw_line_graph(axis, data_df, conds, 
                    style_info = {},
                    label_info = {},
                    spine_info = {},
                    tick_info = {},
                    title_info = {},
                    legend_info = {}):
    """
    :param axis:
    :param data_df(pd.DataFrame):
        -column x, 
        -column group, 
        -column y, 
    """
    line_alpha = style_info.get("line_alpha", 0.5)
    line_fmt = style_info.get("line_fmt", "-o")
    
    error_bars = []
    for cond in conds:
        sel_df = data_df[data_df["group"] == cond].sort_values(by = "x")
        
        p = axis.errorbar(x = sel_df["x"],
                          y = sel_df["y"],
                          yerr = sel_df["err"],
                          alpha = line_alpha,
                          fmt = line_fmt)
        error_bars.append(p)

    draw_label(axis, label_info)
    draw_spine(axis, spine_info)
    draw_ticks(axis, tick_info)
    draw_title(axis, title_info)
    
    cp_legend_info = legend_info.copy()
    cp_legend_info["legends"] = error_bars

    draw_legend(axis, cp_legend_info)
    
def draw_stack_graph(data_sets, 
                     legends = None,
                     title = None,
                     x_marks = None,
                     x_rotation = None,
                     x_label = None,
                     y_label = None):
    longest_set_length = 0
    for data_set in data_sets:
        data_length = len(data_set)
        if longest_set_length < data_length:
            longest_set_length = data_length
    
    indexes = range(0, longest_set_length)
    
    width = 0.35
    
    import matplotlib.pyplot as plt
    
    plts = []
    for i, data_set in enumerate(data_sets):
        if i == 0:
            c_plt = plt.bar(indexes, data_set, width)
        else:
            c_plt = plt.bar(indexes, data_set, width, bottom = data_sets[i-1])
        plts.append(c_plt)
    
    ps = [plt[0] for plt in plts]
    
    if legends is not None:
        plt.legend(tuple(ps), tuple(legends))
    
    if title is not None:
        plt.title(title)
    
    if x_rotation is not None and x_marks is not None:
        plt.xticks(indexes, x_marks, rotation = x_rotation)
    else:
        if x_marks is not None:
            plt.xticks(indexes, x_marks)
    
    if x_label is not None:
        plt.xlabel(x_label)
        
    if y_label is not None:
        plt.ylabel(y_label)
    
    return plt

def draw_function(x, function):
    """
    Drawing graph of function

    :param x: numpy array ex) np.linespace(-100, 100, 1000)
    :param function: function ex) lambda x: x+1
    """
    plt.plot(x, list(map(lambda element: function(element), x)))

def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))

    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)

def make_meshgrid(x1, x2, h=.02):
    """
    make meshgrid

    :param x1(np.array): data
    :param x2(np.array): data
    :param h: distance

    return X(grid data), Y(grid data)
    """
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    return xx1, xx2


def plot_contours(ax, clf, xx1, xx2, **params):
    """
    plot contour

    :param ax: AxesSubplot
    :param clf: classifier
    :param xx: grid data
    :param yy: grid data
    """
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    out = ax.contourf(xx1,
                      xx2,
                      Z,  # predict value
                      **params)

    return out

def draw_subplots(figsize, n_cols, draw_functions):
    """
    draw multi subplot
    
    :param figsize(tuple): tuple ex) (15,2)
    :param n_cols: set fixed column
    :param draw_functions: drawing function list ex) [ lambda: plt.bar([1],[1]) ]
    """
    plt.figure(figsize=figsize)
    
    while(True):
        draw_count = len(draw_functions)
        if draw_count % n_cols == 0:
            break
        else:
            draw_functions += [lambda: plt.cla()]
            
    
    count = 1
    for draw_function in draw_functions:
        plt.subplot(int(draw_count / n_cols), n_cols, count)
        draw_function()
        count += 1
    plt.tight_layout()

def one_to_many_scatters(data, data_colors, fix_col_name, subplot_size = 4, figsize = (25,25)):
    """
    Drawing many scatter plot representing fix_col to other columns
    
    :param data(DataFrame): 
    :param data_colors: color for visualizing data point ex) ["red", "blue", "red"...]
    :param fix_col_name: x-axis column_name
    :param subplot_size: #suplot_size ex) 4
    :param figsize: figure size
    """
    fig = plt.figure(figsize = figsize)

    for i, colname in enumerate(data.columns):
        if colname == fix_col_name:
            continue
        row = int(i / subplot_size)
        col = i % subplot_size

        ax = fig.add_subplot(subplot_size, subplot_size, row*subplot_size + col + 1)
        ax.scatter(x = data.loc[:,fix_col_name], y = data.loc[:,colname], c=data_labels);
        ax.set(xlabel=fix_col_name, ylabel= colname)
        
def plot_timeseries(axis, 
                    data,
                    tr, 
                    showing_x_interval = 10, 
                    style_info = {},
                    label_info = {},
                    title_info = {},
                    tick_info = {},
                    legend_info = {}):
    """
    Plotting data as time series
    
    :params axis: plt axis
    :params data(np.ndarray): signal varying over time, where each column is a different signal.
    :params tr: time resolution(float) ex) 2
    :params showing_x_interval: x-axis interval to show in axis
    :params linewidth: line width which plot draws line as
    """
    
    # Plot
    linewidth = style_info.get("plot_linewidth", 1)

    legends = []
    n_column = data.shape[1]
    line_colors = style_info.get("line_colors", None)
    for i in range(n_column):
        if line_colors == None:
            legend = axis.plot(data.iloc[:, i], linewidth = linewidth)[0]
        else:
            legend = axis.plot(data.iloc[:, i], linewidth = linewidth, color = line_colors[i])[0]
        legends.append(legend)
        
    # Label
    draw_label(axis, label_info)
    
    # x-ticks
    tick_info["x_data"] = np.arange(0, len(data) * tr, showing_x_interval)
    tick_info["x_names"] = np.arange(0, len(data) * tr, showing_x_interval)
    draw_ticks(axis, tick_info)
    
    # lim
    xlim = style_info.get("xlim", (np.min(tick_info["x_data"]), np.max(tick_info["x_data"])))
    axis.set_xlim(xlim)
        
    ylim = style_info.get("ylim", (np.min(data), np.max(data)))
    axis.set_ylim(ylim)
    
    # Title
    draw_title(axis, title_info)
    
    # Legends
    legend_info["legends"] = legends
    draw_legend(axis, legend_info)
            
    # Spine
    draw_spine(axis, style_info)

def plot_design_matrix(design_mat_array, regressor_indexes, figsize = (6,10), regressor_labels=None):
    """
    Plotting Design matrix
    
    :param design_mat_array(numpy array): design matrix columns: regressor, rows: data
    :param regressor_indexes(list): regressor index for plotting ex) [1,2,3]
    :param figsize(tuple): figure size
    :param regressor_labels(list): regressor labels for showing to title
    """
    X = design_mat_array
    
    data_length = X.shape[0]
    
    f, a = plt.subplots(ncols = len(regressor_indexes), figsize=figsize, sharey=True)
    plt.gca().invert_yaxis()
    
    for axis_i in range(len(regressor_indexes)):
        regressor_index = regressor_indexes[axis_i]
        a[axis_i].plot(X[:,regressor_index], range(data_length))
        
        if regressor_labels != None:
            regressor_label = regressor_labels[axis_i]
            a[axis_i].set_title(regressor_label, rotation = 45, fontsize=20)
    plt.tight_layout()

def plotting(axis,
             names, 
             values,
             style_info = {},
             threshold_info = {},
             tick_info = {},
             label_info = {},
             grid_info = {},
             title_info = {},
             search_names = [], 
             rank_ranges = None, 
             exclude_names = [],
             is_sort = True):
    """
    Plot decoding results
    
    :param names(list): names
    :param values(2d - list): accuracies
    :param threshold: threshold
    :param search_names: search key for searching keyword in roi_decoding_results
    """
    name_index = 0
    accuracy_index = 1
    
    results = list(zip(names, values))
    
    filtered_results = list(filter(lambda result: search_string(target = result[name_index], 
                                                                search_keys = search_names, 
                                                                search_type="all",
                                                                exclude_keys = exclude_names), 
                                   results))
    
    
    # Sort by accuracy
    if is_sort:
        filtered_results.sort(key=lambda x: np.mean(x[1]), reverse=True)
    
    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    xmin = 0
    xmax = len(filtered_results)
    
    x_data = np.arange(xmin, xmax)
    x_names = [name for name, _ in filtered_results]
    y_data = [accs for _, accs in filtered_results]
    
    # x-axis
    plot_style = style_info.get("plot_style", Visualizing.scatter.bar)
    
    if plot_style.value & Visualizing.bar.value != 0:
        palette = style_info.get('palette', ("tab10"))
        single_color = style_info.get('single_color', None)
        ci = style_info.get("ci", "deprecated")
        
        if single_color != None:
            sns.barplot(data = y_data,
                        color = single_color,
                        ci = ci)
        else:
            sns.barplot(data = y_data,
                        palette = palette,
                        ci = ci)

    if plot_style.value & Visualizing.swarm.value != 0:
        swarm_alpha = style_info.get('swarm_alpha', .35)
        swarm_color = style_info.get("swarm_color", "0")
        
        sns.swarmplot(data = y_data, color = swarm_color, alpha = swarm_alpha)
   
    # threshold
    draw_threshold(axis, threshold_info)
    
    # ticks
    y_lim = style_info.get('y_lim', (np.min(y_data), np.max(y_data)))
    y_interval = style_info.get('y_interval', (y_lim[1] - y_lim[0]) / 10)
    
    cp_tick_info = tick_info.copy()
    cp_tick_info["x_data"] = x_data
    cp_tick_info["x_names"] = x_names
    cp_tick_info.get('y_data', np.arange(y_lim[0], y_lim[1], y_interval))
    cp_tick_info.get('y_names', [np.round(e, 4) for e in cp_tick_info["y_data"]])
    draw_ticks(axis, cp_tick_info)
    
    axis.set_ylim(y_lim)

    draw_label(axis, label_info)
    draw_spine(axis, style_info)
    draw_grid(axis, grid_info)
    draw_title(axis, title_info)
    
    return x_names, y_data
    
def plot_stats(axis,
               names, 
               stats,
               p_values,
               style_info = {},
               threshold_info = {},
               search_names = [], 
               rank_ranges = None, 
               exclude_names = [],
               tick_info = {},
               label_info = {},
               is_sort = True,
               title_info = {},
               grid_info = {},
               legend_info = {},
               errors = None):
    """
    Plot stats
    
    :param names(list): names
    :param stats(1d - list): stats
    :param threshold_info(dictionary): threshold information
    :param search_names: search key for searching keyword in roi_decoding_results
    :param is_sort(boolean): 
    :param style_info(dictionary): 
        -k, x_adds(list): x-difference between previous data and next data
        -k, color_style(string): bar color style
            -ex: multi_color, palette, single_color
        -k, single_color(string): single bar color 
        -k, palette(string): pallete style
        -k, multi_color: colors corresponding to the datas
        -k, sig_y_diff: y-difference between significant level representation and barplot
    :param tick_info: tick style configuration(dictionary)
    :param label_info(dictionary): 
    :param grid_info(dictionary): 
    :param title_info(dictionary): 
    :param legend_info(dictionary): 
        
    """
    # cp
    tick_info = tick_info.copy()
    threshold_info = threshold_info.copy()
    
    # 
    name_index = 0
    accuracy_index = 1
    
    if type(errors) == type(None):
        errors = np.repeat(0, len(stats))
        
    results = list(zip(names, stats, p_values, errors))
    
    filtered_results = list(filter(lambda result: search_string(target = result[name_index], 
                                                                search_keys = search_names, 
                                                                search_type="all",
                                                                exclude_keys = exclude_names), 
                                   results))
    
    # Sort by accuracy
    if is_sort:
        filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    # bat plot
    n_data = len(filtered_results)
    origin_x_data = np.ones(n_data)
    
    y_tick_round = tick_info.get("y_tick_round", 2)
    x_adds = style_info.get("x_adds", np.zeros(n_data))
    bar_width = style_info.get("bar_width", 0.8)
    
    x_data = np.cumsum(origin_x_data) + np.cumsum(x_adds) 
    x_data -= 1
    
    y_data = [stat for _, stat, _, _ in filtered_results]
    y_data = np.round(y_data, y_tick_round)
    p_data = [p for _, _, p, _ in filtered_results]
    error_data = [err for _, _, _, err in filtered_results]
    
    bar_style = style_info.get("bar_style", "default")
    color_style = style_info.get("color_style", "palette")
    if color_style == "single_color":
        single_color = style_info.get('single_color', "gray")
        
        axis.bar(x_data, y_data, color = single_color, width = bar_width)
    elif color_style == "palette":
        palette = style_info.get('palette', ("tab10"))

        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        my_cmap = plt.get_cmap(palette)
        
        axis.bar(x_data, y_data, color = my_cmap(rescale(y_data)), width = bar_width)
    elif color_style == "multi_color":
        multi_color = style_info.get('multi_color', np.repeat("gray", n_data))
        
        axis.bar(x_data, y_data, color = multi_color, width = bar_width)

    if bar_style == "error":
        axis.errorbar(x = x_data, 
                      y = y_data, 
                      fmt = 'none', 
                      ecolor = 'black',
                      capsize = 4,
                      yerr = error_data)
        
        data_y_min = min(y_data - error_data)
        data_y_max = max(y_data + error_data)
    else:
        data_y_min = min(y_data)
        data_y_max = max(y_data)
    data_y_min = min(tick_info.get("y_min", data_y_min), data_y_min)
    data_y_max = max(tick_info.get("y_max", data_y_max), data_y_max)
    
    sig_style = style_info.get("sig_style", "default")
    sig_y_diff = style_info.get("sig_y_diff", (data_y_max - data_y_min) / 30)
    
    # Iterrating over the bars one-by-one to represent significance level
    i = 0
    
    sig_annot_pos = []
    for bar, p in zip(axis.patches, p_data):    
        bar_height = bar.get_height()
        x_loc = bar.get_x() + (bar.get_width() / 2)
        
        if sig_style == "default":
            if bar_style == "error":
                extra = error_data[i]
            else:
                extra = 0
                
            if bar_height >= 0:
                y_loc = bar_height + sig_y_diff + extra
            else:
                y_loc = bar_height - sig_y_diff - extra
        elif sig_style == "y_max":
            y_loc = data_y_max + sig_y_diff
            
        axis.annotate(convert_pvalue_to_asterisks(p),
                       (x_loc, y_loc), 
                       ha = "center", 
                       va = "top",
                       size = style_info.get('star_size', 8), 
                       xytext = (0, 5),
                       textcoords = 'offset points',
                       weight = "bold")
        
        sig_annot_pos.append([x_loc, y_loc])
        i += 1
        
    sig_annot_pos = np.array(sig_annot_pos)
    
    # x-ticks
    filtered_names = [name for name, _, _, _ in filtered_results]
    tick_info["x_data"] = tick_info.get("x_data", x_data)
    tick_info["x_names"] = tick_info.get("x_names", filtered_names)
    
    # y-ticks
    y_divided = tick_info.get("y_divided", 3)
    y_need_tick = tick_info.get("y_need_tick", None)
    
    y_interval = (data_y_max - data_y_min) / y_divided
    y_range = np.append(np.arange(data_y_min, data_y_max, y_interval), data_y_max)
    
    ys = np.array(list(y_range))
    filter_conditions = np.logical_and(ys <= data_y_max, ys >= data_y_min)
    ys = np.array(ys)[filter_conditions]
    ys = np.round(ys, y_tick_round)
    ys = np.unique(ys)
    if y_need_tick != None:
        ys = np.append(ys, y_need_tick)
        min_y = min(np.min(sig_annot_pos[:, 1]), np.min(y_need_tick), data_y_min)
        max_y = max(np.max(sig_annot_pos[:, 1]), np.max(y_need_tick), data_y_max)
    else:
        min_y = min(np.min(sig_annot_pos[:, 1]), data_y_min)
        max_y = max(np.max(sig_annot_pos[:, 1]), data_y_max)
    
    tick_info["y_data"] = tick_info.get("y_data", ys)
    tick_info["y_names"] = tick_info.get("y_names", ys)
    
    draw_ticks(axis, tick_info)
    
    # ylim
    yticks = axis.yaxis.get_major_ticks()
    sig_annot_pos = np.round(sig_annot_pos, y_tick_round)
    y_margin = (max_y - min_y) / 10
    
    axis.set_ylim(min_y - y_margin, max_y + y_margin)
    
    # threshold
    threshold_info["x_min"] = threshold_info.get("x_min", np.min(x_data))
    threshold_info["x_max"] = threshold_info.get("x_max", np.max(x_data))
    threshold_info["y_min"] = threshold_info.get("y_min", min_y)
    threshold_info["y_max"] = threshold_info.get("y_max", max_y)
    
    draw_threshold(axis, threshold_info)
    
    # Label
    draw_label(axis, label_info)
    
    # Spines
    draw_spine(axis, style_info)
    
    # grid
    draw_grid(axis, grid_info)
    
    # Title
    draw_title(axis, title_info)
    
    draw_legend(axis, legend_info)
    
    # Legend

    return filtered_names, y_data

def draw_plot_box(axis, names, accuracies, search_names = [], exclude_names = [], rank_ranges = (0, 10)):
    """
    Draw box pot
    
    :param axis: axis
    :param names(list - string): name of accuracy ex) ["hippocampus", "gyrus", "cerebellum"]
    :param accuracies(list): accuracy ex) [0.1, 0.1, 0.1]
    :param search_names(list - string): search name ex) ["hippocampus"]
    :param exclude_names(list - string): exclude name ex) ["hippocampus"]
    :param rank_ranges(tuple): range ex) (0, 10)
    """
    results = list(zip(names, accuracies))
    rank_ranges = (0, 10)
    
    name_index = 0
    filtered_results = list(filter(lambda result: search_string(target = result[name_index], 
                                                                search_keys = search_names, 
                                                                search_type="all",
                                                                exclude_keys = exclude_names), 
                                   results))


    # Sort by accuracy
    filtered_results.sort(key=lambda x: np.mean(x[1]), reverse=True)

    # ranging rank
    if rank_ranges != None:
        filtered_results = filtered_results[rank_ranges[0]: rank_ranges[1]]

    
    
    # data
    target_names = list(map(lambda result: result[0], filtered_results))
    y_data = [accs for _, accs in filtered_results]
    
    # plot
    axis.boxplot(y_data)
    
    axis.set_xticks(np.arange(1, len(target_names) + 1), target_names, rotation=90)
    
    # label
    axis.set_xlabel("design property")
    axis.set_ylabel("accuracy")
    
    # Title
    if len(search_names) != 0:
        search_title = "search: " + "_".join(search_names)
    else:
        search_title = "search: All"
    
    if len(exclude_names) != 0:
        exclude_title = "exclude: " + "_".join(exclude_names)
    else:
        exclude_title = ""
    range_title = "range: " + str(rank_ranges[0]) + " ~ " + str(rank_ranges[1])

    title = "_".join([search_title, exclude_title, range_title], ", ")
    axis.set_title(title)

def draw_regPlot(axis,
                 xs, 
                 ys,
                 threshold_info = {},
                 label_info = {},
                 title_info = {},
                 spine_info = {},
                 tick_info = {},
                 grid_info = {},
                 text_info = {}):
    """
    Draw linear plot
    
    :param axis: axis
    :param xs(list): x_values
    :param ys(list): y_values
    """
    sns.regplot(x = xs, y = ys, ax = axis)
    
    draw_threshold(axis, threshold_info)
    draw_label(axis, label_info)
    draw_title(axis, title_info)
    draw_spine(axis, spine_info)
    draw_ticks(axis, tick_info)
    draw_grid(axis, grid_info)
    
    result_correlation = pearsonr(xs, ys)
    stat = np.round(result_correlation.statistic, 2)
    p_value = result_correlation.pvalue
    
    text = f"r = {stat}" + "\n" + format_p_value(p_value)
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    cp_text_info = text_info.copy()
    cp_text_info["x"] = cp_text_info.get("x", max_x - (max_x - min_x) / 5)
    cp_text_info["y"] = cp_text_info.get("y", max_y - (min_y - max_y) / 5)
    cp_text_info["text"] = cp_text_info.get("text", text)
    draw_text(axis, cp_text_info)
    
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"

    return ""

def format_p_value(p_value, threshold = 0.01):
    n_zero = int(-math.log(p_value, 10))
    if p_value < threshold:
        return  f"p < $10^" + '{' + f"-{n_zero}" + '}$' 
    else:
        return f"p = {p_value:.2f}"

def draw_onImg(img, 
               xs_series,
               ys_series,
               cmap_name = "tab20",
               legend_loc = (1.1, -0.5)):
    """
    Draw point on img
    
    :param img(np.array - 3d): img data ex) the img's shape: 480, 928, 3
    :param xs_series(pd.Series): x datas, index is the name of each data
    :param ys_series(pd.Series): y datas, index is the name of each data    
    """
    assert np.all(xs_series.index == ys_series.index), "X and Y datas are not matched"
    
    # img
    plt.imshow(img)
    
    # Marker
    my_cmap = plt.get_cmap(cmap_name)
    
    for i in range(len(xs_series)):
        name = xs_series.index[i]
        plt.scatter(xs_series.iloc[i], ys_series.iloc[i], s = 4, label = name, color = my_cmap(i))

    plt.legend(loc = legend_loc)
    
def compare_frames(*args, titles, fig_info = { "fig_width" : 10 }, cmap = "gray"):
    """
    Compare frame by frame
    
    :param *args: frames (np.array - shape: n_t, n_y, n_x)
    :param titles(list - string): each title of video ex) "a", "b"
    :param cmap(string): color map for visualizing video
    """
    prev1_btn = Button(description="Prev_1")
    next1_btn = Button(description="Next_1")
    prev10_btn = Button(description="Prev_10")
    next10_btn = Button(description="Next_10")
    buttons = HBox(children=[prev1_btn, next1_btn, prev10_btn, next10_btn])

    video_frames = [value for value in args]
    n_video = len(video_frames)
    
    shapes = [video.shape for video in video_frames]
    
    assert np.all([shape[0] == shapes[0][0] for shape in shapes]), "the number of time must be same"
    n_t, n_y, n_x = shapes[0]
    
    slider = IntSlider(min = 0, 
                       max = n_t - 1, 
                       step = 1, 
                       layout = {'width': '900px'})
    
    @interact
    def func1(frame = slider):
        fig, axises = plt.subplots(1, n_video)
        if not isinstance(axises, Iterable):
            axises = [axises]
            
        fig.set_figwidth(fig_info["fig_width"])

        for axis, title, frames in zip(axises, titles, video_frames):
            axis.axis('off')
            axis.set_title(title)

            axis.imshow(frames[frame], cmap = cmap)

        plt.show()    

    # Callbacks
    def onPrev1(s):
        slider.value = slider.value - 1

    def onNext1(s):
        slider.value = slider.value + 1

    def onPrev10(s):
        slider.value = slider.value - 10

    def onNext10(s):
        slider.value = slider.value + 10

    prev1_btn.on_click(onPrev1)
    next1_btn.on_click(onNext1)
    prev10_btn.on_click(onPrev10)
    next10_btn.on_click(onNext10)

    display(buttons)

def draw_errorlines(mean_df,
                    error_df,
                    title,
                    subtitles,
                    other_mean_df = None,
                    fig = None,
                    axes = None,
                    xlabel = "",
                    ylabel = "",
                    save_path = None):
    """
    Draw error lines
    
    :param mean_df(pd.DataFrame - shape: (#roi, #time)): mean activation across roi
    :param error_df(pd.DataFrame - shape: (#roi, #time)): standard deviation across roi
    :param other_mean_df(np.array - shape: (#roi, #time)): model prediction(GLM) result
    :param title(string): title
    :param save_path(string): path
    """
    # Figure
    if type(fig) == None and type(axes) == None:
        n_col = 4
        fig, axes = plt.subplots(int(len(mean_df) / n_col + 1), n_col)
        fig.set_figheight(30)
        fig.set_figwidth(15)

    axes = axes.flatten()
    for roi_i in range(len(mean_df)):
        mean = mean_df.iloc[roi_i].to_numpy()
        error = error_df.iloc[roi_i].to_numpy()

        xs = np.arange(len(mean))
        axes[roi_i].plot(mean, color = "black")
        axes[roi_i].fill_between(xs, 
                                 mean - error, 
                                 mean + error,
                                 color = "black",
                                 alpha = 0.1)
        
        if type(other_mean_df) != None:
            axes[roi_i].plot(other_mean_df.iloc[roi_i].to_numpy(), linestyle='dashed', color = "orange")
        axes[roi_i].set_xticks(np.arange(len(mean_df.columns)), mean_df)
        axes[roi_i].set_title(subtitles[roi_i], weight = "bold")
    fig.supxlabel(xlabel, weight = "bold")
    fig.supylabel(ylabel, weight = "bold")
    fig.suptitle(f"{title}", fontsize=20, y = 1.00, weight = "bold")
    fig.tight_layout()
    
    # Save figure
    if save_path != None:
        plt.savefig(save_path)
        print(f"save: {save_path}")
    
    return fig, axes

def draw_profile_datas(ax,
                       sample_datas: np.ndarray, 
                       cmap: str = "tab10",
                       cond_spread_width: float = 0.4,
                       p_threshold: float = 0.05,
                       y_minmax: tuple = None,
                       is_scatter = False):
    """
    Draw profile roi results
    
    :param ax: matplotlib axis
    :param sample_data(shape - (#cond, #roi, #subj)): data arrays
    :param cmap: color map (ex: "tab10", "viridis")
    :param cond_spread_width: total width of data across conditions per tick 
    :param p_threshold: p-value for thresholding significance representation
    :param y_minmax: y-axis range
    """
    n_cond, n_roi, n_subj = sample_datas.shape

    if n_cond > 1:
        cond_x_spacing = cond_spread_width / (n_cond-1)
    else:
        cond_x_spacing = 0
        
    x = np.arange(n_roi)
    
    # Draw datas
    cmap = plt.cm.get_cmap(cmap)
    for cond_i, sample_data in enumerate(sample_datas):
        if is_scatter:
            cond_x = (x - (cond_spread_width/2)) + (cond_i * cond_x_spacing)
        else:
            cond_x = (x - (cond_spread_width/2))
            
        mean = np.mean(sample_data, axis = 1)
        error = sem(sample_data, axis = 1)

        scatter_xs = np.repeat(cond_x, n_subj)
        
        cond_color = cmap(cond_i)

        if is_scatter:
            ax.scatter(scatter_xs, sample_data.flatten(), s = 10, alpha = 0.2, color = cond_color)
            
        ax.plot(cond_x, mean, color = cond_color)
        ax.fill_between(cond_x, mean - error, mean + error, alpha = 0.2, color = cond_color)

    # Show significant areas
    significant_index_info = {}
    if type(y_minmax) != type(None):
        y_min_, y_max_ = y_minmax[0], y_minmax[1]
        
        y_height = (y_max_ - y_min_)
        rect_height = y_height / 30
        
        max_height_forSig = n_cond * rect_height
        for cond_i, sampling_data in enumerate(sample_datas):
            cond_color = cmap(cond_i)
            
            stat_result = ttest_1samp(sampling_data, popmean = 0, axis = 1)
            significant_indexes = np.where(stat_result.pvalue < p_threshold)[0]
            significant_index_info[cond_i] = significant_indexes
            
            cond_number = cond_i + 1
            y = y_min_ + max_height_forSig - (rect_height * cond_number)
            
            for sig_i in significant_indexes:
                ax.add_patch(Rectangle(xy = (sig_i - 0.5, y), 
                                       width = 1, 
                                       height = rect_height, 
                                       color = cond_color))
    
    return x

class ImageSelector:
    def __init__(self, image_paths, fig_info={"fig_width": 20, "fig_height": 10}):
        self.image_paths = image_paths
        self.fig_info = fig_info
        self.selected_indices = [] # 클래스 내부에서 리스트 관리
        
        # 위젯 생성
        self.slider = IntSlider(
            min=0, max=len(image_paths)-1, step=1, 
            layout={'width': '700px'}, description='Index'
        )
        self.btn = Button(description="Select Image", button_style='info')
        self.out = Output()
        
        # 이벤트 연결
        self.btn.on_click(self.on_click_select)
        self.slider.observe(self.display_image, names='value')
        
    def display_image(self, change):
        index = change['new']
        with self.out:
            clear_output(wait=True)
            fig, axis = plt.subplots(1, 1, figsize=(self.fig_info["fig_width"], self.fig_info["fig_height"]))
            
            img = mpimg.imread(self.image_paths[index])
            axis.imshow(img)
            axis.set_title(f"Index: {index} | {self.image_paths[index]}")
            axis.axis("off")
            plt.show()

    def on_click_select(self, b):
        current_idx = self.slider.value
        if current_idx not in self.selected_indices:
            self.selected_indices.append(current_idx)
            with self.out:
                print(f"✅ Index {current_idx} 추가됨. 현재 선택: {self.selected_indices}")
        else:
            with self.out:
                print(f"ℹ️ Index {current_idx}는 이미 선택되어 있습니다.")

    def show(self):
        ui = VBox([HBox([self.slider, self.btn]), self.out])
        display(ui)
        self.display_image({'new': 0}) # 초기 이미지 로드

def create_3d_time_series_plot(dataset_3d, obj_info={}, skeletons = []):
    """
    Create 3d time series plot using plotly
    
    :param dataset_3d: xarray Dataset containing the 3D marker data. Expected dimensions include Times, Labels, and Coords, with variable "3D" storing the coordinates.
    :param obj_info: Dictionary describing static objects to visualize. Each key is an object name and the value must contain a "points" entry specifying the object's 3D coordinates.
    :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
    """
    # 1. Initialization
    times = dataset_3d["Times"].to_numpy()
    n_frame = len(times)
    labels = list(dataset_3d["Labels"].to_numpy())

    x_index, y_index, z_index = 0, 1, 2
    
    # Helper function
    def make_traces(step_i: int):
        sel_times = times[:step_i + 1]
        
        marker_coordinates = dataset_3d.sel(
            Times=sel_times, 
            Labels=labels
        )["3D"].to_numpy()
            
        # Visualize - object (Static or dependent on points)
        obj_traces = []
        for obj_name in obj_info:
            kinds = obj_info[obj_name]["points"]
            for kind in kinds:
                pts = np.array(obj_info[obj_name]["points"][kind])
                obj_trace = go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 2], # Mapping Y to Z for Plotly orientation
                    z=pts[:, 1], # Mapping Z to Y for Plotly orientation
                    mode="lines",
                    line=dict(color="black", width=2),
                    visible=True,    # Set to True to display objects by default
                    showlegend=True,
                    name=kind,
                    hoverinfo="name"
                )
                obj_traces.append(obj_trace)

        # Visualize - data markers
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(labels)))
        
        marker_traces = []
        for target, color in zip(labels, colors):
            target_index = labels.index(target)
            
            color_str = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
            trace = go.Scatter3d(
                x=marker_coordinates[:, target_index, x_index],
                y=marker_coordinates[:, target_index, z_index],
                z=marker_coordinates[:, target_index, y_index],
                mode="markers",
                marker=dict(size=2, opacity=0.6, color=color_str),
                visible=False,
                name=target
            )
            marker_traces.append(trace)

        # Visualize - skeleton
        skeleton_traces = []
        for p1, p2 in skeletons:
            i1 = labels.index(p1)
            i2 = labels.index(p2)
        
            trace = go.Scatter3d(
                x=[marker_coordinates[-1, i1, x_index], marker_coordinates[-1, i2, x_index]],
                y=[marker_coordinates[-1, i1, z_index], marker_coordinates[-1, i2, z_index]],
                z=[marker_coordinates[-1, i1, y_index], marker_coordinates[-1, i2, y_index]],
                mode="lines",
                line=dict(color="gray", width=4),
                visible=False,
                showlegend=False,
                hoverinfo="skip"
            )
        
            skeleton_traces.append(trace)
        return skeleton_traces + obj_traces + marker_traces

    # 2. Create all traced over all frames
    all_traces = []
    for step_i in range(n_frame):
        traces = make_traces(step_i)
        n_trace_per_frame = len(traces)
        all_traces.extend(traces)
    
    total_traces_count = len(all_traces)

    # Make slider step
    steps = []
    for frame_i in range(n_frame):
        step = dict(
            method="restyle",
            args=["visible", [False] * total_traces_count],
            label=f"Frame {frame_i}"
        )
        # Change related trace per frame
        start_idx = frame_i * n_trace_per_frame
        for j in range(n_trace_per_frame):
            step["args"][1][start_idx + j] = True  
        steps.append(step)

    # Make figure and layout
    fig = go.Figure(data=all_traces)
    
    # Active first frame
    for i in range(n_trace_per_frame):
        fig.data[i].visible = True

    # Calculate range of axis
    range_candidates = []

    if obj_info:
        all_corners = []
        for obj in obj_info:
            obj_data = obj_info[obj]
            for kind in obj_data["corners"]:
                all_corners.append([obj_data["corners"][kind][corner_name] for corner_name in obj_data["corners"][kind]])
        all_corners = np.concatenate(all_corners, axis = 0)
        
        min_obj_x, min_obj_y, min_obj_z = np.min(all_corners, axis = 0)
        max_obj_x, max_obj_y, max_obj_z = np.max(all_corners, axis = 0)

        range_candidates.append([min_obj_x, min_obj_y, min_obj_z])
        range_candidates.append([max_obj_x, max_obj_y, max_obj_z])
        
    mins = dataset_3d["3D"].min(dim=[d for d in dataset_3d["3D"].dims if d != "Coords"], skipna=True)
    maxs = dataset_3d["3D"].max(dim=[d for d in dataset_3d["3D"].dims if d != "Coords"], skipna=True)
    
    min_x, min_y, min_z = mins.sel(Coords="X").item(), mins.sel(Coords="Y").item(), mins.sel(Coords="Z").item()
    max_x, max_y, max_z = maxs.sel(Coords="X").item(), maxs.sel(Coords="Y").item(), maxs.sel(Coords="Z").item()
    range_candidates.append([min_x, min_y, min_z])
    range_candidates.append([max_x, max_y, max_z])
    
    range_candidates = np.array(range_candidates)
    
    scene_min_range = np.min(range_candidates, axis = 0)
    scene_max_range = np.max(range_candidates, axis = 0)
    
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Time Step: "},
            pad={"t": 50},
            steps=steps
        )],
        scene=dict(
            xaxis=dict(title='X-axis', range=(scene_min_range[x_index], scene_max_range[x_index])),
            yaxis=dict(title='Z-axis', range=(scene_min_range[z_index], scene_max_range[z_index])),
            zaxis=dict(title='Y-axis', range=(scene_min_range[y_index], scene_max_range[y_index])),
        ),
        showlegend=True,
        height=800
    )

    return HTML(fig.to_html(include_plotlyjs="cdn"))

def plot_3D_dataset(position_ds: xarray.Dataset,
                    ds_coord_order: str,
                    targets: list = [],
                    obj_info: dict = {},
                    skeletons: list = [],
                    vis_info: dict = {},
                    axis_info: dict = {}):
    """
    Plot 3D coordinates from an xarray dataset.

    :param position_ds: Dataset containing '3D' variable with 'Times', 'Labels', 'Coords'.
    :param ds_coord_order: the direction of x,y,z coord ex) IAL, LPI, ...
    :param targets: List of marker labels (Targets) to visualize.
    :param obj_info: Dictionary for static objects. Format: {'obj_name': {'points': [[x,y,z], ...]}}.
    :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
    :param vis_info: Dictionary containing configuration for visualization.
    :param axis_info: Dictionary containing configuration for the origin axes visualization.
        * show_origin_axes (bool): Whether to display the coordinate axes at the origin. (default: True)
        * axis_length (float): The length of the line for each axis. (default: 0.2)
        * cone_size (float): The size of the arrowhead (cone) at the end of each axis. (default: 0.1)
        * axis_origin (tuple): The (x, y, z) coordinates where the axes will be centered. (default: (0, 0, 0))
    
    :return: (IPython.display.HTML) Rendered Plotly 3D visualization.
    """
    visualize_coord_order = "LPI"
    
    # Preprocessing
    position_ds = copy.deepcopy(position_ds)
    position_ds["3D"].data = reorient_array(position_ds["3D"].data, ds_coord_order, visualize_coord_order)

    obj_coord_order = ds_coord_order
    obj_info = copy.deepcopy(obj_info)
    for obj_name in obj_info:
        kinds = obj_info[obj_name]["points"]
        for kind in kinds:
            pts = np.array(obj_info[obj_name]["points"][kind])
            pts = reorient_array(pts[None, :, :], obj_coord_order, visualize_coord_order)
            obj_info[obj_name]["points"][kind] = pts[0]
            
    # Index mapping for coordinates
    x_index, y_index, z_index = 0, 1, 2

    # Time
    times = position_ds["Times"].to_numpy()

    # Target
    targets = list(position_ds.Labels.to_numpy()) if len(targets) == 0 else list(targets)
    
    # Calculate axis ranges based on the entire dataset for consistency
    min_x = np.nanmin(position_ds.sel(Coords="X")["3D"].to_numpy())
    min_y = np.nanmin(position_ds.sel(Coords="Y")["3D"].to_numpy())
    min_z = np.nanmin(position_ds.sel(Coords="Z")["3D"].to_numpy())
    
    max_x = np.nanmax(position_ds.sel(Coords="X")["3D"].to_numpy())
    max_y = np.nanmax(position_ds.sel(Coords="Y")["3D"].to_numpy())
    max_z = np.nanmax(position_ds.sel(Coords="Z")["3D"].to_numpy())
    
    """
    1. Visualize - Static Objects (e.g., table, environment boundaries)
    """
    obj_traces = []
    for obj_name in obj_info:
        kinds = obj_info[obj_name]["points"]
        for kind in kinds:
            pts = np.array(obj_info[obj_name]["points"][kind])
            
            obj_trace = go.Scatter3d(
                x=pts[:, x_index],
                y=pts[:, y_index],
                z=pts[:, z_index],
                mode="lines",
                line=dict(color="black", width=2),
                visible=True,
                showlegend=True,
                name=kind,
                hoverinfo="name"
            )
            obj_traces.append(obj_trace)

    """
    2. Visualize - Coordinate Axes (Origin)
    """
    show_origin_axes = axis_info.get("show_origin_axes", False)
    axis_length = axis_info.get("axis_length", 0.1)
    cone_size = axis_info.get("cone_size", 0.05)
    axis_origin = axis_info.get("axis_origin", (0,0,0))

    axis_traces = []
    if show_origin_axes:
        x_origin, y_origin, z_origin = axis_origin[x_index], axis_origin[y_index], axis_origin[z_index]
        
        # X-axis
        axis_traces.append(go.Scatter3d(x=[x_origin, x_origin + axis_length], 
                                        y=[y_origin, y_origin], 
                                        z=[z_origin, z_origin],
                                        mode="lines+text", 
                                        line=dict(color="red", width=8),
                                        text=["", "X"], 
                                        textposition="top center", 
                                        name="X-axis", 
                                        showlegend=False))
        axis_traces.append(go.Cone(
            x=[x_origin + axis_length], y=[y_origin], z=[z_origin],
            u=[axis_length * 0.3], v=[0], w=[0], # 방향 벡터
            colorscale=[[0, 'red'], [1, 'red']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))

        # Y-axis
        axis_traces.append(go.Scatter3d(
            x=[x_origin, x_origin], y=[y_origin, y_origin + axis_length], z=[z_origin, z_origin],
            mode="lines+text", line=dict(color="green", width=8),
            text=["", "Y"], textposition="top center", name="Y-axis", showlegend=False
        ))
        axis_traces.append(go.Cone(
            x=[x_origin], y=[y_origin + axis_length], z=[z_origin],
            u=[0], v=[axis_length * 0.3], w=[0],
            colorscale=[[0, 'green'], [1, 'green']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))

        # Z-axis
        axis_traces.append(go.Scatter3d(
            x=[x_origin, x_origin], y=[y_origin, y_origin], z=[z_origin, z_origin + axis_length],
            mode="lines+text", line=dict(color="blue", width=8),
            text=["", "Z"], textposition="top center", name="Z-axis", showlegend=False
        ))
        axis_traces.append(go.Cone(
            x=[x_origin], y=[y_origin], z=[z_origin + axis_length],
            u=[0], v=[0], w=[axis_length * 0.3],
            colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))
        
    """
    3. Visualize - Markers (Dynamic time-series data)
    """
    marker_traces = []
    
    # Extract coordinates for the selected times and targets
    marker_coordinates = position_ds.sel(Times=times, Labels=targets)["3D"].to_numpy()
    
    # Define color gradient based on time progression (Coolwarm colormap)
    num_colors = len(times)
    my_cmap = plt.get_cmap("coolwarm")
    
    # Convert colormap to Plotly-compatible RGBA strings and handle ZeroDivisionError
    if num_colors > 1:
        colors = [f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})" 
                  for i in range(num_colors) for c in [my_cmap(i / (num_colors - 1))]]
    else:
        colors = ["rgba(0, 0, 255, 1)"] # Default blue if only one time step exists

    mode = vis_info.get("marker_mode", "markers")
    for target in targets:
        target_index = targets.index(target)
        trace = go.Scatter3d(
            x = marker_coordinates[:, target_index, x_index],
            y = marker_coordinates[:, target_index, y_index],
            z = marker_coordinates[:, target_index, z_index],
            showlegend=False,
            mode = mode,
            marker = dict(
                size = 4,
                opacity = 0.8,
                color = colors, # Apply time-based color gradient
            ),
            name = target,
            text = [f"{target} {t}" for t in times],
        )
        marker_traces.append(trace)

    """
    4. Skeleton
    """
    skeleton_traces = []
    labels = list(position_ds.Labels)
    
    for p1, p2 in skeletons:
        i1 = labels.index(p1)
        i2 = labels.index(p2)
        trace = go.Scatter3d(
            x=[marker_coordinates[-1, i1, x_index], marker_coordinates[-1, i2, x_index]],
            y=[marker_coordinates[-1, i1, y_index], marker_coordinates[-1, i2, y_index]],
            z=[marker_coordinates[-1, i1, z_index], marker_coordinates[-1, i2, z_index]],
            mode="lines",
            line=dict(color="gray", width=4),
            visible=True,
            showlegend=False,
            hoverinfo="skip"
        )
        skeleton_traces.append(trace)
            
    """
    5. Layout Configuration
    """
    range_candidates = []
    
    if show_origin_axes:
        min_axis_x = axis_origin[x_index] - axis_length
        min_axis_y = axis_origin[y_index] - axis_length
        min_axis_z = axis_origin[z_index] - axis_length

        max_axis_x = axis_origin[x_index] + axis_length
        max_axis_y = axis_origin[y_index] + axis_length
        max_axis_z = axis_origin[z_index] + axis_length
        
        range_candidates.append([min_axis_x, min_axis_y, min_axis_z])
        range_candidates.append([max_axis_x, max_axis_y, max_axis_z])
    
    if obj_info:
        all_corners = []
        for obj in obj_info:
            obj_data = obj_info[obj]
            for kind in obj_data["corners"]:
                all_corners.append([obj_data["corners"][kind][corner_name] for corner_name in obj_data["corners"][kind]])
        all_corners = np.concatenate(all_corners, axis = 0)
        
        min_obj_x, min_obj_y, min_obj_z = np.min(all_corners, axis = 0)
        max_obj_x, max_obj_y, max_obj_z = np.max(all_corners, axis = 0)

        range_candidates.append([min_obj_x, min_obj_y, min_obj_z])
        range_candidates.append([max_obj_x, max_obj_y, max_obj_z])

    range_candidates.append([min_x, min_y, min_z])
    range_candidates.append([max_x, max_y, max_z])
    range_candidates = np.array(range_candidates)
    
    scene_min_range = np.min(range_candidates, axis = 0)
    scene_max_range = np.max(range_candidates, axis = 0)

    x_range_width = scene_max_range[x_index] - scene_min_range[x_index]
    y_range_width = scene_max_range[y_index] - scene_min_range[y_index]
    z_range_width = scene_max_range[z_index] - scene_min_range[z_index]
    
    axis_bg = vis_info.get("axis_bg", "rgba(230,230,230,30)")
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
        title=f"3D Estimation Traces ({len(times)} frames)",
        scene=dict(
            xaxis=dict(title="L->R", 
                       range=(scene_min_range[x_index] - x_range_width * 2/10, 
                              scene_max_range[x_index] + x_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            yaxis=dict(title="P->A", 
                       range=(scene_min_range[y_index] - y_range_width * 2/10, 
                              scene_max_range[y_index] + y_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            zaxis=dict(title="I->S", 
                       range=(scene_min_range[z_index] - z_range_width * 2/10, 
                              scene_max_range[z_index] + z_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            aspectmode='data'
        ),
        showlegend=True
    )
    
    """
    5. Construct Figure and Render to HTML
    """
    data = axis_traces + obj_traces + marker_traces + skeleton_traces
    plot_figure = go.Figure(data=data, layout=layout)

    return HTML(plot_figure.to_html(include_plotlyjs="cdn"))

def plot_3D_datasets(
    position_ds_list,
    ds_coord_order,
    targets: list = [],
    skeletons: list = [],
    obj_info = {},
    dataset_names = [],
    vis_info = {},
    axis_info = {},
):
    """
    Plot 3D coordinates from multiple xarray datasets.
    
    :param position_ds_list: (list[xarray.Dataset]) List of datasets containing the '3D' variable with
                          'Times', 'Labels', 'Coords' dimensions.
    :param ds_coord_order: the direction of x,y,z coord ex) IAL, LPI, ...
    :param targets: List of marker labels (Targets) to visualize.
    :param skeletons: skeleton information ex) [("Shoulder", "Elbow"), ("Elbow", "Wrist")]
    :param obj_info: Dictionary for static objects.
                     Format: {'obj_name': {'points': [[x, y, z], ...]}}
    :param dataset_names: Names for each dataset to display in the legend.
                          Must have the same length as position_ds_list.
    :param vis_info: Dictionary containing configuration for visualization.
    :param axis_info: Dictionary containing configuration for the origin axes visualization.
        * show_origin_axes: Whether to display the coordinate axes at the origin. (default: True)
        * axis_length: The length of the line for each axis. (default: 0.2)
        * cone_size: The size of the arrowhead (cone) at the end of each axis. (default: 0.1)
        * axis_origin: The (x, y, z) coordinates where the axes will be centered. (default: (0, 0, 0))
    
    :return: (IPython.display.HTML) Rendered Plotly 3D visualization.
    """
    visualize_coord_order = "LPI"
    
    # Preprocessing
    position_ds_list = copy.deepcopy(position_ds_list)
    for position_ds in position_ds_list:
            position_ds["3D"].data = reorient_array(position_ds["3D"].data, ds_coord_order, visualize_coord_order)
        
    obj_coord_order = ds_coord_order
    obj_info = copy.deepcopy(obj_info)
    for obj_name in obj_info:
        kinds = obj_info[obj_name]["points"]
        for kind in kinds:
            pts = np.array(obj_info[obj_name]["points"][kind])
            pts = reorient_array(pts[None, :, :], obj_coord_order, visualize_coord_order)
            obj_info[obj_name]["points"][kind] = pts[0]
            
    # Validation check
    n_ds = len(position_ds_list)
    dataset_names = [f"{i}" for i in range(n_ds)] if len(dataset_names) == 0 else dataset_names
    assert len(dataset_names) == n_ds, "dataset_names and position_ds_list must have the same length"
    
    # Config
    x_index, y_index, z_index = 0, 1, 2
    
    # 1. Static objects
    colors = plotly.colors.qualitative.Plotly
    
    obj_traces = []
    for i, obj_name in enumerate(obj_info):
        kinds = obj_info[obj_name]["points"]
        color = colors[i % len(colors)]
        for j, kind in enumerate(kinds):
            pts = np.array(obj_info[obj_name]["points"][kind])
            obj_trace = go.Scatter3d(
                x=pts[:, x_index],
                y=pts[:, y_index],
                z=pts[:, z_index], 
                mode="lines",
                line=dict(color=color, width=2),
                visible=True,
                showlegend=(j == 0),
                name=obj_name,
                hoverinfo="name"
            )
            obj_traces.append(obj_trace)

    # 2. Marker traces for multiple datasets
    marker_traces = []

    # dataset-level colors
    cmap_dataset = plt.get_cmap("tab10")
    dataset_rgbs = [cmap_dataset(i % 10)[:3] for i in range(n_ds)]
    
    all_min_x, all_min_y, all_min_z = [], [], []
    all_max_x, all_max_y, all_max_z = [], [], []

    mode = vis_info.get("marker_mode", "markers")
    for ds_idx, position_ds in enumerate(position_ds_list):
        ds_name = dataset_names[ds_idx]
        rgb = dataset_rgbs[ds_idx]
        r, g, b = [int(v * 255) for v in rgb]

        times = position_ds["Times"].to_numpy()
        alphas = np.linspace(1.0, 0.15, len(times))
        point_colors = [f"rgba({r},{g},{b},{a})" for a in alphas]
        sel_t = list(position_ds.Labels.to_numpy()) if len(targets) == 0 else targets
        marker_coordinates = position_ds.sel(Times=times, Labels=sel_t)["3D"].to_numpy()

        for target_idx, target in enumerate(sel_t):
            trace = go.Scatter3d(
                x=marker_coordinates[:, target_idx, x_index],
                y=marker_coordinates[:, target_idx, y_index],
                z=marker_coordinates[:, target_idx, z_index],
                mode=mode,
                marker=dict(
                    size=4,
                    opacity=0.8,
                    color=point_colors,
                ),
                name=ds_name,
                legendgroup=ds_name,
                showlegend=(target_idx == 0),
                text = [f"{target} {t}" for t in times],
            )
            marker_traces.append(trace)

        # collect axis ranges across all datasets
        all_min_x.append(np.nanmin(position_ds.sel(Coords="X")["3D"].to_numpy()))
        all_min_y.append(np.nanmin(position_ds.sel(Coords="Y")["3D"].to_numpy()))
        all_min_z.append(np.nanmin(position_ds.sel(Coords="Z")["3D"].to_numpy()))

        all_max_x.append(np.nanmax(position_ds.sel(Coords="X")["3D"].to_numpy()))
        all_max_y.append(np.nanmax(position_ds.sel(Coords="Y")["3D"].to_numpy()))
        all_max_z.append(np.nanmax(position_ds.sel(Coords="Z")["3D"].to_numpy()))

    min_x, min_y, min_z = np.min(all_min_x), np.min(all_min_y), np.min(all_min_z)
    max_x, max_y, max_z = np.max(all_max_x), np.max(all_max_y), np.max(all_max_z)

    # 3. Skeleton
    skeleton_traces = []
    for ds_idx, position_ds in enumerate(position_ds_list):
        times = position_ds["Times"].to_numpy()
        marker_coordinates = position_ds.sel(Times=times)["3D"].to_numpy()
        labels = list(position_ds.Labels.to_numpy())
        
        for p1, p2 in skeletons:
            i1 = labels.index(p1)
            i2 = labels.index(p2)

            trace = go.Scatter3d(
                x=[marker_coordinates[-1, i1, x_index], marker_coordinates[-1, i2, x_index]],
                y=[marker_coordinates[-1, i1, y_index], marker_coordinates[-1, i2, y_index]],
                z=[marker_coordinates[-1, i1, z_index], marker_coordinates[-1, i2, z_index]],
                mode="lines",
                line=dict(color="gray", width=4),
                visible=True,
                showlegend=False,
                hoverinfo="skip"
            )
        
            skeleton_traces.append(trace)
    
    # 4. Axis
    show_origin_axes = axis_info.get("show_origin_axes", False)
    axis_length = axis_info.get("axis_length", 0.2)
    cone_size = axis_info.get("cone_size", 0.1)
    axis_origin = axis_info.get("axis_origin", (0,0,0))

    axis_traces = []
    if show_origin_axes:
        x_origin, y_origin, z_origin = axis_origin[x_index], axis_origin[y_index], axis_origin[z_index]
        
        # X-axis
        axis_traces.append(go.Scatter3d(x=[x_origin, x_origin + axis_length], 
                                        y=[y_origin, y_origin], 
                                        z=[z_origin, z_origin],
                                        mode="lines+text", 
                                        line=dict(color="red", width=8),
                                        text=["", "X"], 
                                        textposition="top center", 
                                        name="X-axis", 
                                        showlegend=False))
        axis_traces.append(go.Cone(
            x=[x_origin + axis_length], y=[y_origin], z=[z_origin],
            u=[axis_length * 0.3], v=[0], w=[0],
            colorscale=[[0, 'red'], [1, 'red']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))

        # Y-axis
        axis_traces.append(go.Scatter3d(
            x=[x_origin, x_origin], y=[y_origin, y_origin + axis_length], z=[z_origin, z_origin],
            mode="lines+text", line=dict(color="green", width=8),
            text=["", "Y"], textposition="top center", name="Y-axis", showlegend=False
        ))
        axis_traces.append(go.Cone(
            x=[x_origin], y=[y_origin + axis_length], z=[z_origin],
            u=[0], v=[axis_length * 0.3], w=[0],
            colorscale=[[0, 'green'], [1, 'green']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))

        # Z-axis
        axis_traces.append(go.Scatter3d(
            x=[x_origin, x_origin], y=[y_origin, y_origin], z=[z_origin, z_origin + axis_length],
            mode="lines+text", line=dict(color="blue", width=8),
            text=["", "Z"], textposition="top center", name="Z-axis", showlegend=False
        ))
        axis_traces.append(go.Cone(
            x=[x_origin], y=[y_origin], z=[z_origin + axis_length],
            u=[0], v=[0], w=[axis_length * 0.3],
            colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, sizemode="absolute", sizeref=cone_size
        ))
        
    # 5. Layout configuration
    range_candidates = []
    if show_origin_axes:
        min_axis_x = axis_origin[x_index] - axis_length
        min_axis_y = axis_origin[y_index] - axis_length
        min_axis_z = axis_origin[z_index] - axis_length
        max_axis_x = axis_origin[x_index] + axis_length
        max_axis_y = axis_origin[y_index] + axis_length
        max_axis_z = axis_origin[z_index] + axis_length
        range_candidates.append([min_axis_x, min_axis_y, min_axis_z])
        range_candidates.append([max_axis_x, max_axis_y, max_axis_z])
        
    if obj_info:
        all_corners = []
        for obj in obj_info:
            obj_data = obj_info[obj]
            for kind in obj_data["corners"]:
                all_corners.append([obj_data["corners"][kind][corner_name] for corner_name in obj_data["corners"][kind]])
        all_corners = np.concatenate(all_corners, axis = 0)

        min_obj_x, min_obj_y, min_obj_z = np.min(all_corners, axis = 0)
        max_obj_x, max_obj_y, max_obj_z = np.max(all_corners, axis = 0)

        range_candidates.append([min_obj_x, min_obj_y, min_obj_z])
        range_candidates.append([max_obj_x, max_obj_y, max_obj_z])

    range_candidates.append([min_x, min_y, min_z])
    range_candidates.append([max_x, max_y, max_z])
    range_candidates = np.array(range_candidates)

    scene_min_range = np.min(range_candidates, axis = 0)
    scene_max_range = np.max(range_candidates, axis = 0)

    x_range_width = scene_max_range[x_index] - scene_min_range[x_index]
    y_range_width = scene_max_range[y_index] - scene_min_range[y_index]
    z_range_width = scene_max_range[z_index] - scene_min_range[z_index]
    
    axis_bg = vis_info.get("axis_bg", "rgba(230,230,230,30)")
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
        title=f"3D Estimation Traces ({len(times)} frames)",
        scene=dict(
            xaxis=dict(title="L->R", 
                       range=(scene_min_range[x_index] - x_range_width * 2/10, 
                              scene_max_range[x_index] + x_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            yaxis=dict(title="P->A", 
                       range=(scene_min_range[y_index] - y_range_width * 2/10, 
                              scene_max_range[y_index] + y_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            zaxis=dict(title="I->S", 
                       range=(scene_min_range[z_index] - z_range_width * 2/10, 
                              scene_max_range[z_index] + z_range_width * 2/10), 
                       backgroundcolor = axis_bg),
            aspectmode='data'
        ),
        showlegend=True
    )

    data = axis_traces + obj_traces + marker_traces + skeleton_traces
    plot_figure = go.Figure(data=data, layout=layout)
    
    return HTML(plot_figure.to_html(include_plotlyjs="cdn"))

def draw_marker(img: np.array, 
                marker_pos: np.array, 
                marker_names: list,
                marker_reference = "UL"):
    """
    Draw marker positions on an image
    
    :param img: image data
    :param marker_pos: position of marker in image (shape: #marker, 2)
    :param marker_names: name of markers (shape: #marker)
    :param marker_reference: origin of coordinate system of marker position
    """
    plt.imshow(img)
    img_height, img_width, _ = img.shape
    
    for i, name in enumerate(marker_names):
        marker_i = list(marker_names).index(name)
        if marker_reference == "UL":
            x = marker_pos[marker_i][0]
            y = marker_pos[marker_i][1]
        elif marker_reference == "DL":
            x = marker_pos[marker_i][0]
            y = img_height - marker_pos[marker_i][1]
        
        plt.scatter(x, y, label = name)
    plt.legend()

def compare_trajectories(trajectories,
                         labels=None,
                         figsize=(6, 6),
                         legend_loc = None):
    """
    Visualize and compare multiple 2D trajectories using an index-based slider.

    Each trajectory is expected to contain columns "X" and "Y", which will be
    converted into a NumPy array of shape (T, 2). The function allows interactive
    visualization of trajectory progression over index (time step).

    :param trajectories: List of trajectory data.
    :type trajectories: list of pandas.DataFrame

    :param labels: Optional list of labels corresponding to each trajectory.
                   If None, default labels ("Traj 0", "Traj 1", ...) are used.
    :type labels: list of str or None

    :param figsize: Size of the matplotlib figure (width, height).
    :type figsize: tuple of int or float

    :return: None. Displays an interactive matplotlib plot using ipywidgets.
    :rtype: None
    """

    if labels is None:
        labels = [f"Traj {i}" for i in range(len(trajectories))]

    max_index = max(len(traj) for traj in trajectories) - 1

    x_min = min(traj[:, 0].min() for traj in trajectories)
    x_max = max(traj[:, 0].max() for traj in trajectories)
    y_min = min(traj[:, 1].min() for traj in trajectories)
    y_max = max(traj[:, 1].max() for traj in trajectories)

    markers = ["o", "s", "^", "D", "x", "*"]

    def plot_index(idx):
        fig, ax = plt.subplots(figsize=figsize)

        for i, traj in enumerate(trajectories):

            # full trajectory
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.2)

            # partial trajectory up to index
            if idx < len(traj):
                ax.plot(traj[:idx+1, 0], traj[:idx+1, 1], linewidth=2)
                ax.scatter(traj[idx, 0],
                           traj[idx, 1],
                           s=100,
                           marker=markers[i % len(markers)],
                           label=labels[i])
            else:
                ax.plot(traj[:, 0], traj[:, 1], linewidth=2)

        ax.text(
            0.02, 0.98,
            f"Index: {idx}",
            transform=ax.transAxes,
            ha="left", va="top"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Trajectory comparison")
        if legend_loc is None:
            ax.legend()
        else:
            ax.legend(legend_loc)

        plt.show()

    interact(
        plot_index,
        idx=IntSlider(min=0, max=max_index, step=1, value=0, description="Index"),
    )
    
if __name__=="__main__":
    import F_Visualize
    test = pd.DataFrame([
        [100, 200, 150],
        [123, 180, 159],
        [130, 190, 182],
        [134, 210, 167],
        [159, 230, 171],
        [160, 235, 180],
        [169, 237, 188]
                ])

    a = draw_line_graph([test[0], test[1], test[2]], 
                        x_marks = ['아', '야', '어', '여', '오', '요'],
                        ylim = [0, 300],
                        xlabel = 'x축',
                        ylabel = 'y축',
                        title = 'abc')
    
    draw_stack_graph([[1, 2, 3, 4], [5, 6, 7, 8]],
                                 legends = ['1234','456'],
                                 title = '1234',
                                 x_marks = ['가','나','다','라'],
                                 x_label = 'Y축!~',
                                 y_label = 'X축!~')

    raw_function(np.linspace(0, 100, 100), lambda x: x + 1)

    data = {
        "a": [1, 2, 3, 2, 1],
        "b": [2, 3, 4, 3, 1],
        "c": [3, 2, 1, 4, 2],
        "d": [5, 9, 2, 1, 8],
        "e": [1, 3, 2, 2, 3],
        "f": [4, 3, 1, 1, 4],
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.show()

    from sklearn import svm
    fig, ax = plt.subplots()
    xx1, xx2 = make_meshgrid(np.array([0, 1]), np.array([1, 2]), h=0.5)
    clf = svm.SVC(kernel='linear')
    plot_contours(ax, clf, xx1, xx2, cmap=plt.cm.coolwarm, alpha=0.8)

    draw_subplots((15,2), 3, [lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1]), lambda: plt.bar([1],[1])])
    
    one_to_many_scatters(data = pd.DataFrame({
        "A" : [1,2,3],
        "B" : [4,5,6],
        "C" : [1,1,1]
    }),
                         data_labels = ["red", "blue", "red"],
                         fix_col_name = "A")

    plot_timeseries(np.c_[
        [1,2,3],
        [4,5,6]
    ])
    
    fig, axis = plt.subplots(1,1)
    draw_plot_box(axis, ["a", "b"], [[1,2,3], [4,5,6]])
    
    fig, axis = plt.subplots(1,1)
    plot_accuracies(axis, ["a", "b"], [[1,2,3], [4,5,6]])
    
    fig, axis = plt.subplots(1,1)
    plot_stats(axis, ["a", "b"], [1,2])
    
    data = np.zeros((100, 120, 3))
    xs = pd.Series({
        "1" : 50,
        "2" : 30,
    })
    ys = pd.Series({
        "1" : 20,
        "2" : 10,
    })
    draw_onImg(data, xs, ys)

    show_images(image_paths = ["/mnt/ext1/seojin/temp/1.jpg"])
