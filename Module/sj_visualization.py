# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:12:48 2019

@author: STU24
"""

# Visualize 관련

# Common Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from enum import Enum
from matplotlib.pyplot import xticks, yticks
from scipy.stats import pearsonr
from collections import Counter
import math
from ipywidgets.widgets import Button, IntSlider, interact, HBox, VBox
from collections.abc import Iterable

# Custom Libraries
from sj_string import str_join
from sj_enum import Visualizing
from sj_string import search_stringAcrossTarget, search_string
from sj_matplotlib import draw_title, draw_grid, draw_threshold, draw_ticks, draw_spine, draw_text
from sj_matplotlib import draw_legend, draw_vlines, vline_pos, draw_label
from sj_color import rgb_to_hex

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
    line_style = style_info.get("line_style", "-")
    
    error_bars = []
    for cond in conds:
        sel_df = data_df[data_df["group"] == cond].sort_values(by = "x")
        
        p = axis.errorbar(x = sel_df["x"],
                          y = sel_df["y"],
                          yerr = sel_df["err"],
                          alpha = line_alpha,
                          fmt = line_fmt,
                          ls = line_style)
        error_bars.append(p)

    draw_label(axis, label_info)
    draw_spine(axis, spine_info)
    draw_ticks(axis, tick_info)
    draw_title(axis, title_info)
    
    cp_legend_info = legend_info.copy()
    cp_legend_info["legends"] = error_bars
    cp_legend_info["names"] = conds

    draw_legend(axis, cp_legend_info)
    
# stack graph를 그림
# data_sets: data_set을 요소로 갖는 리스트, 
# legends: 범례로 들어갈 리스트
# x_marks: x축 눈금에 들어갈 이름
# x_label: x축 이름
# y_label: y축 이름
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
    
    n_column = data.shape[1]
    line_colors = style_info.get("line_colors", None)
    for i in range(n_column):
        if line_colors == None:
            axis.plot(data.iloc[:, i], linewidth = linewidth)
        else:
            axis.plot(data.iloc[:, i], linewidth = linewidth, color = line_colors[i])
    
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
        search_title = "search: " + str_join(search_names)
    else:
        search_title = "search: All"
    
    if len(exclude_names) != 0:
        exclude_title = "exclude: " + str_join(exclude_names)
    else:
        exclude_title = ""
    range_title = "range: " + str(rank_ranges[0]) + " ~ " + str(rank_ranges[1])

    title = str_join([search_title, exclude_title, range_title], ", ")
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
    assert np.alltrue(xs_series.index == ys_series.index), "X and Y datas are not matched"
    
    # img
    plt.imshow(img)
    
    # Marker
    my_cmap = plt.get_cmap(cmap_name)
    
    for i in range(len(xs_series)):
        name = xs_series.index[i]
        plt.scatter(xs_series[i], ys_series[i], s = 4, label = name, color = my_cmap(i))

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
    prev30_btn = Button(description="Prev_30")
    next30_btn = Button(description="Next_30")
    buttons = HBox(children=[prev1_btn, next1_btn, prev10_btn, next10_btn, prev30_btn, next30_btn])

    video_frames = [value for value in args]
    n_video = len(video_frames)
    
    shapes = [video.shape for video in video_frames]
    
    assert np.alltrue([shape[0] == shapes[0][0] for shape in shapes]), "the number of time must be same"
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

    def onPrev30(s):
        slider.value = slider.value - 30

    def onNext30(s):
        slider.value = slider.value + 30

    prev1_btn.on_click(onPrev1)
    next1_btn.on_click(onNext1)
    prev10_btn.on_click(onPrev10)
    next10_btn.on_click(onNext10)
    prev30_btn.on_click(onPrev30)
    next30_btn.on_click(onNext30)

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

