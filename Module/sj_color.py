
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from PIL import ImageColor

def rgb_to_hex(rgb):
    return "#" + '%02x%02x%02x' % rgb

def l_cmap(cmap_style, names, add_number = 0.5, axis = None):
    """
    Get listed cmap
    
    :param cmap_style: cmap_style
    :param names: x tick names
    :param add_number: add x value
    :param axis: axis
    """
    palette = {}
    n_color = len(names)
    
    # colors
    cmap = matplotlib.cm.get_cmap(cmap_style, n_color)
    for i in range(n_color):
        palette[names[i]] = matplotlib.colors.rgb2hex(cmap(i)[:3])
        
    # x-values
    n_color_div2 = n_color / 2
    
    min_x = -n_color_div2 + add_number
    min_y = n_color_div2 + add_number
    
    # mapping
    x_values = np.arange(min_x, min_y, 1)
    for i, key in enumerate(list(palette.keys())):
        palette[key] = [x_values[i], palette[key]]
    
    # Visualize
    colors = []
    for name in names:
        index = names.index(name)
    
        colors.append(ImageColor.getcolor(palette[name][1], "RGB"))
    colors = np.array(colors)

    # Create the ListedColormap
    cmap = matplotlib.colors.ListedColormap([palette[key][1] for key in palette])
    
    if axis == None:
        fig, axis = plt.subplots(1)
    axis.imshow([np.array(colors)])
    axis.set_xticks(np.arange(0, len(colors)), names)

    return cmap, palette

if __name__ == "__main__":
    rgb_to_hex((255, 255, 195))
    
    cmap, palette = l_cmap("Pastel2", ["early", "late", "whole"])
    