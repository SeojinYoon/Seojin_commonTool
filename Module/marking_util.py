
import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

def create_point_clicker(img: np.array, 
                         labels: list, 
                         markers = None,
                         figsize = None):
    """
    Display an image and enable interactive point selection.

    :param img: Image array to display (e.g., numpy array)
    :param tablet_labels: Labels for clickable points
    :param markers: Marker styles for each label (default: '*' for all)
    :param figsize: Size of the figure (width, height)

    :return: Matplotlib figure, axis, and clicker object
    :rtype: tuple (fig, ax, klicker)
    """

    if markers is None:
        markers = ["*"] * len(labels)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(img)

    klicker = clicker(ax, labels, markers=markers)

    return fig, ax, klicker

if __name__ == "__main__":
    # %matplotlib widget
    import numpy as np

    frame = np.random.rand(300, 300)
    tablet_labels = ["point1", "point2", "point3", "point4"]
    fig, ax, klicker = create_point_clicker(
        frame=frame,
        tablet_labels=tablet_labels,
        title="Click 4 points"
    )
    pass
    