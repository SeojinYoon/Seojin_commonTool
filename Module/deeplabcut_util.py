
# Common Libraaries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D

# Functions
def dynamic_marker_pos(video_path: str, 
                       dlc_predictions: np.ndarray, 
                       bodyparts: np.ndarray):
    """
    Visualize marker positions on a video frame.

    :param video_path: Path to the video file.
    :param dlc_predictions(shape: (#step, #marker, 2)): predicted body part positions of DeepLabCut
    :param bodyparts(shape: #marker): body part information
    """
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    slider = IntSlider(
        min=0,
        max=frame_count - 1,
        step=1,
        layout={'width': '900px'}
    )

    @interact(step_i=slider)
    def func1(step_i):
        fig, axis = plt.subplots(1, 1, figsize=(12, 6))

        video.set(cv2.CAP_PROP_POS_FRAMES, step_i)
        ret, frame = video.read()

        if not ret:
            print(f"Failed to read frame {step_i}.")
            plt.close(fig)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axis.imshow(frame)

        colors = ["red", "blue", "green", "orange"]
        for marker_i, bodypart in enumerate(bodyparts):
            x = dlc_predictions[step_i, marker_i, 0]
            y = dlc_predictions[step_i, marker_i, 1]
        
            axis.scatter(x, y, s = 50, color = colors[marker_i], label = bodypart)
            axis.text(x, y, bodypart, fontsize = 10, color = colors[marker_i])
            
        axis.set_title(f"Frame {step_i}")
        plt.show()

def load_dlc_h5(path):
    df = pd.read_hdf(path)
    df = df.droplevel(0, axis=1)
    return df
    