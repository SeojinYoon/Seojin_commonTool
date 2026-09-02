
# Common Libraries
import os
import cv2
import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import Union
import matplotlib.pylab as plt
from multiprocessing import Pool
from joblib import Parallel, delayed
from moviepy import VideoFileClip, VideoClip, concatenate_videoclips

# Functions
def get_video_info(video_path: str) -> dict:
    """
    Get video's information
    
    :param video_path: video path
    
    return video information
        -key: frame_count
        -key: width
        -key: height
        -key: fps
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    return {
        "frame_count" : frame_count,
        "width" : w,
        "height" : h,
        "fps" : fps,
    }

def process_background_subtraction(args: tuple):
    """
    Do background subtraction
    
    :param args: video_path, start_index, end_index of frame
    
    return (np.array - shape: (#frame, #y, #x))
    """
    video_path, start_i, end_i, dir_path = args
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    results = []
    for i in range(start_i, end_i):
        ret, frame = cap.read()
        if not ret:
            break
        fgMask = backSub.apply(frame)
        results.append(cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB))
    cap.release()
    
    save_path = os.path.join(dir_path, f"bg_sub_{start_i}.mp4")
    save_video(rgb_arrays = np.array(results), fps = fps, output_path = save_path)

def parallel_background_subtraction(video_path: str, 
                                    output_dir_path: str, 
                                    split_window: int,
                                    n_process: int = 5):
    """
    Process parallel background subtraction
    
    :param video_path: video apth
    :param output_dir_path: output direcotry path after preprocessing
    :param split_window: number of frames per chunk for parallel processing
    :param n_process: the number of process
    """
    video_info = get_video_info(video_path)
    frame_count = video_info["frame_count"]
    fps = video_info["fps"]
    
    args = [(video_path, i, min(i + split_window, frame_count), output_dir_path) for i in range(0, frame_count, split_window)]
    with Pool(processes = n_process) as pool:
        for member in tqdm(pool.imap(process_background_subtraction, args), total = len(args)):
            pass

def do_background_subtraction(video_path: str, 
                              output_video_path: str):
    """
    Do background subtraction on video
    
    :param video_path: target video path to do background subtraction
    :param output_video_path: processed video path
    """
    if os.path.exists(output_video_path):
        print(f"Output already exists: {output_video_path}")    
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Files
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    #create Background Subtractor objects
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Convert origin video to background subtraction video
    for i in tqdm(range(frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        ret, frame = cap.read()    
        #update the background model
        fgMask = backSub.apply(frame)

        fg = cv2.copyTo(frame,fgMask)
        out.write(fg)
    cap.release()
    out.release()
    
def calc_pixel_sum(video_path: str, 
                   pixel_sum_path: str, 
                   roi_x: tuple = (0, 100), 
                   roi_y: tuple = (400, 480)) -> list:
    """
    Calculate pixel sum over roi across video

    :param video_path: video_path
    :param pixel_sum_path(string): Path to save this process' result
    :param roi_x: roi over x-axis (from,to)
    :param roi_y: roi over y-axis (from,to)

    return pixel_sums: sum of the pixel of each frame's roi
    """
    if os.path.exists(pixel_sum_path):
        pixel_sums = np.load(pixel_sum_path)
    else:
        bgs_cap = cv2.VideoCapture(video_path)
        frame_count = int(bgs_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pixel_sums = []

        ranges = range(frame_count)
        for i in tqdm(ranges):
            ret, frame = bgs_cap.read()
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corner_image = imgGray[roi_y[0]: roi_y[1], roi_x[0]: roi_x[1]]
            
            pixel_sums.append(np.sum(corner_image))
            
        # Save pixel sum
        pixel_sums = np.array(pixel_sums)
        np.save(pixel_sum_path, pixel_sums)
        bgs_cap.release()
    return pixel_sums

def calc_n_active_pixels(video_path: str, 
                         save_path: str, 
                         roi_x: tuple = (0, 100), 
                         roi_y: tuple = (400, 480)):
    """
    Calculate pixel sum over roi across video

    :param video_path: video_path
    :param save_path: Path to save this process' result
    :param roi_x: roi over x-axis (from,to) ex) (0, 100)
    :param roi_y: roi over y-axis (from,to) ex) (400, 480)

    return pixel_sums(list): sum of the pixel of each frame's roi
    """
    if os.path.exists(save_path):
        n_active_pixels = np.load(save_path)
    else:
        bgs_cap = cv2.VideoCapture(video_path)
        frame_count = int(bgs_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_active_pixels = []

        ranges = range(frame_count)
        for i in tqdm(ranges):
            ret, frame = bgs_cap.read()
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corner_image = imgGray[roi_y[0]: roi_y[1], roi_x[0]: roi_x[1]]
            
            n_active_pixel = len(np.where(corner_image > 0)[0])
            n_active_pixels.append(n_active_pixel)
            
        # Save pixel sum
        n_active_pixels = np.array(n_active_pixels)
        np.save(save_path, n_active_pixels)
        bgs_cap.release()
    return n_active_pixels

def cut_video_usingTime(video_path: str,
                        output_path: str,
                        start_sec: float,
                        end_sec: float):
    """
    Cut video and save the result to output path
    
    :param video_path: origin video path
    :param output_path: output path
    :param start_sec: start time(seconds)
    :param end_sec: end time(seconds)
    """
    
    # Load video
    clip = VideoFileClip(video_path)
    
    # Create the subclip
    subclip = clip.subclipped(start_sec, end_sec)
    
    # Write the subclip to a file
    subclip.write_videofile(output_path)
    
def cut_video_usingFrame(video_path: str, 
                         output_path: str,
                         start_frame: int,
                         end_frame: int,
                         fps: int):
    """
    Cut video and save the result to output path
    
    :param video_path(string): origin video path
    :param output_path(string): output path
    :param start_frame(int): start cut frame
    :param end_frame(int): end cut frame
    :param fps(int): frame per second
    """
        
    # Calculate start and end times in seconds
    start_sec = start_frame / fps
    end_sec = end_frame / fps
    
    # Cut video
    cut_video_usingTime(video_path, output_path, start_sec, end_sec)

def estimate_depth_monocular(video_path: str, 
                             model_dir_path: str,
                             model_type: str = "DPT_Large") -> np.array:
    """
    Estimate depth from video
    
    :param video_path(string): video path
    :param model_dir_path(string): depth estimation model path
    :param model_type(string): depth estimation model type
    
    :return: depth estimation result of each images
    """
    import torch
    
    # MiDaS
    torch.hub.set_dir(model_dir_path)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    # Video information
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    results = []
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = transform(frame).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction_ = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        results.append(prediction_.cpu().numpy())
    return np.c_[results]

def get_video_frames(video_path: str) -> np.array:
    """
    Get video frames (RGB)

    :param video_path: path for video
    :return: [n_frames, height, width, 3]
    """

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results.append(frame)

    cap.release()

    return np.stack(results)

def convert_gray(video_path: str):
    """
    Convert image to gray scale

    :param video_path: video_path

    return frames
    """
    if os.path.exists(video_path):
        results = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ranges = range(frame_count)
        for i in tqdm(ranges):
            ret, frame = cap.read()
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results.append(imgGray)
            
        cap.release()
    return np.c_[results]

def save_video(rgb_arrays: np.array, 
               fps: int,
               output_path: str,
               is_progress_bar: bool = False):
    """
    Save video from numpy array
    
    :param rgb_arrays: rgb array frame (#frame, #y, #x, rgb)
    :param fps: fps os video
    :param is_progress_bar: is progress bar showing
    """
    frame_count, ny, nx, nrgb = rgb_arrays.shape
    time_duration = frame_count / fps
    
    def make_frame(t):
        index = int(t * fps)
        return rgb_arrays[index]
    
    animation = VideoClip(make_frame, duration = time_duration)

    my_logger = 'bar' if is_progress_bar else None
    animation.write_videofile(output_path, 
                              fps = fps,
                              logger = my_logger)
    print(f"save: {output_path}")

def append_frames(video_path: str,
                  rgb_frames: np.array, 
                  fps: int, 
                  is_progress_bar: bool = False):
    """
    Append video frames on the video_path
    
    :param video_path: video path
    :param rgb_frames: rgb image frames (#frame, #y, #x, 3)
    :param fps: frame per second
    :param is_progress_bar: is showing progress bar
    """
    frame_count, ny, nx, nrgb = rgb_frames.shape
    time_duration = frame_count / fps
    
    def make_frame(t):
        index = int(t * fps)
        return rgb_frames[index]
    
    animation = VideoClip(make_frame, duration = time_duration)
    if os.path.exists(video_path):
        video = VideoFileClip(video_path)
        clip = concatenate_videoclips([video, animation])
    else:
        clip = animation
    clip.write_videofile(video_path, 
                         fps = fps, 
                         verbose = is_progress_bar, 
                         logger = None)

def pixel_sum(args):
    video_path, start_i, end_i, roi_x, roi_y = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
   
    results = []
    for i in range(start_i, end_i):
        ret, frame = cap.read()
        if not ret:
            break
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_image = imgGray[roi_y[0]: roi_y[1], roi_x[0]: roi_x[1]]
        results.append(np.sum(corner_image))
    cap.release()
    
    return results

def calc_pixel_sum(video_path: str, 
                   pixel_sum_path: str, 
                   roi_x: tuple, 
                   roi_y: tuple, 
                   split_window: int = 100,
                   num_jobs: int = -1) -> list:
    """
    Calculate pixel sum over roi across video and accumulate frame by frame

    :param video_path: video_path
    :param pixel_sum_path: Path to save this process' result
    :param roi_x: roi over x-axis (from,to) ex) (0, 100)
    :param roi_y: roi over y-axis (from,to) ex) (400, 480)
    :param split_window: number of frames per chunk for parallel processing
    :param num_jobs: number of parallel jobs to run (-1 uses all processors)

    return cumulative sum of the pixel of each frame's roi
    """
    if os.path.exists(pixel_sum_path):
        pixel_sums = np.load(pixel_sum_path)
    else:
        video_info = get_video_info(video_path)
        frame_count = video_info["frame_count"]
        
        args = []
        for start_i in range(0, frame_count, split_window):
            args.append((video_path, start_i, min(start_i + split_window, frame_count), roi_x, roi_y))
        
        with Pool(processes=num_jobs) as pool:
            pixel_sums = list(tqdm(pool.imap(pixel_sum, args), total = len(args), desc = "Processing frames"))
        
        pixel_sums = np.concatenate(pixel_sums)
        
        # Save pixel sum 
        np.save(pixel_sum_path, pixel_sums)
        
    return pixel_sums

def compare_trajectories_to_video(trajectories: list[np.ndarray],
                                  output_path: str,
                                  labels: list[str] = None,
                                  figsize: tuple[Union[int, float], Union[int, float]] =(6, 6),
                                  fps: int = 10):
    """
    Save multiple 2D trajectories as a video using cv2.VideoWriter.

    Each trajectory must be a NumPy array of shape (T, 2), where:
        - T is the number of time points
        - column 0 is X
        - column 1 is Y

    :param trajectories: List of trajectories. Each element must be a NumPy array of shape (T, 2).
    :param output_path: Output video file path.
    :param labels: Optional list of labels corresponding to each trajectory.
                   If None, default labels ("Traj 0", "Traj 1", ...) are used.
    :param figsize: Size of the matplotlib figure (width, height).
    :param fps: Frames per second for output video.

    :return: None
    """

    if labels is None:
        labels = [f"Traj {i}" for i in range(len(trajectories))]

    max_index = max(traj.shape[0] for traj in trajectories) - 1

    x_min = min(traj[:, 0].min() for traj in trajectories)
    x_max = max(traj[:, 0].max() for traj in trajectories)
    y_min = min(traj[:, 1].min() for traj in trajectories)
    y_max = max(traj[:, 1].max() for traj in trajectories)

    markers = ["o", "s", "^", "D", "x", "*"]

    # Determine frame size
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    plt.close(fig)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def make_frame(idx):
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
                ax.scatter(traj[-1, 0],
                           traj[-1, 1],
                           s=100,
                           marker=markers[i % len(markers)],
                           label=labels[i])

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
        ax.legend()

        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        plt.close(fig)
        return frame

    for idx in range(max_index + 1):
        frame = make_frame(idx)
        writer.write(frame)

    writer.release()
    print(f"Saved to: {output_path}")

def extract_frames(video: cv2.VideoCapture, frame_idx: np.ndarray) -> np.ndarray:
    """
    Extract sub frames from a video 

    :param video: video
    :param frame_idx: Indices of the frames to extract

    return frames (#frame, height, width, 3)
    """
    n_frames = len(frame_idx)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.zeros((n_frames, height, width, 3), dtype = np.uint8)
    for i, frame_i in enumerate(frame_idx):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = video.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[i] = frame
    return frames
    
if __name__ == "__main__":
    parallel_background_subtraction(video_path, output_video_path, split_window = 30, n_process = 5)
    
    dir_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Camera1/video_perTrial"
    video_path = os.path.join(dir_path, "trial2.mp4")
    result = estimate_depth_monocular(video_path)
    
    frames = get_video_frames(video_path)
        
    rgb_array = np.random.rand(135, 480, 928, 3)
    fps = 30
    output_path = "output.mp4"
    save_video(rgb_array, fps, output_path)
    
    append_frames(video_path, rgb_array, 30)
    