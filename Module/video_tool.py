
# Common Libraries
import os
import cv2
import numpy as np
import xarray as xr
from tqdm import tqdm
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

def save_3d_pos_video(pos_3d: xr.core.dataset.Dataset,
                      bodyparts: list,
                      output_path: str,
                      obj_info: dict,
                      continuous_body_parts: list = [],
                      skeletons = [],
                      fps: int = 30,
                      figsize = (15, 5),
                      dpi = 100):
    """
    Save pose video

    :param pos_3d: (#frame, #marker, 3)
    :param bodyparts: list or array of bodypart names
    :param continuous_body_parts: list of bodypart for continous plot
    :param output_path: output mp4 path
    :param tablet_angle: angle of table
    :param skeletons: list of tuples consisting of bodypart, e.g. [("Shoulder", "Elbow"), ...]
    :param fps: output video fps
    :param figsize: matplotlib figure size
    :param dpi: figure dpi
    :parma table_size: tablet width and height
    """

    # Data information
    n_frames = len(pos_3d["Times"])
    pos_array = pos_3d["3D"].to_numpy()

    x_index, y_index, z_index = 0, 1, 2
    
    # Plot configuration
    colors = plt.cm.tab10(np.linspace(0, 1, len(bodyparts)))
    x_min, x_max = np.nanmin(pos_array[:, :, 0]), np.nanmax(pos_array[:, :, 0])
    y_min, y_max = np.nanmin(pos_array[:, :, 1]), np.nanmax(pos_array[:, :, 1])
    z_min, z_max = np.nanmin(pos_array[:, :, 2]), np.nanmax(pos_array[:, :, 2])
    
    fig_w = int(figsize[0] * dpi)
    fig_h = int(figsize[1] * dpi)
    
    # Video configuration
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fig_w, fig_h))

    for step_i in range(n_frames):
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # 3D
        for i, bodypart in enumerate(bodyparts):
            if bodypart in continuous_body_parts:
                xyz = pos_array[0:step_i+1, i, :]
                ax.scatter(xyz[:,x_index], xyz[:,z_index], xyz[:,y_index], s=20, color=colors[i])
            else:
                xyz = pos_array[step_i, i, :]
                ax.scatter(xyz[x_index], xyz[z_index], xyz[y_index], s=20, color=colors[i])
            
        for body_part1, body_part2 in skeletons:
            idx1 = bodyparts.index(body_part1)
            idx2 = bodyparts.index(body_part2)
            dlc3d = np.array([pos_array[step_i, idx1, :], pos_array[step_i, idx2, :]])
            ax.plot(dlc3d[:, x_index], dlc3d[:, z_index], dlc3d[:, y_index], c="black")

        # Visualize - Static Objects (e.g., table, environment boundaries)
        obj_traces = []
        for obj_name in obj_info:
            for type_ in obj_info[obj_name]["points"]:
                obj_pts = np.array(obj_info[obj_name]["points"][type_])
                ax.plot(obj_pts[:, x_index], obj_pts[:, z_index], obj_pts[:, y_index], color="black")
            
        # Others        
        ax.set_title("3D")
        ax.set_xlabel("X"), ax.set_ylabel("Z"), ax.set_zlabel("Y")
        ax.set_xlim(x_min, x_max), ax.set_ylim(z_min, z_max), ax.set_zlim(y_min, y_max)

        handles = [plt.Line2D([0], 
                              [0],
                              marker = "o",
                              color = "w",
                              markerfacecolor = colors[i],
                              markersize = 8,
                              label = bodyparts[i]) for i in range(len(bodyparts))]
        fig.legend(handles = handles, loc = "upper right")
        plt.tight_layout()

        # figure -> numpy image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        writer.write(img)
        plt.close(fig)

        if step_i % 50 == 0:
            print(f"Saved frame {step_i}/{n_frames}")
    
    writer.release()
    print(f"Video saved to: {output_path}")

def compare_trajectories_to_video(trajectories: list[np.ndarray],
                                  output_path: str,
                                  labels: list[str] = None,
                                  figsize: tuple[int | float, int | float] =(6, 6),
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

def save_3d_poses_video(
    position_ds_list,
    output_path: str,
    targets: list = [],
    skeletons: list = [],
    obj_info: dict = {},
    dataset_names: list = [],
    continuous_targets: list = [],
    fps: int = 30,
    figsize=(10, 8),
    dpi=100,
    elev=20,    # Elevation angle in degrees
    azim=-80    # Azimuthal angle in degrees
):
    """
    Generate and save a 3D pose estimation video from multiple xarray datasets.

    :param position_ds_list: List of xarray.Dataset containing the '3D' variable.
    :param output_path: String, the file path to save the resulting .mp4 video.
    :param targets: List of marker labels (strings) to visualize.
    :param skeletons: List of tuples defining connections, e.g., [("Shoulder", "Elbow")].
    :param obj_info: Dict containing static object points to render in the environment.
    :param dataset_names: List of names for each dataset to display in the legend.
    :param continuous_targets: List of markers that should leave a trajectory trail.
    :param fps: Integer, frames per second for the output video.
    :param figsize: Tuple (width, height) for the matplotlib figure size.
    :param dpi: Integer, dots per inch for the figure resolution.
    :param elev: Float, camera elevation angle in degrees.
    :param azim: Float, camera azimuthal angle in degrees.
    :return: None
    """

    # 1. Initialization
    n_ds = len(position_ds_list)
    if not dataset_names:
        dataset_names = [f"Dataset_{i}" for i in range(n_ds)]
    
    if not targets:
        targets = list(position_ds_list[0].Labels.to_numpy())

    # Data Indices
    x_idx, y_idx, z_idx = 0, 1, 2
    
    frame_counts = [len(ds["Times"]) for ds in position_ds_list]
    max_frames = max(frame_counts) 
    
    # 2. Calculate Global Axis Limits for stable camera focus
    all_min, all_max = [], []
    for ds in position_ds_list:
        arr = ds.sel(Labels=targets)["3D"].to_numpy()
        all_min.append(np.nanmin(arr, axis=(0, 1)))
        all_max.append(np.nanmax(arr, axis=(0, 1)))
    
    global_min = np.nanmin(all_min, axis=0)
    global_max = np.nanmax(all_max, axis=0)
    
    # 3. Color Configuration
    cmap = plt.get_cmap("tab10")
    ds_colors = [cmap(i % 10) for i in range(n_ds)]

    # 4. Video Writer Setup
    fig_w, fig_h = int(figsize[0] * dpi), int(figsize[1] * dpi)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fig_w, fig_h))

    # 5. Rendering Loop
    for step_i in range(max_frames):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # --- CAMERA VIEW CONFIGURATION ---
        ax.view_init(elev=elev, azim=azim)
        
        # Set aspect ratio based on mapping: (Data X, Data Z, Data Y)
        ax.set_box_aspect((
            global_max[x_idx] - global_min[x_idx], 
            global_max[z_idx] - global_min[z_idx], 
            global_max[y_idx] - global_min[y_idx]
        )) 

        # --- A. Visualize Static Objects ---
        for obj_name in obj_info:
            points_dict = obj_info[obj_name].get("points", {})
            for kind in points_dict:
                pts = np.array(points_dict[kind])
                # Plotting mapping: (X, Z, Y)
                ax.plot(pts[:, x_idx], pts[:, z_idx], pts[:, y_idx], color="black", alpha=0.3)

        # --- B. Visualize Multiple Datasets ---
        for ds_idx, ds in enumerate(position_ds_list):
            color = ds_colors[ds_idx]
            pos_array = ds["3D"].to_numpy()
            labels = list(ds.Labels.to_numpy())
            ds_n_frames = frame_counts[ds_idx]
            
            # Freeze shorter datasets at their last frame
            curr_idx = min(step_i, ds_n_frames - 1)

            for target in targets:
                if target not in labels: continue
                l_idx = labels.index(target)

                # Full static trajectory "shadow" (All frames)
                if target in continuous_targets:
                    full_path = pos_array[:, l_idx, :]
                    ax.plot(full_path[:, x_idx], full_path[:, z_idx], full_path[:, y_idx], 
                            color=color, alpha=0.1, linewidth=0.5)

                # Active trajectory trail (Up to current frozen frame)
                if target in continuous_targets:
                    current_path = pos_array[0 : curr_idx + 1, l_idx, :]
                    ax.plot(current_path[:, x_idx], current_path[:, z_idx], current_path[:, y_idx], 
                            color=color, alpha=0.6, linewidth=1.5)
                
                # Moving marker
                curr_xyz = pos_array[curr_idx, l_idx, :]
                ax.scatter(curr_xyz[x_idx], curr_xyz[z_idx], curr_xyz[y_idx], 
                           color=color, s=35, edgecolors='white', linewidth=0.5)

            # Draw Skeleton
            for p1, p2 in skeletons:
                if p1 in labels and p2 in labels:
                    idx1, idx2 = labels.index(p1), labels.index(p2)
                    bone_pts = np.array([pos_array[curr_idx, idx1, :], 
                                         pos_array[curr_idx, idx2, :]])
                    ax.plot(bone_pts[:, x_idx], bone_pts[:, z_idx], bone_pts[:, y_idx], 
                            color=color, linewidth=3, alpha=0.8)

        # --- C. Layout and Axis Labels ---
        ax.set_title(f"3D Comparison | Frame {step_i}/{max_frames}", pad=20)
        
        # Set limits based on mapping: X=DataX, Y=DataZ, Z=DataY
        ax.set_xlim(global_min[x_idx], global_max[x_idx])
        ax.set_ylim(global_min[z_idx], global_max[z_idx])
        ax.set_zlim(global_min[y_idx], global_max[y_idx])
        
        # Labels reflecting the Matplotlib axes relative to Data
        ax.set_xlabel("Data X-axis")
        ax.set_ylabel("Data Z-axis")
        ax.set_zlabel("Data Y-axis")
        
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ds_colors[i], 
                       markersize=10, label=dataset_names[i]) for i in range(n_ds)
        ]
        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        # Render Figure to Image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        writer.write(img)
        plt.close(fig)

        if step_i % 50 == 0:
            print(f"Rendering frame {step_i}/{max_frames}...")

    writer.release()
    print(f"Success! Video saved at: {output_path}")
    
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
    