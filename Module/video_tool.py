
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
def get_video_info(video_path):
    """
    Get video's information
    
    :param video_path(string): video path
    
    return (Dictionary)
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

def process_background_subtraction(args):
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

def parallel_background_subtraction(video_path, 
                                    output_dir_path, 
                                    split_window,
                                    n_process = 5):
    """
    Process parallel background subtraction
    
    :param video_path(string): video apth
    :param output_dir_path(string): output direcotry path after preprocessing
    :param split_window(int): the number of frames to be processed per one time
    :param n_process(int): the number of process
    """
    video_info = get_video_info(video_path)
    frame_count = video_info["frame_count"]
    fps = video_info["fps"]
    
    args = [(video_path, i, min(i + split_window, frame_count), output_dir_path) for i in range(0, frame_count, split_window)]
    with Pool(processes = n_process) as pool:
        for member in tqdm(pool.imap(process_background_subtraction, args), total = len(args)):
            pass

def do_background_subtraction(video_path, output_video_path):
    """
    Do background subtraction on video
    
    :param video_path(string): target video path to do background subtraction
    :param output_video_path(string): processed video path
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
    
def calc_pixel_sum(video_path, 
                   pixel_sum_path, 
                   roi_x = (0, 100), 
                   roi_y = (400, 480)):
    """
    Calculate pixel sum over roi across video

    :param video_path(string): video_path
    :param pixel_sum_path(string): Path to save this process' result
    :param roi_x(tuple - (from,to)): roi over x-axis
    :param roi_y(tuple - (from,to)): roi over y-axis

    return pixel_sums(list): sum of the pixel of each frame's roi
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

def calc_n_active_pixels(video_path, 
                         save_path, 
                         roi_x = (0, 100), 
                         roi_y = (400, 480)):
    """
    Calculate pixel sum over roi across video

    :param video_path(string): video_path
    :param save_path(string): Path to save this process' result
    :param roi_x(tuple - (from,to)): roi over x-axis
    :param roi_y(tuple - (from,to)): roi over y-axis

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

def cut_video_usingTime(video_path, output_path, start_sec, end_sec):
    """
    Cut video and save the result to output path
    
    :param video_path(string): origin video path
    :param output_path(string): output path
    :param start_sec(float): start time(seconds)
    :param end_sec(float): end time(seconds)
    """
    
    # Load video
    clip = VideoFileClip(video_path)
    
    # Create the subclip
    subclip = clip.subclipped(start_sec, end_sec)
    
    # Write the subclip to a file
    subclip.write_videofile(output_path)
    
def cut_video_usingFrame(video_path, output_path, start_frame, end_frame, fps):
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

def estimate_depth_monocular(video_path, 
                             model_dir_path = "/mnt/ext1/seojin/ComputerVision",
                             model_type = "DPT_Large"):
    """
    Estimate depth from video
    
    :param video_path(string): video path
    :param model_dir_path(string): depth estimation model path
    :param model_type(string): depth estimation model type
    
    return (np.array)
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

def get_video_frames(video_path):
    """
    Get video frames (RGB)

    :param video_path (str): path for video
    :return: np.array (n_frames, height, width, 3)
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

def convert_gray(video_path):
    """
    Convert image to gray scale

    :param video_path(string): video_path

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

def save_video(rgb_arrays, fps, output_path, is_progress_bar = False):
    """
    Save video from numpy array
    
    :param rgb_arrays(np.array - shape: (#frame, #y, #x, rgb)): rgb array frame
    :param fps: fps os video
    :param frame_count: frame count of video
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

def append_frames(video_path, rgb_frames, fps_of_rgb_frames, is_progress_bar = False):
    """
    Append video frames on the video_path
    
    :param video_path(string): video path
    :param rgb_frames(np.array - shape: (#frame, #y, #x, 3)): rgb image frames
    :param fps_of_rgb_frames(int): fps of rgb frames
    :param is_progress_bar(bool): is visualize progress bar
    """
    frame_count, ny, nx, nrgb = rgb_frames.shape
    time_duration = frame_count / fps_of_rgb_frames
    
    def make_frame(t):
        index = int(t * fps_of_rgb_frames)
        return rgb_frames[index]
    
    animation = VideoClip(make_frame, duration = time_duration)
    if os.path.exists(video_path):
        video = VideoFileClip(video_path)
        clip = concatenate_videoclips([video, animation])
    else:
        clip = animation
    clip.write_videofile(video_path, 
                         fps = fps_of_rgb_frames, 
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

def calc_pixel_sum(video_path, 
                   pixel_sum_path, 
                   roi_x = (0, 100), 
                   roi_y = (400, 480), 
                   split_window = 100,
                   num_jobs = -1):
    """
    Calculate pixel sum over roi across video and accumulate frame by frame

    :param video_path(string): video_path
    :param pixel_sum_path(string): Path to save this process' result
    :param roi_x(tuple - (from,to)): roi over x-axis
    :param roi_y(tuple - (from,to)): roi over y-axis
    :param num_jobs(int): Number of parallel jobs to run (-1 uses all processors)

    return pixel_sums(list): cumulative sum of the pixel of each frame's roi
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
                ax.scatter(xyz[:,0], xyz[:,2], xyz[:,1], s=20, color=colors[i])
            else:
                xyz = pos_array[step_i, i, :]
                ax.scatter(xyz[0], xyz[2], xyz[1], s=20, color=colors[i])
            
        for body_part1, body_part2 in skeletons:
            idx1 = bodyparts.index(body_part1)
            idx2 = bodyparts.index(body_part2)
            dlc3d = np.array([pos_array[step_i, idx1, :], pos_array[step_i, idx2, :]])
            ax.plot(dlc3d[:, 0], dlc3d[:, 2], dlc3d[:, 1], c="black")

        # Visualize - Static Objects (e.g., table, environment boundaries)
        obj_traces = []
        for obj_name in obj_info:
            obj_pts = np.array(obj_info[obj_name]["points"])
            ax.plot(obj_pts[:, 0], obj_pts[:, 2], obj_pts[:, 1], color="black")
            
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