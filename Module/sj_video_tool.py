
# Common Libraries
import os
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, VideoClip, concatenate_videoclips
import torch

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

def do_background_subtraction(video_path, output_video_path, split_window):
    """
    Do background subtraction on video
    
    :param video_path(string): target video path to do background subtraction
    :param output_video_path(string): processed video path
    :param split_window(int): the unit of processing
    """
    video_info = get_video_info(video_path)
    fps = video_info["fps"]
    frame_count = video_info["frame_count"]
    width = video_info["width"]
    height = video_info["height"]
    
    # Capture video
    cap = cv2.VideoCapture(video_path)
    
    # Create Background Subtractor objects
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    for start_i in tqdm(range(0, frame_count, split_window)):
        end_i = start_i + split_window

        result = np.zeros((end_i - start_i, height, width), dtype = np.uint8)
        for i in range(start_i, end_i):
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()    

            # Update the background model
            fgMask = backSub.apply(frame)
            result[i - start_i] = fgMask
        rgb_array = np.stack((result,)*3, axis=-1)
        append_frames(output_video_path, rgb_array, fps)

def calc_pixel_sum(video_path, pixel_sum_path, roi_x = (0, 100), roi_y = (400, 480)):
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
            bgs_cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = bgs_cap.read()
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corner_image = imgGray[roi_y[0]: roi_y[1], roi_x[0]: roi_x[1]]
            pixel_sums.append(np.sum(corner_image))

        # Save pixel sum
        pixel_sums = np.array(pixel_sums)
        np.save(pixel_sum_path, pixel_sums)
        bgs_cap.release()
    return pixel_sums

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
    subclip = clip.subclip(start_sec, end_sec)
    
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

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
    Get video frames
    
    :param video_path(string): path for video
    
    return (np.array)
    """
    # Video information
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = []
    for i in tqdm(range(frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        ret, frame = cap.read()
        results.append(frame)
    return np.c_[results]

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

def save_video(rgb_arrays, fps, output_path):
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
    animation.write_videofile(output_path, fps = fps)
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
        final_clip = concatenate_videoclips([video, animation])
        final_clip.write_videofile(video_path, fps = fps_of_rgb_frames, verbose=False, logger=None)
    else:
        animation.write_videofile(video_path, fps = fps_of_rgb_frames, verbose=False, logger=None)
    
if __name__ == "__main__":
    do_background_subtraction(video_path, output_video_path)
    
    dir_path = "/mnt/sdb2/DeepDraw/Projects/20220801_DP02_mri/Camera1/video_perTrial"
    video_path = os.path.join(dir_path, "trial2.mp4")
    result = estimate_depth_monocular(video_path)
    
    frames = get_video_frames(video_path)
        
    rgb_array = np.random.rand(135, 480, 928, 3)
    fps = 30
    output_path = "output.mp4"
    save_video(rgb_array, fps, output_path)
    
    append_frames(video_path, rgb_array, 30)