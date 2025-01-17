# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:39:39 2020

@author: seojin
"""

# Common Libraries
import cv2
import numpy as np
import matplotlib.pylab as plt
import sys
from skimage import measure
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from collections import defaultdict, deque

# Custom Libraries
import sj_higher_function
from sj_timer import convert_time_to_second, convert_second_to_time, convert_second_to_frame
from sj_sequence import slice_list_usingDiff

# Functions
def shift_image(X, dx, dy):
    """
    shift numpy image

    :param X: image
    :param dx: move distance about x
    :param dy: move distance about y
    :return: moved image
    """
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def specify_color(img, rgb):
    """
    find pixel of image matched color

    :param img: image
    :param rgb: rgb
    :return: image(extracted specific color)
    """
    return (img[:, :, 0] == rgb[0]) & (img[:, :, 1] == rgb[1]) & (img[:, :, 2] == rgb[2])

def sizing_aspect_ratio(origin_w, origin_h, width = None, height = None):
    """
    adjusted image size streched by ratio

    In arguments, width or height is only specified one

    :param origin_w: origin width
    :param origin_h: origin height
    :param width: target width
    :param height: target height
    :return:
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = (origin_w, origin_h)

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return (origin_w, origin_h)

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        r = height / float(h)
        return (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        r = width / float(w)
        return (width, int(h * r))
    
def upright_camera_img(img, is_transpose_first = True):
    if is_transpose_first:
        result = cv2.transpose(img)
        result = cv2.flip(result, 1)
    else:
        result = cv2.flip(img, 1)
        result = cv2.transpose(result)
    return result

def remove_overlapped_pixels(pivot_img, compare_img):
    """
    remove overlapped pixel between pivor and compare image

    :param pivot_img: pivot image
    :param compare_img: compare image
    :return: image
    """
    if pivot_img.shape != compare_img.shape:
        raise Exception('이미지간의 shape이 맞아야 합니다.')

    pivot_img_gray = cv2.cvtColor(pivot_img, cv2.COLOR_RGB2GRAY) # 3ms
    compare_img_gray = cv2.cvtColor(compare_img, cv2.COLOR_RGB2GRAY) # 3ms

    pivot_thres1 = cv2.threshold(pivot_img_gray, 1, 255, cv2.THRESH_BINARY)[1] # 3ms
    compare_thres2 = cv2.threshold(compare_img_gray, 1, 255, cv2.THRESH_BINARY)[1] # 3ms

    overlap_mask = cv2.bitwise_and(pivot_thres1, compare_thres2) # 16ms

    return cv2.bitwise_and(pivot_img, pivot_img, mask=cv2.bitwise_not(overlap_mask)) # 16ms

def sobel_edge_img(img, ksize = 3):
    """
    apply sobel edge in img

    :param img: image
    :param ksize: sobel filter size
    :return: filtered image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x) 
    img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y) 

    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

    return img_sobel

def canny(img):
    """
    apply canny in img

    :param img: image
    :return: filtered image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2.Canny(img, 170, 200)
    
    return [edge1, edge2, edge3]

'''
Input:
    img: 0과 255로 이루어진 이미지(255가 object이어야 한다)

Return:
    object의 rectangle 반환
'''
def find_obj_rectangle(img):
    """
    find rectangle of non zero area
    In this function, the obj means non zero area

    :param img: image
    :return: ((x,y) width, height)
    """
    l_padding = find_padding_size(img, 'L')
    r_padding = find_padding_size(img, 'R')
    t_padding = find_padding_size(img, 'T')
    b_padding = find_padding_size(img, 'B')
    
    x_start = l_padding + 1
    y_start = t_padding + 1

    total_w = img.shape[1]
    total_h = img.shape[0]

    obj_w = total_w - (l_padding + r_padding)
    obj_h = total_h - (t_padding + b_padding)

    return ((x_start, y_start), obj_w, obj_h)

def find_padding_size(img, direction):
    """
    find padding size along direction about image
    In this function, padding means zero areas
    To find the padding, find zero pixels along direction

    :param img: image
    :param direction: direction of finding, Left, Right, Top, Bottom ex) "L", "R", "T", "B"
    :return: padding size
    """
    img_cp = img[:]
    
    padding_count = 0
    
    if direction == 'L':
        while len(img_cp) != 0:
            cursor = img_cp[:,0]
            if sum(cursor) == 0:
                img_cp = img_cp[:,1:]
                padding_count += 1
            else:
                break          
    elif direction == 'R':
        while len(img_cp) != 0:
            cursor = img_cp[:,-1]
            if sum(cursor) == 0:
                img_cp = img_cp[:,:-1]
                padding_count += 1
            else:
                break
    elif direction == 'T':
        while len(img_cp) != 0:
            cursor = img_cp[0,:]
            if sum(cursor) == 0:
                img_cp = img_cp[1:,:]
                padding_count += 1
            else:
                break
    elif direction == 'B':
        while len(img_cp) != 0:
            cursor = img_cp[-1,:]
            if sum(cursor) == 0:
                img_cp = img_cp[:-1,:]
                padding_count += 1
            else:
                break
    return padding_count

class Image_preprocessing:
    # degree: absolute value
    def rotate_center(img, degree, rot_center, is_clockwise, scale = 1):
        """
        rotate image

        :param degree: degree ex) 90
        :param rot_center: rotating center position ex) [10, 10]
        :param is_clockwise: rotating direction ex) True, False
        :param scale: scale factor
        :return: rotate image
        """
        height, width, channel = img.shape

        # getRotationMatrix2D: Positive value mean counter-clockwise rotation

        # Rotation Matrix 2D할때 어디로 Rotation 되었는지 뽑아야함

        degree = np.abs(degree)
        if is_clockwise == True:
            matrix = cv2.getRotationMatrix2D((rot_center[0], rot_center[1]), -degree, scale)
        else:
            matrix = cv2.getRotationMatrix2D((rot_center[0], rot_center[1]), degree, scale)
            
        result = img.copy()
        result = cv2.warpAffine(result, matrix, (width, height))

        return result
    
    def get_rotate_point(pt, center_pt, degree, is_clockwise):
        """
        rotate point

        :param center_pt:rotating center position ex) [10, 10]
        :param degree: degree ex) 90
        :param is_clockwise: rotating direction ex) True, False
        :return: point
        """
        pt = np.array(pt)
        center_pt = np.array(center_pt)
        
        degree = np.abs(degree)
        if is_clockwise == True:
            degree = degree
        else:
            degree = -degree
        
        trans_pt = pt - center_pt
        rot_pt = np.array(Image_preprocessing.rotate_point(trans_pt, degree))
        fin_pt = rot_pt + center_pt
            
        return fin_pt

    def rotate_point(pts, center_pt, degree, scale):
        """
        rotate point clock wise

        :param center_pt:rotating center position ex) [10, 10]
        :param degree: degree ex) 90
        :param scale: scale
        :return: point
        """
        pts = np.expand_dims(pts, axis=0)
        
        M = cv2.getRotationMatrix2D((center_pt[0], center_pt[1]), degree, scale)
        
        result = cv2.transform(pts, M)
        return np.squeeze(result, axis = 0)

def show_image_byFrame(cap, frame_scan_interval, start_time, target_time, fig_width = 20, fig_height = 80):
    """
    Show image using frame
    
    :param cap: video capture
    :param frame_scan_interval: scanning interval to show image
    :param start_time: Start video time(h, m, s)
    :param target_time: target video time(h, m, s)
    :param fig_width: figure width
    :param fig_height: figure height
    """
    # Start time
    start_h = start_time[0]
    start_m = start_time[1]
    start_s = start_time[2]
    
    start_seconds = convert_time_to_second(h = start_h, m = start_m, s = start_s)
    
    # Target time
    target_h = target_time[0]
    target_m = target_time[1]
    target_s = target_time[2]

    target_seconds = convert_time_to_second(h = target_h, m = target_m, s = target_s)
    
    # Time(second) interval
    sec_interval = target_seconds - start_seconds

    # Target frame
    target_frame = convert_second_to_frame(sec_interval, frame_per_second)

    # Scans
    scans = np.array(list(range(0, frame_scan_interval)))
    scans = sorted(np.append(scans, -scans))

    fig, axises = plt.subplots(frame_scan_interval, 2)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)
    
    axises = sj_higher_function.flatten_2d(axises)
    
    i = 0
    for scan in scans:
        axis = axises[i]

        frame_number = int(target_frame + scan)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number);
        success, image = cap.read()

        axis.imshow(image)
        axis.set_title("frame number: " + str(frame_number) + " " + "scan number: " + str(scan))

        i += 1

def split_video_byFrame(cap, fps, start_frame, end_frame, video_size, output_path):
    """
    split video using frame
    
    :param cap: video capture
    :param start_frame: start frame
    :param end_frame: end frame
    :param video_size: video size
    :param output_path: output path
    """    
    frame_duration = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame);
    
    out = cv2.VideoWriter(filename = output_path, 
                          fourcc = cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps = fps,
                          frameSize = video_size)
    
    for i in range(0, frame_duration):
        success, image = cap.read()
        
        out.write(image)
        
    print("save: ", output_path)

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports, working_ports

def show_camera_spec(cap):
    return {
        "frame_width" : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height" : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count" : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fourcc" : int(cap.get(cv2.CAP_PROP_FOURCC)),
        "fps" : int(cap.get(cv2.CAP_PROP_FPS)),
        "buffer_size" : int(cap.get(cv2.CAP_PROP_BUFFERSIZE)),
    }

def find_cluster_onBinaryImage(image, thres_n_neighbor):
    """
    Find cluster on binary image
    
    :param image(np.array - 2d): image array
    :param thres_n_neighbor(int); threshold of the number of neighbor
    
    return cluster data(np.array)
    """
    labels = measure.label(image, connectivity=2)
    cluster_numbers = np.unique(labels)
    
    # Constraint
    for number in cluster_numbers:
        cluster = labels == number
        n_neighbor = np.sum(cluster)
        if n_neighbor < thres_n_neighbor:
            labels[cluster] = 0
    
    # Renumbering
    cluster_numbers = np.unique(labels)
    re_numbering = np.arange(len(cluster_numbers))
    for cluster_number, re_number in zip(cluster_numbers, re_numbering):
        mask = labels == cluster_number
        labels[mask] = re_number
    return labels

def draw_temporal_correlation(frames, fix_x, fix_y, is_vis = False):
    """
    Draw temporal correlation between pixels
    
    :param frames(np.array - shape: n_t, n_y, n_x): frames
    :param fix_x(int): reference pixel x-index
    :param fix_y(int): reference pixel y-index
    """
    spatial_x_pearsons = []
    for x in range(n_x):
        stat = pearsonr(frames[:, fix_y, fix_x], frames[:, fix_y, x]).statistic
        spatial_x_pearsons.append(stat)
    spatial_x_pearsons = np.array(spatial_x_pearsons)
    
    spatial_y_pearsons = []
    for y in range(n_y):
        stat = pearsonr(frames[:, fix_y, fix_x], frames[:, y, fix_x]).statistic
        spatial_y_pearsons.append(stat)
    spatial_y_pearsons = np.array(spatial_y_pearsons)
    
    if is_vis:
        fig, axes = plt.subplots(2)
        axes[0].plot(spatial_x_pearsons)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("corr")
        valid_corrs = spatial_x_pearsons[np.logical_not(np.isnan(spatial_x_pearsons))]
        axes[0].axvline(x = fix_x, 
                        ymin = np.min(valid_corrs), 
                        ymax = np.max(valid_corrs), 
                        color = "red")
        
        axes[1].plot(spatial_y_pearsons)
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("corr")
        axes[1].axvline(x = fix_y, 
                        ymin = np.min(spatial_y_pearsons), 
                        ymax = np.max(spatial_y_pearsons), 
                        color = "red")
        
        fig.tight_layout()

def divide_screen(frame, n_rect_x, n_rect_y):
    """
    Divide screen using rect
    
    :param n_rect_x(int): the number of rect across x-axis
    :param n_rect_y(int): the number of rect across y-axis
    
    return rect information(pd.DataFrame)
        - index: rect_x_index, rect_y_index
        - column: x1(left up), x2(right down), y1(left up), y2(right down)
    """
    fig, axis = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(10)

    dx = width / n_rect_x
    dy = height / n_rect_y

    rect_width = dx
    rect_height = dy

    x1 = np.arange(0, width, rect_width).astype(np.int16)
    x2 = np.arange(0 + rect_width, width + 1, rect_width).astype(np.int16)

    y1 = np.arange(0, height, rect_height).astype(np.int16)
    y2 = np.arange(0 + rect_height, height + 1, rect_height).astype(np.int16)

    rect_pos_datas = []
    rect_pos_data_indexes = []
    for rect_x_i, rect_y_i in product(np.arange(n_rect_x), np.arange(n_rect_y)):
        rect_pos_datas.append([x1[rect_x_i], x2[rect_x_i], y1[rect_y_i], y2[rect_y_i]])
        rect_pos_data_indexes.append((rect_x_i, rect_y_i))

    rect_pos_df = pd.DataFrame(rect_pos_datas)
    rect_pos_df.columns = ["x1", "x2", "y1", "y2"]
    rect_pos_df.index = rect_pos_data_indexes

    axis.imshow(frame, cmap = "gray")
    for rect_data in rect_pos_df.iterrows():
        i, rect = rect_data

        rect_width = rect["x2"] - rect["x1"]
        rect_height = rect["y2"] - rect["y1"]

        axis.add_patch(plt.Rectangle(xy = (rect["x1"], rect["y1"]), 
                                     width = rect_width, 
                                     height = rect_height, 
                                     fill = False, 
                                     edgecolor = 'red', 
                                     linewidth = 3))
    return rect_pos_df

def detect_high_gradients_regions(frames,
                                  time_index,
                                  y_index, 
                                  grad_threshold,
                                  n_region_threshold = 10,
                                  region_continue_threshold = 5,
                                  is_vis = False):
    """
    Detect high gradient regions across x-axis given y index
    
    This function is for detecting edge of an image
    
    :param frames(np.array - shape: n_t, n_y, n_x): frames
    :param time_index(int): target index of frames
    :param y_index(int): y index  
    :param grad_threshold(int): gradient threshold
    :param n_region_threshold(int): region width to filter thin regions
    :param region_continue_threshold(int): threshold to detect continuous region
    :param is_vis(bool): is visualized a result
    """
    n_t, n_y, n_x = frames.shape
    
    # High gradient
    gradients = np.gradient(frames[time_index, y_index, :])
    high_grad_indexes = np.where(np.abs(gradients) > grad_threshold)[0]
    t_high_grad_indexes = list(high_grad_indexes)
    
    # Sort
    t_high_grad_indexes = np.array(sorted(t_high_grad_indexes))
    
    # Detect region
    diff_indexes = t_high_grad_indexes[1:] - t_high_grad_indexes[:-1]
    
    previous_stop_i = 0
    region_indexes = []
    for slicing_start_i, slicing_stop_i in slice_list_usingDiff(diff_indexes):
        if slicing_stop_i - slicing_start_i > n_region_threshold:
            start_i = t_high_grad_indexes[slicing_start_i]
            stop_i = t_high_grad_indexes[slicing_stop_i]
            
            
            if (start_i - previous_stop_i) > region_continue_threshold:
                region_indexes.append((start_i, stop_i))
                previous_stop_i = stop_i

    if is_vis:
        cmap = get_cmap("viridis", len(region_indexes))        

        plt.hlines(y_index, xmin = 0, xmax = n_x - 1, color = "red")
        
        i = 0
        for r_start, r_stop in region_indexes:
            color = rgb2hex(cmap(i)[:3])
            plt.fill_betweenx(y = [0,n_y], x1 = r_start, x2 = r_stop, alpha = 0.3, color = color)
            i += 1
            
        plt.imshow(frames[time_index, :, :], cmap = "gray")
            
    return region_indexes

def find_connected_components_faces(faces):
    """
    Find connected components based on faces
    
    :param faces(np.array - shape: (#face, 3)): each containing 3 vertex indices

    return list
    """
    # Step 1: Build an adjacency list for faces based on shared vertices
    face_adjacency = defaultdict(list)
    
    # Map vertices to the faces they belong to
    vertex_to_faces = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_index)
    
    # Build the adjacency list for faces
    for face_index, face in enumerate(faces):
        neighbors = set()
        for vertex in face:
            neighbors.update(vertex_to_faces[vertex])
        neighbors.discard(face_index)  # Remove the face itself from its neighbors
        face_adjacency[face_index] = list(neighbors)
    
    # Step 2: Find connected components using BFS
    visited = set()
    connected_components = []
    
    def bfs(start):
        """Perform BFS to find all faces in the connected component."""
        queue = deque([start])
        component = []
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            component.append(current)
            
            for neighbor in face_adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return component
    
    # Iterate through all faces to find all connected components
    for face_index in range(len(faces)):
        if face_index not in visited:
            # Start a BFS for this component
            component = bfs(face_index)
            connected_components.append(component)
    
    return connected_components

if __name__ == "__main__":
    frames = convert_gray(video_path)
    draw_temporal_correlation(frames = frames, fix_x = 0, fix_y = 0, is_vis = True)
    divide_screen(frames, n_rect_x = 2, n_rect_y = 3)
    
    # Detect edge
    n_t, n_y, n_x = frames.shape
    edges = np.repeat(False, n_t * n_y * n_x).reshape(n_t, n_y, n_x)
    for time_index in range(n_t):
        for y_index in range(n_y):
            high_gradient_regions = detect_high_gradients_regions(frames, 
                                                                  time_index = time_index,
                                                                  y_index = y_index, 
                                                                  n_region_threshold = 1,
                                                                  grad_threshold = 4,
                                                                  region_continue_threshold = 1,
                                                                  is_vis=False)
            for region in high_gradient_regions:
                edges[time_index, y_index, region[0]] = True
