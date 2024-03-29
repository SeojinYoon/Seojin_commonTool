# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:39:39 2020

@author: seojin
"""

import cv2
import numpy as np
import prj_info
import matplotlib.pylab as plt
import sys

import sj_higher_function
from sj_timer import convert_time_to_second, convert_second_to_time, convert_second_to_frame

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
