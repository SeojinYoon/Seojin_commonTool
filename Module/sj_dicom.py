import os
import sys
import numpy as np
from pydicom import dcmread
from datetime import datetime

def convert_dicom_time(dicom_date, dicom_time):
    """
    Converts DICOM DA (Date) and TM (Time) strings 
    into a single Python datetime object.
    """
    if not dicom_date or not dicom_time:
        return None
    
    # DICOM format is YYYYMMDD and HHMMSS.FFFFFF (or HHMMSS)
    # We strip the fractional seconds for basic parsing
    clean_time = dicom_time.split('.')[0]
    dt_string = f"{dicom_date}{clean_time}"
    
    return datetime.strptime(dt_string, "%Y%m%d%H%M%S")
    
def get_phillips_dicom_info(dicomdir_path: str, series_name: str) -> dict:
    """
    :param dicomdir_path: dicrectory path
    :param series_name: name of series (ex: run1, T1, survey)

    return information for dicom
        -k scan_technique(str): Protocol-specific scan technique (e.g., TFE, EPI)
        -k tr(float): Repetition Time (ms)
        -k dynScan(int): Number of dynamic scans (time points in fMRI)
        -k date(str): Series acquisition date (YYYYMMDD)
        -k time(str): Series acquisition time (HHMMSS.fraction)
        -k slice(int): Number of slices in the stack
        -k coverage(list[str]): Field of View coverage [X, Y, Z] in mm
        -k thickness(str): Slice thickness (voxel height) in mm
        -k stack_view_axis(str): stack method for slices
        -k row(int): Number of rows (image height in pixels)
        -k column(int): Number of columns (image width in pixels)
        -k frame(str): Total number of frames (slices * dynamics)
        -k image_path(str): Relative file path to the DICOM pixel data
        -k image_type(str): DICOM Image Type metadata (ORIGINAL/PRIMARY etc.)
    """
    # Load
    dicomdir = dcmread(dicomdir_path)
    
    # Get informations
    datas = []
    for patient in dicomdir.patient_records:
        for study in patient.children:
            for series in study.children:
                is_presentation = series["0008", "0060"].value == "PR"
                series_desc = getattr(series, 'SeriesDescription', 'No Description')
                protocol_name = getattr(series, 'ProtocolName', '')
                
                if (f"{series_name}" in series_desc) & (not is_presentation):
                    private_tag = series["2001", "105f"][0]
                    n_slice = int(series["2001", "1018"].value)
                    n_dynScan = int(series["2001", "1081"].value)
                    tr = series["2005", "1030"].value
                    scan_technique = series["2001", "1020"].value

                    # https://github.com/malaterre/dicom-private-dicts/blob/master/PMS-R32-dict.txt
                    x_coverage = private_tag["2005", "1074"].value
                    y_coverage = private_tag["2005", "1075"].value
                    z_coverage = private_tag["2005", "1076"].value
                    thickness = private_tag["2005", "107e"].value
                    stack_view_axis = private_tag["2005", "1081"].value

                    images = [image for image in series.children if image.DirectoryRecordType == "IMAGE"]
                    assert len(images) == 1, "Multiple image paths"
                    image = images[0]

                    meta_images = [image for image in series.children if image.DirectoryRecordType == "PRIVATE"]
                    assert len(meta_images) == 1, "Multiple meta data of image paths"
                    meta_image = meta_images[0]
                    
                    datas.append({
                        "scan_technique" : scan_technique,
                        "tr" : tr,
                        "#dynScan" : n_dynScan,
                        "series_time" : str(convert_dicom_time(series.SeriesDate, series.SeriesTime)),
                        "#slice" : n_slice,
                        "coverage" : [ f"{x_coverage}mm", f"{y_coverage}mm", f"{z_coverage}mm"],
                        "thickness" : f"{thickness}mm", # height of voxel
                        "stack_view_axis" : stack_view_axis,
                        "#row" : image.Rows,
                        "#column" : image.Columns,
                        "#frame" : image.NumberOfFrames,
                        "image_path" : os.sep.join(image.ReferencedFileID),
                        "image_type" : image.ImageType,
                        "meta_image_path" : os.sep.join(meta_image.ReferencedFileID),
                        
                    })
    assert len(datas) == 1, f"Multiple datas {series_name}"
    return datas[0]

def get_image_meta_info(file_path):
    ds = dcmread(file_path)
    n_slice = ds["2001", "1018"].value
    n_dynScan = ds["2001", "1081"].value
    
    return {
        "manufacturer" : ds.Manufacturer,
        "scan_date" : ds.StudyDate,
        "gender" : ds.PatientSex,
        "patient_name" : ds.PatientName,
        "birth_date" : ds.PatientBirthDate,
        "age" : ds.PatientAge,
        "protocol_name" : ds.ProtocolName,
        "pulse_sequence" : ds.PulseSequenceName,
        "patient_position" : ds.PatientPosition,
        "#row" : ds.Rows,
        "#column" : ds.Columns,
        "#data_row" : ds.DataPointRows,
        "#data_column" : ds.DataPointColumns,
        "#slice" : n_slice,
        "#dynscan" : n_dynScan,
    }

def get_image_info(path, is_phillips_scale = True):
    ds = dcmread(path)
    n_slice = ds["2001", "1018"].value
    n_dynScan = ds["2001", "1081"].value
    n_row = ds.Rows
    n_col = ds.Columns

    
    # image orientation
    image_orientation_vector = ds["5200", "9230"][0]["0020", "9116"][0]["0020", "0037"].value
    row_cosine = image_orientation_vector[:3]
    col_cosine = image_orientation_vector[3:]
    pixel_orientation = ("RL" if int(row_cosine[0]) > 0 else "LR", "AP" if int(col_cosine[1]) > 0 else "PA")
    
    # xyz position per slice
    xyzs = []
    for frame in ds["5200", "9230"]:
        xyz = frame.PlanePositionSequence[0]["0020", "0032"].value
        xyzs.append(xyz)
    xyzs = np.array(xyzs)

    # sorted slice along the z-axis
    sorted_indices = []
    for i in range(0, len(xyzs), n_slice):
        sorted_indices += list(np.argsort(xyzs[i:i+n_slice, 2]) + i)
    sorted_indices = np.array(sorted_indices)
    sorted_pixel_array = ds.pixel_array[sorted_indices]

    # make time series format (checked date: 2026.01.15)
    sorted_pixel_array = sorted_pixel_array.reshape(int(n_dynScan), int(n_slice), n_row, n_col)

    if is_phillips_scale:
        slice_i = 0
        rescale_intercept = float(ds["5200", "9230"][slice_i]["0028", "9145"][0]["0028", "1052"].value)
        rescale_slope = float(ds["5200", "9230"][slice_i]["0028", "9145"][0]["0028", "1053"].value)
        mr_scale_intercept = ds["5200", "9230"][slice_i]["2005", "140f"][0]["2005", "100d"].value
        mr_scale_slope = ds["5200", "9230"][slice_i]["2005", "140f"][0]["2005", "100e"].value
        sorted_pixel_array = (sorted_pixel_array * rescale_slope + rescale_intercept) / (rescale_slope * mr_scale_slope)
        
    # return
    return {
        "study_time" : str(convert_dicom_time(ds.StudyDate, ds.StudyTime)), # Patient enters the scanner
        "series_time" : str(convert_dicom_time(ds.SeriesDate, ds.SeriesTime)), # The protocol is selected.
        "acqusition_time" : ds.AcquisitionDateTime.split(".")[0], # The pulses start; data is being collected
        "content_time" : str(convert_dicom_time(ds.ContentDate, ds.ContentTime)), # The time to save disk
        "#slice" : n_slice,
        "#dynscan" : n_dynScan,
        "pixel_orientation" : pixel_orientation,
        "pixel_array" : sorted_pixel_array,
    }

def extract_mosaic_to_3d(dicom_path):
    ds = dcmread(dicom_path)
    matrix_text = ds["0018", "1310"].value
    rows_per_slice = matrix_text[0]
    cols_per_slice = matrix_text[-1]
    
    num_slices = ds["0019", "100a"].value
    
    mosaic_pixel_array = ds.pixel_array
    mosaic_rows, mosaic_cols = mosaic_pixel_array.shape
    
    # Mosaic 이미지 내의 슬라이스 배치 계산
    # Mosaic는 바둑판 형태이므로 한 행에 몇 개의 슬라이스가 들어가는지 계산합니다.
    tiles_per_row = mosaic_cols // cols_per_slice
    
    # 3D 배열 생성을 위한 빈 공간 준비 (#slice, #row, #column)
    volume_3d = np.zeros((num_slices, rows_per_slice, cols_per_slice), dtype=mosaic_pixel_array.dtype)
    
    # Mosaic에서 개별 슬라이스를 잘라내어 3D 배열에 저장
    for i in range(num_slices):
        # 현재 슬라이스의 Mosaic 내 타일 위치 계산
        tile_row = i // tiles_per_row
        tile_col = i % tiles_per_row
        
        # 시작/끝 인덱스 계산
        start_r = tile_row * rows_per_slice
        start_c = tile_col * cols_per_slice
        
        # 슬라이스 추출 및 저장
        volume_3d[i, :, :] = mosaic_pixel_array[
            start_r : start_r + rows_per_slice,
            start_c : start_c + cols_per_slice
        ]
        
    return volume_3d
    
if __name__ == "__main__":
    dicomdir_path = "/mnt/ext1/seojin/temp/DO11/DICOMDIR"
    get_phillips_dicom_info(dicomdir_path, "Survey")

    file_path = "/mnt/ext1/seojin/temp/DO11/DICOM/XX_0001"
    get_image_meta_info(file_path)

    path = "/mnt/ext1/seojin/HR/exp_blueprint_0324v4/fMRI_data/raw_data/HP01/head/RUN1_REVERSE/20210712_KSH.MR.HEAD_PI_CNIR_IBS.0005.0001.2021.07.12.12.54.57.884893.355952101.IMA"
    extract_mosaic_to_3d(path)