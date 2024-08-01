
# Cummon Libraries
import time
from ast import literal_eval
from os import stat, path
import os
import csv
import pandas as pd
import pickle 
import glob
import tqdm
from multiprocessing import Pool
import subprocess
import numpy as np
import pathlib
from pathlib import Path
import h5py
import re

# Custom Libraries
from sj_enum import File_comparison
from sj_string import str_join

# Sources
def get_fileName(path):
    """
    Get file name of path
    
    :param path(string): path ex) "/mnt/sdb2/DeepDraw/mri_mask/targetROIs/Lt_BA6_ventrolateral.nii.gz"
    """
    regex_pattern = r'[^/]+(?=\.nii\.gz)'
    match = re.search(regex_pattern, path)
    extracted_with_regex = match.group() if match else None
    
    return extracted_with_regex

def file_name(path):
    """
    :param path: file_path
    :return(string): file name
    """
    from sys import platform

    if platform == 'win32':
        delimiter = '\\'
    else:
        delimiter = '/'

    return path.split(delimiter)[-1]

def wait_for_write_finish_download(file_path, wait_seconds, exception):
    """
    process is waited until file is not updated

    :param file_path: file path
    :param wait_seconds: waiting timeout seconds if wait time is elapsed until the seconds then raise exception ex) 5
    :param exception: timeout exception ex) Exception("Timeout")
    """
    # it determines download completed if the file's size is same before and after per 1 second
    count = 0
    last_size, size= -1, 0
    while size != last_size:
        time.sleep(1)
        count += 1
        last_size, size = size, stat(file_path).st_size
        if count >= wait_seconds:
            raise exception

def wait_for_file_download(file_info_path, file_path, wait_seconds, exception):
    """
    process is waited until file is fully downloaded

    :param file_info_path: file path containing file size
    :param file_path: need to wait downloading file path
    :param wait_seconds: waiting timeout seconds if wait time is elapsed until the seconds then raise exception ex) 5
    :param exception: timeout exception ex) Exception("Timeout")
    """
    count = 0
    while compare_file(file_info_path, file_path) != True:
        time.sleep(1)
        count += 1
        if count >= wait_seconds:
            raise exception
    return count

def read_file_size(file_info_path, file_name):
    """
    This function reads file size of file_name in the file located file_info_path

    :param file_info_path: file path containing file size
    :param file_name: file_name ex) asahi.jpg
    :return: integer(bytes of file_size)
    """
    with open(file_info_path, "r") as f:
        str_file_info = f.read()
        file_info = literal_eval(str_file_info)
        try:
            size = file_info[file_name]
        except:
            return -1
        return size

def compare_file(file_info_path, file_path):
    """
    This function compares file size between file_path and file_info_path

    :param file_info_path: file path containing file size
    :param file_path: file path for checking
    :return:
    """
    file_info_size = read_file_size(file_info_path, file_name(file_path))
    if file_info_size == -1:
        return True # if file_info is not existed, return true for convenience
    else:
        file_current_size = stat(file_path).st_size
        return file_info_size == file_current_size

class CsvManager:
    def __init__(self, file_path = None, dir_path="", file_name=""):
        if file_path == None:
            self.dir_path = dir_path
            self.file_path = os.path.join(self.dir_path, file_name)
        else:
            self.dir_path = os.path.dirname(file_path)            
            self.file_path = file_path

        path = pathlib.Path(self.file_path)  
        if path.suffix != ".csv":
            self.file_path += ".csv"

        self.file_name = os.path.basename(self.file_path)

    def read_csv_from_pandas(self):
        return pd.read_csv(self.file_path)

    def read_csv(self):
        try:
            with open(self.file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                return [line for line in reader]
        except FileNotFoundError as e:
            return []

    def write_header(self, headers):
        if len(self.read_csv()) == 0:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        else:
            print("There exists header already")

    def write_row(self, row):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def save(obj, file_path):
    """
    Save file using pickle
    
    :param obj: object to save
    :param file_path: save file path
    """
    filehandler = open(file_path, 'wb') 
    pickle.dump(obj, filehandler)
   
def load(file_path):
    """
    Load file using pickle
    
    :param file_path: load file path
    """
    filehandler = open(file_path, 'rb') 
    return pickle.load(filehandler)

def exist_all_files(file_paths):
    """
    Check all file is exists
    
    :param file_paths(list - string): file paths
    
    return (boolean):
    """
    return sum([os.path.exists(file_path) for file_path in file_paths]) == len(file_paths)

def rename(dir_path, target_str, replace_str):
    """
    Rename file name
    
    :param dir_path: directory path 
    :param target_str: target file name under dir_path
    :param target_str: replace file name under dir_path
    """
    target = os.path.join(dir_path, target_str)
    replace = os.path.join(dir_path, replace_str)
        
    command = str_join(["mv", target, replace], delimiter = " ")
    print(command)
    os.system(command)

def remove(dir_path, startFileName):
    for file_name in os.listdir(dir_path):
        if file_name.startswith(startFileName):
            os.system("rm " + os.path.join(dir_path, file_name))

def checksum(file_path):
    """
    Checksu md5
    
    :param file_path: (string)
    
    return checksum result(string)
    """
    command = f"md5sum {file_path}"
    
    output = subprocess.check_output(command, shell=True).decode("utf-8").split(" ")[0]
    return output.strip()

def get_file_count(dir_path):
    """
    Get file count within directory
    
    :param dir_path: directory_path(string)
    
    return (string): #file
    """
    command = f"find {dir_path} -type f | wc -l"
    
    output = subprocess.check_output(f"find {dir_path} -type f | wc -l", shell=True).decode("utf-8")
    return output.strip()

def is_same_file_count(dirA_path, dirB_path):
    """
    Check file count is same between dirA and dirB
    
    :param dirA_path(string): pathA 
    :param dirB_path(string): pathB 
    
    return (boolean):
    """
    n_dirA_files = get_file_count(dirA_path)
    n_dirB_files = get_file_count(dirB_path)
    
    return n_dirA_files == n_dirB_files

def compare_file_name(file_path1, file_path2):
    """
    Compare file name
    
    :param file_path1(string): path1 
    :param file_path2(string): path2 
    
    return boolean
    """
    file_name1 = file_path1.split(os.sep)[-1]
    file_name2 = file_path2.split(os.sep)[-1]
    
    return file_name1 == file_name2

def compare_file_type(file_path1, file_path2):
    """
    Compare file type
    
    :param file_path1(string): path1 
    :param file_path2(string): path2 
    
    return boolean
    """
    if os.path.isdir(file_path1) and os.path.isdir(file_path2):
        return "directory"
    elif not os.path.isdir(file_path1) and not os.path.isdir(file_path2):
        return "file"
    else:
        return "invalid"

def compare_file_size(file_path1, file_path2):
    """
    Compare file size
    
    :param file_path1(string): path1 
    :param file_path2(string): path2 
    
    return boolean
    """
    file_size1 = os.path.getsize(file_path1)
    file_size2 = os.path.getsize(file_path2)
    
    return file_size1 == file_size2

def compare_file_content(file_path1, file_path2):
    """
    Compare file content using checksum
    
    :param file_path1(string): path1 
    :param file_path2(string): path2 
    
    return boolean
    """
    checksum_file1 = checksum(file_path1)
    checksum_file2 = checksum(file_path2)
    
    return checksum_file1 == checksum_file2

def compare_file(i, file_paths1, file_paths2, file_comparisons):
    """
    Compare file
    
    :param file_paths1(list - string): file paths
    :param file_paths2(list - string): file paths
    :param file_comparisons(list): compare method, ex) ["name", "type", "size", "checksum"]
    
    return invalid_type(string), (file_path1, file_path2)
    """
    file_path1 = file_paths1[i]
    file_path2 = file_paths2[i]
    
    comparison = sum(File_comparison.numbers(file_comparisons))
    
    is_compare_file_name = (comparison & File_comparison.file_name.value) != 0
    is_compare_file_type = True
    is_compare_file_size = (comparison & File_comparison.file_size.value) != 0
    is_compare_file_checksum = (comparison & File_comparison.file_checksum.value) != 0
    
    invalid_type = None
    
    try:
        # 1. Compare file name
        if is_compare_file_name and not compare_file_name(file_path1, file_path2):
            print(f"file name is not same {file_path1}, {file_path2}")
            invalid_type = File_comparison.name(File_comparison.file_name.value)

        # 2. Compare file type
        if is_compare_file_type and invalid_type == None:
            file_type = compare_file_type(file_path1, file_path2)
            if file_type == "invalid":
                print(f"file type is not same {file_path1}, {file_path2}")
                invalid_type = File_comparison.name(File_comparison.file_type.value)

        if file_type == "file":
            # 3. Compare file size
            if is_compare_file_size and invalid_type == None and not compare_file_size(file_path1, file_path2):
                print(f"file size is not same {file_path1}, {file_path2}")
                invalid_type = File_comparison.name(File_comparison.file_size.value)

            # 4. Compare file content
            if is_compare_file_checksum and invalid_type == None and not compare_file_content(file_path1, file_path2):
                print(print(f"file content is not same {file_path1}, {file_path2}"))
                invalid_type = File_comparison.name(File_comparison.file_checksum.value)
    except Exception as e:
        invalid_type = e.args
        
    return invalid_type, (file_path1, file_path2)

def compare_directory(dirA_path, dirB_path, file_comparisons, n_job = 1):
    """
    Compare directory the way is to compare each file within directory
    
    :param dirA_path(string): directory path, last chracter of which must include / ex) "/Users/yoonseojin/Downloads/ 
    :param dirB_path(string): directory path, last chracter of which must include / ex) "/Users/yoonseojin/Downloads/
    :param file_comparisons(list): compare method, ex) ["name", "type", "size", "checksum"]
    
    return: result(list - invalid_type, file_pathA, file_pathB)
    """
    dirA_name_generator = glob.iglob(dirA_path + "**/*", recursive=True)
    dirB_name_generator = glob.iglob(dirB_path + "**/*", recursive=True)

    # Search file paths within directory recursively
    dirA_file_paths = list(dirA_name_generator)
    dirB_file_paths = list(dirB_name_generator)
    
    # Sort file paths
    dirA_file_paths = sorted(dirA_file_paths)
    dirB_file_paths = sorted(dirB_file_paths)
    
    # check number of files
    n_dirA_files = len(dirA_file_paths)
    n_dirB_files = len(dirB_file_paths)

    # Check whether #dirA_file and #dirB_file is same
    assert len(dirA_file_paths) == len(dirB_file_paths), "n_files is not same between dirA and dirB"

    # Check files
    with tqdm.tqdm(total=n_dirA_files) as pbar:
        with Pool(processes=n_job) as pool:
            def callback(*args):
                # callback
                pbar.update()
                return
            
            results = [
                pool.apply_async(compare_file,
                                 args=(i, dirA_file_paths, dirB_file_paths, file_comparisons, ), 
                                 callback=callback) for i in np.arange(0, n_dirA_files)]
            
            results = [r.get() for r in results]

    return results

def check_is_hidden(p):
    """
    Check if file is hidden
    
    :param p(string): path
    
    return (boolean):
    """
    p = Path(p).name
    if os.name== 'nt':
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith('.') #linux-osx

def get_all_files(dir_path, filename_pattern):
    """
    Get all file with matching filename_pattern
    
    return generator
    """
    root = Path(dir_path)
    
    return root.rglob(filename_pattern)

def get_hdf5_info(file_path):
    """
    Get all data keys within hdf5 file.
    
    :param file_path(string): The file path of hdf5
    
    return (string)
    """
    with h5py.File(file_path) as hdf:
        dataset = list(hdf.keys())
    return dataset

def get_hdf5_data(file_path, key):
    """
    Get datas from hdf5 file using key
    
    :param file_path(string): The file path of hdf5
    :param key(string)
    
    return data(np.array)
    """
    with h5py.File(file_path) as hdf:
        if key in hdf:
            dataset = hdf[key]
            result = dataset[:]
            return result
        else:
            raise KeyError(f"{key} not found in HDF5 file.")

def print_hdf5_keys(name, obj, indent=0, max_depth=None, current_depth=0):
    """
    Recursively prints the keys of an HDF5 file up to a specified depth.

    Parameters:
    - name: The name of the current object being processed.
    - obj: The HDF5 object (group or dataset).
    - indent: The current indentation level for printing.
    - max_depth: The maximum depth to explore. If None, explores all depths.
    - current_depth: The current depth level in the recursion.
    """
    # Print the current object's name with indentation
    print('    ' * indent + f"{name}: [{type(obj).__name__}]")
    
    # Stop if the maximum depth has been reached
    if max_depth is not None and current_depth >= max_depth:
        return
    
    # If the object is a group, iterate through its items and call the function recursively
    if isinstance(obj, h5py.Group):
        for sub_name, sub_obj in obj.items():
            print_hdf5_keys(sub_name, sub_obj, indent + 1, max_depth, current_depth + 1)
    elif isinstance(obj, h5py.Dataset):
        # Optionally, print dataset details here
        pass

def explore_hdf5_file(file_path, max_depth=None):
    """
    Opens an HDF5 file and prints its structure up to a given depth.

    Parameters:
    - file_path: Path to the HDF5 file.
    - max_depth: The maximum depth to explore. If None, explores all depths.
    """
    with h5py.File(file_path, 'r') as hdf:
        print_hdf5_keys('/', hdf, max_depth=max_depth)

def is_jupyter():
    """
    Check current file is whether jupyter or not
    
    return (boolean)
    """
    try:
        # get_ipython() is only available in Jupyter notebooks.
        if get_ipython():
            return True
    except NameError:
        # If get_ipython() is not defined, we are in a regular Python script.
        return False
    
if __name__ == "__main__":
    """
    Example of CsvManager 
    """
    csv_m = CsvManager(dir_path="/Users/yoonseojin/Downloads",
                       file_name="test4")

    csv_m.write_header(["a","b","cdefgh"])
    csv_m.write_row(["얍", "얍얍", "얍얍얍"])

    sj_file_system.save(pd.DataFrame({"A:1"}), "./dd")
    
    sj_file_system.load("./dd")
    
    sj_file_system.rename("/sdafsd", "abc", "def")
    
    sj_file_system.remove(dir_path = "/acscb/dfes", startFileName = "test")
    checksum("/Users/yoonseojin/Downloads/test")
    get_file_count("/Users/yoonseojin/Downloads")
    
    is_same_file_count("/Users/yoonseojin/Downloads", "/Users/yoonseojin/Downloads")
  
    compare_file_name("/Users/yoonseojin/Downloads/test", "/Users/yoonseojin/Downloads/test")
    compare_file_type("/Users/yoonseojin/Downloads/test", "/Users/yoonseojin/Downloads/test")
    compare_file_size("/Users/yoonseojin/Downloads/test", "/Users/yoonseojin/Downloads/test")
    compare_file_content("/Users/yoonseojin/Downloads/test", "/Users/yoonseojin/Downloads/test")
    
    compare_directory("/Users/yoonseojin/Downloads/", "/Users/yoonseojin/Downloads/")
    
    file_is_hidden("/mnt/sdb2/DeepDraw/Projects/20230109_DP21_mri/mri/raw_data/HEAD/._PRE_REST")
    
    get_all_files(".", "*")
    
    dir_path = "/mnt/sdb2/DeepDraw/Deepdraw_dataset/20220801_DP02_mri_converted_data/Original"
    file_name = "dataset_all_20220801_DP02_mri_with_fmri.hdf5"
    get_hdf5_info(os.path.join(dir_path, file_name))
    
    is_jupyter()