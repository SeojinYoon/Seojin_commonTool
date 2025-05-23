
# Common Libraries
import sys
import numpy as np
import os
from collections import Counter
from enum import Enum
import glob
from operator import itemgetter
import glob
import tqdm
from multiprocessing import Pool
import subprocess
import numpy as np

# Functions

class File_comparison(Enum):
    file_name = 1 << 0
    file_type = 1 << 1
    file_size = 1 << 2
    file_checksum = 1 << 3

    all = file_name | file_type | file_size | file_checksum

    @staticmethod
    def name(number):
        if number == File_comparison.file_name.value:
            return "name"
        elif number == File_comparison.file_type.value:
            return "type"
        elif number == File_comparison.file_size.value:
            return "size"
        elif number == File_comparison.file_checksum.value:
            return "checksum"

    @staticmethod
    def number(name):
        if name == File_comparison.name(File_comparison.file_name.value):
            return File_comparison.file_name.value
        elif name == File_comparison.name(File_comparison.file_type.value):
            return File_comparison.file_type.value
        elif name == File_comparison.name(File_comparison.file_size.value):
            return File_comparison.file_size.value
        elif name == File_comparison.name(File_comparison.file_checksum.value):
            return File_comparison.file_checksum.value

    @staticmethod
    def names(numbers):
        return [File_comparison.name(number) for number in numbers]

    @staticmethod
    def numbers(names):
        return [File_comparison.number(name) for name in names]

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
    
    return #file(string)
    """
    command = f"find {dir_path} -type f | wc -l"
    
    output = subprocess.check_output(f"find {dir_path} -type f | wc -l", shell=True).decode("utf-8")
    return output.strip()

def is_same_file_count(dirA_path, dirB_path):
    """
    Check file count is same between dirA and dirB
    
    :param dirA_path: pathA (string)
    :param dirB_path: pathB (string)
    
    return (boolean)
    """
    n_dirA_files = get_file_count(dirA_path)
    n_dirB_files = get_file_count(dirB_path)
    
    return n_dirA_files == n_dirB_files

def compare_file_name(file_path1, file_path2):
    """
    Compare file name
    
    :param file_path1: path1 (string)
    :param file_path2: path2 (string)
    
    return boolean
    """
    file_name1 = file_path1.split(os.sep)[-1]
    file_name2 = file_path2.split(os.sep)[-1]
    
    return file_name1 == file_name2

def compare_file_type(file_path1, file_path2):
    """
    Compare file type
    
    :param file_path1: path1 (string)
    :param file_path2: path2 (string)
    
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
    
    :param file_path1: path1 (string)
    :param file_path2: path2 (string)
    
    return boolean
    """
    file_size1 = os.path.getsize(file_path1)
    file_size2 = os.path.getsize(file_path2)
    
    return file_size1 == file_size2

def compare_file_content(file_path1, file_path2):
    """
    Compare file content using checksum
    
    :param file_path1: path1 (string)
    :param file_path2: path2 (string)
    
    return boolean
    """
    checksum_file1 = checksum(file_path1)
    checksum_file2 = checksum(file_path2)
    
    return checksum_file1 == checksum_file2

def compare_file(i, file_paths1, file_paths2, file_comparisons, is_print_invalid = False):
    """
    Compare file
    
    :param file_paths1: file paths(list - string)
    :param file_paths2: file paths(list - string)
    :param file_comparisons: compare method(list), ex) ["name", "type", "size", "checksum"]
    
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
        
    if is_print_invalid and invalid_type != None:
        print(invalid_type, (file_path1, file_path2))
        
    return invalid_type, (file_path1, file_path2)

def compare_directory(dirA_path, dirB_path, file_comparisons, n_job = 1, is_report_invalid_while_searching = False):
    """
    Compare directory the way is to compare each file within directory
    
    :param dirA_path: directory path(string), last chracter of which must include / ex) "/Users/yoonseojin/Downloads/ 
    :param dirB_path: directory path(string), last chracter of which must include / ex) "/Users/yoonseojin/Downloads/
    :param file_comparisons: compare method(list), ex) ["name", "type", "size", "checksum"]
    
    return: result(list - invalid_type, file_pathA, file_pathB)
    """
    if dirA_path[-1] != os.sep:
        dirA_path += os.sep
    if dirB_path[-1] != os.sep:
        dirB_path += os.sep
        
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
    assert len(dirA_file_paths) == len(dirB_file_paths), f"n_files is not same between {dirA_path} and {dirB_path}"

    # Check files
    with tqdm.tqdm(total=n_dirA_files) as pbar:
        with Pool(processes=n_job) as pool:
            def callback(*args):
                # callback
                pbar.update()
                return
            
            results = [
                pool.apply_async(compare_file,
                                 args = (i, dirA_file_paths, dirB_file_paths, file_comparisons, is_report_invalid_while_searching), 
                                 callback=callback) for i in np.arange(0, n_dirA_files)]
            
            results = [r.get() for r in results]

    return results

def get_itemsFromAxis(axis, datas):
    """
    Get items from specific axis over datas
    
    :param axis: axis number(int)
    :param datas: datas(list)
    
    return list
    """
    return list(map(itemgetter(axis), datas))

def filterUsingFlags(target_list, flag_list, flag_value):
    """
    Filter list using flags
    target list is filtered by using flag_list which is matched with flag_value
    
    :param target_list: target list(list)
    :param flag_list: list(element - boolean)
    :param flag_value: filter value(True or False)
    
    return target_list
    """
    return get_multiple_elements_in_list(target_list, find_indexes(flag_list, flag_value))

def get_multiple_elements_in_list(in_list, in_indices):
    """
    Get multiple element using indexes
    
    :param in_list: data(list)
    :param in_indices: indexes to be extracted from in_list
    """
    return [in_list[i] for i in in_indices]

def find_indexes(list, find_value, method = "equal"):
    """
    Search value from list
    
    :param list: list(list)
    :param find_value: value
    
    return: index of list matched to find_value
    """
    indexes = []
    for i in range(0, len(list)):
        if method == "equal":
            if list[i] == find_value:
                indexes.append(i)
        elif method == "in":
            if find_value in list[i]:
                indexes.append(i)
        elif method == "not_in":
            if find_value not in list[i]:
                indexes.append(i)
    return indexes

if __name__ == "__main__":
    # Constants
    comparison_types = np.array([
        File_comparison.file_name.value,
        File_comparison.file_type.value,
        File_comparison.file_size.value,
        File_comparison.file_checksum.value
    ])
    
    # Input
    dirA_path = input("Input direcotry A path: ")
    dirB_path = input("Input direcotry B path: ")
    
    # Correct path
    if dirA_path[-1] != os.path.sep:
        dirA_path += os.path.sep
    
    if dirB_path[-1] != os.path.sep:
        dirB_path += os.path.sep
    
    print("Select comparison type")
    
    print(f"1. name: {File_comparison.name(comparison_types[0])}")
    print(f"2. file_type: {File_comparison.name(comparison_types[1])}")
    print(f"3. file_size: {File_comparison.name(comparison_types[2])}")
    print(f"4. file_checksum: {File_comparison.name(comparison_types[3])}")
    
    print("If you want stop selecting loop, please input s")
    
    targets = []
    while True:    
        input_value = input("What do you want...? 1, 2, 3, 4, s: ")
        
        if input_value == "s":
            break
        else:
            targets.append(input_value)
        
    
    # compare direcotry
    target_comparison_types = comparison_types[[int(target) -1 for target in targets]]
    target_comparison_types = [File_comparison.name(comparison_type) for comparison_type in target_comparison_types]
    
    results = compare_directory(dirA_path,
                                dirB_path,
                                target_comparison_types,
                                n_job = 30)
    
    # Filter invalid result
    np_invalid_types = np.array(get_itemsFromAxis(0, results), dtype="object")
    invalid_results = filterUsingFlags(target_list = results,
                                       flag_list = np_invalid_types != None,
                                       flag_value = True)
    
    for invalid_result in invalid_results:
        invalid_type = invalid_result[0]
        invalid_pathA = invalid_result[1][0]
        invalid_pathB = invalid_result[1][1]
    
        print(f"invalid type: {invalid_type}, pathA: {invalid_pathA}, pathB: {invalid_pathB} \n")

