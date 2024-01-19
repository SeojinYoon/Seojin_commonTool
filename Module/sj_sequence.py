
# Common Libraries
from collections.abc import Sequence
import pandas as pd
import numpy as np
import collections
from operator import itemgetter
import math

# Custom Libraries
from sj_higher_function import recursive_map, flatten

# Sources

sample_array = np.array([
    [[-1, -1, -1],[-1, -1, -1]],
    [[1, 2, 3],[4, 5, 6]],
    [[7, 8, 9],[0, 1, 2]],
    [[4, 4, 4],[4, 4, 4]],
])

sample_3d = [
    [[1,2],[3,4],[5,6]],
    [[7,8],[9,10],[11,12]],
    [[13,14],[15,16],[17,18]],
    [[19,20],[21,22],[23,24]],
]
sample_3d = np.array(sample_3d)

def check_duplication(data):
    """
    check duplication from list
    
    :param data: data(list)
    
    return: is_duplicated
    """
    counters = collections.Counter(data)
    
    for count in counters.values():
        if count > 1:
            return True
    
    return False
    
def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

def make_2d_list(w, h, init_value=0):
    return [[0 for _ in range(w)] for _ in range(h)]

def is_same(array1, array2):
    """
    Checking array1 and array2 is same.
    The method to check same is to compare element by element
    
    return (boolean)
    """
    if array1.shape != array2.shape:
        return False
    
    comparison = array1 == array2
    equal_arrays = comparison.all()
    return equal_arrays

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

def search_dict(dictionary, keywords):
    """
    Search keywords over values in dictionary 
    
    This function iterates dictionary using keys and searchs keyword in value
    
    return: searched dictionary
    """
    keys = []
    infos = []
    for key in dictionary:
        is_not_matching = False
        for keyword in keywords:
            if keyword not in dictionary[key]:
                is_not_matching = True

        if is_not_matching == False:
            keys.append(key)
            infos.append(dictionary[key])
    return dict(zip(keys, infos))

def prob_intersection(a1, a2, threshold = False, denomiator = "union"):
    """
    calculate intersection prob from array1 and array2

    :param a1: first array
    :param a2: second array
    :param threshold: cutting value about array

    return: probability
    """
    if threshold == False:
        return np.sum(a1 == a2) / a1.size
    else:
        # 두 합집합 중에서 교집합이 얼마나 되는가에 대한 확률
        if denomiator == "union":
            return np.sum(np.logical_and(a1 > threshold, a2 > threshold)) / np.sum(np.logical_or(a1 > threshold, a2 > threshold))
        elif denomiator == "a1":
            return np.sum(np.logical_and(a1 > threshold, a2 > threshold)) / np.sum(a1 > threshold)
        elif denomiator == "a2":
            return np.sum(np.logical_and(a1 > threshold, a2 > threshold)) / np.sum(a2 > threshold)

def is_in(array, population, all_true = True):
    """
    check all element is in population

    :param array: target list
    :param population: population array

    return: True of False

    ex)
    is_in([1,2,1,2], [1,2,3]) -> True
    """
    is_in_flags = np.array(list(map(lambda x: x in population, array)))
    
    if all_true:
        return is_in_flags
    else:
        return is_in_flags

def get_transition(seq):
    """
    :param seq: sequence(list)
    
    return: transitions(tuples) 
    ex) 
    [('4', '1'),
     ('1', '3'),
     ('3', '2'),
     ('2', '4'),
     ('4', '3'),
     ('3', '1'),
     ('1', '2')]
    """
    transitions = []
    for unit_i in range(0, len(seq)):
        if unit_i == 0:
            continue
        else:
            previous_unit_index = unit_i - 1
            transitions.append((seq[previous_unit_index], seq[unit_i]))
    return transitions

def get_reverse_transition(seq):
    """
    :param seq: sequence(list)
    
    return: transitions(tuples) 
    ex) 
    [('4', '1'),
     ('1', '3'),
     ('3', '2'),
     ('2', '4'),
     ('4', '3'),
     ('3', '1'),
     ('1', '2')]
    """
    transitions = []
    for unit_i in range(len(seq)-1, -1, -1):
        if unit_i == len(seq) - 1:
            continue
        else:
            previous_unit_index = unit_i + 1
            transitions.append((seq[previous_unit_index], seq[unit_i]))
    return transitions

def number_of_same_transition(seq1, seq2, is_reverse=False, debug=False):
    """
    :param seq1: list
    :param seq2: list
    
    return: number of same transition
    """
    from Module import sj_datastructure

    if is_reverse == True:
        seq1_transitions = get_reverse_transition(seq1)
        seq2_transitions = get_reverse_transition(seq2)
        
    else:
        seq1_transitions = get_transition(seq1)
        seq2_transitions = get_transition(seq2)
        
    sets = sj_datastructure.Sets(seq1_transitions, seq2_transitions)
    intersection = sets.intersection()

    seq1_value_counts = dict(pd.Series(seq1_transitions).value_counts())
    seq2_value_counts = dict(pd.Series(seq2_transitions).value_counts())
    
    n_same_transition = 0
    for intersection_element in intersection:
        same_count = min(seq1_value_counts[intersection_element], seq2_value_counts[intersection_element])
        if debug == True:
            print("intersection: ", intersection_element, "count: ",same_count)
        n_same_transition += same_count
        
    return n_same_transition

def construct_layer_list(shape, init_value = 0):
    """
    construct initial layered list
    
    :param shape:
    :param init_value: initial value
    
    """
    if len(shape) == 0:
        return init_value
    else:
        result = []
        layer_iterate_count = shape[0]
        for info in range(0, layer_iterate_count):
            result.append(construct_layer_list(shape[1:], init_value))
        return result
    
def set_entry(target_list, entry_indexes, value):
    """
    set list's entry
    
    :param target_list: target list
    :param entry_indexes: index list ex) [0,0]
    :param value: value
    """
    assert len(np.array(target_list).shape) == len(entry_indexes), "list shape and indexes not matched!"
    if len(entry_indexes) == 1:
        target_list[entry_indexes[0]] = value
    else:
        set_entry(target_list[entry_indexes[0]], entry_indexes[1:], value)

def calc_2dmat_agg(mat, aggregate_func, target="all", ):
    """
    calculate 2d matrix using aggregate_func
    
    :param target: matrix calculation target / all, upper_tr, lower_tr
    :param aggregate_func: aggregate function over target entities
    """
    
    if target == "upper_tr":
        mat = np.triu(mat, 1)
    elif target == "lower_tr":
        mat = np.tril(mat, -1)

    return aggregate_func(mat.reshape(-1))

def filter_list(axis, datas, value):
    """
    Filter list
    
    :param axis: filter axis number ex) 0
    :param dats: datas(list)
    :param value: filtered by corresponding value
    
    return filtered list
    """

    return list(filter(lambda data: data[axis] == value, datas))

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

def filterUsingExclude(target_list, exclude_list):
    """
    Filter list using exclude values
    
    :param target_list: target list(list)
    :param exclude_list: exclude list(list)
    
    return list
    """
    return list(filter(lambda x: x not in exclude_list, target_list))
    
def get_itemsFromAxis(axis, datas):
    """
    Get items from specific axis over datas
    
    :param axis: axis number(int)
    :param datas: datas(list)
    
    return list
    """
    return list(map(itemgetter(axis), datas))

def interleave_array(array1, array2, interleave_count):
    """
    Interleave array2 to array1
    
    :param array1: array1(list)
    :param array2: array2(list)
    :param interleave_count: To count how many interleave
    
    return interleaved array
    """
    temp = []
    for i in np.arange(0, len(array1), interleave_count):
        for count in range(interleave_count):            
            temp.append(array1[i + count])
        for count in range(interleave_count):            
            temp.append(array2[i + count])
            
    return temp

def slice_list_usingDiff(data):
    """
    Slice list when a different value happens
    
    :param data: (list)
    
    return [(start index, stop index)]
    """
    if len(data) == 0:
        return []
    elif len(data) == 1:
        return [(data[0],1)]
    
    check_elements = []
    lengths = []
    
    length = 1
    for i, element in enumerate(data):
        if len(check_elements) == 0:
            length = 1
            check_elements.append(element)
        elif i == len(data) - 1:
            if element == check_elements[-1]:
                length += 1
                lengths.append(length)
            else:
                lengths.append(length)
                
                check_elements.append(element)
                length = 1
                lengths.append(length)
            
        elif element != check_elements[-1]:
            lengths.append(length)

            length = 1
            check_elements.append(element)
        else:
            length += 1
        
    start_indexes = []
    stop_indexes = []
    
    start_index = 0
    for element, length in list(zip(check_elements, lengths)):
        start_indexes.append(start_index)
        
        stop_index = start_index + length - 1
        stop_indexes.append(stop_index)
        
        start_index = stop_index + 1
    return list(zip(start_indexes, stop_indexes))

def replace_element(data, from_, to_):
    """
    Replace element from list
    
    :param data: target data(list)
    :param from_: the value will be transformed(list - 1d)
    :param to_: the value will go into the element(list - 1d)
    
    return (list) This list is transformed by the mapping from_ -> to_
    """
    from_ = np.array(from_)
    to_ = np.array(to_)
    
    def replacer(element):
        checked = (element == from_)

        if np.sum(checked) == 0:
            return element
        elif np.sum(checked) == 1:
            return to_[checked][0]
        else:
            raise Exception("from_'s elements are duplicated!!")
    
    return recursive_map(data, replacer)

def remove_duplicate_series(data, return_type = "check_result"):
    """
    Remove the element if previous element is same the element
    
    :param data: (list)
    :param return_type: (string) / data: , check_result:
    
    return (list)
    """
    
    slicing_data_byDup = slice_list_usingDiff(data)
    
    start_indexes = np.array([e[0] for e in slicing_data_byDup])
    stop_indexes = np.array([e[1] for e in slicing_data_byDup])
    
    dup_lengths = (stop_indexes - start_indexes) + 1
    
    check_result = flatten([list(np.repeat(False, dup_length)) for dup_length in dup_lengths])
    
    # Check starting value
    for start_index in start_indexes:
        check_result[start_index] = True
    
    if return_type == "data":
        return filterUsingFlags(target_list = data, flag_list = check_result, flag_value = True)
    elif return_type == "check_result":
        return check_result

def find_nearest(array, value, check_type = "previous", return_type = "idx"):
    """
    find nearest value from array
    
    :param array: array to explore(list - 1d)
    :param value: baseline value(float)
    :param check_type: (string) - ex) 'previous', 'next'
    :param return_type: (string) - ex) 'data', 'idx'
    """
    array = np.asarray(array)
    diff_array = array - value
    
    if check_type == "previous":
        target_array = array[diff_array < 0]
        diff_array = diff_array[diff_array < 0]
        
    elif check_type == "next":
        target_array = array[diff_array > 0]
        diff_array = diff_array[diff_array > 0]
    
    if len(diff_array) == 0:
        return None
    else:
        idx = (np.abs(diff_array)).argmin()
        
        return_value = target_array[idx]
        if return_type == "data":
            return return_value
        else:
            return list(array).index(return_value)

def get_unique_values(data):
    """
    Get unique values from data
    
    :param data: (list)
    
    return (list)
    """
    
    unique_values = []
    for e in data:
        if e in unique_values:
            pass
        else:
            unique_values.append(e)
    return unique_values

def convert_1d_to_symmertic(a_1d, size, k = 0):
    """
    Convert 1d array to symmetric matrix
    
    :param a_1d: (1d array)
    :param size: matrix size
    :param k: offset (int)
    
    return (np.array)
    """

    # put it back into a 2D symmetric array
    if k == 0:
        X = np.zeros((size,size))
        X[np.triu_indices(size, k = 0)] = a_1d
        X = X + X.T - np.diag(np.diag(X))
    else:
        X = np.zeros((size_X,size_X))
        X[np.triu_indices(size, k = 1)] = a_1d
        X = X + X.T

    return X

def upper_tri_1d_index(i, j, n_col):
    """
    Get upper triangle 1d index
    
    (0,1), (0,2), (0,3), (0,4) -> 0, 1, 2, 3
           (1,2), (1,3), (1,4) -> 4, 5, 6
                  (2,3), (2,4) -> 7, 8
                         (3,3) -> 9
                         
    :param i: row index
    :param j: column index
    :param n_col: column number
    
    return index(int)
    """
    if i > j:
        return None
    else:        
        sum_val = 0
        for z in range(0, i):
            sum_val += (n_col - 1) + (-1) * z
        return sum_val + (j - i - 1)
    
def lower_tri_1d_index(i, j):
    """
    Get lower triangle 1d index
    
    :param i: row index
    :param j: column index
    
    return index(int)
    """
    
    if i < j:
        return None
    else:        
        total_fill = 0
        for pr_row_i in range(1, i + 1):
            total_fill += (pr_row_i - 1)
        return total_fill + j

def group_byValue(list_):
    """
    Grouping list by unique value of list
    
    :param list_: (list)
    
    return [group_name, index of list]
    """
    unique_list = np.unique(list_)
    
    return [(e, find_indexes(list_, e)) for e in unique_list]
    
def squareMatDim_2_lowerTriangleMatDim(n):
    """
    Calculate lower triangle dimension from square matrix dimension
    
    return (int), lower triangle dim(1d)
    """
    result = ((n * n) - n) / 2
    frac, whole = math.modf(result)
    
    assert frac == 0, "The dimension is not integer"
    return int(result)

def lowerTriangleMatDim_2_SquareMatDim(lt_d):
    """
    Calculate square matrix dimension from lower triangle dimension
    """
    result = (1 + np.sqrt(1 + 8*lt_d)) / 2
    frac, whole = math.modf(result)
    
    assert frac == 0, "The dimension is not integer"
    return int(result)

def sort_2d_array(corr, orig_order, new_order):
    '''
    First convert 1D array (upper triangle) to 2D array (square)
    Then sort correlation.
    
    Parameters
    ----------
    corr : 1D array
        eg. np.array([1x1, 1x2, 1x3, ..., 5x5, 5x7, 7x7])
    orig_order : 1D array
        eg. np.array([1, 2, 3, 5, 7])
    new_order : 1D array
        eg. np.array([2, 7, 1, 5, 3])
    
    Returns
    -------
    sorted_corr : 2D array
        eg. np.array([[2x2, ..., 2x3], ..., [3x2, ..., 3x3]])
    '''
    
    # Convert
    mat_dim = len(new_order)
    matrix = np.zeros((mat_dim, mat_dim))
    
    upper_triangle_indexes = np.triu_indices(mat_dim, 1)
    r_indexes = upper_triangle_indexes[0]
    c_indexes = upper_triangle_indexes[1]
    for e, r_i, c_i in zip(corr, r_indexes, c_indexes):
        matrix[r_i, c_i] = e
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 1)
    # Sort
    mapping = {new_order[i]: orig_order[i] for i in range(len(orig_order))}
    mapping_keys = sorted(mapping.keys())
    sorted_corr = np.zeros((len(new_order), len(new_order)))
    for i in range(len(new_order)):
        for j in range(len(new_order)):
            old_i = mapping_keys.index(new_order[i])
            old_j = mapping_keys.index(new_order[j])
            sorted_corr[i, j] = matrix[old_i, old_j]
    
    return sorted_corr

if __name__=="__main__":
    get_transition([1,2,3])
    
    number_of_same_transition([1,2,3], [4,1,2])
    
    a = [1,2,3]
    set_entry(a, [2], 4)
    print(a)
    
    construct_layer_list((2,3))
    
    calc_2dmat_agg(np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]), sum, "upper_tr")
    
    interleave_array([1,2,3], [4,5,6], 1)
    
    slice_list_usingDiff([1,1,1,2])
    
    replace_element([1,2,3], [1], [4])
    
    remove_duplicate_series(data = [1,1,2], return_type = "check_result")
    
    find_nearest([1,2,3], 2, return_type = "data")
    
    get_unique_values([1,1,2,3])
    
    convert_1d_to_symmertic([1,2,3], size = 3, k=1)
    
    upper_tri_1d_index(0, 1, 4)
    
    group_byValue([1,1,1,2])
    
    squareMatDim_2_lowerTriangleMatDim(4)
    lowerTriangleMatDim_2_SquareMatDim(6)
    