
# MARK: - Common Libraries
from enum import Enum
import numpy as np
import pandas as pd

# MARK: - Custom Libraries
from sj_sequence import find_nearest

# MARK: - Functions
class Aggregate(Enum):
    mean = 1 << 0
    median = 1 << 1

def flags(list, condition):
    """
    return data of flags

    :param list: data
    :param condition: check data function
    :return: flags of data
    """
    return [True if condition(e) else False for e in list]

def choice(list1, list2, flags):
    """
    if flag is True then select list1's component
    otherwise select list2's component

    :param list1: first set of data
    :param list2: second set of data
    :param flags: criteria of choice algorithm
    :return:
    """
    if len(list1) == len(list2) == len(flags):
        result = []
        for i in range(0, len(flags)):
            if flags[i] == True:
                result.append(list1[i])
            else:
                result.append(list2[i])
        return result
    else:
        raise Exception("length is not compatible among list1, list2 and flags")

def counter(data, bin_ranges, equal_direction = "None"):
    """
    It counts number of data according to a range of bin
    if equal_direction is Left, equal sign of inequality is attached at lower bound
        so it means that if we check 3 and equal_direction is Left, we check the condition like lower_bound <= 3 < upperbound
        if equal_direction is None, we check the condition like lower_bound < 3 < upperbound

    :param data: list ex) [1,2,3,4]
    :param bin_ranges: range of bin ex) [(0,1), (1,2), (2,3)]
    :param equal_direction: direction of equal(None, Left, Right) ex) "Left"
    :return: number of data ex) { (0,1) : 3, (1,2) : 2 }
    """
    method_left_equal = lambda x, lower_bound, upper_bound: True if (x >= lower_bound and x < upper_bound) else False
    method_right_equal = lambda x, lower_bound, upper_bound: True if (x > lower_bound and x <= upper_bound) else False
    method_none_equal = lambda x, lower_bound, upper_bound: True if (x > lower_bound and x < upper_bound) else False


    if equal_direction == "None":
        selected_method = method_none_equal
    elif equal_direction == "Left":
        selected_method = method_left_equal
    elif equal_direction == "Right":
        selected_method = method_right_equal

    result = {}
    for key in bin_ranges:
        result[key] = 0

    for e in data:
        for range in bin_ranges:
            lower_b = range[0]
            upper_b = range[1]
            if selected_method(e, lower_b, upper_b) == True:
                result[range] = result[range] + 1
                break
    return result

def df_to_series(data):
    return list(map(lambda x: x[1], list(data.iterrows())))

def reverse_dict(dictionary):
    new_dict = { dictionary[k]:k for k in dictionary}
    return new_dict

def is_nan(x):
    if type(x) == str:
        return x == "nan"
    else:
        return np.isnan(x)

def is_not_nan(x):
    return is_nan(x) == False

def to_series(one_column_df):
    return one_column_df.iloc[:,0]

def replace_str(entry, replace_targets, replace_results):
    """
    :param entry: data
    :param replace_targets: replace target string
    :param replace_results: target <-> string
    """
    assert type(entry) == str, "Please input string to entry"
    assert len(replace_targets) == len(replace_results), "Please match length between replace_targets, replace_results"
    
    if len(replace_targets) == 0:
        return entry
    else:
        replace_result = entry.replace(replace_targets[0], replace_results[0])
        return replace_str(replace_result, replace_targets[1:], replace_results[1:])

def one_hot_encoding(string, split_str, variables):
    """
    return one-hot encoding 
    
    :param string: string
    :param split_str: split string
    :param variables: one_hot_encoding variables
    
    return: one-hot encoding(DataFrame)
    """
    result = np.zeros(len(variables))
    
    targets = string.split(split_str)
    for target in targets:
        for variable_i in range(0, len(variables)):
            if target in variables[variable_i] and len(target)>0:
                result[variable_i] = 1

    result_df = pd.DataFrame(np.array(result, dtype=int)).T
    result_df.columns = variables
    return result_df
            
def one_hot_encodings(strings, split_str, variables):
    """
    return one-hot encoding 
    
    :param string: string
    :param split_str: split string
    :param variables: one_hot_encoding variables
    
    return: one-hot encoding(DataFrame)
    """
    return pd.concat(list(map(lambda string: one_hot_encoding(string, split_str, variables),
                              strings)), 
                     axis=0)

def make_grouping_indexes(n_group, n_data, postProcessing = "absorbEndGroup"):
    """
    Make indexes for grouping elements of list

    :param n_group(int): the number of group
    :param n_data(int): the number of data
    :param postProcessing(string): post processing method

    return (list - (start_group_index, end_group_index)
    """
    n_element_perGroup = int(np.trunc(n_data / n_group))
    grouping_indexes = [[i, i + n_element_perGroup] for i in range(0, n_data, n_element_perGroup)]
    
    for i in range(len(grouping_indexes)):
        start_i = grouping_indexes[i][0]
        end_i = grouping_indexes[i][1]

        if postProcessing == "absorbEndGroup":
            if end_i > n_data -1:
                grouping_indexes[i-1][1] = n_data
                del grouping_indexes[i]

    return grouping_indexes

# MARK: - Examples
if __name__ == "__main__":
    is_nan(np.NaN)
    is_not_nan(np.NaN)

    replace_str("abc", ["a","b"], ["d", "f"])

    one_hot_encoding("a b c", " ", ["a", "b", "d"])
    one_hot_encodings(["a b c", "b"], " ", ["a", "b", "d"])
    
    
    
    make_grouping_indexes(12, 202)