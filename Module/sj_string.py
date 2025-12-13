
# Common Libraries
import re
import numpy as np

# Custom Libraries
from sj_enum import File_validation

# Functions
def str_join(strs, delimiter = "_"):
    """
    join string

    :param strs: list of string
    :param delimiter: delimiter

    return: combination string
    """
    strs = list(filter(lambda str: str != "", strs))
    strs = list(filter(lambda str: str != None, strs))
    
    if len(strs) == 0:
        return ""
    elif len(strs) == 1:
        return str(strs[0])
    else:
        return strs[0] + delimiter + str_join(strs[1:], delimiter)

def str_join_multi_delimiters(strs, delimiters = ["_"]):
    """
    join string

    :param strs: list of string
    :param delimiter: delimiters

    return: combination string
    """
    strs = list(filter(lambda str: str != "", strs))
    
    if len(strs) == 1:
        return str(strs[0])
    else:
        current_delimiter = delimiters[0]
        next_delimiters = delimiters[1:] + [current_delimiter]
        return strs[0] + delimiters[0] + str_join_multi_delimiters(strs[1:], next_delimiters)
    
def search_string(target, search_keys, search_type = "any", exclude_keys = []):
    """
    Search string with keys in target
    
    :param target(str): target string
    :param keys(list - str): search key
    :param search_type(str): search type - 'any', 'all', any is 'or' condition, all is 'and' condition
    :param exclude_keys(list - str): exclude key
    
    return boolean
    """
    if search_type == "any":
        search_result = any([key in target for key in search_keys])
    elif search_type == "all":
        search_result = all([key in target for key in search_keys])
    elif search_type == "correct":
        assert(len(search_keys) == 1), "please input only one search key"
        search_result = search_keys[0] == target
    
    if exclude_keys != None and exclude_keys != []:
        exclude_result = not any([key in target for key in exclude_keys])
    
        return search_result and exclude_result
    else:
        return search_result

def search_stringAcrossTarget(targets, 
                              search_keys,
                              search_type = "any", 
                              exclude_keys = [], 
                              validation_type = None,
                              return_type = "string"):
    """
    Search string across target strings
    
    :param target(list): target string
    :param keys(str): search key
    :param search_type(str): search type - 'any', 'all', any is or condition, all is and condition
    :param exclude_keys(list - str): exclude key
    :param validation_type(File_validation): kinds of validation checking from search result
    :param return_type(string): 'string' or 'index'
    
    return list of searched string
    """
    search_results = [search_string(target = target, 
                                    search_keys = search_keys, 
                                    search_type = search_type,
                                    exclude_keys = exclude_keys) for target in targets]
    search_flags = np.array(search_results)
    indexes = np.where(search_flags == True)[0]
    result = list(np.array(targets)[indexes])
    
    # search validation
    if validation_type is not None:
        if validation_type.value & File_validation.exist.value != 0: # check existence
            assert len(result) != 0, "Please check to exist file"
        if validation_type.value & File_validation.only.value != 0: # check the number of files
            if len(result) > 1:
                raise Exception("Multiple similar files")
    
    # return
    if return_type == "index":
        out = indexes
    elif return_type == "flag":
        out = search_flags
    else:
        out = results

    # only 옵션이면 scalar로 축약
    if (validation_type is not None
        and (validation_type.value & File_validation.only.value)
        and return_type != "flag"
    ):
        return out[0]

    return out

def replace_all(text, dic):
    for old, new in dic.items():
        text = text.replace(old, new)
    return text

def make_pad_fromInt(integer, n_digit):
    return str(integer).zfill(n_digit)

# Examples                      
if __name__ == "__main__":
    a_string = "A string is more than its parts!"
    matches = ["more", "d"]
    
    search_string(a_string, matches, search_type = "all")
    
    replace_all("a b c", {"a" : "aa"})
    search_stringAcrossTarget(targets = ["1","2","3"], search_keys=["1"], return_type = "flag")
    
    make_pad_fromInt(2, 3)
    