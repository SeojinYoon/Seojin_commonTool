
# Cummon Libraries
import numpy as np
from operator import itemgetter
import copy

# Custom Libraries

# Sources

def init_dic(keys, initial_value):
    """
    Initialize dictionary in accordance with keys
    
    :param keys: dictionary keys
    :param initial_value: initial value to the key
    """
    
    dic = {}
    for key in keys:
        dic[key] = copy.copy(initial_value)
    return dic


def find_valueInDict(info, search_value):
    """
    find value in dictionray
    
    :param info: (dictionary)

    return: key of dict corresponding the value
    """
    for key, value in info.items():
        if search_value in value:
            return key

    return None

def access_keys(keys, dic):
    """
    Access dictionary using keys
    
    :param keys: key of dictionary(list)
    :param dic: dictioanry
    
    return values
    """
    return itemgetter(*keys)(dic)

def filter_from_key(dic, filter_func):
    """
    Filter dictionary based on keys
    
    :param dic: (dictionary)
    :param filter_func: filtering function ex) lambda key: key == "a"
    
    return filtered dictionary
    """
    keys = np.array(list(dic.keys()))
    flags = list(map(filter_func, keys))
    
    filtered_result = dict(np.array(list(dic.items()), dtype = object)[flags, :])
    
    return filtered_result

def df_to_dict(df):
    """
    convert dataframe to dictionary
    
    :param df: (DataFrame) This data frame must include key and value column.
        
    return: (dictionary)
    """
    info = {}
    for row in df.itertuples(index=True):
        info[row.key] = row.value
        
    return info

def search_dict(dictionary, keywords):
    """
    Search keywords over values in dictionary 
    
    This function iterates dictionary using keys for searching keyword in value
    
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

if __name__ == "__main__":
    df = pd.DataFrame({ "key" : [1,2,3], "value" : ["a", "b", "c"]} )
    df_to_dict(bn_df)
    
    sj_dictionary.search_dict({ 1 : "a", 2 : "b"}, "a")
    