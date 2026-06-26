
# MARK: - Common Libraries
import numpy as np
import pandas as pd

# MARK: - Custom Libraries
from sj_higher_function import get_index_from_nested_list

# MARK: - functions related to columns
def get_column_keywords(data, keywords, search_mode = 1):
    """
    search data from passed condition
    
    :param data: dataframe
    :param keywords: search keywords, if multiple index the keywords are list
    :param search_mode: 1 <-> correct, 0 <-> contain
    
    # get_column_keywords(grade_data_1_2, ["Cognitive", "습", None, None])
    
    return dataframe
    """
    columns = list(data.columns)
    column_indexes = search_column_index(data=data, keywords=keywords, search_mode=search_mode)

    result = data.iloc[:, column_indexes]
    
    return result

def get_columns_keywords(data, keywords_list, mode=0):
    col_data_list = []
    for keywords in keywords_list:
        col_data = get_column_keywords(data, keywords=keywords, search_mode=mode)
        col_data_list.append(col_data)
    
    return pd.concat(col_data_list, axis=1)
    
def remove_non_need_column(data, non_need_keywords):
    """
    remove non need column
    
    :param data: dataframe
    :param non_need_keywords: search keyword for column
    
    return: DataFrame
    """
    search_non_need_column_indexes = []
    for keywords in non_need_keywords:
        search_non_need_column_indexes.append(list(search_column_index(data,
                                                                       keywords = keywords)))
        search_non_need_column_indexes = flatten(search_non_need_column_indexes)
    
    # Extract need index
    need_indexes = []
    col_indexes = np.arange(0, len(data.columns))
    for col_index in col_indexes:
        if col_index not in search_non_need_column_indexes:
            need_indexes.append(col_index)

    return data.iloc[:, need_indexes]
    
def sort_columns(df, sort_column_array):
    """
    Sort columns from df using sort_column_array
    
    :param df: target data(dataframe)
    :param sort_column_array: column list(list)
    
    return sorted dataframe(dataframe)
    """
    access_first_entry = 0

    column_index_info = {}
    for column_i in range(0, len(df.columns)):
        column = df.columns[column_i]
        column_index_info[column] = column_i
    
    return df.iloc[:, list(map(lambda column: column_index_info[column], sort_column_array))]

# MARK: - functions related to change data
def change_df(data, column_name, apply_func):
    """
    change data
    
    :param data: DataFrame
    :param column_name: name of column. if you use MultiIndex, column_name is list
    :param apply_func: function to apply column data
    
    return DataFrame
    """
    data_ = data.copy()
    
    target_column_index = search_column_index(data, column_name)
    assert len(target_column_index) == 1, "searched multiple columns!!"

    applied_data = to_series(data_.iloc[:, target_column_index]).apply(apply_func)

    data_.iloc[:, target_column_index] = pd.DataFrame(applied_data)
    return data_

# MARK: - functions related to compare
def compare_df(df1, df2, ignore_index=True):
    if ignore_index:
        df1_ = df1.reset_index(drop=True)
        df2_ = df2.reset_index(drop=True)
    else:
        df1_ = df1
        df2_ = df2
    
    return df1_.compare(df2_, 
                align_axis=1, 
                keep_equal=True, 
                keep_shape=True)

# MARK: - functions related to search
def search_column_index(data, keywords, search_mode = 1):
    """
    search column_index from passed condition
    
    :param data: dataframe
    :param keywords: search keywords, if multiple index the keywords are list
    :param search_mode: 1 <-> correct, 0 <-> contain
    return index array
    """
    def search_method(keyword):
        if keyword == None or keyword == "":
            return lambda element: True
        else:
            if search_mode == 1:
                return lambda element: keyword == element
            else:
                return lambda element: keyword in element
    
    conditions = [search_method(keyword) for keyword in keywords]
    
    return get_index_from_nested_list(data, conditions)[0]

def search_multi_conditions(data, search_columns, filter_funcs):
    """
    filter data using filter funcs over search_columns
    
    :param search_columns: list of column name ex [ ["학생코드", "", "", ""], ...] <- if you use multindex column
    :param filter_funcs: Apply filter function over the column vector
    
    return DataFrame
    """
    
    conditions = []
    for search_column, filter_func in zip(search_columns, filter_funcs):
        column_data = to_series(get_column_keywords(data, search_column, search_mode=1))
        condition = column_data.apply(filter_func)
        
        conditions.append(condition)
        
    condition_matrix = pd.concat(conditions, axis = 1)
    final_condition = condition_matrix.apply(lambda x: sum(x), axis=1) == len(search_columns)

    return data.iloc[list(final_condition),:]

def search(data,
           search_columns,
           filter_funcs,
           mode=1,
           showing_columns = None,):
    searched = search_multi_conditions(data=data, 
                                       search_columns=search_columns,
                                       filter_funcs=filter_funcs)
    if showing_columns == None:
        return searched
    else:
        return get_columns_keywords(searched, keywords_list=showing_columns, mode=mode)

# MARK: - functions for grouping
def group(data, by, group_function, filter_func=None):
    """
    grouping data
    
    :param by: grouping column_name
    :param group_function: function to apply group data
    
    return DataFrame
    """
    grouping_targets = []
    group_func_result = []
    for grouping_target, group_data in data.groupby(by):
        grouping_targets.append(grouping_target)
        group_func_result.append(group_function(group_data))

    result = pd.DataFrame([group_func_result])
    result = result.T
    result.index = grouping_targets
    
    if filter_func != None:
        result = result[list(map(filter_func, group_func_result))]
    
    return result

# MARK: - Examples
if __name__ == "__main__":
    sort_columns(pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }),
            ["B","A"])
    
    get_column_keywords(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }),
                          keywords= ["A"])
    
    compare_df(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }), 
        pd.DataFrame({
                                "A" : [2,3,4],
                                "B": [4,5,6]
                            }))
    
    remove_non_need_column(pd.DataFrame({
                                    "A" : [1,2,3],
                                    "B": [4,5,6]
                                }),
                          non_need_keywords=["A"])

    change_df(pd.DataFrame({
                                "A" : [1,2,3],
                                "B": [4,5,6]
                            }),
             column_name="A",
             apply_func = lambda x: x+1)

    search_column_index(pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }), 
                    ["B"],
                    search_mode=1)
    
    search_multi_conditions(data=pd.DataFrame({
                            "A" : [1,2,3],
                            "B": [4,5,6]
                        }), 
                        search_columns=[["A"]],
                        filter_funcs=[lambda x: x>=2]
                       )
    
    get_columns_keywords(pd.DataFrame({
        "A" : [1,2,3],
        "B" : [4,5,6],
        "C": [5,5,5]
    }), keywords_list = [
                ["A"],
                ["B"]
            ])

    group(pd.DataFrame({
        "A" : [1,2,3,1],
        "B": [4,5,6,1]}),
        by = "A",
        group_function=lambda x: sum(x["B"]),
        filter_func=lambda x: x > 5
    )

    search(pd.DataFrame({
        "A": [1,2,3],
        "B": [4,5,6],
        "C": [7,8,9]}
        ),
           search_columns = ["A"],
           filter_funcs=[lambda x: x==1],
           showing_columns = [["A"], ["B"]],
           mode=0
          )
    