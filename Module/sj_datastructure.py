
# Common Libraries
import copy
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Custom Libraries
from sj_linux import exec_command, make_command
from sj_string import search_stringAcrossTarget

# Custom Libraries

# Sources

# 정렬관련 #############################################################################################
# 퀵 정렬
def quick_sort(itr, cmp):
    if len(itr) <= 1:
        return itr
    else:
        pivot = itr[0]
        
        left_pivot = [e for e in itr[1:] if cmp(e,pivot) == True]
        right_pivot = [e for e in itr[1:] if cmp(e,pivot) == False]
        
        return quick_sort(left_pivot, cmp) + [pivot] + quick_sort(right_pivot, cmp)

def sort_usingRef(targets, refs, cmp):
    """
    Sort list using reference by compare method
    
    :param targets: population to be sorted(list)
    :param refs: reference population to sort(list)
    :param cmp: compare method(function) ex) lambda a, b: a < b
    
    return (list)
    """
    # Validation check - all element must be unique
    assert len(np.unique(refs)) == len(refs), "Each element must be unique"
    
    # Sort reference population
    sorted_refs = quick_sort(refs, cmp)
    
    # Mapping target and reference using index
    ref_indexes = [refs.index(ref) for ref in sorted_refs]
    return [targets[index] for index in ref_indexes]

# 검색 관련 #############################################################################################
def google_search(target, start):
    from bs4 import BeautifulSoup as BS 
    import ssl, urllib 
    import traceback
    import re
    base_url = 'https://www.google.co.kr/search' 

    #: 검색조건 설정 
    values = { 'q': target, # 검색할 내용 
              'oq': target, 
              'aqs': 'chrome..69i57.35694j0j7', 
              'sourceid': 'chrome', 
              'start' : str(start),
              'ie': 'UTF-8', 
              } 

    # Google에서는 Header 설정 필요 
    hdr = {'User-Agent': 'Mozilla/5.0'} 

    query_string = urllib.parse.urlencode(values) 
    req = urllib.request.Request(base_url + '?' + query_string, headers=hdr) 
    context = ssl._create_unverified_context() 

    try: 
        res = urllib.request.urlopen(req, context=context) 
    except: 
        traceback.print_exc() 

    html_data = BS(res.read(), 'html.parser')
    divs = html_data.select('#main > div')[2:]
    datas = []
    for d in divs:
        title = d.find(class_ = 'BNeawe vvjwJb AP7Wnd')
        url = None
        find_url_tag = d.find(class_ = 'kCrYT').select_one('a')
        if find_url_tag is not None:
            url = find_url_tag.get('href')
            url = re.search('h.+' , url).group()
            
        if not (title is None or url is None):
            datas = datas + [ (title.get_text(), url) ]    

    return datas


# Set 관련 함수(리스트로 구현) #############################################################################################
class Sets:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def union(self):
        return Sets_util.union(self.x, self.y)

    def intersection(self):
        return Sets_util.intersection(self.x, self.y)

    def difference(self):
        return Sets_util.intersection(self.x, Sets_util.complement(self.y, Sets_util.union(self.x, self.y)))

class Sets_util:
    @staticmethod
    def union(x, y):
        s_x = Sets_util.sort_unique(x)
        s_y = Sets_util.sort_unique(y)

        i = 0
        j = 0
        result = []
        while (True):
            x_last_i = len(s_x) - 1
            y_last_j = len(s_y) - 1

            # -1은 index의 끝을 의미
            if j > y_last_j:
                j = -1

            if i > x_last_i:
                i = -1

            if i == -1 and j == -1:
                break

            if i == -1:
                y_e = s_y[j]
                result.append(y_e)
                j += 1
            elif j == -1:
                x_e == s_x[i]
                result.append(x_e)
                i += 1
            else:
                x_e = s_x[i]
                y_e = s_y[j]

                if x_e == y_e:
                    j += 1
                else:
                    result.append(x_e)
                    i += 1
        return result

    @staticmethod
    def intersection(x, y):
        # 정렬해서 비교하면 더 쉬워짐
        s_x = Sets_util.sort_unique(x)
        s_y = Sets_util.sort_unique(y)

        i = 0
        j = 0
        result = []
        while (True):
            x_last_i = len(s_x) - 1
            y_last_j = len(s_y) - 1

            # 비교 대상이 없는 경우 비교 중단
            if j > y_last_j or i > x_last_i:
                break

            x_e = s_x[i]
            y_e = s_y[j]

            # 값이 같지 않을 경우 작은 집합의 index를 옮김
            # 값이 같을 경우 result에 추가
            if x_e < y_e:
                i += 1
            elif x_e == y_e:
                i += 1
                j += 1
                result.append(y_e)
            else:  # x_i > y_j
                j += 1
        return result

    @staticmethod
    def complement(x, u):
        result = []
        for u_e in u:
            if u_e not in x:
                result.append(u_e)
        return result

    @staticmethod
    def sort_unique(x):
        result = []
        s_x = sorted(x, key=lambda x: x)
        for i, e in enumerate(s_x):
            p_i = i - 1
            if p_i >= 0:
                if s_x[p_i] != e:
                    result.append(e)
            else:
                result.append(e)
        return result

class Tree:
    def __init__(self, data):
        self.data = data
        self.subtree = []

    def p_order(self):
        # 서브 트리의 데이터를 가지고 묶어야함
        if self.subtree == []:
            return [self.data]
        else:
            sub_tr_datas = []
            for s_tr in self.subtree:
                sub_tr_datas += s_tr.p_order()

            result = []
            temp = []
            for s_tr_d_l in sub_tr_datas:
                if type(s_tr_d_l) is list:
                    temp = [self.data] + s_tr_d_l
                else:
                    temp = [self.data] + [s_tr_d_l]

                result.append(temp)

            return result

    # For permutation helper
    @staticmethod
    def parse(structured_datas):
        trees = []
        for e in structured_datas:
            if type(e) is list:
                last_tree = trees[len(trees) - 1]
                last_tree.subtree = Tree.parse(e)
            else:
                trees.append(Tree(e))

        return trees

# DataFrame관련 ################################################################################################
# DataFrame에서 특정한 column을 지정해서 그 column의 unique한 데이터가 column으로 붙고 해당하는 데이터가 있는지를 체크하는 DataFrame을 반환
def append_unique_checked_column(df, col_names):
    # column 특정하기(반복) - col_names의 개수만큼 돌려야함
    # 특정된 column의 unique한 값을 뽑아내야함(반복)
    # unique한 값이 row에 들어있는지 판단후 있으면 True, 없으면 False로 데이터를 생성
    # 생성된 column을 dataframe에 집어넣는다.
    cp_df = copy.deepcopy(df)
    for sp_col in col_names:
        try:
            for uq_d in cp_df[sp_col].unique():
                n_col = cp_df[sp_col] == uq_d
                n_col_name = str(sp_col) + '_' + str(uq_d)
                cp_df[n_col_name] = n_col
        except KeyError as err:
            print('key Error occured! {}'.format(err))
            return df
    return cp_df


# DataFrame에서 특정한 수치형 데이터인 column을 지정해서 해당 column의 데이터의 범주를 체크하고 범주를 column으로 한 새로운 DataFrame 반환
def checked_column_inequalities(df, col_names, inequalities):
    # column 특정하기(반복) - col_names의 개수만큼 돌려야함
    # 부등식의 개수만큼 반복
    # 특정한 column의 값이 부등식에 속하는지 판단 속하면 True, 안속하면 False로 데이터 생성
    # 생성된 column을 dataframe에 집어넣는다.
    
    cp_df = copy.deepcopy(df)
    for sp_col in col_names:
        try:
            for inequality in inequalities:
                n_col = (inequality[0] <= cp_df[sp_col]) & (cp_df[sp_col] <= inequality[1])
                n_col_name = str(sp_col) + '_' + str(inequality[0]) + '~' + str(inequality[1])
                cp_df[n_col_name] = n_col
        except KeyError as err:
            print('key Error occured! {}'.format(err))
            return df
    return cp_df


# 특정한 데이터가 dataframe의 row에 얼마나 존재하는 지를 체크해주는 함수(dataframe의 첫번째 컬럼 대상)
def check_frequent_df(df, split_c, sp_datas):
    c_df = copy.deepcopy(df)

    for sp_d in sp_datas:
        counts = []  # 각 sp_d 에 대한 count 값을 누적하는 변수
        for row in c_df.values:
            count = 0  # 특정한 sp_d에 대한 row의 count값
            first_column_row = row[0]
            splited_row = first_column_row.split(split_c)
            for splited_row_e in splited_row:
                if splited_row_e == sp_d:
                    count += 1
            counts += [count]
        c_df[str(sp_d)] = counts
    return c_df

class Permutation:
    @staticmethod
    def permutation(ds, r):
        # data set이 들어오면 데이터 요소에서 r개를 선택하는 순열 생성
        result = []
        for d in Tree.parse(Permutation.permutation_helper(ds, r)):
            result += d.p_order()
        return result

    @staticmethod
    def permutation_helper(ds, r):
        # data 선택하기
        # data set에서 선택된 data 빼기
        # 밑의 데이터랑 모아서 결합한다음에 상위로 전달
        if r == 0 or ds == []:
            return []

        result = []
        for i, selected_d in enumerate(ds):
            selected_d2 = Permutation.permutation_helper(Permutation.except_data(ds, i), r - 1)
            result.append(selected_d)
            if selected_d2 != []:  # 빈값이 아니라면
                result.append(selected_d2)
        return result

    @staticmethod
    def except_data(datas, removal_index):
        return [e for i, e in enumerate(datas) if i != removal_index]



def make_3d_dataset(data, 
                    wrapping_dataset_name,
                    element_dataset_names,
                    dataset1_dim_names, 
                    dataset2_dim_names, 
                    dataset3_dim_names):
    """
    Make 3D dataset from 3D numpy array
    
    :param data(numpy array - shape(3d)): numpy data ex) (#time, #marker, #coord)
    :param wrapping_dataset_name(string): Wrapping name of total dataset
    :param element_dataset_names(list - string): dataset name list of each dataset within total dataset
    :param dataset1_dim_names(list - string): dimension name list of dataset1
    :param dataset2_dim_names(list - string): dimension name list of dataset2
    :param dataset3_dim_names(list - string): dimension name list of dataset3
    
    return (xarray.Dataset)
    """
    # Create the xarray Dataset
    ds = xr.Dataset(
        {
            wrapping_dataset_name : (element_dataset_names, data)
        },
        coords={
            element_dataset_names[0]: dataset1_dim_names,
            element_dataset_names[1]: dataset2_dim_names,
            element_dataset_names[2]: dataset3_dim_names
        }
    )
    return ds

# Test  #############################################################################################
if __name__=="__main__":
    import my_function

    # quick sort
    my_function.quick_sort( [4,1,2,3], lambda x1, x2: x1 < x2 )

    # Google Search
    my_function.google_search('hi', 0)

    # Set
    s = my_function.Sets([1,2,3,4], [1,3,4,5,6])
    s.union()
    s.intersection()
    s.difference()

    my_function.Sets_util.complement([2,3], [1,2,3,4,5,6,7,8,9])
    my_function.Sets_util.intersection([1,2, 5, 3, 9], [4,3,6, 2])

    # Permutation
    Permutation.permutation([1,2,3], 2)

    # DataFrame
    import pandas as pd
    df = pd.DataFrame(['ice cream', 'other vegetables'])
    ss = check_frequent_df(df, ',', ['ice cream', 'other vegetables'])

    # 3D dataframe
    ## Define the dimensions
    companies = ["MSFT", "APPL", "TSLA"]
    dates = pd.date_range('2024-01-01', periods=10)  # 10 days of data
    prices = ['Open', 'Close']

    ## Generate random data - Shape: (companies, dates, prices)
    data = np.random.rand(len(companies), len(dates), len(prices))

    ## Make dataset
    ds = make_3d_dataset(data = data,
                         wrapping_dataset_name = "Stock Prices",
                         element_dataset_names = ["Company", "Dates", "Prices"],
                         datset1_dim_names = companies,
                         datset2_dim_names = dates,
                         datset3_dim_names = prices)

    n_t = 10
    n_marker = 3
    n_coord = 3
    dummy_marker_pos = np.random.random((n_t, n_marker, n_coord))
    estim_3D_dataSet = make_3d_dataset(data = dummy_marker_pos,
                                       wrapping_dataset_name = "3D",
                                       element_dataset_names = ["Times", "Labels", "Coords"],
                                       dataset1_dim_names = np.arange(n_t),
                                       dataset2_dim_names = [marker_i for marker_i in range(n_marker)],
                                       dataset3_dim_names = ["X", "Y", "Z"])
    