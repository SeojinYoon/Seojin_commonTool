#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Nov  9 17:48:12 2019

@author: yoonseojin
"""

'''
map_draw 함수 설명
    geo_path: 지역별 경계를 나누는 json 데이터가 있는 path
    place_data_set: 지역의 채도를 결정하는 숫자데이터 list(숫자가 크면 채도가 커짐)  ex) [1,2]
    place_names: 숫자데이터와 매핑할 행정동 이름 리스트 - geo_path에 존재하는 데이터만 place_name으로 지정할수 있음 ex) ['사직동', '혜화동']
    legend_name: 범례 이름 ex) ['사용자 밀집도']
    save_path: 저장될 path 
    circle_data_set: 원의 크기를 결정하는 숫자데이터 ex) [1,2]
    place_marker_names: 지역에 marking할 이름 ex) ['여기는 사직동입니다.', '여기는 혜화동입니다.']
    place_marker_coordinates: marking 위치 ex) [[37.57079395579531, 126.9716479937459], [37.58424893112498, 126.99787004917903]]
    circle_marker_coordinates: 상점위치 표시 
        ex) 
         [[37.57079395579531, 126.9716479937459]]

         만약 상점의 범주마다 color를 다르게 주고 싶다면 다음과 같이 입력한다.
         {
         '#FF0000' : [[37.558918299999995, 126.9962879], [37.558918299999995, 126.8962879]],
         '#0000FF' : [[37.548918299999995, 126.9962879], [37.548918299999995, 126.8962879]]
         }
    reference_image_path: 참조가 필요한 이미지 경로
    additional_legend_html: 지도에 관해 추가 설명이 필요한 경우 html 코드를 입력하여 설명할수 있음
'''
def map_draw(geo_path, 
             save_path = None,
             place_data_set = [], 
             place_names = [],
             legend_name = '',
             legend_categories_range = [],
             circle_data_set = [],
             circle_coordinates = [],
             place_marker_names = [],
             place_marker_coordinates = [],
             circle_marker_coordinates = [],
             reference_image_path = None,
             additional_legend_html = None):
    import json
    import folium
    from folium.features import DivIcon
    import warnings
    import pandas as pd
    from folium.plugins import FloatImage
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    
    geo_str = json.load(open(geo_path, encoding='CP949'))
    seoul_map = folium.Map(location = [37.5502, 126.982], zoom_start=10.5, tiles='cartodbpositron')

    if (len(place_data_set) == len(place_names)):
        if len(place_names) != 0:
            # 지역(동) 채색
            df = pd.DataFrame({'행정동': place_names, 'value': list(place_data_set)})
            
            if len(legend_categories_range):
                seoul_map.choropleth(
                    geo_data = geo_str,
                    data = df,
                    columns = ['행정동', 'value'],
                    fill_color = 'PuRd',
                    key_on = 'feature.properties.name',
                    bins = legend_categories_range,
                    legend_name = legend_name)
            else:
                seoul_map.choropleth(
                    geo_data = geo_str,
                    data = df,
                    columns = ['행정동', 'value'],
                    fill_color = 'PuRd',
                    key_on = 'feature.properties.name',
                    legend_name = legend_name)
    else:
        raise Exception('place 데이터 데이터 셋과 place 이름의 길이가 같지 않습니다!')
        
    # Marker 표현
    if (len(place_marker_coordinates) == len(place_marker_names)):
        if len(place_marker_names) != 0:
            for i, place_marker_coordinate in enumerate(place_marker_coordinates):
                folium.map.Marker(place_marker_coordinate,
                                  icon = DivIcon(icon_size=(150,36), 
                                                 icon_anchor=(0,0),
                                                 html='<div style="font-size: 6pt">{}</div>'.format(place_marker_names[i]))).add_to(seoul_map)
    else:
        raise Exception('place marker 데이터 셋과 place marker 위치 정보의 길이가 같지 않습니다!')
    
    # circle marker 경위도 위치 나타내기
    if type(circle_marker_coordinates) == list:
        for coord in circle_marker_coordinates:
            folium.Circle(radius = 1,
                          location = coord,
                          color = '#F6CECE',
                          fill = True).add_to(seoul_map)
    else:
        for color in circle_marker_coordinates:
            for coord in circle_marker_coordinates[color]:
                folium.Circle(radius = 1,
                              location = coord,
                              color = color,
                              fill = True).add_to(seoul_map)
    
    # 원그리기
        # 최대 자리수를 구한후
        # circle의 반경이 1000이 넘지 않도록(1000넘어가면 너무 커짐)
    def count_digits_str(n):
        return len(str(n))
    
    # counter digits
    def matching_digit(max_digit, target_digit):
        need_div = 1
        if max_digit >= target_digit:    
            while(True):
                if max_digit == target_digit:
                    break
                else:
                    max_digit -= 1
                    need_div *= 10
        else:
            while(True):
                if max_digit == target_digit:
                    break
                else:
                    max_digit += 1
                    need_div /= 10
        return need_div

    
    if (len(circle_coordinates) == len(circle_data_set)):
        if len(circle_data_set) != 0:
            find_circle_max_digits = max([count_digits_str(e) for e in circle_data_set])
            need_div = matching_digit(find_circle_max_digits, 3)
    
            if need_div > 0:
                flat_digits_circle_data_set = pd.Series(circle_data_set) / need_div
            else:
                flat_digits_circle_data_set = pd.Series(circle_data_set) * need_div
            
            for i, circle_data in enumerate(flat_digits_circle_data_set):
                folium.Circle(radius = circle_data,
                              location = circle_coordinates[i],
                              color = '#0000FF',
                              weight = 1,
                              fill = True).add_to(seoul_map)
                
                '''
                folium.map.Marker([circle_coordinates[i][0], circle_coordinates[i][1] - 0.1], #  circle_coordinates[i]
                              icon = DivIcon(icon_size=(150,36), 
                                             icon_anchor=(0,0),
                                             html='<div style="font-size: 10pt">{}</div>'.format(str(circle_data)))
                              ).add_to(seoul_map)
                              '''
    else:
        raise Exception('Circle 데이터 셋과 Circle 위치 정보의 길이가 같지 않습니다!')
 
    if reference_image_path is not None:
        FloatImage(reference_image_path, bottom=0, left=0).add_to(seoul_map)           
    
    if additional_legend_html is not None:
        seoul_map.get_root().html.add_child(folium.Element(additional_legend_html))
        
    # 저장
    if save_path != None:
        seoul_map.save(save_path)
        
    return seoul_map

def mean_location_per_area(geo_path):
    import numpy as np
    import json
    from functools import reduce
    geo_str = json.load(open(geo_path, encoding='CP949'))
    mean_locs = {}
    for geo in geo_str['features']:
        lats = []
        longs = []
        name = geo['properties']['name']
        for loc in reduce(lambda x, y: x+ y, geo['geometry']['coordinates']):
            lats.append(loc[0])
            longs.append(loc[1])
        
        if mean_locs.get(name) is not None:
            raise Exception('행정동 이름이 같은게 있습니다!!!')
        else:
            mean_locs[name] = [np.mean(longs), np.mean(lats)]
    return mean_locs

def find_loc(pivot_locs, find_areas):
    result = []
    for area in find_areas:
        result.append(pivot_locs[area])
    return result

# Test  #############################################################################################
if __name__== "__main__":
    import Folium_Extension
    
    result = convert_loc_to_region(long = 126.870338, lat = 37.631455, dong_kind = '법정동')

    # 예제1 - 맵 색칠만 
    Folium_Extension.map_draw(geo_path ='c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json',
                              place_data_set = [100, 200, 300],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              save_path = 'c:/Users/STU24/Desktop/공부/finance_data_project/맵_색칠만.html')
    
    # 예제2 - 맵 색칠 + 지역 marker 띄우기 
    place_centers = mean_location_per_area('/Users/yoonseojin/Desktop/아이티윌_공부/finance_data_project/필요행정동경계.json')
    Folium_Extension.map_draw(geo_path ='/Users/yoonseojin/Desktop/아이티윌_공부/finance_data_project/필요행정동경계.json',
                              place_data_set = [100, 200, 300],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              save_path = '/Users/yoonseojin/Desktop/아이티윌_공부/finance_data_project/맵_색칠_마커.html',
                              place_marker_names = ['여기는 종로1.2.3.4가동!!', '여기는 광희동!!', '여기는 교남동!!'],
                              place_marker_coordinates = find_loc(place_centers , ['종로1.2.3.4가동', '광희동', '교남동']))
    
    # 예제3 - 맵 색칠 + 지역 marker + 원 그리기
    place_centers = mean_location_per_area('c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json')
    
    Folium_Extension.map_draw(geo_path ='c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json',
                              place_data_set = [101, 201, 301],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              save_path = 'c:/Users/STU24/Desktop/공부/finance_data_project/맵_색칠_마커_원.html',
                              circle_data_set = [100, 200, 300],
                              circle_coordinates = find_loc(place_centers , ['명동', '장충동', '을지로동']),
                              place_marker_names = ['여기는 종로1.2.3.4가동!!', '여기는 광희동!!', '여기는 교남동!!', '명동', '장충동', '을지로동'],
                              place_marker_coordinates = find_loc(place_centers , ['종로1.2.3.4가동', '광희동', '교남동', '명동', '장충동', '을지로동']))
    
    # 예제4 - 맵 색칠 + 지역 marker + 원 그리기 + 상점위치표시
    place_centers = mean_location_per_area('c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json')
    Folium_Extension.map_draw(geo_path ='c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json',
                              place_data_set = [100, 200, 300],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              save_path = 'c:/Users/STU24/Desktop/공부/finance_data_project/맵_색칠_마커_원2.html',
                              circle_data_set = [100, 200, 300],
                              circle_coordinates = find_loc(place_centers , ['명동', '장충동', '을지로동']),
                              place_marker_names = ['여기는 종로1.2.3.4가동!!', '여기는 광희동!!', '여기는 교남동!!', '명동', '장충동', '을지로동'],
                              circle_marker_coordinates = [[37.558918299999995, 126.9962879]],  # 여기에 상점 위치 표시!
                              place_marker_coordinates = find_loc(place_centers , ['종로1.2.3.4가동', '광희동', '교남동', '명동', '장충동', '을지로동'])
                              )

    # 예제5 - 맵 색칠 + 카테고리 + 지역 marker + 원 그리기 
    place_centers = mean_location_per_area('c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json')
    
    Folium_Extension.map_draw(geo_path ='c:/Users/STU24/Desktop/공부/finance_data_project/필요행정동경계.json',
                              save_path = 'c:/Users/STU24/Desktop/공부/finance_data_project/맵_색칠_마커_원_카테고리.html',
                              place_data_set = [101, 201, 301],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              legend_categories_range = [100, 200, 300, 400],
                              circle_data_set = [100, 200, 300],
                              circle_coordinates = find_loc(place_centers , ['명동', '장충동', '을지로동']),
                              place_marker_names = ['여기는 종로1.2.3.4가동!!', '여기는 광희동!!', '여기는 교남동!!', '명동', '장충동', '을지로동'],
                              place_marker_coordinates = find_loc(place_centers , ['종로1.2.3.4가동', '광희동', '교남동', '명동', '장충동', '을지로동']))
    
    # 예제6 - 맵 색칠 + 카테고리 + 지역 marker(색상 다르게) + 원 그리기 + 범례 추가
    legend_html = '''
    <div style=”position: fixed; 
     bottom: 50px; left: 50px; width: 100px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     “>&nbsp;
     <span style="color:#FF0000">빨간색 글자입니다.</span>
     &nbsp; 
     West <br>
      </div>
     '''
    Folium_Extension.map_draw(geo_path ='/Users/yoonseojin/Desktop/아이티윌_공부/finance_data_project/필요행정동경계.json',
                              place_data_set = [100, 200, 300],
                              place_names = ['종로1.2.3.4가동', '광희동', '교남동'],
                              legend_name = '유동인구수',
                              save_path = '/Users/yoonseojin/Desktop/아이티윌_공부/finance_data_project/맵_색칠_마커_원_카테고리.html',
                              circle_data_set = [100, 200, 300],
                              circle_coordinates = find_loc(place_centers , ['명동', '장충동', '을지로동']),
                              place_marker_names = ['여기는 종로1.2.3.4가동!!', '여기는 광희동!!', '여기는 교남동!!', '명동', '장충동', '을지로동'],
                              circle_marker_coordinates = [[37.558918299999995, 126.9962879]],  #{
                              #                                       '#FF0000' : [[37.558918299999995, 126.9962879], [37.558918299999995, 126.8962879]],
                              #                                       '#0000FF' : [[37.548918299999995, 126.9962879], [37.548918299999995, 126.8962879]]
                              #                                       },
                              place_marker_coordinates = find_loc(place_centers , ['종로1.2.3.4가동', '광희동', '교남동', '명동', '장충동', '을지로동']),
                              #                               reference_image_path = '/Users/yoonseojin/Desktop/이나영.jpg',
                              additional_legend_html = legend_html)



