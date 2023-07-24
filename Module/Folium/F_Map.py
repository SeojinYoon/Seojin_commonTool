#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:57:49 2019

@author: yoonseojin
"""

'''
Kakao API를 사용해 Location과 행정동을 연결

참조 - 
https://developers.kakao.com/docs/restapi/getting-started
https://developers.kakao.com/docs/restapi/local#좌표-행정구역정보-변환
'''

import numpy as np
import requests

# Kakao 주소
Kakao_host = 'https://dapi.kakao.com/'

# Main 주소 뒤에 붙는 주소
kakao_coord2regioncode_address = 'v2/local/geo/coord2regioncode.json'

# 서진 키
seojin_admin_key = '029dfb82a3340c777cdf6cca56cd2cd8'
seojin_javascript_key = 'bca11ccd82d391d0069821f920921267'
seojin_resetapi_key = '38be4ad7d78a013508a0c80badf97e7f'
seojin_native_app_key = '981708af2bdf686988023b5516ac2d94'


# 용준 키
yoongjun_admin_key = 'a8e850c8affbbdd6200e7c19c62bdee0'
yoongjun_javascript_key = 'e1b0fd440d264176dc6d8105d37f2c44'
yoongjun_resetapi_key = '3a454ad7fa287f0f070d842f83b774ba'
yoongjun_native_app_key = 'cfad3503e37ba4784be7a4855fb8f871'

# 윤아 키
yoona_analysis_admin_key = 'bd427d717b1cd873692b0e3e5a832540'
yoona_analysis_javascript_key = '1861984fe286623e8352a851a3f46c16'
yoona_analysis_resetapi_key = 'f659071a8f3af77649d677a8077b9465'
yoona_analysis_native_app_key = '416a6cc47243c7595a90c8b8f0256869'

# 키세팅
Finance_data_analysis_admin_key = seojin_admin_key # <=> KakaoAk
Finance_data_analysis_javascript_key = seojin_javascript_key
Finance_data_analysis_resetapi_key = seojin_resetapi_key
Finance_data_analysis_native_app_key = seojin_native_app_key



# Reqeust 설명 
'''
Request ---
GET /v2/local/geo/coord2regioncode.{format} HTTP/1.1
Host: dapi.kakao.com
Authorization: KakaoAK {app_key}

x	x 좌표로 경위도인 경우 longitude	O	String
y	y 좌표로 경위도인 경우 latitude	O	String
input_coord	x, y 로 입력되는 값에 대한 좌표 체계	X(기본 WGS84)	WGS84 or WCONGNAMUL or CONGNAMUL or WTM or TM
output_coord	결과에 출력될 좌표 체계	X(기본 WGS84)	WGS84 or WCONGNAMUL or CONGNAMUL or WTM or TM
lang	응답을 받을 언어를 설정. ko 나 en 중 하나를 설정	X(기본 ko)	String
'''

# Response 설명
'''
Response ---
meta
키	설명	타입
total_count	매칭된 문서수	Integer

document
키	설명	타입
region_type	지상/지하/공중 여부	H(행정동) or B(법정동)
address_name	전체 지역 명칭	String
region_1depth_name	지역 1Depth명 - 시도 단위(바다지역시 존재안함)	String
region_2depth_name	지역 2Depth명 - 구 단위(바다지역시 존재안함)	String
region_3depth_name	지역 3Depth명 - 동 단위(바다지역시 존재안함)	String
region_4depth_name	지역 4Depth명 - region_type 이 법정동이며, 리 영역인 경우만 존재	String
code	region 코드	String
x	X 좌표값 혹은 longitude	String
y	Y 좌표값 혹은 latitude	String
'''

# public function

'''
convert_loc_to_region 호출 파라미터 정리
dong_king: '행정동', '법정동'
'''

def convert_loc_to_region(long, lat, dong_kind = '행정동'):
    # 행정동: 법정동의 크고 작음에 따라 효율적인 행정편의 및 관리를 위하여 재편성한 후 주민센터를 설치한 단위(즉, 법정동을 행정적 편의를 위해 재편성한것)
    # 법정동: 법으로 정해진 동(과거로부터 내려온 동네 명칭)
    
    city_arg = 'city'
    district_arg = 'district'
    dong_arg = 'dong'
    
    city = 'region_1depth_name'
    district = 'region_2depth_name'
    dong = 'region_3depth_name'
          
    admin_dong = 'H'    
    law_dong = 'B'
    try:
        res = _req_loc_to_region(long, lat)['documents']
        data = None
        if dong_kind == '행정동':
            data = list(filter(lambda x: x['region_type'] == admin_dong, res))[0]
        elif dong_kind == '법정동':
            data = list(filter(lambda x: x['region_type'] == law_dong, res))[0]
        else:
            raise Exception('유효하지 않은 동 종류입니다.')
    
        return {
                city_arg : data[city],
                district_arg : data[district],
                dong_arg : data[dong]
                }
    except Exception as err:
        print('error occured! : {}'.format(err))
        return {
                city_arg : np.NaN,
                district_arg : np.NaN,
                dong_arg : np.NaN
                }
        
def _req_loc_to_region(long, lat):    
    headers = {
        'Authorization' : 'KakaoAK' + ' ' + Finance_data_analysis_admin_key
        }
    
    params = {
        'x' : long,
        'y' : lat
        }
    url = Kakao_host + kakao_coord2regioncode_address
    req = requests.get(url, params = params, headers = headers)
    
    return req.json()


def _append_str(target_str, append_str, sep = ' '):
    if target_str == '':
        return append_str
    else:
        return target_str + sep + append_str

# Test  #############################################################################################
if __name__== "__main__":
    import F_Map
    
    result = convert_loc_to_region(long = 126.870338, lat = 37.631455, dong_kind = '법정동')

