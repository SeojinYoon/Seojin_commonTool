# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:10:38 2019

@author: STU24
"""

# Common Libraries

# Custom Libraries

# Sources

def convert_weekday_to_string(weekday):
    if weekday == 0:
        return '월'
    elif weekday == 1:
        return '화'
    elif weekday == 2:
        return '수'
    elif weekday == 3:
        return '목'
    elif weekday == 4:
        return '금'
    elif weekday == 5:
        return '토'
    elif weekday == 6:
        return '일'
    else:
        return None
    
    
