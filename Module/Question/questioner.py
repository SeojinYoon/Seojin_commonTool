#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 04:57:30 2021

@author: yoonseojin
"""

import re
import numpy as np
import sys

def load_text_and_seperate(file_path, seperater):
    txts = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        txts = f.readlines()
    
    partitions1 = []
    partitions2 = []
    for txt in txts:
        seq_1 = re.findall('.*' + seperater, txt)
        seq_2 = re.findall(seperater+ '.*', txt)
        
        if len(seq_1) > 0 and len(seq_2) > 0:        
            partitions1.append(seq_1[0].replace(seperater, ''))
            partitions2.append(seq_2[0].replace(seperater, ''))
    return (partitions1, partitions2)
   
def check_randomly(file_path, seperater):
    partitions1, partitions2 = load_text_and_seperate(file_path = file_path, seperater = "//")
    
    incorrects = []
    while(True):
        index = int(np.random.uniform(0, len(partitions1)))
        seq_1 = partitions1[index]
        seq_2 = partitions2[index]
        print()
        print(seq_1)
        
        my_seq_2 = input()   
        
        print(seq_2)
        if my_seq_2 == '끝':
            break
        else:
            if my_seq_2 == 'x':
                incorrects.append(seq_1)
            else:
                continue
    return incorrects

def check_consecutive(file_path, seperater):
    partitions1, partitions2 = load_text_and_seperate(file_path = file_path, seperater = "//")
    
    incorrects = []
    for i in range(0, len(partitions1)):
        print()
        print(partitions1[i])
        
        my_seq_2 = input()   
        
        print(partitions2[i])
        if my_seq_2 == '끝':
            break
        else:
            if my_seq_2 == 'x':
                incorrects.append(partitions1[i])
            else:
                continue
    return incorrects

def remover(file_path, seperater):
    partitions1, partitions2 = load_text_and_seperate(file_path = file_path, seperater = "//")
    
    removes = []
    for i in range(0, len(partitions1)):
        print()
        print(partitions1[i])
        
        my_answer = input()   
        
        print(partitions2[i])
        if my_answer == '끝':
            break
        else:
            if my_answer == 'x':
                removes.append((i, partitions1[i]))
            else:
                continue
    return removes

def removes(file_path, indexes):
    partitions1, partitions2 = load_text_and_seperate(file_path = file_path, seperater = "//")
        
    with open(file_path, "w") as outfile:
        for pos, line in enumerate(partitions1):
            if pos not in indexes:
                outfile.write(line)
            else:
                print(pos, line)

# file_path = "/Users/clmn/statistics_sj/Module/Question/english_sentence"

program_name = sys.argv[0]
file_path = sys.argv[1]
option = sys.argv[2]

print(program_name)
print(file_path)
print(option)

if option == "random":
    check_randomly(file_path, "//")
elif option == "consecutive":
    check_consecutive(file_path, "//")
elif option == "remove":
    rms = remover(file_path, "//")
    removes(file_path, list(map(lambda x: x[0], rms))) 



    
