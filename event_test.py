# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:33:54 2016

@author: Xiaoqing
"""

import csv
with open('data_loc.csv','rb') as originaldata:
    event_info = csv.reader(originaldata)
    event = list(event_info)
title_row = event[0]
#event.pop(0)

# select useful columns
ind = []
col_remain = ['EVENT_DATE', 'EVENT_NAME', 'EVENT_TYPE', 'LOCATION']
for word in col_remain:
    for index, f_name in enumerate(title_row):
        if word == f_name:
            ind.append(index)
# discard repeated rows            
new_dict = []
for row in event:
    new_row = [f for i, f in enumerate(row) if i in ind]
    if new_row not in new_dict:
        new_dict.append(new_row)


with open('dictionary_event.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row2 in new_dict:
        wr.writerow(row2)
 