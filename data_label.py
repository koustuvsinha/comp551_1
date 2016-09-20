# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:56:10 2016

@author: Xiaoqing
"""

import csv
with open('data_loc.csv','rb') as originaldata:
    event_info = csv.reader(originaldata)
    event = list(event_info)
title_row = event[0]
event.pop(0)

key_word = ['Marathon', 'Full Marathon', 'Challenge Marathon', 'Ottawa Marathon', 'Scotiabank Full Marathon', 'Scotiabank Ottawa Marathon', '42.2km', '42 KM Solo']

# label 1
for row in event:
    row.insert(0,'')
    for word in key_word:
        if word == row[5]:
            row[0] = '1'
            break
# label 0
for row2 in event:
    if row2[0] == '':
        row2[0] = '0'

title_row.insert(0,'LABEL')
event.insert(0,title_row)

with open('data_withLabel.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row3 in event:
        wr.writerow(row3)
