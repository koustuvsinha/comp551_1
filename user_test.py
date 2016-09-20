# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 16:16:52 2016

@author: Xiaoqing
"""
import re
import csv
with open('Project1_data.csv','rb') as originaldata:
    athlete_info = csv.reader(originaldata)
    athlete = list(athlete_info)
    
title_row = athlete[0]

# delete event information
ind = []
for index, f_name in enumerate(title_row):
    if f_name == 'CATEGORY':
        ind.append(index)
new_userid = []
for row in athlete:
    user_temp = [item for i, item in enumerate(row) if i in ind]
    new_userid.append(user_temp)
new_userid.pop(0)

# gender generation
for row in new_userid:
    row.insert(0, '')
    for categ in row:
        if categ != '' and categ[0] in ['F', 'H', 'M']:
            gender = categ[0]
            if gender == 'H':
                gender = 'M'
            row[0] = gender
            break

# age generation
for row in new_userid:
    row.insert(1,'')
    for index, categ in enumerate(row, 2):
        if re.match(r'\w\d{2}-\d{2}', categ):
            row[1] = categ[1:]
            break

# pop useless columns & add participant id
final_userid = []
for index, row in enumerate(new_userid):
    user_row = [item for i, item in enumerate(row) if i == 0 or i == 1]  
    user_row.insert(0,str(index))
    final_userid.append(user_row)  
    
# attach title
NEWtitle_row = ['PARTICIPANT_ID','GENDER','AGE']

final_userid.insert(0, NEWtitle_row)

with open('dictionary_user.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row2 in final_userid:
        wr.writerow(row2)
