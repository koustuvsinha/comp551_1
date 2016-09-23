#!/usr/bin/python
# -*- coding: utf-8 -*-

# File to Clean and Process the data


import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import unicodedata

COLS = ['PARTICIPANT_ID','EVENT_DATE','EVENT_NAME','EVENT_TYPE','TIME','CATEGORY']
URL = "https://www.sportstats.ca/events.xhtml"
PAGE_LOAD_WAIT_TIME = 5

class Process:
	def __init__(self, fileName='Project1_data.csv'):
		self.fileName = fileName
		convertColtoRows()
		self.df = clean_event(self.df)
		extractFeatures()
		self.df.to_csv('clean_data_17_09_16.csv')
		#extractLocations()

	def convertColtoRows():
		""" Convert Column to Rows """
		df = pd.DataFrame(columns=COLS)
		unclean_data = csv.reader(open(self.fileName,"rb"))
		ct = 0
		for row in unclean_data:
		    ct = ct + 1
		    if ct == 1:
		        continue
		    #print row
		    r = row
		    n = []
		    idn = r.pop(0)
		    #print idn,
		    t = 0
		    for j in r:
		        #print j
		        #print t
		        if t % 5 == 0:
		            if t != 0:
		                df = df.append(pd.DataFrame([n],columns=COLS))
		            n = []
		            n.append(idn)
		            n.append(j)
		        else:
		            n.append(j)
		        t = t + 1
		    df = df.append(pd.DataFrame([n],columns=COLS))
		self.df = df
		print 'Total Count of Rows = ',len(df)

	def clean_event(self,df):
		""" Clean the Event Type to more Generalized form """
    replace_dict = {
    '10 km':'10K Marathon',
    '5 km':'5 KM',
    'Demi Marathon':'Demi-Marathon',
    'Demi-marathon':'Demi-Marathon',
    '21 km':'21 KM',
    '20 Km Route':'20 KM',
    '20 km Route':'20 KM',
    '10km':'10K Marathon',
    '20 km':'20 KM',
    '21.1 km':'21.1 KM',
    '21,1 km':'21.1 KM',
    '20 km route':'20 KM',
    'Individuel':'Individual',
    '10 KM':'10K Marathon',
    '5 km':'5 KM',
    'Demi Marathon':'Demi-Marathon',
    'Demi-marathon':'Demi-Marathon',
    '21 KM':'Half-Marathon',
    '20 Km Route':'20 KM',
    '20 km Route':'20 KM',
    '10km':'10K Marathon',
    '20 km':'20 KM',
    '21.1 km':'21.1 KM',
    '21,1 km':'21.1 KM',
    '20 km route':'20 KM',
    'Half Marathon':'Half-Marathon',
    'Ottawa Marathon':'Marathon',
    'Ottawa Half Marathon':'Half-Marathon',
    'Scotiabank Ottawa Marathon':'Marathon',
    'Scotiabank Ottawa Half Marathon':'Half-Marathon',
    '10 KM':'10K Marathon',
    '10 Km Route':'10K Marathon',
    'Full Marathon':'Marathon',
    '19 km':'Half-Marathon',
    '11 km':'10K Marathon',
    '10 Km':'10K Marathon',
    '8 km':'10K Marathon',
    'Demi marathon':'Demi-Marathon',
    'Demi Marathon Course ':'Demi-Marathon',
    '10km Course':'10K Marathon',
    '21km Course':'Half-Marathon',
    '21.1km':'Half-Marathon',
    '23.5 KM The North Face':'Half-Marathon',
    '23 km':'Half-Marathon',
    '21 km - course':'Half-Marathon',
    'Individuel':'Individual',
    'Ottawa Half-Marathon - Demi-marathon':'Half-Marathon - Demi-Marathon',
    '1/2 Marathon - Demi-marathon':'Half-Marathon - Demi-Marathon',
    'Half Marathon - Demi Marathon':'Half-Marathon - Demi-Marathon',
    '5 KM':'5K Road Race',
    '15 km':'15K Marathon',
    '30 km':'30K Marathon',
    '30km Race':'30K Marathon',
    '22.2 Km':'Half-Marathon',
    '21,1 km Course':'Half-Marathon',
    '21 km (Course)':'Half-Marathon',
    '22.2 Km Chemin de Campagne':'Half-Marathon',
    '22.2 km chemin de campagne':'Half-Marathon',
    '21.1 km course':'Half-Marathon',
    '25 KM':'Half-Marathon',
    '21,1 km Course':'Half-Marathon',
    '22.2 Km':'Half-Marathon',
    '21 km (Course)':'Half-Marathon',
    '25 KM SkyRace®':'Half-Marathon',
    '18 km':'Half-Marathon',
    '21km':'Half-Marathon',
    '21 Km':'Half-Marathon',
    '21.1K':'Half-Marathon',
    '21 km Media Challenge':'Half-Marathon',
    '25km Classique':'Half-Marathon',
    '21.1 km Trail Run':'Half-Marathon'
    }
    for key in replace_dict:
        df.EVENT_TYPE = df.EVENT_TYPE.replace(key,replace_dict[key])
        return df

	def extractFeatures(self):
		""" Extract common Features """
		df = self.df
  	df['GENDER'] = df.CATEGORY.apply(lambda x : (str(x)+" ")[0])
  	df['EVENT_DT'] = pd.to_datetime(df.EVENT_DATE)
  	df = df.reset_index()
  	df = df.drop('index',1)
  	df['DAY'] = df.EVENT_DT.apply(lambda x : x.day)
  	df['YEAR'] = df.EVENT_DT.apply(lambda x : x.year)
  	df['DAY_OF_WEEK'] = df.EVENT_DT.apply(lambda x : x.dayofweek)
  	df['MONTH'] = df.EVENT_DT.apply(lambda x : x.month)
  	self.df = df

  def extractLocations(self):
  	""" Extract Location of events from Sportstats """
		df = self.df
		driver = webdriver.Chrome()
		locations = []
		loc_d = {}
		issues = []
		ct = 0
		df['LOCATION'] = ''
		for index, row in df.iterrows():
	    #print index,
	  if row['LOCATION'] != '' :
	  	continue
		if row['EVENT_NAME'] not in loc_d:
		    d = {}
		    box = driver.find_elements_by_id("mainForm:raceOrEventNameSearchField")[0]
		    box.clear()
		    box.send_keys(row['EVENT_NAME'].decode('unicode_escape').encode('ascii','ignore'))
		    box.send_keys(Keys.ENTER)
		    time.sleep(PAGE_LOAD_WAIT_TIME)
		    table_div = driver.find_elements_by_xpath("//*[@class='cal-results']/tbody/tr/td/div/div[1]/div[2]")
		    try:
		        for div in table_div:
		            #print div.text
		            fields = div.text.split(" (")
		            d[fields[0]] = fields[1][0:-1]
		        loc_d[row['EVENT_NAME']] = d
		    except:
		        issues.append(index)
		else:
		    d = loc_d[row['EVENT_NAME']]
	    #print d
	    loc = ''
	    # if cant match by date, then get the first one
	    if row['EVENT_DATE'] not in d:
	        if len(d.values()) > 0:
	            loc = d.values()[0]
	    else:
	        loc = d[row['EVENT_DATE']]
	    locations.append(loc)
	    df.set_value(index,'LOCATION',loc)
	    ct = ct + 1
	    #if ct > 3:
	    #    sys.exit(0)
	  df['LOCATION'] = df.LOCATION.apply(lambda x : unicode(x).encode("utf-8"))
    self.df = df
