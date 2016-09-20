# Runner - To run the algorithms and provide the output

import pandas as pd
import numpy as np
import learner as ln

YEARS = [2012,2013,2014,2015]
TOPREDICT = 2016
## Cleaned data file for classification (not feature engineered)
DATA_FILE = 'final_data_v2.csv'
## Cleaned and feature engineered file for regression
REG_DATA_FILE = 'RegressionTest1.CSV'
GenderMap = {'M' : 0, 'F' : 1}
UNIQUE_IDS = 8711
FINAL_SUBMISSION_FILE = 'FinalSubmission.csv'

class Runner():
    """docstring for Runner"""
    def __init__(self):
        self.data = pd.read_csv(DATA_FILE)
        self.dataReg = pd.read_csv(REG_DATA_FILE)
        self.validation()
        self.prediction()


    def transformYearData(self,num_users):
        """ Years in which user has participated in a Montreal Marathon (0 no, 1 yes) """
        data = self.data
        ndr = []
        mon_mara = data[(data.EveTyp == 6) & data.LOCATION.str.contains('montreal',case=False)]
        for x in range(num_users):
            nr = np.zeros(len(YEARS)) # 2012, 2013, 2014, 2015
            for i,rec in mon_mara[mon_mara['PARTICIPANT_ID'] == x].iterrows():
                nr[YEARS.index(rec['YEAR'])] = 1
            ndr.append(nr)
        ndr = np.vstack(ndr)
        return np.array(ndr)

    def transformTotalMaraMon(self,num_users):
        """ Total Montreal Marathons participated by a user 
            Also accounts for all marathons in Montreal (including demi,half etc, which were labelled LABEL = 1) """
        data = self.data
        dt = data[(data.LABEL == 1) & data.LOCATION.str.contains('montreal',case=False)].groupby(['PARTICIPANT_ID']).agg('count').reset_index()
        numr = []
        for x in range(num_users):
            v = dt[dt.PARTICIPANT_ID == x]['EVENT_DATE']
            if len(v) > 0:
                numr.append(v.iloc[0])
            else:
                numr.append(0)
        return np.array(numr).reshape(-1,1)

    def transformTotalMara(self,num_users):
        """ Total Marathons participated by a user """
        data = self.data
        dt = data[data.LABEL == 1].groupby(['PARTICIPANT_ID']).agg('count').reset_index()
        numr = []
        for x in range(num_users):
            v = dt[dt.PARTICIPANT_ID == x]['EVENT_DATE']
            if len(v) > 0:
                numr.append(v.iloc[0])
            else:
                numr.append(0)
        return np.array(numr).reshape(-1,1)

    def transformTotal(self,num_users):
        """ Total Events participated by a user """
        data = self.data
        dt = data.groupby(['PARTICIPANT_ID']).agg('count').reset_index()
        numr = []
        for x in range(num_users):
            v = dt[dt.PARTICIPANT_ID == x]['EVENT_DATE']
            if len(v) > 0:
                numr.append(v.iloc[0])
            else:
                numr.append(0)
        return np.array(numr).reshape(-1,1)

    def transformYearTotalData(self,num_users):
        """ Total Events Participated by User in each year """
        years = [2012,2013,2014,2015,2016]
        data = self.data
        ndr = []
        dt = data.groupby(['PARTICIPANT_ID','YEAR']).agg('count').reset_index().sort_values('EVENT_DATE',ascending=False)
        for x in range(num_users):
            nr = np.zeros(len(years)) # 2012, 2013, 2014, 2015 and 2016
            for i,rec in dt[dt['PARTICIPANT_ID'] == x].iterrows():
                nr[years.index(rec['YEAR'])] = rec['EVENT_DATE']
            ndr.append(nr)
        return np.array(np.vstack(ndr))

    def meanAge(self):
        """ Get Mean Age of participant throughout his/her career """
        data = self.data
        ds = data.groupby(['PARTICIPANT_ID']).agg('mean').reset_index()
        ndr_age = ds.AvgAge.tolist()
        return np.array(ndr_age).reshape(-1,1)

    def transformGender(self,num_users):
        """ Transform Gender into 0 and 1 """
        data = self.data
        ndr = []
        for x in range(num_users):
            done = False
            for i,rec in data[data['PARTICIPANT_ID'] == x].iterrows():
                if rec['GENDER'] == 'M' or rec['GENDER'] == 'F':
                    ndr.append(GenderMap[rec['GENDER']])
                    done = True
                    break
            if not done:
                ndr.append(-1)
        return np.array(ndr).reshape(-1,1)

    def split_test_train(self,X,y,per=0.7):
        """ Split Test Train """
        msk = np.random.rand(len(X)) < per
        X_train = X[msk]
        X_test = X[~msk]
        y_train = y[msk]
        y_test = y[~msk]
        return X_train,X_test,y_train,y_test

    def aggregateColumns(self,stype='test'):
        """ Aggregate all columns and return X and y """
        ndr_age = self.meanAge()
        ndr_gender = self.transformGender(UNIQUE_IDS)
        ndr_mara = self.transformTotalMara(UNIQUE_IDS)
        ndr_total = self.transformTotal(UNIQUE_IDS)
        ndr_total_m = self.transformTotalMaraMon(UNIQUE_IDS)
        ndr_total_year = self.transformYearTotalData(UNIQUE_IDS)
        ndr_year = self.transformYearData(UNIQUE_IDS)
        ndr_all_but_yr = np.concatenate((ndr_age,ndr_gender,ndr_mara,ndr_total,ndr_total_m,ndr_total_year),axis=1)
        ndr_year_X = []
        ndr_year_y = []
        if stype == 'test':
            ndr_year_X = ndr_year[:,[0,1,2]] # taking 2012,2013 and 2014 for X
            ndr_year_y = ndr_year[:,[3]].reshape((-1,)) # taking 2015
        else:
            ndr_year_X = ndr_year[:,[1,2,3]] # prediction case, 2013, 2014 and 2015
        X = np.concatenate((ndr_all_but_yr, ndr_year_X), axis=1)
        y = ndr_year_y
        return X,y

    def accuracy(self,y_pred,y_true):
        """ Calculate Classification Accuracy """
        return np.mean(y_pred == y_true)

    def mse(self,y_pred, y_true):
        """ Calculate Mean Squared Error for Regression """
        return ((y_pred - y_true) ** 2).mean()

    def getRegData(self, stype='train'):
        """ Pass X,y needed for regression """
        X = []
        y = []
        reg = self.dataReg
        if stype == 'train':
            X = reg[['AGE','GENDER','AvgTimeForMontrealMarathon2012','AvgTimeForMontrealMarathon2013','AvgTimeForMontrealMarathon2014','AvgTimeInAllMarathons','AvgTimeInAllEvents','TotalNoOfMarathonEvents']]
            y = reg.AvgTimeForMontrealMarathon2015
        else:
            X = reg[['AGE','GENDER','AvgTimeForMontrealMarathon2013','AvgTimeForMontrealMarathon2014','AvgTimeForMontrealMarathon2015','AvgTimeInAllMarathons','AvgTimeInAllEvents','TotalNoOfMarathonEvents']]
        return X,y

    def validation(self):
        """ Validate all algorithms for train test set """
        ## feature engineering for classification
        X,y = self.aggregateColumns()
        X_train,X_test,y_train,y_test = self.split_test_train(X,y)
        # Logistic Regression
        print 'Running Logistic Regression'
        lg = ln.logistic(iter_n=100)
        lg.fit(X_train,y_train)
        y_p = lg.predict(X_test)
        print 'Accuracy:',self.accuracy(y_p,y_test)
        print 'Running Naive Bayes'
        lnb = ln.NaiveBayes()
        lnb.fit(X_train,y_train)
        y_p1 = lnb.predict(X_test)
        print 'Accuracy:',self.accuracy(y_p1,y_test)
        X,y = self.getRegData()
        X_train,X_test,y_train,y_test = self.split_test_train(X,y)
        print 'Running Linear Regression'
        lr = ln.linearRegression(iter_n=500)
        lr.fit(X_train,y_train)
        y_p2 = lr.predict(X_test)
        print 'Mean Squared Error : ',self.mse(y_p2,y_test)
        ## store thetas for future prediction
        self.logistic = lg
        self.linear = lr
        self.naive = lnb

    def prediction(self):
        """ Final Prediction for Montreal 2016 Event """
        final_df = pd.DataFrame(columns=['PARTICIPANT_ID','Y1_LOGISTIC','Y1_NAIVEBAYES','Y2_REGRESSION'])
        X,y = self.aggregateColumns(stype='predict')
        print 'Predicting by Logistic Regression'
        y_log = self.logistic.predict(X)
        print 'Predicted Rows : ', len(y_log)
        print 'Predicting by Naive Bayes'
        y_naive = self.naive.predict(X)
        print 'Predicted Rows :', len(y_naive)
        X,y = self.getRegData(stype='predict')
        print 'Predicting Linear Regression'
        y_lin = self.linear.predict(X)
        print 'Predicted Rows :',len(y_lin)
        final_df.PARTICIPANT_ID = range(UNIQUE_IDS)
        final_df.Y1_LOGISTIC = y_log
        final_df.Y1_NAIVEBAYES = y_naive
        final_df.Y2_REGRESSION = y_lin
        final_df.to_csv(FINAL_SUBMISSION_FILE)


def run():
    print 'Running'
    Runner()

run()



