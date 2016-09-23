## Test File for Comp 551
# Simple test script to check our implementation

import unittest
import imp
learner = imp.load_source('learner', 'src/learner.py')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

X = np.array([[0,0],[0,2],[2,0],[2,2],[5,0],[5,2],[7,0],[7,1]])
y_cat = np.array([0,0,0,0,1,1,1,1])
y_reg = np.array([0.2,0.5,0.8,0.9,1.2,2.4,3.1,3.1])
X_test = np.array([[0,2],[7,1],[4,0],[1,2]])

class LearnerTest(unittest.TestCase):

	def test_sigmoid(self):
		#self.assertEqual(l.sigmoid(0),0.5)
		pass

	def test_logistic(self):
		print 'Logistic Regression'
		l = learner.logistic(iter_n=100)
		l.fit(X,y_cat)
		l.predict(X_test)
		print "Now comparing with scikit learn"
		lg = LogisticRegression(penalty='l2',max_iter=100)
		lg.fit(X,y_cat)
		print lg.predict(X_test)

	def test_linear(self):
		print 'Linear Regression'
		lr = learner.linearRegression(iter_n=100)
		lr.fit(X,y_reg)
		lr.predict(X_test)

	def test_naive(self):
		print 'Naive Bayes'
		ln = learner.NaiveBayes(alpha=0.00000001)
		ln.fit(X,y_cat)
		ln.predict(X_test)

	def test_crossvalid(self):
		print 'Cross Validation'
		l = learner.logistic(iter_n=100)
		lc = learner.cross_validation()
		Xt = np.array([[0,0],[0,2],[2,0],[2,2],[5,0],[5,2],[7,0],[7,1],[0,2],[7,1],[4,0],[1,2]])
		yt = np.array([0,0,0,0,1,1,1,1,0,1,1,0])
		scores = lc.cross_val_score(l,Xt,yt,3)
		print scores

if __name__ == '__main__':
    unittest.main()