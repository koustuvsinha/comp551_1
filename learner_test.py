## Test File for Comp 551
# Simple test script to check our implementation

import unittest
import learner
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
		l = learner.logistic(X,y_cat,iter_n=100)
		l.fit()
		l.predict(X_test)
		print "Now comparing with scikit learn"
		lg = LogisticRegression(penalty='l2',max_iter=100)
		lg.fit(X,y_cat)
		print lg.predict(X_test)

	def test_linear(self):
		lr = learner.linearRegression(X,y_reg,iter_n=100)
		lr.fit()
		lr.predict(X_test)


if __name__ == '__main__':
    unittest.main()