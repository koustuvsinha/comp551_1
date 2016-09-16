## Class to perform the following algorithms for Comp 551 course
#	1. Logistic Regression with Gradient Descent
#	2. Naive Bayes
#	3. Linear Regression
#	Also with regularization


import numpy as np

class logistic:

	def __init__(self, X=[0], y=0, alpha=0.07,iter_n=9000, W=0, lambd=0.01):
		# store dimensions for future use
		# m : Number of rows
		# n : Number of features
		self.m, self.n = X.shape
		# padding of X with 1's
		ones = np.array([1] * self.m).reshape(self.m, 1)
		self.X = np.append(ones,X,axis=1)
		self.y = y
		# Initialize W
		# Can be modified to random values
		self.W = np.array([0.0] * (self.n + 1))
		self.alpha = alpha
		self.iter_n = iter_n # Number of iterations for gradient descent
		self.lambd = lambd # regularization

	# Sigmoid function
	def sigmoid(self,a):
		return (1 / (1 + np.exp(-a)))

	def loss(self):
		# calculate wTx
		wTx = self.sigmoid(self.X.dot(self.W))
		#
		c1 = (-self.y.dot(np.log(wTx)))
		c2 = (1 - self.y).dot(np.log(1 - wTx))
		# Regularization
		r = 0.5 * 0.01 * np.sum(self.W[1:,] ** 2)
		loss = (c1 - c2 + r) / self.m
		return np.sum(loss)

	def gradient_descent(self):
		for i in range(0,self.iter_n):
			w = self.W
			wTx = self.sigmoid(np.dot(self.X,np.transpose(self.W)))
			diff = wTx - self.y
			# Update W. For first row skip
			self.W[0] = w[0] - self.alpha * (1.0 / self.m) * np.sum(diff.dot(self.X[:,0]))
			n = self.X.shape[1]
			# Can be done in vector operation
			for j in range(1,n):
				self.W[j] = w[j] - self.alpha * (1.0 / self.m) * (np.sum(diff.dot(self.X[:,j])) + self.lambd * self.m * w[j])
			#grad = np.sum(np.dot(diff,self.X))
			#self.W = self.W - self.alpha * (1.0 / self.m) * grad
			cost = self.loss()
			print 'Iteration',i,'\tCost',cost

	def fit(self):
		self.gradient_descent()

	def predict(self, X):
		X = np.array(X)
		m = X.shape[0]
		ones = np.array([1] * m).reshape(m, 1)
		X = np.append(ones,X,axis=1)
		p = self.sigmoid(np.dot(X,np.transpose(self.W)))
		np.putmask(p, p >= 0.5, 1.0)
		np.putmask(p, p < 0.5, 0.0)
		print p

class linearRegression():
	def __init__(self, X=[0], y=0, alpha=0.07, model=1,iter_n=100, W=0, lambd=0.01):
		self.m, self.n = X.shape
		ones = np.array([1] * self.m).reshape(self.m, 1)
		self.X = np.append(ones,X,axis=1)
		self.y = y
		self.W = np.array([0.0] * (self.n + 1))
		self.alpha = alpha
		self.iter_n = iter_n # Number of iterations for gradient descent
		self.lambd = lambd # regularization

	def loss(self):
		wTx = np.dot(self.X,np.transpose(self.W))
		diff = wTx - self.y
		loss = (1. / (2 * self.m)) * (diff ** 2)
		return np.sum(loss)

	def gradient_descent(self):
		for i in range(0,self.iter_n):
			w = self.W
			wTx = np.dot(self.X,np.transpose(self.W))
			diff = wTx - self.y
			self.W = w - self.alpha * (1. / self.m) * diff.dot(self.X)
			#grad = np.sum(np.dot(diff,self.X))
			#self.W = self.W - self.alpha * (1.0 / self.m) * grad
			cost = self.loss()
			print 'Iteration',i,'\tCost',cost

	def fit(self):
		self.gradient_descent()

	def predict(self, X):
		X = np.array(X)
		m = X.shape[0]
		ones = np.array([1] * m).reshape(m, 1)
		X = np.append(ones,X,axis=1)
		p = np.dot(X,np.transpose(self.W))
		print p

class NaiveBayes():
	def __init__(self, X=[0],y=0,alpha=0.1,cutoff=0):
		self.X = X
		self.y = y
		self.m, self.n = X.shape
		self.alpha = alpha
		self.cutoff = cutoff

	def value_counts(self,k):
		uniq = set(list(k))
		d = {}
		for u in uniq:
			d[u] = np.sum(k==u)
		return d

	def fit(self):
		self.count_pos = [self.value_counts(k) for k in np.transpose(self.X[self.y==1])]
		self.count_neg = [self.value_counts(k) for k in np.transpose(self.X[self.y==0])]
		self.total_pos = float(sum(self.y==1))
		self.total_neg = float(sum(self.y==0))
		total = self.total_pos + self.total_neg
		self.prior_prob_pos = self.total_pos / total
		self.prior_prob_neg = self.total_pos / total
		#print self.count_pos
		#print self.count_neg

	def predict(self,X_test):
		m,n = X_test.shape
		predictions = np.zeros(m)

		for i,rows in enumerate(X_test):
			probXneg = np.zeros(n)
			probXpos = np.zeros(n)
			for j, value in enumerate(rows):
				#print value
				n_count = self.count_neg[j].get(value,0)
				p_count = self.count_pos[j].get(value,0)
				probXpos = (p_count + self.alpha) / (self.total_pos + self.alpha * len(self.count_pos[j]))
				probXneg = (n_count + self.alpha) / (self.total_neg + self.alpha * len(self.count_neg[j]))
			predictions[i] = np.log(self.prior_prob_pos) + np.sum(np.log(probXpos)) - np.log(self.prior_prob_neg) - np.sum(np.log(probXneg))
		p = predictions
		#print p
		np.putmask(p, p >= self.cutoff, 1.0)
		np.putmask(p, p < self.cutoff, 0.0)
		print p



