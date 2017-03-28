import math

class NaiveBayesClassifier(object):
	def __init__(self, x, y): 
		self.classes = set(y)
		self.class_count = len(set(y))
		self.train = zip(x,y)
	
	def classProb(self, clss, input, method='regular'):
		try:
			x_probab = []
			clssProbability = len(filter(lambda u: u[1] == clss, self.train))/float(len(self.train))
			if method == 'regular':
				clss_instances = filter(lambda u: u[1] == clss, self.train)
				for i in range(len(input)):
					x_occurrence = len(filter(lambda u: u[0][i]== input[i], clss_instances))
					x_probab.append(x_occurrence/len(clss_instances))	
			if method == 'gaussian':
				x_mean, x_stdev = self._classMean(clss), self._classStd(clss)
				for i in range(len(input)):
					x, mean, stdev = input[i], x_mean[i], x_stdev[i]
					if stdev == 0:
						continue
					exponent = math.exp(-(math.pow(x-mean,2)/float(2*math.pow(stdev,2))))
					x_probab.append((1.0 / (math.sqrt(2*math.pi) * stdev)) * exponent)
			return (reduce(lambda x,y: x*y, x_probab) * clssProbability)
		except Exception as e:
			print e
	
	def _classMean(self, clss):
		def mean(values):
			return sum(values)/len(values)
		clss_instances = filter(lambda u: u[1] == clss, self.train)
		X = [u[0] for u in clss_instances]
		x_mean = tuple([mean(attr) for attr in zip(*X)])
		return x_mean

	def _classStd(self, clss):
		def stdev(values):
			avg = sum(values)/len(values)
			variance = sum([pow(x-avg,2) for x in values])/float(len(values)-1)
			return math.sqrt(variance)
		clss_instances = filter(lambda u: u[1] == clss, self.train)
		X = [u[0] for u in clss_instances]
		x_stdev = tuple([stdev(attr) for attr in zip(*X)])
		return x_stdev

	def predictClass(self, input, method='regular'):
		probabilityMap = {}
		for clss in self.classes:
			probabilityMap[clss] = classProb(clss, input, method)
		print 'Prediction:', max(probabilityMap, key=probabilityMap.get)