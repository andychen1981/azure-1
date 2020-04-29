import numpy as np
import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, confusion_matrix, f1_score, make_scorer
from sklearn.feature_selection import SelectFromModel

def showAccuracy(model, cls_name, test_d, test_l, kPlot=True):
	predict = model.predict(test_d)		#this is not needed - only for initial verification

	proba   = model.predict_proba(test_d)
	probability = proba[:,1]		#we are interested in the class 1 (see extractClass)
	prediction = [round(value).astype(int) for value in probability]	#threshold at 0.5

	#predict and prediction should be identical; now you can threshold probability using something != 0.5
	accuracy = accuracy_score(test_l, prediction)
	balanced = balanced_accuracy_score(test_l, prediction)

class myDb(object):
	def __init__(
		self, 
		datasetroot=kDatasetPath, 
		datasetname='training', 
		csvname='reference.csv'
	):
		self._datasetroot = datasetroot
		self._datasetname = datasetname
		self._csvname	  = csvname
		self._filelist    = []
		self._labels      = []
		self._cases		  = defaultdict(list)

	#generator for a subset specified by 'indices'
	def subset(self, indices=None):
		filelist = self.filelist
		dir = self.datasetpath()

		if indices is None:
			for i in range(0, len(filelist), 1):
				yield dir + filelist[i]
		else:
			for i in indices:
				yield dir + filelist[i]

	#generator for the entire dataset
	def __iter__(self):
		cases = self.cases		# a dictionary keyed by 'caseno'
		dir = self.datasetpath()
		for k, v in cases.items():
			for f in v:
				yield dir + f

	def validate(self, logging):
		count = 0
		failed = 0

		iter = (self)	#create a generator instance for ourselves

		for f in iter:
			result = validateOne(f)
			if result:
				count += 1
			else:
				failed += 1

		if logging:
			print("validate: count=%d, failed=%d" % (count, failed))