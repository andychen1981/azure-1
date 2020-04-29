import os, sys
import time
import pprint, pickle
from collections import defaultdict

import wave
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import imblearn
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.metrics import classification_report_imbalanced
import lightgbm

import pyutils.folderiter as folderiter
#from Cost_sensitive_boost import trainCalibratedAdaMEC # Our calibrated AdaMEC implementation 
#import Cost_sensitive_boost

#project includes
import wavutil as wu
import fft
import heartSound
import extractFeatures as eF
import hyperparams

kLogging=False
kXGBOOST=True
kEnableImb=False


def classiferSet(pre_cost_weight=20):
	#xgt = xgb.XGBClassifier(learning_rate=0.1, scale_pos_weight=10, n_estimators=100, random_state=1) #80.77%
	xgt = xgb.XGBClassifier(
		learning_rate=0.1,
		#subsample=0.99,
		max_depth=3,
		scale_pos_weight=pre_cost_weight, 
		n_estimators=80,
		#cv=5,
		#subsample=.99,
		random_state=27,
		nthread=2			#use more threads only for large dataset
	)	#84.62%

	ada = AdaBoostClassifier(
		n_estimators=100, 
		learning_rate=.1, 
		random_state=1234
	)	#(0,130): .815
	
	#gbt = GradientBoostingClassifier(n_estimators=100, subsample=1.0, learning_rate=1, random_state=1234)		#(0,130): .830
	gbt = GradientBoostingClassifier(
		n_estimators=100, 
		subsample=0.99, 
		learning_rate=.1, 
		random_state=1234
	)		#(0,130): .861															#
	
	rf  = RandomForestClassifier(
		n_estimators=100,
		#max_depth=10,
		oob_score=True,
		class_weight={0:1,1:pre_cost_weight},
		#class_weight='balanced',
		random_state=1234
	)				#.846

	brf = BalancedRandomForestClassifier(
		n_estimators=100, 
		oob_score=True,
		class_weight={0:1,1:pre_cost_weight},
		random_state=1234
	)

	rus = RUSBoostClassifier(
		n_estimators=100, 
		random_state=1234
	)
	#https://www.kaggle.com/c/home-credit-default-risk/discussion/60921
	#https://sites.google.com/view/lauraepp/parameters
	lgbm = lightgbm.LGBMClassifier(
		boosting_type='dart',		#'gbdt', 'goss', 'dart'
		num_leaves=31, max_depth=-1, learning_rate=0.1,
		class_weight=None,		#{0:1,1:pre_cost_weight}, using this is inferior to default
		random_state=1234
	)

	ourmodels = dict(
		{'AdaBoost': 	 ada,
		'GradientBoost': gbt, 
		'RandomForest':  rf,
		'BalancedRandomForest': brf,
		'RUSBoost':		 rus,
		'XGBoost': 		 xgt,
		'LightGBM':		 lgbm
		}
	)
	return ourmodels


class classifiers(object):	
	def __init__(
		self,
		pre_cost_weight=20
	):
		self._models = classiferSet(pre_cost_weight)

	@staticmethod
	def classid(cls):
		classstr = type(cls)
		tokens = str(classstr).split('.')
		#internals = dir(cls)
		return tokens[-1][0:-2]

	def getClassifer(self, classiferName='GradientBoost'):
		return self._models[classiferName]			


def showFeatureImportance(importance, kHistogram=True, kSavePNG=None):
	#thresholds = sort(importance)
	#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
	print("xgt.feature_importances", importance)
	plt.bar(range(len(importance)), importance)

	if kHistogram:
		if kSavePNG:
			plt.savefig(kSavePNG)
		plt.show()

#process the wave files in 'heartDb' which is an instance of heartSound()
def processFeatures(
	heartDb, 
	indices,
	outputdir='output/',
	kMT=1,
	logging=False
):
	count = 0
	failed = 0
	kNumFeatures = 2 + 13*1

	def onefile(i, f):
		if logging:
			print("'%s'" % (f))
		features = eF.extractFeatures(f, kNumFeatures=kNumFeatures, shape='1D')
		assert(len(features) == kNumFeatures)
	
		return features

	#iter = (heartDb)	#create a generator instance for ourselves
	iter = heartDb.subset(indices)
	numentries = heartDb.subsetsize(indices)
	feature_table = np.ndarray((numentries, kNumFeatures), dtype=np.float)

	if kMT > 1:
		feature_table = eF.extractFeaturesMT(heartDb.filelist, heartDb.datasetpath(), kMT=kMT)
	else:
		for i, f in enumerate(iter):
			features = onefile(i, f)	#(framerate, nf)

			if features is not None:
				feature_table[i] = features
				count += 1
			else:
				print("'%s' failed." % (f))
				failed += 1

	if count == -1:
		wavarr = wu.loadWav(f)	#(framerate, nf)
		wu.plotSpectrogram(wavarr)
		fourier = fft.doFFT(wavarr[1])
		fft.plotFourierSignal(fourier, wavarr[0], caption="Fourier Real", kSavePNG=outputdir + "plot2.png")

	print("processFeatures: count=%d, failed=%d" % (count, failed))
	return np.asarray(feature_table)

	
def selectFeatures(model, importances, train_d, test_d, train_l, test_l):
	thresholds = np.sort(importances)
	maxaccuracy = 0
	
	for thresh in thresholds:
		selection = SelectFromModel(model, threshold=thresh, prefit=True)
		select_X_train  = selection.transform(train)
		#train model
		selection_model = XGBClassifier()
		selection_model.fit(select_X_train, train_l)
		# eval model
		select_X_test = selection.transform(test_d)
		y_pred = selection_model.predict(select_X_test)
		predictions = [round(value) for value in y_pred]
		accuracy = accuracy_score(test_l, predictions)
		maxaccuracy = max(maxaccuracy, accuracy)
		print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

	return maxaccuracy

def visualize_result(heartSoundDb, test_d_idx, test_l, prediction):
	files  = heartSoundDb.getFiles(test_d_idx)		#not needed for training (for user feedback)
	labels = heartSoundDb.getLabels(test_d_idx)
	alllables = heartSoundDb.getLabels(None)
	u, indices, counts = np.unique(test_l, return_index=True, return_counts=True)
	test_files = dict()
	for idx, f in enumerate(files):
		test_files[f] = idx

	cases = heartSoundDb.cases
	result = defaultdict(list)
	status = [0, 0]

	#build a dict organized by 'caseno' for the files in 'test_d_idx'
	for k, v in cases.items():
		for f in v:
			ouridx = test_files.get(f, -1)
			if ouridx >= 0:
				record = (f, labels[ouridx], prediction[ouridx])
				result[k].append(record)
				status[labels[ouridx]] += 1
	print("status: %s" % status)		#should match _, _, counts = np.unique(test_l)

	return result

def train(
	heartSoundDb,
	cls_name='GradientBoost',
	subset=None,
	outputdir='output/',
	modeldir='models/',
	optimize=False,
	kMT=1,
	logging=False,
	report_imbalanced=False
):
	logFile=open(outputdir + 'mylogfile'+'.txt', 'w')
	pp = pprint.PrettyPrinter(indent=1, width=120, stream=logFile)
	start = time.time()

	if heartSoundDb.isempty():
		print("WARNING: heartSoundDb '%s' is empty!" % heartSoundDb.datasetpath())
		return

	#1: process the sound db for 'subset' -> feature_table
	feature_table = processFeatures(heartSoundDb, subset, outputdir=outputdir, kMT=kMT, logging=logging)

	end = time.time()
	print("processFeatures %.2fsec" % (end - start))
	start = end

	files  = heartSoundDb.getFiles(subset)		#not needed for training (for user feedback)
	labels = heartSoundDb.getLabels(subset)

	u, indices, counts = np.unique(labels, return_index=True, return_counts=True)
	osscounts = []

	if kEnableImb:
		oss = imblearn.under_sampling.OneSidedSelection(sampling_strategy='auto', random_state=12)
		X_res, y_res = oss.fit_resample(feature_table, labels)
		_, osscount  = np.unique(y_res, return_index=False, return_counts=True)
		feature_table = X_res
		labels 		  = y_res

	print("train[subset=%s], oss[%s]" % (counts, osscounts))

	train_d_idx, test_d_idx, train_l, test_l = train_test_split(range(0,len(feature_table)), labels, test_size=.2, stratify=labels, random_state=42)
	train_d, test_d, train_l, test_l = train_test_split(feature_table, labels, test_size=.2, stratify=labels, random_state=42)

	models = classifiers(pre_cost_weight=20)

	if kXGBOOST:
		dtrain = xgb.DMatrix(train_d, label=train_l)
		dtest  = xgb.DMatrix(test_d,  label=test_l)

		#https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
		param = {'max_depth':3, 'eta':.3, 'silent':1, 'objective':'binary:logistic'}
		num_round = 200
		bst = xgb.train(param, dtrain, num_round)
		pred = bst.predict(dtest)
		#hyperparams.showAccuracy(bst, test_d, test_l)
		
		xgt = xgb.XGBClassifier(
			learning_rate=0.1,
			#subsample=0.99,
			max_depth=3,
			scale_pos_weight=20, 
			n_estimators=100,
			#cv=5,
			random_state=27,
			nthread=2			#use more threads only for large dataset
		)
		param_x = xgt.get_xgb_params()

		xgt.fit(train_d, train_l)
		end = time.time()
		start = end
		
		pp.pprint(param_x)
		print("XGBClassifier %.2fsec" % (end - start))
		print("xgboost.score", xgt.score(train_d, train_l))

		#https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_evals_result.py

		roc_auc, confusion, prediction = hyperparams.showAccuracy(xgt, 'XGBoost', test_d, test_l, kPlot=False)
		showFeatureImportance(xgt.feature_importances_, kHistogram=False, kSavePNG=outputdir+"XGBoostfeatures")

		#xgb.plot_importance(xgt)
		#plt.show()

		#DMselectFeatures(xgt, xgt.feature_importances_, train_d, test_d, train_l, test_l)
		#print("xgt.predict", predict)
		#print("test_l", test_l)

	# Assign higher cost for misclassification of abnormal heart sounds
	Cost = [[0, 20], [1, 0]]

	ada = AdaBoostClassifier(n_estimators=60, learning_rate=1, random_state=1234)				#(0,130): .815
	gbt = GradientBoostingClassifier(n_estimators=100, subsample=0.99, learning_rate=.1, random_state=1234)		#(0,130): .861															#
	rf  = RandomForestClassifier(n_estimators=100, random_state=1234)				#.846

	#https://stackoverflow.com/questions/32615429/k-fold-stratified-cross-validation-with-imbalanced-classes
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)		#0.829
	kf  = KFold(n_splits=5, shuffle=True, random_state=1234)				#.633
	if None:
		for train_set, leave_set in skf.split(allfiles, labels):
			print("train:", train_set, "leave_set:", leave_set)
			t_files  = heartSoundDb.getFiles(train_set)
			t_labels = heartSoundDb.getLabels(train_set)

	#2: use kfold cross validate helper
	cls_name = 'LightGBM'		#GradientBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'AdaBoost', 'BalancedRandomForest', 
	cls = models.getClassifer(cls_name)
	#print(models.classid(cls))

	scores = cross_val_score(
		cls, 
		feature_table, labels, 
		cv=skf, 
		n_jobs=1
	)
	end = time.time()
	print("cross_val_score(%s) %.2fsec" % (cls_name, (end - start)), "mean: %.3f" % scores.mean(), "max %.3f" % scores.max(), "min %.3f" % scores.min())

	#cls = imblearn.pipeline.make_pipeline(imblearn.under_sampling.OneSidedSelection, cls)
	cls.fit(train_d, train_l)

	print(cls)
	roc_auc, confusion, prediction = hyperparams.showAccuracy(cls, cls_name, test_d, test_l, kPlot=False)
	logFile.write("confusion: ")
	pp.pprint(confusion)

	if report_imbalanced:
		y_pred_bal = cls.predict(test_d)
		print(classification_report_imbalanced(test_l, y_pred_bal))

	# save model to file
	hyperparams.saveModel(cls, cls_name, modeldir)
	best_cls = cls
	
	#use GridSearchCV to optimize hyperparams for 'cls'
	if optimize:
		best_cls = hyperparams.optimizeParameters(cls, cls_name, train_d, train_l, test_d, test_l, params=None)
		# save model to file
		hyperparams.saveModel(best_cls, cls_name, modeldir, tag='_best')
		result_by_case = visualize_result(heartSoundDb, test_d_idx, test_l, prediction)
		print("visualize_result:", file=logFile)
		pp.pprint(result_by_case)

	#end train()

