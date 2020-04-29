import time, os
import copy
import pprint, pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, confusion_matrix, f1_score, make_scorer
from sklearn.feature_selection import SelectFromModel

from pyutils import dirutils, dictutils
from stats import roc

#https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

DefParams = {'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bytree': 1,
 'cv': 5,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 3,
 'min_child_weight': 1,
 'missing': None,
 'n_estimators': 100,
 'nthread': 1,
 'nthreads': 1,
 'objective': 'binary:logistic',
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 20,
 'seed': 27,
 'silent': True,
 'subsample': 1,
 'verbosity': 0}


def saveModel(cls, cls_name, outputdir, tag=''):
	dirutils.mkdir(outputdir)
	pickle.dump(cls, open(outputdir+cls_name+tag+".pickle.dat", "wb"))

def loadModel(cls_name, outputdir, tag=''):
	cls = None
	modelname = outputdir+cls_name+tag+".pickle.dat"
	with open(modelname, "rb") as f:
		cls = pickle.load(f)
	return cls

def showAccuracy(model, cls_name, test_d, test_l, kPlot=True, outputdir='output/'):
	predict = model.predict(test_d)
	proba   = model.predict_proba(test_d)
	probability = proba[:,1]		#we are interested in the class 1 (see extractClass)
	prediction = [round(value).astype(int) for value in probability]	#threshold at 0.5
	accuracy = accuracy_score(test_l, prediction)
	balanced = balanced_accuracy_score(test_l, prediction)

	u, indices, counts = np.unique(test_l, return_index=True, return_counts=True)

	#https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
	confusion= confusion_matrix(test_l, prediction)
	tn, fp, fn, tp = confusion.ravel()

	#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
	#roc_auc = roc.roc4classes(test_l, prediction)
	roc_auc = roc.plotROC(test_l, probability, kPlot, kSavePNG=outputdir+cls_name+'ROC.png')

	print("Accuracy(%s): %.2f%%, roc_auc %.2f, test[%s], confusion:" % (cls_name, accuracy * 100.0, roc_auc*100, counts))
	print(confusion)
	plotPrecisionRecall = roc.plotPrecisionRecall(test_l, probability, kPlot=kPlot, kSavePNG=outputdir+cls_name+'PR_curve.png')

	return roc_auc, confusion, prediction

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

def cutoff_predict(clf, X, cutoff):
	prob = clf.predict_proba(X)[:1]
	return (prob > cutoff).astype(int)	#0|1, lookup 'cutoff' in dynamic scope

def custom_f1(y, y_pred, cutoff):
	ypred2 = (y_pred > cutoff).astype(int)	#0|1, lookup 'cutoff' in dynamic scope
	#confusion= confusion_matrix(y, ypred2)
	#print(confusion)
	return f1_score(y, ypred2)

def custom_auc(y, y_pred, cutoff):
	ypred2 = (y_pred > cutoff).astype(int)	#0|1, lookup 'cutoff' in dynamic scope
	fpr, tpr, thresholds = roc_curve(y_test, ypred2)
	roc_auc = auc(fpr, tpr)
	return roc_auc

def optimize_cutoff(cls, train_d, train_l, score=custom_f1, logging=True):
	if logging:
		print("Optimize cutoff..")
	scores = []

	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
	f1_scores = []
	best_f1_score = 0.
	best_cls = None
	best_cutoff = .5

	for cutoff in np.arange(0.1, 0.9, 0.1):
		thres = cutoff
		#clf = copy.deepcopy(cls)	#might not need to do deep copy
		clf = cls
		our_f1 = make_scorer(custom_f1, needs_proba=True, cutoff=thres)
		validated = cross_val_score(clf, train_d, train_l, cv=skf, scoring=our_f1)
		scores.append(validated)
		avg_val = np.average(validated)
		#TODO: argmax(validated) to select best_cls
		print("cross_val_score[cutoff=%f] %s, avg %f" % (cutoff, validated, avg_val))
		f1_scores.append(avg_val)
		if avg_val > best_f1_score:
			best_f1_score = avg_val
			best_cutoff = cutoff
	return best_cutoff

def optimizeParameters(
	cls, 
	cls_name,
	train_d,
	train_l,
	test_d,
	test_l,
	params,
	kOptCutoff=False,
	logging=False
):
	pp = pprint.PrettyPrinter(indent=1, width=120)
	no_learnrate = {'RandomForest', 'BalancedRandomForest'}

	#1: scoring methods
	scores = {'AUC': 'roc_auc'}		#['precision', 'recall']

	start = time.time()
	best_cutoff = .5

	if not (cls_name in no_learnrate):
		srchgrid = dict(
			learning_rate=[.02, .04, .06, .08, .1, .15, .2, .5, .9],
			#class_weight=[1, 2, 4, 8, 16, 20],		#LightGBM
			scale_pos_weight=[1, 2, 4, 8, 16, 20],	#XGBoost
			n_estimators=[20, 40, 60, 80, 100],
			#min_child_samples=[2, 4, 6, 8, 10],
			#min_samples_leaf=[1, 2, 3, 4, 5],
			#num_leaves=[5, 10, 15, 20, 25, 31]
		)
		#per-classifier hacks
		if cls_name == 'GradientBoost':
			srchgrid['min_samples_leaf']=[1, 2, 3, 4, 5]
		if cls_name == 'LightGBM':
			srchgrid['min_child_samples']=[2, 4, 6, 8, 10]
			srchgrid.pop('class_weight', None)		#this cause a numerical error

		#remove parameters that are not in the classifer
		valid_params = cls.get_params()
		srchgrid, removed = dictutils.dict_sub(srchgrid, valid_params)
		print("GridSearchCV.. '%s' ignored" % removed)

		#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9876)
		best_score = 0
		cv_result = None

		clf = GridSearchCV(
			estimator=cls, 
			param_grid=srchgrid,
			scoring=scores, 
			iid=False, 
			cv=5,		#skf
			refit='AUC',
			n_jobs=-1
		)
		clf.fit(train_d, train_l)

		if clf.best_score_ > best_score:
			best_cls = clf.best_estimator_
			best_params = clf.best_params_
			best_score = clf.best_score_
			cv_result = clf.cv_results_
		#print("  best_estimator for '%s' from GridSearchCV(): %s, %f" % (score, best_params, best_score))
		print("  best_params from GridSearchCV(): %s, %f" % (best_params, best_score))
		#print(cv_result)
	else:
		pass

	if kOptCutoff:
		best_cutoff = optimize_cutoff(cls, train_d, train_l, score=custom_auc, logging=True)
		print("  optimize_cutoff(): '%f'" % best_cutoff)

	end = time.time()

	best_score = clf.score(train_d, train_l)
	print("  optimizeParameters(%s) %.2fsec, best_score %.3f" % (cls_name, end - start, best_score))

	roc_auc, confusion, prediction = showAccuracy(best_cls, cls_name + "_best", test_d, test_l, kPlot=False)

	return best_cls

