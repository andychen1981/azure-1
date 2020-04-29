import time
import argparse
import numpy as np
import pprint, pickle
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import imblearn

import heartSound, train
import extractFeatures
import hyperparams
import stats.roc

kTestCache=True
kDumpCache=False
kMTFeature=True


def data_processing_mean_using_cache(column):
	 return print(column)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='testModel.py')
	parser.add_argument('--i', type=str, default='all', help='WAV file')
	parser.add_argument('--dataset', type=list, default=['m_3m_wf_3s'], help='dataset root')	#, 'm_v1_wf_4k'
	args = parser.parse_args()

	pp = pprint.PrettyPrinter(indent=1, width=120)
	datadir = 'data/'
	outputdir = 'output/'
	modeldir = 'models/'
	kLogging = False

	datasetname = args.dataset
	wavfile = args.i

	heartSoundDb = heartSound.heartSound(
		datasetroot=datadir + 'test_data/', 
		datasetname=datasetname, 	#'HD Training Data', 'training_3s'
		csvname=''					#'reference.csv'
	)
	wavelist, alllabels = heartSoundDb.loadmetadata(layout='flat', exclude={'80'}, logging=kLogging)	#kLogging
	print('numsamples %d, num_patient %d' %(heartSoundDb.numsamples(), heartSoundDb.numcases()))

	#get a small subset for testing
	subset = np.arange(0, 130, 1)
	subset = None 		#whole dataset

	start = time.time()

	if kMTFeature:
		pprint.pprint(list(chunks(range(0, 130), 16)))

		feature_tableMT = train.processFeatures(heartSoundDb, subset, outputdir=outputdir, kMT=4, logging=False)

		end = time.time()
		print("processFeaturesMT %.2fsec" % (end - start))
		start = end

	feature_table = train.processFeatures(heartSoundDb, subset, outputdir=outputdir, kMT=1, logging=False)

	end = time.time()
	print("processFeatures %.2fsec" % (end - start))
	start = end

	if kMTFeature:
		print("feature_table(serial, MT)", extractFeatures.verifyFeatures(feature_table, feature_tableMT))

	files  = heartSoundDb.getFiles(subset)		#not needed for training (for user feedback)
	labels = heartSoundDb.getLabels(subset)

	train_d, test_d, train_l, test_l = train_test_split(feature_table, labels, test_size=.2, stratify=labels, random_state=42)

	cls_name = 'GradientBoost'	#'GradientBoost', 'XGBoost'
	best = '_best'				#'_best': load optimized version of the models or last run

	cls = hyperparams.loadModel(cls_name + best, modeldir)
	
	if cls != None:
		print("Loaded '%s'" % (cls_name+best))
		pp.pprint(cls)

		hyperparams.showAccuracy(cls, cls_name, test_d, test_l, kPlot=False)

		#test feature cache
		if kTestCache:
			if kDumpCache:
				ourcache = extractFeatures.featureCache(folder='features/', fname='featureCache.joblib')
				ourcache.setChecksums(heartSoundDb.checksums)
				ourcache.setFeatures(feature_table)
				numentries = ourcache.size
				ourcache.dump()

			ourcache2 = extractFeatures.featureCache(folder='features/', fname='featureCache.joblib')
			ourcache2.load()
			print("verifyChecksums", ourcache2.verifyChecksums(heartSoundDb.checksums))
			print("verifyFeatures", extractFeatures.verifyFeatures(ourcache2.features, feature_table))

			feature_table = ourcache2.features

			_, test_d2, _, test_l2 = train_test_split(feature_table, labels, test_size=.2, stratify=labels, random_state=42)

			hyperparams.showAccuracy(cls, cls_name, test_d2, test_l2, kPlot=False)



	#Parallel(n_jobs=2)(
		#delayed(data_processing_mean_using_cache)(col) for col in files)
	