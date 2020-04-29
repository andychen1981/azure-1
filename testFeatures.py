import argparse
import os, sys
import time
import joblib
import multiprocessing as mp
import pprint

import numpy as np
import matplotlib.pyplot as plt

#project local includes
import pyutils.mt as mt
import wavutil as wu
import mymfcc, fft
import heartSound
from extractFeatures import extractFeatures
import extractFeatures as eF

PLAYWAVE=False
kLogging=False
kMP=False
kJoblib=True


# define a persistent thread processing function
# this is not really used but is an example of what a threadproc should look like
def threadprocPT(length, i, q, outputQ):
	mt.info("PT %d" % i)
	while not q.empty():
	    item = q.get()
	    #do_work(item)
	    #print("thread(%d): " % i)
	    print(item)

	    #q.task_done()
	    outputQ.put(item)
	print("exiting threadprocPT %d" % i)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='testFeatures.py')
	args = parser.parse_args()
	#print(args)
	pp = pprint.PrettyPrinter(indent=1, width=120)

	#TODO: get these from argparse
	datadir = 'data/'
	outputdir = 'output/'

	#wavefile = base + '.wav'
	#wavefile = datadir + 'pre1.wav'		#'post1.wav'
	# wavefile = 'data/test_data/training_3s/100post1_1.wav'
   	# wavefile = 'data/test_data/training_3s/114post2_2.wav'
   	wavefile = 'data/test_data/m_3m/80post1.wav'
   	wavefile = 'data/test_data/m_3m/80post2.wav'

	if PLAYWAVE:
		wu.playWavFile(wavefile, kLog=True)

	if kLogging:
		wavarr = wu.loadWav(wavefile)	#(framerate, np)
		print(wavarr)

	features = extractFeatures(wavefile)

	if kLogging:
		pp.pprint(features)
	else:
		if features != None:
			print("features.size %d" % (len(features)))
			pp.pprint(features)
		else:
			print("Failed to load '%s'" % wavefile)

	kNumThreads = 2
	datasetroot = datadir + 'test_data/'
	datasetname = 'm_3m_wf_3s/'

	heartSoundDb = heartSound.heartSound(
		datasetroot=datasetroot, 
		datasetname=datasetname, 	#'HD Training Data', 'training_3s'
		csvname=''					#'reference.csv'
	)
	wavelist, alllabels = heartSoundDb.loadmetadata(layout='flat', exclude={'80'}, logging=kLogging)	#kLogging
	print('numsamples %d, num_patient %d' %(heartSoundDb.numsamples(), heartSoundDb.numcases()))

	if kMP:
		worktodo = dict()
		queue = mp.Queue(maxsize=len(worktodo))

		print('# of workitems', len(wavelist))
		for i, key in enumerate(wavelist):
			queue.put(key)

		#workdone = mt.processbatches(queue, worktodo, threadproc, kNumThreads, False)
		start = time.time()

		workdone = mt.processbatchesPT(queue, threadprocPT, kNumThreads, False)

		end = time.time()
		print("processFeatures %.2fsec" % (end - start))
		print(workdone)

	if kJoblib:
		results1 = eF.extractFeaturesMT(wavelist, datasetroot, kMT=1)
		results2 = eF.extractFeaturesMT(wavelist, datasetroot, kMT=4)
		print("serial and MT are %s" % np.array_equal(results1, results2))
		#print(results1)
