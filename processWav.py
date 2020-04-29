import argparse
import os, sys
from collections import defaultdict
import pprint
import joblib

import wave
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

#from python_speech_features import mfcc, delta, logfbank
import pyutils.folderiter as folderiter

#project local includes
import wavutil as wu
import fft
import heartSound, train
#from extractFeatures import extractFeatures


kHD_TRAINING_DATA=False
PLAYWAVE=False
kSAVEPLOT=True
kMEL=True
kLogging=False


def buildWavData(inputfolder, wavfilter=folderiter.wavfile_filter, logging=True):
	case2audio = defaultdict(list)

	def onefile(ifile, context):
		current_class = heartSound.extractClass(ifile)

		basename = os.path.basename(ifile)
		caseno = heartSound.extractCaseNo(ifile)

		print("Andy2: case[%d] '%s %s':%d" % (caseno, basename, ifile, current_class))
		if logging:
			print("case[%d] '%s':%d" % (caseno, ifile, current_class))
	
		if context != None:
			context[caseno].append(basename)
		return
	
	folderiter.folder_iter(inputfolder, onefile, case2audio, file_filter=wavfilter, logging=False)
	return case2audio


# Command-line arguments to the system -- you can extend these if you want, but you shouldn't need to modify any of them
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='GradientBoost', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--model_path', type=str, default='models/', help='folder for trained models')

    # parser.add_argument('--dataset', type=list, default='[m_3m/,m_v1_4k/,m_v2_4k/]', help='dataset root')	#, 'm_v1_wf_4k'

    parser.add_argument('--dataset', type=list, default='[m_3m_3s/,m_v1_4k_3s/,m_v2_4k_3s/]', help='dataset root')	#, 'm_v1_wf_4k'
   	# parser.add_argument('--dataset', type=list, default='[m_3m/,m_v1_4k/,m_v2_4k/]', help='dataset root')	#, 'm_v1_wf_4k'
    # parser.add_argument('--dataset', type=list, default='[m_v1_4k/,m_v2_4k/]', help='dataset root')	#, 'm_v1_wf_4k'
    # parser.add_argument('--dataset', type=list, default='[m_3m/]', help='dataset root')	#, 'm_v1_wf_4k'

    parser.add_argument('--output_path', type=str, default='output/', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    parser.add_argument('-threads', type=int, default=1, help='number of threads to use for MT')
    args = parser.parse_args()
    return args

if __name__=='__main__':
	args = _parse_args()
	print(args)
	pp = pprint.PrettyPrinter(indent=1, width=120)

	#sns.set()

	#TODO: get these from argparse
	datadir = '../dialysis/data/' + 'test_data/'
	outputdir = 'output/'
	modeldir = 'models/'

	modelid = args.model
	datasetname = args.dataset
	#datasetname = ['m_3m_wf_3s', 'm_v2_wf_4k']
	nthreads = args.threads

	Db = heartSound.heartSound(
		datasetroot=datadir, 
		datasetname=datasetname, 	#'HD Training Data', 'training_3s'
		csvname=''					#'reference.csv'
	)
	wavelist, alllabels = Db.loadmetadata(layout='flat', exclude={'80'}, logging=kLogging)	#kLogging
	print('numsamples %d, num_patient %d' %(Db.numsamples(), Db.numcases()))
	#Db.dumpChecksums("features/checksums.joblib", Db.checksums)

	if kLogging:
		print(Db.cases)

	#get a small subset for testing
	# subset = np.arange(0, 130, 1)
	# subset = None 		#whole dataset
	subset_num = 180
	subset = np.arange(0, subset_num, 1)
	#subset = None 		#whole dataset
	print('Subset Numer: ',subset_num)
	
	train.train(
		Db, 
		cls_name='GradientBoost', 
		subset=subset, 
		optimize=True, 
		outputdir=outputdir, 
		modeldir=modeldir,
		kMT=nthreads, 
		logging=kLogging
	)

	#end of our real code.....
	
	#wavefile = base + '.wav'
	# wavefile = datadir + 'pre1.wav'		#'post1.wav'

	if wavelist:
		wavefile = Db.getFilePath(0)
		print("wavefile '%s'" % wavefile)

	if PLAYWAVE:
		wu.playWavFile(wavefile, kLog=True)

	wavarr = wu.loadWav(wavefile)	#(framerate, np)
	if kLogging:
		print(wavarr)

	if kHD_TRAINING_DATA:
		wavedir = 'HD Training Data/'
		case2audio = buildWavData(wavedir, logging=True)
		pp.pprint(case2audio)

		if kLogging:
			pp.pprint(case2audio)
		#quit()

		signal = wavarr[1]
		# rate, aud_data = scipy.io.wavfile.read(file)
		rate = wavarr[0]

		wu.plotSpectrogram(wavarr)
		
		#MFCC
		if kMEL:
			# Window length for feature extraction in seconds - run_orig.m
			# win_len = 5
			win_len = 6
			# Specify the overlap between adjacent windows for feature extraction in percentage - run_orig.m
			# win_overlap = 0
			win_overlap = 1

			mfcc_feat, d_mfcc_feat, fbank_feat = mfcc.mfcc_features(wnarr, win_len, win_overlap)
			#d_mfcc_feat = delta(mfcc_feat, 2)		#compute delta features from a feature vector
			#fbank_feat = logfbank(signal, rate)		#compute log Mel-filterbank energy features from an audio signal

		aud_data = signal
		ii = np.arange(0, len(aud_data))
		t = ii / rate
		aud_data = np.zeros(len(t))
		for w in [1000, 5000, 10000, 15000]:
		    aud_data += np.cos(2 * np.pi * w * t)

		#padded to power-of-2 of FFT:
		fourier = fft.doFFT(aud_data)

		#wu.testspectro()
		fft.plotFourierSignal(fourier, rate, caption="Fourier Real")

		Time=np.linspace(0, len(signal)/rate, num=len(signal))
		plt.figure(1)
		plt.title('Signal Wave...')
		plt.plot(Time, signal)
		if kSAVEPLOT:
			plt.savefig(outputdir + "plot1.png")
		plt.show()

		#kHD_TRAINING_DATA

