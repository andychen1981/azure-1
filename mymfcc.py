import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import base, sigproc		#mfcc, delta, logfbank
import fft

def mfcc_features(
	wavarr, 
	win_len=5, 		# window length for feature extraction in secs - run_orig.m
	win_overlap=0, 	# specify the overlap between adjacent windows for feature extraction in percentage - run_orig.m
	nfft=0,
	lowfreq=5, 
	highfreq=1000,
	kDelta=False, 
	logging=False
):
	# rate, aud_data = scipy.io.wavfile.read(file)
	rate = wavarr[0]
	signal = wavarr[1]
	d_mfcc_feat = None

	if nfft == 0:
		nfft = fft.calculate_nfft(signal.size)		#FFT size as the padded next power-of-two

	mfcc_feat = base.mfcc(signal, rate,
		winlen=win_len,						#window_length*1000 in extractFeatures.m
		winstep=win_len-win_overlap,		#10ms shift; Ts = 10 in extractFeatures.m
		numcep=13,			 				#C=12; in extractFeatures.m
		nfilt=20, 							#M=20; in extractFeatures.m
		nfft=nfft,							#pad to next power-of-2
		lowfreq=5, highfreq=1000, 			#LF=5; HF=1000; in extractFeatures.m
		preemph=0.97, ceplifter=22,			#alpha=0.97; L=22; in extractFeatures.m
		winfunc=np.hamming,					#@hamming
		appendEnergy=False					# replace first cepstral coefficient with log of frame energy
	)
	if kDelta:
		d_mfcc_feat = base.delta(mfcc_feat, 2)		#compute delta features from a feature vector
	#fbank_feat = sigproc.logfbank(signal, rate)	#compute log Mel-filterbank energy features from an audio signal

	return mfcc_feat, d_mfcc_feat
