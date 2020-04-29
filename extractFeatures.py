import os, sys
import numpy as np
import joblib
from multiprocessing import cpu_count	#use psutil later
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import SelectFromModel

import wavutil as wu
import heartSound, mymfcc, fft
import pyutils.checksum as checksum
 
from python_speech_features import base, sigproc


class featureVec(object):
	def __init__(
		self,
		kNumFeatures,
		shapeId='1D'
	):
		self._feature = None
		self._numfeatures = kNumFeatures
		self._shapeId = shapeId

	@property
	def numfeatures(self):
		return self._numfeatures
	
	@property
	def shapeId(self):
		return self._shapeId
	
	@property
	def feature(self):
		return self._feature 	#ndarray

	@property
	def crc(self):
		return self._crc
	
class featureCache(object):
	def __init__(
		self,
		folder='features/',				#output folder
		fname='featureCache.joblib'		#filename for the cache itself
	):
		self._folder = folder
		self._fname  = fname
		self._checksums = None
		self._cache = dict()

	def isValid(checksums, params=None):
		return verifyChecksums(checksums)

	@property
	def size(self):
		return len(self.checksums)
	
	@property
	def folder(self):
		return self._folder
	
	@property
	def filename(self):
		return self._fname

	@property
	def checksums(self):
		return self._checksums

	def setChecksums(self, checksums):
		self._checksums = checksums

	@property
	def features(self):
		return self._features

	def setFeatures(self, nfeatures):
		self._features = nfeatures

	def checkSumsName(self):
		barename, ext = os.path.splitext(self.filename)
		return self.folder + barename + '-checksums.joblib'

	def dump(self):
		joblib.dump(self.checksums, self.checkSumsName())
		joblib.dump(self.features, self.folder+self.filename, compress=3)

	def load(self):
		self._checksums = joblib.load(self.checkSumsName())
		self._features = joblib.load(self.folder+self.filename)

	def verifyChecksums(self, newchecksums):
		if self.checksums.shape != newchecksums.shape:
			return False
		return np.array_equal(newchecksums, self.checksums)

	def finalize(self, mtresults, wavelist, datadir):
		self._cache = dict(mtresults)
		result = []
		for f in wavelist:
			result.append(self.lookup(datadir+f))
		return result

	@staticmethod
	def hash(filename):
		#return checksum.crc32str(filename)
		return filename

	def lookup(self, filename):
		id = featureCache.hash(filename)
		return self._cache[id]

	def insert(self, filename, feature):
		id = hash(filename)
		self._cache[id] = feature

#https://www.mathworks.com/matlabcentral/fileexchange/65286-heart-sound-classifier
def dominant_frequency_features(signal, rate, cutoff, nfft=4096, logging=False):
	f = rate/2*np.linspace(0, 1, nfft/2, dtype=np.float32)
	cutoffidx = np.searchsorted(f, cutoff)		#length(find(f <= cutoff))
	f = f[0:cutoffidx+1]
	frames = sigproc.framesig(f, signal.size, signal.size) #sig, frame_len, frame_step - (1, 12000)
	powersp = sigproc.powspec(frames, nfft)		#(1, 2049)

	#print(powersp[0])
	if logging:
		print("frames.shape %s, powersp.shape %s" % (str(frames.shape), str(powersp.shape)))
	maxidx = powersp[0].argmax()
	maxval = f[maxidx]
	if logging:
		print("maxidx %d maxval %f" % (maxidx, maxval))
	maxfreq = 0		#TODO:
   	#% Extract features from the power spectrum
#    [~, maxval, ~] = dominant_frequency_features(current_signal, fs, 256/*cutoff*/, 0);
	return maxfreq, maxval

#see 'extractFeaturesCodegen.m'
def extractMFCC(
	kNumFeatures,
	shape,				#only supports 1D features for now, will add 2D support next
	wavarr, 
	win_len, 
	win_overlap, 
	nfft, 
	cutoff=256,		#for dominant_frequency_features only 
	kDelta=False 	#frame-deltas (need more than 1 frame)
):
	rate   = wavarr[0]
	signal = wavarr[1]
	kNumScalars = 2

	#see 'extractFeaturesCodegen.m'
	mfcc_feat, _ = mymfcc.mfcc_features(wavarr, win_len=win_len, win_overlap=win_overlap, nfft=nfft, lowfreq=5, highfreq=1000)
	#print("  # mfcc frames %d, mfcc_feat[0].shape: %s" % (mfcc_feat.shape[0], str(mfcc_feat[0].shape)))
	nframes = mfcc_feat.shape[0]

	if shape=='1D':
		features = [None] * kNumScalars 	#np.zeros(shape=(kNumFeatures,), dtype=np.float)

	if shape=='2D':
		ndfeatures = np.zeros((nframes, kNumFeatures), dtype=np.float32)

	step_length = win_len - win_overlap
	offset = kNumScalars

	for f in range(nframes):	#this can be replaced by one np flattening
		mfcc = mfcc_feat[f]		#a frame

		current_start_sample = f * rate * step_length
		current_end_sample = current_start_sample + win_len * rate
		current_signal = signal[current_start_sample:current_end_sample]	#current window

		#[kurtosis, dominant_frequency_features]
		kurt = kurtosis(current_signal)
		#dom_nfft = 4096					#Matlab code is using nfft=4096
		#maxfreq, domfreq = dominant_frequency_features(current_signal[0:dom_nfft], rate, cutoff=cutoff, nfft=dom_nfft)

		if shape=='1D':		#only supports 1D features for now, will add 2D support next
			features[0] = kurt
			features[1] = 0 	#domfreq
			#features[2] = skew(current_signal)		#this led to a drop of > 4% accuracy
			features.extend(mfcc) 					#TODO: check for overflow here if we use ndarray for features
			offset += len(mfcc)

		if shape=='2D':
			fvec = [kurt, 0]
			fvec.extend(mfcc)
			ndfeatures[f] = fvec

	if kDelta and (nframes > 1):
		assert(False)		#TODO: fix this code
		d_mfcc_feat = base.delta(mfcc_feat, 2)		#compute delta features from a feature vector
		ndfeatures = np.append(features, d_mfcc_feat)

	if shape=='1D':		#only supports 1D features for now, will add 2D support next
		ndfeatures = np.zeros(kNumFeatures, dtype=np.float32)
		n = min(kNumFeatures, len(features))
		ndfeatures[0:n] = features[:n]

	return ndfeatures

def extractFeatures(
	f,
	kNumFeatures=15,
	shape='1D',		#only supports 1D features for now, will add 2D support next
	logging=False
):
	wavarr = wu.loadWav(f)	#(framerate, nf)
	rate   = wavarr[0]
	signal = wavarr[1]
	win_len = 3			#3secs wide window
	win_overlap = 0		#among of overlap between frames

	nfft = fft.calculate_nfft(signal.size)		#FFT size as the padded next power-of-two
	if logging:
		print("  signal.size %d, rate: %d, nfft %d" % (signal.size, rate, nfft))

	features = extractMFCC(
		kNumFeatures,
		shape,
		wavarr, 
		win_len, 
		win_overlap, 
		nfft=nfft, 
		cutoff=256, 
		kDelta=False
	)
	#fbank_feat = logfbank(signal, rate)		#compute log Mel-filterbank energy features from an audio signal

	return features

def work(f, datadir, featurecache):
	filename = datadir + f
	#nonlocal datafolder
	#print("Function receives the arguments as a list:", arg)
	features = extractFeatures(filename)
 	#time.sleep(1)    
	# ... and prints a string containing the inputs:
	#print("%s" % (f))
	return (filename, features)

def extractFeaturesMT(wavelist, datadir, kMT=4):
	featurecache = featureCache(folder='features/', fname='featureCache.joblib')

	if kMT > cpu_count():
		print("WARNING: using more threads %d than #cores %d" % (kMT, cpu_count()))
	datafolder = datadir
	results = joblib.Parallel(n_jobs=kMT, verbose=1, backend="loky")(
		joblib.delayed(work)(f, datadir, featurecache) for f in wavelist)
	print("numEntries %d" % (len(results)))
	featureVecs = featurecache.finalize(results, wavelist, datadir)
	print("featurecache.keys %d" % len(featurecache._cache.keys()))
	print("numEntries %d" % (len(featureVecs)))
	return featureVecs



