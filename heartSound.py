import os, sys
from collections import defaultdict
import operator
import joblib
import pprint
import numpy as np
from scipy.stats import kurtosis
import csv
import parse
#from my python library
import pyutils.dirutils as dirutils
import pyutils.folderiter as folderiter
import pyutils.checksum as checksum

#project local includes
import wavutil as wu
import fft


kDatasetPath='data/test_data/'

#infer class label from pre|post in the filename
def extractClass(filename):
	basename = os.path.basename(filename)
	return 1 if 'pre' in basename.lower() else 0	#Note: update heartSound.label2str to match

def extractCaseNo(filename):
	dirname = os.path.dirname(filename)
	dirn = os.path.basename(dirname)
	caseno = int(dirn)
	return caseno

def extractFileName(filename):
	barename, ext = os.path.splitext(filename)
	#infer <caseno> + class label from pre|post in the filename
	digits = parse.parse('{caseno}pre{sample}', barename)
	if digits == None:
		digits = parse.parse('{caseno}post{sample}', barename)
	return digits

def error(errstr):
	print(errstr)	

def iterate_testfolder(
	datafolder, 
	layout='flat', 
	exclude={}, 
	case2audio=defaultdict(list),
	kFileList=False, 
	logging=False
):
	def onefile(ifile, context):
		#print("ifile '%s'" % ifile)
		basename = os.path.basename(ifile)
		dirname  = os.path.dirname(ifile)

		if dirname in exclude:
			print("excluding '%s'" % dirname)
			return
		dirn = os.path.basename(dirname)

		current_class = extractClass(ifile)

		if pass_n == 1:		#in pass 1?
			pdf[current_class] += 1

		if pass_n == 2:		#in pass 2?
			nonlocal count
			caseno = 0
			
			if layout=='byCaseno':
				caseno = extractCaseNo(ifile)
			if layout=='flat':
				#infer class label from pre|post in the filename
				digits = extractFileName(basename)

				if digits != None:
					caseno = int(digits['caseno'])
					sample = int(digits['sample'])		#TODO: handle '82post1_2.wav' => digits['sample'] == '1_2'
					print("Andy: case[%d], sample[%d], '%s':%d" % (caseno, sample, ifile, current_class)) ## Andy added
					if logging:
						print("case[%d], sample[%d], '%s':%d" % (caseno, sample, ifile, current_class))
			
			filename = dirn + '/' + basename

			if kFileList:
				filelist.append(filename)
			labels[count] = current_class
			count += 1

			if context != None:
				context[caseno].append(filename)
		return
	
	pdf = [0,0]		#count of negative and positive labels

	pass_n = 1
	for dataset in datafolder:
		print("datafolder '%s'" % dataset)
		folderiter.folder_iter(dataset, onefile, case2audio, file_filter=folderiter.wavfile_filter, logging=False)

	numinput = sum(pdf)
	filelist = []
	labels	 = np.ndarray((numinput,), dtype=np.int)

	pass_n = 2
	count  = 0		#'count' is used to index label[] for appending in the file iterator callback
	for dataset in datafolder:
		folderiter.folder_iter(dataset, onefile, case2audio, file_filter=folderiter.wavfile_filter, logging=False)

	if logging:
		pp = pprint.PrettyPrinter(indent=1, width=120)
		pp.pprint(case2audio)

	return filelist, labels, pdf

#descriptor for the metadata associated with each 'heartSound' dataset
class heartSoundDesc(object):
	def __init__(self, layout='flat', samplingrate=4*1000, duration=3.):
		self._samplingrate = samplingrate
		self._duration     = duration
		self._layout	   = layout		#'byCaseno': each folder contains the samples for 1 patient with case-no encoded
										#'flat': single folder with filename <caseno>pre|post<visit_location>
	@property
	def layout(self):
		return self._layout

	@property
	def samplingrate(self):
		return self._samplingrate
	
	@property
	def duration(self):
		return self._duration

class heartSoundFeatures(object):
	def __init__(self, desc):
		self._desc = desc
		self._features = {}

	@property
	def features(self):
		return self._features
	
	@property
	def desc(self):
		return self._desc

class heartSound(object):	
	def __init__(self, datasetroot=kDatasetPath, datasetname=['training'], csvname='reference.csv'):
		self._datasetroot = datasetroot
		self._csvname	  = csvname
		self._filelist    = []
		self._labels      = []
		self._cases		  = None
		self._desc		  = heartSoundDesc(layout='flat')
		self._pdf 		   = [0,0]	
		self._checksums   = None

		if type(datasetname) == str:
			datasetname = [datasetname]

		datasetnames = list(map(lambda f: datasetroot + f, datasetname))
		self._datasetnames = datasetnames

	@property
	def datasetroot(self):
		return self._datasetroot

	@property
	def datasetnames(self):
		return self._datasetnames
	
	@property
	def csvname(self):
		return self._csvname

	@property
	#list of class label (an array) for effciently indexing while iterating the filelist
	def labels(self):
		return self._labels
	
	@property
	#list of wave files (an array)
	def filelist(self):
		return self._filelist

	@property
	#1 case for each patient:
	def cases(self):
		return self._cases

	@property
	def desc(self):
		return self._desc

	@property
	def pdf(self, label):
		return self._pdf[label]

	def setpdf(self, counts):
		total = sum(counts)
		self._pdf = [0,0] if total == 0 else [float(i)/total for i in counts]

	@property
	def checksums(self):
		return self._checksums

	@staticmethod
	def label2str(label):
		labelstr = ['Normal', 'Abnormal']
		return labelstr[label]

	def numcases(self):
		return len(self.cases.keys())

	def numsamples(self):
		return len(self.filelist)

	def isempty(self):
		return self.numsamples() == 0

	def datasetpath(self):
		return self.datasetroot		#TODO: fixme

	def getFilePath(self, sample):
		filepath = self.datasetpath() + self.filelist[sample] 
		return filepath

	def getLabel(self, sample):
		return self.labels[sample]

	def getLabels(self, indices):
		output = []
		if self.isempty():
			return output
		if indices is None:
			output = self.labels
		else:
			output = np.take(self.labels, indices)
		return output

	def getFiles(self, indices):
		output = []
		if self.isempty():
			return output
		if indices is None:
			output = self.filelist	#return a handle to our internal list. This break encapsulation 
									#but is a compromise for large datasets
		else:
			for i in indices:	#TODO: use indices directly (np.take?)
				output.append(self._filelist[i])
		return output

	#our actual Builder:
	def loadmetadata(self, layout='flat', exclude={}, kFileList=True, logging=False):
		pp = pprint.PrettyPrinter(indent=1, width=120)

		#load metadata from 'csvname'
		case2audio = defaultdict(list)
		filelist = []
		labels	 = []
		count 	 = [0,0]
		#open 'reference.csv'
		csvfilename = self.datasetroot + self.csvname

		if not os.path.isfile(csvfilename):
			filelist, labels, count = iterate_testfolder(self.datasetnames, layout, exclude={'80'}, case2audio=case2audio, kFileList=kFileList)
			self.desc._layout = layout
			#print("labels %s" % labels)
		else:	
			print("loading dataset '%s'.." % csvfilename)
			with open(csvfilename, 'r') as f:
				reader = csv.reader(f)
				for row in reader:
					fname = row[0]
					label = row[1]
					current_class = extractClass(fname)		#1: post, 0: pre
					filelist.append(fname + '.wav')
					labels.append(label)
					#basename = os.path.basename(fname)
					count[current_class] += 1

					#infer class label from pre|post in the filename
					digits = extractFileName(fname)

					if digits != None:
						if logging:
							print("%s -> %s:%s" % (fname, digits['caseno'], digits['sample']))
						caseno = int(digits['caseno'])
						sample = int(digits['sample'])
						case2audio[caseno].append(fname + '.wav')
					else:
						error('no valid class label in filename')

			self.desc._layout = 'flat'

		self._filelist = filelist
		self._labels   = labels
		self._cases	   = case2audio
		self.setpdf(count)

		self.finalize(logging)

		return filelist, labels

	#generator for the entire dataset
	def __iter__(self):
		case2audio = self.cases
		dir = self.datasetpath()
		for k, v in case2audio.items():
			for f in v:
				yield dir + f

	def subsetsize(self, indices=None):	
		filelist = self.filelist

		if indices is None:
			return len(filelist)
		else:
			return len(indices)

	#generator for a subset specified by 'indices'
	def subset(self, indices=None):
		if self.isempty():
			return

		filelist = self.filelist
		dir = self.datasetpath()

		if indices is None:
			for i in range(0, len(filelist), 1):
				yield dir + filelist[i]
		else:
			for i in indices:
				yield dir + filelist[i]

	def dumpChecksums(self, filepath, checksums):
		#basename = os.path.basename(filepath)
		dirname = os.path.dirname(filepath)
		dirutils.mkdir(dirname)
		filenames = joblib.dump(checksums, filepath, compress=2)
		print("checksums cache '%s'" % filenames)

	def finalize(self, logging=False):
		dir = self.datasetpath()
		filelist = self.filelist
		labels 	 = self.labels
		checksums = []

		if self.numsamples() == 0:
			return

		filelist_sorted, labels_sorted = zip(*sorted(zip(filelist, labels),
  											key=operator.itemgetter(0), reverse=False))
		for idx, f in enumerate(filelist_sorted):
			current_class = extractClass(f)
			assert(current_class == labels_sorted[idx])
			crc = checksum.crc32sum(dir + f)
			checksums.append(crc)

		self._filelist = filelist_sorted
		self._labels   = labels_sorted
		self._checksums = checksums

		self.validate(logging)

	def validate(self, logging):
		count = 0
		failed = 0
		wavarr0 = None
		crc = None

		iter = (self)	#create a generator instance for ourselves

		for f in iter:
			wavarr = wu.loadWav(f)	#(framerate, data)
			if wavarr:
				if logging:
					print("'%s'" % (f))
				if wavarr0:
					if wavarr0[0] != wavarr[0]:
						print("framerate mismatch %d,%d" % (wavarr0[0], wavarr[0]))
				else:
					wavarr0 = wavarr
					self.desc._samplingrate = wavarr[0]
					self.desc._duration 	= wavarr[1].size

				count += 1
			else:
				failed += 1

		if logging:
			print("validate: count=%d, failed=%d" % (count, failed))

		u, indices, counts = np.unique(self.labels, return_index=True, return_counts=True)
		#print("%s: %d" % (heartSound.label2str(u[0]), counts[0]), "%s: %d" % (heartSound.label2str(u[1]), counts[1]))

		return
		
