## Loading WAV Files and Showing Frequency Response
## Posted on August 1, 2016 by Rob Elder

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq

import scipy
import scipy.fftpack
from scipy.signal import argrelextrema
import numpy as np

import wave

# To read in wave file
from pylab import *

# import wave
import sys
import os
import matplotlib as mpl

## Plot Pulse
def my_fft(wav2):

	mpl.rcParams['agg.path.chunksize'] = 10000
	rate2, data2 = wavfile.read(wav2)
	print('Rate: ', rate2)
	# print('Data: ', data2)

	string1 = wav2.split('.')
	# print('String1: ',string1)
	string2 = string1[0]
	# print('String2: ',string2)
	string3 = string2.strip('data/new_data/')
	# print('String3: ',string3)
	string4 = os.path.join('static/fft/' + string3 + '.png')
	# print('String4: ',string4)

	samples = data2.shape[0]
	datafft = fft(data2)
	#Get the absolute value of real and complex component:
	fftabs = abs(datafft)
	freqs = fftfreq(samples,1/rate2)
	plt.xlim( [10, rate2/2] )
	plt.xscale( 'log' )
	plt.grid( True )
	plt.xlabel( 'Frequency (Hz)' )
	plt.ylabel( 'Amplitude' )
	plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
	# plt.show()
	fig = plt.savefig(string4, dpi=1200)
	plt.close(fig)
