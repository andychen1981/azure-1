import scipy
import scipy.fftpack
from scipy.signal import argrelextrema
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave

# To read in wave file
from pylab import *

# import wave
import sys
import os
import matplotlib as mpl

## Plot Pulse
def my_pulse(wav1):

	mpl.rcParams['agg.path.chunksize'] = 10000 ## Fix "OverflowError: Exceeded cell block limit (set 'agg.path.chunksize' rcparam)"

	# print('\nWav1: ', wav1)
	rate1, data1 = wavfile.read(wav1)

	print('Rate: ', rate1)
	# print('Data: ', data1)

	string1 = wav1.split('.')
	# print('String1: ',string1)
	string2 = string1[0]
	# print('String2: ',string2)
	string3 = string2.strip('data/new_data/')
	# print('String3: ',string3)
	# Creating PNG file
	string4 = os.path.join('static/pulse/' + string3 + '.png') # Local file directory
	# print('String4: ',string4)
	
	plt.plot(data1)
	plt.xlabel('Sample')
	plt.ylabel('Amplitude')
	fig = plt.savefig(string4, dpi=2400)
	plt.close(fig)
	# plt.show()
	# return none

