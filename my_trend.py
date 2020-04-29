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
def my_trend(wav3):

	string1 = wav3.split('.')
	print('String1: ',string1)
	string2 = string1[0]
	print('String2: ',string2)
	string3 = string2.strip('data/new_data/')
	print('String3: ',string3)
	string4 = os.path.join('static/trend/' + string3 + '.png')
	print('String4: ',string4)

	# plt.show()
	
	fig = plt.savefig(string4, dpi=1200)
	plt.close(fig)