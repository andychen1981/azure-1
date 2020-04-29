import wave
import playsound
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


def playWavFile(wavefile, kCHUNK=1024, kLog=True):
	playsound.playsound(wavefile)

#import pyaudio
def playWavFilePyAudio(wavefile, kCHUNK=1024, kLog=True):
	try:
		wf = wave.open(wavefile, 'rb')
	except:
		print("Fail to play wav file '%s" % wavefile)
		return

	nchans 	  = wf.getnchannels()
	framerate = wf.getframerate()	#samples/sec
	numframes = wf.getnframes()

	if kLog:
		print("nchans", nchans)
		print("framerate", framerate)
		print("numframes", numframes)

	# instantiate PyAudio (1)
	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	                channels=wf.getnchannels(),
	                rate=wf.getframerate(),
	                output=True)

	# read data
	data = wf.readframes(kCHUNK)

	# play stream (3)
	while len(data) > 0:
	    stream.write(data)
	    data = wf.readframes(kCHUNK)

	# stop stream (4)
	stream.stop_stream()
	stream.close()

	# close PyAudio (5)
	p.terminate()

def plotSpectrogram(wavdata, kPlot=True):
	# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.spectrogram.html
	# http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html
	sample_rate = wavdata[0]
	samples = wavdata[1]
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)	

	if kPlot:
		plt.title('Spectrogram')
		plt.pcolormesh(times, frequencies, spectrogram)
		plt.imshow(spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()

def testspectro():
	time1 = np.arange(0,  5, 0.0001)
	time  = np.arange(0, 15, 0.0001)
	data1 = np.sin(2*np.pi*300*time1)
	data2 = np.sin(2*np.pi*600*time1)
	data3 = np.sin(2*np.pi*900*time1)
	data = np.append(data1, data2 )
	data = np.append(data, data3)
	print(len(time))
	print(len(data))

	NFFT = 200     # the length of the windowing segments
	Fs = 500  # the sampling rate

	# plot signal and spectrogram

	ax1 = plt.subplot(211)
	plt.plot(time, data)   # for this one has to either undersample or zoom in 
	plt.xlim([0, 15])
	plt.subplot(212 )  # don't share the axis
	Pxx, freqs, bins, im = plt.specgram(data, NFFT=NFFT, Fs=Fs, noverlap=100, cmap=plt.cm.gist_heat)
	plt.show() 

def loadWav(wavefile, kLog=False):
	arr = None
	try:
		arr = wavfile.read(wavefile)		#(framerate, data)
		if kLog:
			print("framerate", arr[0])
	except:
		print("Fail to load wav file '%s'" % wavefile)

	return arr		#(framerate, data)
