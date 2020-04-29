import numpy as np
import matplotlib.pyplot as plt


def calculate_nfft(samplelen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    """
    nfft = 1
    while nfft < samplelen:
        nfft *= 2
    return nfft

def doFFT(wavsignal, kPad=True):
	len_data = len(wavsignal)
	if kPad:
		inputbuf = np.zeros(2**(int(np.ceil(np.log2(len_data)))))	#padded to power-of-2 of FFT:
		inputbuf[0:len_data] = wavsignal
	else:
		inputbuf = wavsignal
		
	fourier = np.fft.fft(inputbuf)
	return fourier
	
def plotFourierSignal(signal, rate, caption="Fourier Real", kSavePNG='', kComplex=False):
	plt.figure(caption)
	w = np.linspace(0, rate, len(signal))

	if kComplex == False:
		# First half is the real component, second half is imaginary
		fourier_to_plot = signal[0:len(signal)//2]
		w = w[0:len(fourier_to_plot)]

		plt.plot(w, fourier_to_plot)
	else:
		plt.plot(signal)
	plt.xlabel('frequency')
	plt.ylabel('amplitude')
	if kSavePNG:
		plt.savefig(kSavePNG)
	plt.show()
