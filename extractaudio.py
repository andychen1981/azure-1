#import skvideo
import argparse
from ffmpy import FFmpeg
import os, sys
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

import wavutil as wu

FFMPEGDIR='c:/Utils/ffmpeg/bin/'
#ffmpeg -i emotions.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 emotions.wav

DEMUX=False			#split video and audio into 2 .mp4 files
EXTRACTWAV=False	#extract audio stream into a .wav file
PLAYWAVE=False

# Command-line arguments to the system -- you can extend these if you want, but you shouldn't need to modify any of them
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/HD Training Data', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/HD Training Data', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


if __name__=='__main__':
	args = _parse_args()
	print(args)

	inputvid = 'test.mp4'
	basename = os.path.basename(inputvid)
	base, ext = os.path.splitext(basename)
	
	if DEMUX:
		ff = FFmpeg(
			inputs={inputvid: None},
			outputs={
				base + '-video.mp4': ['-map', '0:0', '-c:a', 'copy', '-f', 'mp4', '-y'],
				base + '-audio.mp4': ['-map', '0:1', '-c:a', 'copy', '-f', 'mp4', '-y']
			}
		)
		print(ff.cmd)
		ff.run()

	wavefile = base + '.wav'

	if EXTRACTWAV:
		ff = FFmpeg(
			inputs={inputvid: None},
			outputs={
				wavefile: ['-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y']
			}
		)
		print(ff.cmd)
		ff.run()

	if PLAYWAVE:
		wu.playWav(wavefile)

	wavarr = wu.loadWav(wavefile)	#(framerate, np)
	print(wavarr)

	fs = wavarr[0]
	signal = wavarr[1]
	Time=np.linspace(0, len(signal)/fs, num=len(signal))

	plt.figure(1)
	plt.title('Signal Wave...')
	plt.plot(Time, signal)
	plt.savefig("plot1.png")
	plt.show()
