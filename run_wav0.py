from flask import Flask, render_template, redirect, \
    url_for, request, session, flash

import scipy
import scipy.fftpack
from scipy.signal import argrelextrema
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import wavfile

# To read in wave file
from scipy.io import wavfile # get the api
from pylab import *
import wave

import sys
import os
from my_pulse import my_pulse
from my_fft import my_fft
import sqlite3

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from testModel import testModel, runClassifier

import shutil

import argparse
import hyperparams

import names
import psycopg2

app = Flask(__name__)

def run_wav0(wav0):
	conn = sqlite3.connect('flow-ez.db')
	cur = conn.cursor()
	
	print('\nProcessing Run Wave 0...\n')
	# wav = os.path.join('data/new_data/fft/' + string2 + '.png')
	# wav0 = os.listdir('data/new_data/')[0]
	wav = wav0.strip('data/new_data/')
	wav1 = wav.strip('.wav') + '.png'
	
	# wav = os.listdir('data/new_data/')[0]
	# wav = next(join('data/new_data/', f) for f in os.listdir('data/new_data/'))

	# print('Wav0: \n', wav0)
	# print('Wav: \n', wav)
	# print('Wav1: \n', wav1)
	word = wav.split('-')
	# print('Word: \n', word)
	mea_date = word[0]
	# print('Measure Date: \n', mea_date)
	disp_date = mea_date[0:4] + '-' + mea_date[4:6] + '-' + mea_date[6:8] + ' ' + mea_date[8:10] + ':' + mea_date[10:12] + ':' + mea_date[12:14]
	# print('Display Date: \n', disp_date)
	dev_id = word[1]
	# print('Dev ID: \n', dev_id)
	qr_code0 = word[3]
	qr_code = qr_code0.strip('.wav')
	# print('RQ Code: \n', qr_code)
	loc = word[2]
	# print('Location: \n', loc)
	
	print('\nProcessing Prob...')
	
	conf = testModel()
	# print('Conf-2: ',conf)
	# run_prob (dev_id)
	print('\nFinished Prob.')

	# path = '/Users/andy/Projects/flow-ez/data' 
	# print("Moving file:")  
	# print(os.listdir(path)) 
	# source = '/Users/andy/Projects/flow-ez/data/new_data'
	# destination = '/Users/andy/Projects/flow-ez/data/old_data'
	# dest = shutil.move(source, destination, copy_function=copy2) 
	
	# print("Finished Moving file:")  
	# print(os.listdir(path))  
	# print("Destination path:", dest)  

	print('\nProcessing Pulse...')
	plt.figure(dpi=2400)
	my_pulse (wav0)
	print('\nFinished Pulse.')

	print('\nProcessing FFT...')
	my_fft (wav0)
	print('\nFinished FFT.')
	
	if (conf > 50):
		res = 'Abnormal'
	else:
		res = 'Normal'

	first_name = names.get_first_name()
	last_name = names.get_last_name()
	
	shutil.copy('static/trend/20191101101100-003004802801-UP-00000001.png', 'static/trend/' + wav1)

	cur.execute("BEGIN TRANSACTION;")
    # cur.execute("Update data_table (mea_date,dev_id,qr_code,loc,pulse,fft,trend) VALUES (?,?,?,?,?,?,?)",(mea_date,dev_id,qr_code,loc,wav1,wav1,wav1) where (last_name='West'))
	# cur.execute("UPDATE data_table set mea_date=?,disp_date=?,dev_id=?,qr_code=?,loc=?,pulse=?,fft=?,trend=?,res=?,prob=? WHERE dev_id=?", (mea_date,disp_date,dev_id,qr_code,loc,wav1,wav1,wav1,res,conf,dev_id,) )
	cur.execute("INSERT INTO data_table (first_name,last_name,mea_date,disp_date,dev_id,qr_code,loc,pulse,fft,trend,res,prob) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (first_name,last_name,mea_date,disp_date,dev_id,qr_code,loc,wav1,wav1,wav1,res,conf) )
	cur.execute("COMMIT TRANSACTION;")
	cur.close
	conn.close()

	source = os.listdir("/Users/andy/Projects/flow-ez/data/new_data/")
	destination = "/Users/andy/Projects/flow-ez/data/old_data/"
	for files in source:
		if files.endswith('.wav'):
			shutil.move("/Users/andy/Projects/flow-ez/data/new_data/" + files,destination)
			# print('\nWave Files Moved.')

	# my_psp (wav0) ## Leadtek's data has "RuntimeWarning: divide by zero" issue