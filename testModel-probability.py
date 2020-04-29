import timeimport argparseimport numpy as npimport pprint, pickleimport joblibfrom joblib import Parallel, delayedfrom sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_splitimport imblearnimport heartSound, trainimport extractFeaturesimport hyperparamsimport stats.rocimport datetimefrom flask import Flask, render_template, redirect, \    url_for, request, session, flash, gfrom functools import wrapsimport sqlite3# with sqlite3.connect("flow.db") as conn: conn = sqlite3.connect('flow.db')c = conn.cursor()conf = 50now = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")# c.execute("CREATE TABLE pat(title TEXT, details TEXT)")# c.execute("CREATE TABLE yours (hd TEXT, msg TEXT)")# c.execute('INSERT INTO pat (hd, msg) VALUES("Good", "I\'m good.")')# c.execute('INSERT INTO pat (hd, msg) VALUES("Well", "I\'m well.")')# c.execute("INSERT INTO yours (dev_id, bed_id, hd, msg, sig) VALUES ('1237', '12', 'DA', 'Abnormal', '135pre1.wav')")# c.execute("INSERT INTO test (hd, msg) VALUES('Good', 'Normal')")# c.execute("BEGIN TRANSACTION;")# c.execute("CREATE TABLE mine (hd TEXT, msg TEXT, dev_id INTEGER (16) UNIQUE NOT NULL, pat_id INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT, bed_id INTEGER (16), sig CHAR (32) UNIQUE, res CHAR (16));")# c.execute("INSERT INTO pat (hd, msg, dev_id, pat_id, bed_id, sig, res) VALUES ('DA', 'Abnormal', 15678, 111, 17, '222pre1.wav', NULL);")# c.execute("INSERT INTO pat (hd, msg, dev_id, pat_id, bed_id, sig, res) VALUES ('DA', 'Abnormal', 15679, 112, 18, '225pre1.wav', NULL);")# c.execute("COMMIT TRANSACTION;")def runClassifier(model, cls_name, test_d, kPlot=True, outputdir='output/'):	global conf	global probability	global prediction	conn = sqlite3.connect('flow.db')	c = conn.cursor()	predict = model.predict(test_d)	proba   = model.predict_proba(test_d)	probability = proba[:,1]		#we are interested in the class 1 (see extractClass)	print("Probability: ", probability)	print(type(probability))	conf = round(float(probability[0]*100),1)	print("Conf: ", conf)		prediction = [round(value).astype(int) for value in probability]	#threshold at 0.5	print("Prediction: ", prediction)	# c.execute("BEGIN TRANSACTION;")	# c.execute("INSERT OR REPLACE INTO meas (dev_id, date, status, sig, res, msg, conf) VALUES (?,?,?,?,?,?,?)",('aaa','9/25','','','','',probability))	# c.execute("COMMIT TRANSACTION;")	return predictionif __name__=='__main__':	parser = argparse.ArgumentParser(description='testModel.py')	parser.add_argument('--i', type=str, default='all', help='WAV file')	parser.add_argument('--testset', type=list, default=['newtestdata'], help='dataset root')	#, 'm_v1_wf_4k'	parser.add_argument('--model', type=str, default='GradientBoost', help='trained model')	args = parser.parse_args()	pp = pprint.PrettyPrinter(indent=1, width=120)	datadir = 'data/'	outputdir = 'output/'	modeldir = 'models/'	kLogging = False	datasetname = args.testset	wavfile = args.i	heartSoundDb = heartSound.heartSound(		datasetroot=datadir, 		datasetname=datasetname, 	#'HD Training Data', 'training_3s'		csvname=''					#'reference.csv'	)	wavelist, alllabels = heartSoundDb.loadmetadata(layout='flat', exclude={'80'}, logging=kLogging)	#kLogging	print('numsamples %d, num_patient %d' %(heartSoundDb.numsamples(), heartSoundDb.numcases()))	#get a small subset for testing	subset = None 		#whole dataset	start = time.time()	feature_table = train.processFeatures(heartSoundDb, subset, outputdir=outputdir, kMT=1, logging=False)	end = time.time()	print("processFeatures %.2fsec" % (end - start))	start = end	files  = heartSoundDb.getFiles(subset)		#not needed for training (for user feedback)	labels = heartSoundDb.getLabels(subset)	test_d = feature_table	cls_name = args.model	#cls_name = 'LightGBM'	#'GradientBoost', 'XGBoost', 'LightGBM'	#best = '_best'				#'_best': load optimized version of the models or last run	best = ''					#use unoptimized version of the model	cls = hyperparams.loadModel(cls_name + best, modeldir)		if cls != None:		print("Loaded '%s'" % (cls_name+best))		pp.pprint(cls)		prediction = runClassifier(cls, cls_name, test_d, kPlot=False)		print("Prediction:", prediction)		# print("Probability: ", probability)		for i, f in enumerate(files):			label = heartSound.heartSound.label2str(prediction[i])			print("[%d]: %s -> %s %s" % (i, f, label, round(float(probability[i]*100),1)))			print("Label: ", label)			print(type(label))			p_id = f[12:]			print("P_ID: ",p_id)			# print("Probability: ", proba[:,1])			# print(type(prediction[i]))			# c.execute("BEGIN TRANSACTION;")			# c.execute("INSERT OR REPLACE INTO meas (dev_id, date, status, sig, res, msg, conf) VALUES (?,?,?,?,?,?,?)",('aaa','09/23','ok','80pre1.wav',label,'','95%'))			# c.execute("COMMIT TRANSACTION;")						c.execute("BEGIN TRANSACTION;")			# c.execute("INSERT OR REPLACE INTO meas (pat_id, my_date, status, sig, res, msg, conf) VALUES (?,?,?,?,?,?,?)",('aaa',now,'ok','118pre1.wav',label,'',conf)) ## Working			c.execute("INSERT OR REPLACE INTO meas (pat_id, my_date, status, sig, res, conf) VALUES (?,?,?,?,?,?)",('aaa',now,'ok',p_id,label,conf)) ## Working			c.execute("COMMIT TRANSACTION;")		c.close		conn.close()				# return render_template('measure.html')  # render a template						# print(pat[0])		# result_by_case = train.visualize_result(heartSoundDb, range(0, len(labels)), labels, prediction)		# pp.pprint(result_by_case)	# cur = conn.execute('select * from pat')	# print(cur)	