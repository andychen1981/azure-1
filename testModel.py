import timeimport argparseimport numpy as npimport pprint, pickleimport joblibfrom joblib import Parallel, delayedfrom sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_splitimport imblearnimport heartSound, trainimport extractFeaturesimport hyperparamsimport stats.roc# From Michael Hermanfrom flask import Flask, render_template, redirect, url_for, request, session, flash, gfrom functools import wrapsimport sqlite3from flask import Flask, render_template, redirect, \    url_for, request, session, flashimport sysimport osfrom my_pulse import my_pulsefrom my_fft import my_fftfrom pydrive.auth import GoogleAuthfrom pydrive.drive import GoogleDrive# from testModel import runClassifier# from testModel import run_probimport shutilimport timefrom watchdog.observers import Observerfrom watchdog.events import FileSystemEventHandler# from run_wav import run_wavprint('\n')# create the application objectapp = Flask(__name__)conf = 50 ## How global var is defiend!!global labelglobal dev_iddef runClassifier(model, cls_name, test_d, kPlot=True):# def runClassifier(model, cls_name, test_d, kPlot=True, outputdir='../flow-ez/output/'):		global conf	predict = model.predict(test_d)	proba   = model.predict_proba(test_d)	probability = proba[:,1]		#we are interested in the class 1 (see extractClass)	## Print confidence level	print("Probability: ", probability)	# print(type(probability))	# conf = round(float(probability[0]*100),1)	conf = int(round(float(probability[0]*100),0))	# for i in probability:		# conf = round(float(probability[i]*100),1)	# print("Conf: ", conf)		prediction = [round(value).astype(int) for value in probability]	#threshold at 0.5	conn = sqlite3.connect('flow-ez.db')	c = conn.cursor()	# global dev_id	# print('Dev_ID: ', dev_id)	if (conf > 0.5):		res = 'Abnormal'	else:		res = 'Normal'	# print('Conf-1: Res: ',conf, res)	# print("Label: ",label)	# sql_edit_insert("UPDATE data_table set last_name=?, msg=? WHERE first_name=? and msg=?", (last_name,msg,first_name,msg))	c.execute("BEGIN TRANSACTION;")	# c.execute("UPDATE data_table set res=?, prob=? WHERE dev_id=?", (res, conf, dev_id))	c.execute("COMMIT TRANSACTION;")	c.close	conn.close()	return predictiondef testModel():	# print('testModel-1: ')	# if __name__=='__main__':		parser = argparse.ArgumentParser(description='testModel.py')	parser.add_argument('--i', type=str, default='all', help='WAV file')		# parser.add_argument('--testset', type=list, default=['newtestdata'], help='dataset root')	#, 'm_v1_wf_4k'	parser.add_argument('--testset', type=list, default=['new_data'], help='dataset root')	#, 'm_v1_wf_4k'	parser.add_argument('--model', type=str, default='GradientBoost', help='trained model')	args = parser.parse_args()	pp = pprint.PrettyPrinter(indent=1, width=120)	datadir = 'data/'	outputdir = 'output/'	modeldir = 'models/'	kLogging = False	datasetname = args.testset	wavfile = args.i	heartSoundDb = heartSound.heartSound(		datasetroot=datadir, 		datasetname=datasetname, 	#'HD Training Data', 'training_3s'		csvname=''					#'reference.csv'	)	wavelist, alllabels = heartSoundDb.loadmetadata(layout='flat', exclude={'80'}, logging=kLogging)	#kLogging	# print('numsamples %d, num_patient %d' %(heartSoundDb.numsamples(), heartSoundDb.numcases()))	#get a small subset for testing	subset = None 		#whole dataset	start = time.time()	feature_table = train.processFeatures(heartSoundDb, subset, outputdir=outputdir, kMT=1, logging=False)	end = time.time()	# print("processFeatures %.2fsec" % (end - start))	start = end	files  = heartSoundDb.getFiles(subset)		#not needed for training (for user feedback)	labels = heartSoundDb.getLabels(subset)	test_d = feature_table	cls_name = args.model	#cls_name = 'LightGBM'	#'GradientBoost', 'XGBoost', 'LightGBM'	#best = '_best'				#'_best': load optimized version of the models or last run	best = ''					#use unoptimized version of the model	cls = hyperparams.loadModel(cls_name + best, modeldir)	# global label	if cls != None:		# print("Loaded '%s'" % (cls_name+best))		# pp.pprint(cls)		prediction = runClassifier(cls, cls_name, test_d, kPlot=False)		# print("Prediction:", prediction[0])		# print('cls: %s cls_name: %s test_d: %s', cls,cls_name,test_d)		for i, f in enumerate(files): 			# global label			label = heartSound.heartSound.label2str(prediction[i])				# print("[%d]: %s -> %s" % (i, f, label))		result_by_case = train.visualize_result(heartSoundDb, range(0, len(labels)), labels, prediction)		pp.pprint(result_by_case)		return(conf)		if __name__=='__main__':		app.run(debug=False)