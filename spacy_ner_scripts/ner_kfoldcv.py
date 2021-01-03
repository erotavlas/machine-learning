#!/usr/bin/env python
# coding: utf8
"""
Performs a K-Fold Cross Validation on spaCy's named entity recognizer, and builds a final model with a test set

Starting off with an existing model or a blank model.

Will train nproc number of folds simultaneously
Use argument -gpu y to enable gpu processing if available

NOTE: Requires train.py to be present in the same folder as this script 

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.8

ssalpietro - July 18, 2019
"""
from __future__ import unicode_literals, print_function
from multiprocessing import Pool
import plac
import random
from pathlib import Path
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from os import listdir
from os.path import isfile, join
import os
import sys
from pathlib import Path

import train #place the train.py file in the same folder you are executing from

@plac.annotations(
	gpu=("Use gpu y = yes to enable gpu", "option", "gpu", str),
	nproc=("Number of simultaneous worker processes to run", "option", "nproc", int),
    	model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
	vocab=("Vocab name. This is the word vectos model", "option", "v", str),
	input_dir=("Required: the location where all the train test sets are located", "option", "i", Path),
    	output_dir=("Requires: where to write the model folders", "option", "o", Path),
    	n_iter=("Number of training iterations", "option", "n", int),
	dropout=("Dropout", "option", "d", float),
	batchsize=("Max compounding batch Size", "option", "b", float)
)
def main(gpu=None, nproc=None, model=None, vocab=None, input_dir=None, output_dir=None, n_iter=100, dropout=0.5, batchsize=32.0):

	arguments = {}
	arguments['model'] = model
	arguments['vocab'] = vocab
	arguments['iterations'] = n_iter
	arguments['dropout'] = dropout
	arguments['maxbatchsize'] = batchsize

	inputpath = os.path.abspath(input_dir)
	outputpath = os.path.abspath(output_dir)

	path = Path(inputpath)
	onlyfiles = [e for e in path.iterdir() if e.is_file()]

	finaltrainfile = None;
	
	#cvfiles = [];
	allfiles = [];

	# collect training files
	for i in onlyfiles:
		if i.name.startswith("TRAIN_"):
			allfiles.append(i);
		elif i.name.startswith("FINALTRAIN"): #set the finel training file
			allfiles.append(i);
			#finaltrainfile = i;
			print("final train data: " , i.name)
	
	# batch process
	#cpus = multiprocessing.cpu_count()
	#print("CPU Count: " , cpus)
	
	maxproc = 1;
	if nproc is not None:
		maxproc = nproc

	args = ((gpu, model, vocab, i, outputpath, n_iter, dropout, batchsize) for i in allfiles)
	print(len(allfiles))
	with Pool(processes = maxproc) as pool:
		results = pool.starmap(train.worker, args)
		print('Done training')
		
	#train final model
	#print("Training final model...")
	#train.worker(gpu, model, vocab, finaltrainfile, outputpath, n_iter, dropout, batchsize)
	
	#evaluate models
	print("Starting evaluation...")
	getscores(inputpath, outputpath, arguments)
	

def average(lst): 
	return round(sum(lst) / len(lst),3) 
	
def evaluate(ner_model, examples):
	scorer = Scorer()
	for input_, annot in examples:
		doc_gold_text = ner_model.make_doc(input_)
		gold = GoldParse(doc_gold_text, entities=annot['entities'])
		pred_value = ner_model(input_)
		scorer.score(pred_value, gold)
	return scorer.scores

def getscores(input_dir=None, models_dir=None, arguments=None):

	#models = {}
	
	finalmodel = None
	
	error = False

	result_f = []
	result_p = []
	result_r = []
	
	result_per_ent = []
	
	final_f = 0;
	final_p = 0;
	final_r = 0;

	inputpath = os.path.abspath(input_dir)
	outputpath = os.path.abspath(models_dir)
	modelspath = os.path.abspath(models_dir)

	mpath = Path(modelspath)
	m_onlydirs = [e for e in mpath.iterdir() if e.is_dir()]

	ipath = Path(inputpath)
	i_onlyfiles = [e for e in mpath.iterdir() if e.is_file()]

	opath = Path(outputpath)

	print("Evaluating cross validation models...")
	for i in m_onlydirs:
		if i.name.startswith("TRAIN_ALLBUT_"):
			fold = i.name.lstrip("TRAIN_ALLBUT_")
			try:
				model = spacy.load(i.as_posix())
				print("Loaded Model: " + i.as_posix())
			except:
				error = True
				print("model not loaded: ", sys.exc_info()[0])
				print("Aborting...")
				break
			path = os.path.abspath(os.path.join(inputpath.rstrip("\\").rstrip("/") , "EVAL_" + fold))
			with open(path, 'r') as myfile:
				data = myfile.read()
				eval_data = eval(data)
				print("Evaluating Model: " + i.as_posix())
				results = evaluate(model, eval_data)
				
				result_f.append(results['ents_f'])
				result_p.append(results['ents_p'])
				result_r.append(results['ents_r'])
				
				result_per_ent.append(results['ents_per_type'])
			
			#models[fold] = (model, eval(data))
		elif i.name.startswith("FINALTRAIN"):
			try:
				finalmodel = spacy.load(i.as_posix())
				print("Loaded Model: " + i.as_posix())
			except:
				error = True
				print("model not loaded: ", sys.exc_info()[0])
				print("Aborting...")
				break
				
	if error is False:
		print("Evaluating Final Model...")
		finalevaldata = os.path.abspath(os.path.join(inputpath.rstrip("\\").rstrip("/") , "FINALTEST"))
		with open(finalevaldata, 'r') as myfile:
			data = myfile.read()
		final_results = evaluate(finalmodel, eval(data))
		final_f = final_results['ents_f']
		final_r = final_results['ents_r']
		final_p = final_results['ents_p']

		# calculate averages for cross validation results
		# this might be wrong - do not print
		avg_recall = sum(result_r) / len(result_r)
		avg_precision = sum(result_p) / len(result_p)
		avg_fscore = sum(result_f) / len(result_f)

		# write summary file
		file.write("-----------------------------------------" + "\n")
		file.write("Version: " + "\n")
		file.write("-----------------------------------------" + "\n")
		# average result for k-fold cross validation - all entities
		file = open(os.path.abspath(os.path.join(outputpath.rstrip("\\").rstrip("/") , "SUMMARY")), 'w')
		file.write(str(len(result_f)) + "-fold " + "cross validation average score..." + "\n")
		file.write("Recall: " + str(avg_recall) + "\n")
		file.write("Precision: " + str(avg_precision) + "\n")
		file.write("FScore: " + str(avg_fscore) + "\n")
		
		# average result for final model - all entities
		file.write("" + "\n")
		file.write("Final model average score..." + "\n")
		file.write("Recall: " + str(final_r) + "\n")
		file.write("Precision: " + str(final_p) + "\n")
		file.write("FScore: " + str(final_f) + "\n")
		
		# average result for k-fold cross validation - per entity type
		# removed - just keep final model results for now
		#ent_dict = {}
		#for item in result_per_ent:
		#	for ent in item:
		#		if(ent not in ent_dict):
		#			ent_dict[ent] = {'p': [], 'r': [], 'f': []}
		#		ent_dict[ent]['p'].append(item[ent]['p'])
		#		ent_dict[ent]['f'].append(item[ent]['f'])
		#		ent_dict[ent]['r'].append(item[ent]['r'])

		#file.write("" + "\n")
		#file.write(str(len(result_f)) + "-fold " + "cross validation average scores per entity type..." + "\n")
		#for ent in ent_dict:
		#	file.write(str(ent) + " p: " + str(average(ent_dict[ent]['p'])) + " r: " + str(average(ent_dict[ent]['r'])) + " f: " + str(average(ent_dict[ent]['f'])) + "\n")	

		# average result for final model - per entity type 
		file.write("" + "\n")
		file.write("Final model average scores per entity type..." + "\n")
		final_results_per_ent = final_results['ents_per_type']
		for ent in final_results_per_ent:
			file.write(str(ent) + ' p: ' + str(round(final_results_per_ent[ent]['p'], 3)) + ' r: ' + str(round(final_results_per_ent[ent]['r'],3)) + ' f: ' + str(round(final_results_per_ent[ent]['f'],3)) + "\n")

		if arguments is not None:
			file.write("" + "\n")
			file.write("spacy.__version__: " + str(spacy.__version__) + "\n")
			file.write("model: " + str(arguments['model']) + "\n")
			file.write("vocab: " + str(arguments['vocab']) + "\n")
			file.write("iterations: " + str(arguments['iterations']) + "\n")
			file.write("dropout: " + str(arguments['dropout']) + "\n")
			file.write("maxbatchsize: " + str(arguments['maxbatchsize']) + "\n")


if __name__ == "__main__":
    plac.call(main)

