import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import sys
import os

"""
Train a single train and test set - outputs model to output directory 
NOTE: evaluation is not performed automatically by this script

Trains spaCy's named entity recognizer, starting off with an
existing model or a blank model.

Main worker that trains an ner model

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.8

ssalpietro - May 31, 2019
"""

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
def worker(gpu=None, model=None, vocab=None, input_file=None, output_dir=None, n_iter=100, dropout=0.5, batchsize=32.0):

	reset_weights = False # if model contains no pre trained weights, then reset and begin training with random weights

	isgpu = False;
	
	if gpu is 'y':
		isgpu = spacy.prefer_gpu()
	
	print("GPU: ", isgpu)
		

	print("begin worker")
	if output_dir is not None:
		output_dir = os.path.join(output_dir, input_file.name.rstrip("\\").rstrip("/"))

	if input_file is None:
		print("No input file - exiting")
		return
	else:
		# training data
		path = input_file.as_posix()
		print("Load training data: " , path)

		with open(path, 'r') as myfile:
			data = myfile.read()
		  
		TRAIN_DATA = eval(data)

		# """Load the model, set up the pipeline and train the entity recognizer."""
		if model is not None:
			nlp = spacy.load(model)  # load existing spaCy model
			print("Loaded model '%s'" % model)
		else:
			nlp = spacy.blank("en")  # create blank Language class
			nlp.vocab.vectors.name = 'spacy_pretrained_vectors' #added this to solve error Unnamed vectors -- this won't allow multiple vectors models to be loaded
			print("Created blank 'en' model")

		if vocab is not None:
			spacy.load(vocab, vocab=nlp.vocab)

		# create the built-in pipeline components and add them to the pipeline
		# nlp.create_pipe works for built-ins that are registered with spaCy
		if "ner" not in nlp.pipe_names:
			ner = nlp.create_pipe("ner")
			nlp.add_pipe(ner, last=True)
			reset_weights = True # model contain no pretrained weights for ner
		else: # otherwise, get it so we can add labels
			ner = nlp.get_pipe("ner")

		# add labels
		for _, annotations in TRAIN_DATA:
			for ent in annotations.get("entities"):
				ner.add_label(ent[2])

		# get names of other pipes to disable them during training
		other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
		with nlp.disable_pipes(*other_pipes):  # only train NER
			# reset and initialize the weights randomly – but only if we're
			# training a new model
			if model is None or reset_weights:
				nlp.begin_training()
			for itn in range(n_iter):
				random.shuffle(TRAIN_DATA)
				losses = {}
				# batch up the examples using spaCy's minibatch
				size=compounding(2.0, batchsize, 1.005)
				batches = minibatch(TRAIN_DATA, size)
				for batch in batches:
					texts, annotations = zip(*batch)
					nlp.update(
						texts,  # batch of texts
						annotations,  # batch of annotations
						drop=dropout,  # dropout - make it harder to memorise data
						losses=losses,
					)
				count = itn + 1
				print("Losses", losses, " Iteration: ", count, " of ", n_iter, " Fold: " , input_file)
				sys.stdout.flush()
				

		# test the trained model
		#for text, _ in TRAIN_DATA:
		#	doc = nlp(text)
		#	print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
		#	print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

		# save model to output directory
		if output_dir is not None:
			output_dir = Path(output_dir)
			if not output_dir.exists():
				output_dir.mkdir()
			nlp.to_disk(output_dir)
			print("Saved model to", output_dir)

			# test the saved model
			#print("Loading from", output_dir)
			#nlp2 = spacy.load(output_dir)
			#for text, _ in TRAIN_DATA:
			#	doc = nlp2(text)
			#	print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
			#	print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
			

if __name__ == "__main__":
	plac.call(worker)