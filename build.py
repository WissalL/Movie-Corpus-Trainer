import numpy as np
import os
from skipthoughts import skipthoughts
from skipthoughts.decoding import vocab, train


def gen_cornell_conversations():
	""" Parse the movie lines from the Cornell Movie Dialogs Corpus and store 
		them in a numpy-ready file as conversations.
	"""

	DATADIR = ["data", "cornell-movie-dialogs-corpus"]
	LINESFILE = os.path.join(*(DATADIR + ["movie_lines.txt"]))
	CONVERSATIONSFILE = os.path.join(*(DATADIR + ["movie_conversations.txt"]))
	DELIMITER = " +++$+++ "

	lines_meta = np.genfromtxt(
		LINESFILE, 
		delimiter=DELIMITER, 
		dtype=str, 
		comments="@", # comments can't be disabled, so I set it to a character
		              # that is never used.
		filling_values="", 
		usecols=(0, 4)
	)

	conversation_locs = np.genfromtxt(
		CONVERSATIONSFILE, 
		delimiter=DELIMITER, 
		dtype=str, 
		usecols=(3,)
	)

	conversation_locs = np.array([
		x.strip("[]").strip("'").split("', '") for x in conversation_locs
	])

	lines = dict(lines_meta)

	def convert_line_ids_to_lines(conversation):
		return np.array([lines[x] for x in conversation], dtype=str)

	conversations = np.array([
		convert_line_ids_to_lines(conversation) 
		for conversation in conversation_locs
	])

	np.save("data/cornell_conversations.npy", conversations)

	return conversations


def gen_cornell_responses(conversations):
	""" Separate conversations into two sequences dividing sources and targets
	"""

	sources, targets = [], []
	for conversation in conversations:
		for i in xrange(len(conversation) - 1):
			sources.append(conversation[i])
			targets.append(conversation[i + 1])

	sources = np.array(sources, dtype=str)
	targets = np.array(targets, dtype=str)

	np.save("data/source_cornell_sequences.npy", sources)
	np.save("data/target_cornell_sequences.npy", targets)

	return sources, targets


def gen_fry_responses():

	return (
		np.genfromtxt("data/fry_sources.txt", delimiter="\n", dtype=str), 
		np.genfromtxt("data/fry_targets.txt", delimiter="\n", dtype=str)
	)


def gen_vocab(targets):

	path = os.path.join("data", "dictionary.pkl")
	if not os.path.isfile(path):
		worddict, wordcount = vocab.build_dictionary(targets)
		vocab.save_dictionary(worddict, wordcount, path)

	return path


def gen_model():

	model = skipthoughts.load_model()
	return model


def gen_encodings(model, sources, category):

	result = skipthoughts.encode(model, sources, use_norm=True, verbose=True)
	np.save("data/source_" + category + "_encodings.npy", result)

	return result


def go_train(sources, targets, model, dictloc, max_epochs):

	train.trainer(targets, sources, model, 
		saveto="data/trainer.npz", dictionary=dictloc, max_epochs, saveFreq=100, 
		reload_=os.path.isfile("data/trainer.npz"))


def gen_cornell(model):
	conversations = (
		np.load("data/cornell_conversations.npy")
		if os.path.isfile("data/cornell_conversations.npy")
		else gen_cornell_conversations()
	)
	print "===== Loaded Cornell Conversations ====="

	sources, targets = (
		np.load("data/source_cornell_sequences.npy"),
		np.load("data/target_cornell_sequences.npy")
		if os.path.isfile("data/source_cornell_sequences.npy")
			and os.path.isfile("data/target_cornell_sequences.npy")
		else gen_cornell_responses(conversations)
	)

	print "===== Loaded Cornell Sources And Targets ====="
	

def gen_fry(model):

	sources, targets = gen_fry_responses();

	print "===== Loaded Fry Sources and Targets ====="

	sources = sources[:200]
	targets = targets[:200]

	return sources, targets


if __name__ == "__main__":

	combinedModel = gen_model()

	print "===== Loaded Model ====="

	frySources, fryTargets = gen_fry(combinedModel)

	fryDictloc = gen_vocab(fryTargets)

	print "====== Loaded Vocabulary - Fry ====="

	cornellSources, cornellTargets = gen_cornell(combinedModel)

	cornellDictloc = gen_vocab(cornellTargets)

	print "====== Loaded Vocabulary - Cornell ====="

	print "===== Loaded Vocabulary ====="

	go_train(cornellSources, cornellTargets, combinedModel, cornellDictloc, 5)

	print "===== Finished Training - Cornell ====="

	go_train(frySources, fryTargets, combinedModel, fryDictloc, 20)

	print "===== Finished Training - Fry ====="
