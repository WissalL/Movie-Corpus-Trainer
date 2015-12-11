import numpy as np
import os
import time
from skipthoughts import skipthoughts
from skipthoughts.decoding import vocab, train


# For the purposes of this demo, the Cornell corpus is never used
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
	""" Fry's conversations are determined by separating the lines spoken to
		Fry (sources) and Fry's responses to those lines (targets). They should
		be located in "data/fry_sources.txt" and "data/fry_targets.txt"
		respectively.

		Data gets parsed as Numpy arrays, with conversations containing blank
		lines or just punctuation removed.
	"""

	if os.path.isfile("data/fry_sources.npy") and os.path.isfile("data/fry_targets.npy"):
		return np.load("data/fry_sources.npy"), np.load("data/fry_targets.npy")

	sources = np.genfromtxt("data/fry_sources.txt", delimiter="\n", dtype=str)
	targets = np.genfromtxt("data/fry_targets.txt", delimiter="\n", dtype=str)

	assert len(sources) == len(targets)

	# remove useless lines
	i = 0
	while i < len(sources) and i < len(targets):
		sources[i] = sources[i].strip()
		targets[i] = targets[i].strip()
		if(
			sources[i] == "..." or targets[i] == "..." or 
			sources[i] == "."   or targets[i] == "."   or
			sources[i] == "?"   or targets[i] == "?"   or
			sources[i] == "!"   or targets[i] == "!"
		):
			sources = np.delete(sources, i)
			targets = np.delete(targets, i)
		else:
			i += 1


	np.save("data/fry_sources.npy", sources)
	np.save("data/fry_targets.npy", targets)

	return sources, targets


def gen_vocab(targets, fname):
	""" Use Fry's target lines to generate a dictionary of words """

	path = os.path.join("data", fname)
	if not os.path.isfile(path):
		worddict, wordcount = vocab.build_dictionary(targets)
		vocab.save_dictionary(worddict, wordcount, path)

	return path


def gen_model():
	""" Get the Skipthoughts model to be used in encoding """

	model = skipthoughts.load_model()
	return model


def gen_encodings(model, sources, category):
	""" Generate encodings in advance. It can save computation time if
		training multiple times. Not used in this demo.
	"""

	result = skipthoughts.encode(model, sources, use_norm=True, verbose=True)
	np.save("data/source_" + category + "_encodings.npy", result)

	return result


def go_train(sources, targets, model, dictloc, max_epochs):
	""" Train the network on the conversations and store them in 
		data/trainer.npz 
	"""

	train.trainer(targets, sources, model, 
		saveto="data/trainer.npz", 
		dictionary=dictloc, 
		max_epochs=max_epochs, 
		saveFreq=100, 
		reload_=os.path.isfile("data/trainer.npz")
	)


def gen_cornell():
	""" Load preparsed Cornell conversations and separate them into sources
		and targets
	"""
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

	return sources, targets


if __name__ == "__main__":

	start = time.time()

	combinedModel = gen_model()

	print "===== Loaded Model ====="

	frySources, fryTargets = gen_fry_responses()

	print "===== Loaded Fry Sources and Targets ====="

	fryDictloc = gen_vocab(fryTargets, "dictionary_fry.pkl")

	print "====== Loaded Fry Vocabulary ====="

	go_train(frySources, fryTargets, combinedModel, fryDictloc, 20)

	end = time.time()

	print "===== Finished Fry Training ====="

	m, s = divmod(end - start, 60)
	h, m = divmod(m, 60)
	print "Finished in %d:%02d:%02d hours" % (h, m, s)
