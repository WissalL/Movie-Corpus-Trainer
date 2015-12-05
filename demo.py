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

	np.save(os.path.join("data", "cornell_conversations.npy"), conversations)

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

	np.save(os.path.join("data", "source_cornell_sequences.npy"), sources)
	np.save(os.path.join("data", "target_cornell_sequences.npy"), targets)

	return sources, targets


def gen_vocab(targets):

	path = os.path.join("data", "dictionary.pkl")
	if not os.path.isfile(path):
		worddict, wordcount = vocab.build_dictionary(targets)
		vocab.save_dictionary(worddict, wordcount, path)

	return path


def gen_model():

	model = skipthoughts.load_model()
	return model


def go_train(sources, targets, model, dictloc):

	train.trainer(targets, sources, model, 
		saveto=os.path.join("data", "trainer.npz"), dictionary=dictloc)


if __name__ == "__main__":

	conversations = (
		np.load(os.path.join("data", "cornell_conversations.npy")) 
		if os.path.isfile(os.path.join("data", "cornell_conversations.npy")) 
		else gen_conversations()
	)
	print "===== Loaded Cornell Conversations ====="

	sources, targets = (
		np.load(os.path.join("data", "source_cornell_sequences.npy")), 
		np.load(os.path.join("data", "target_cornell_sequences.npy"))
		if os.path.isfile(os.path.join("data", "source_cornell_sequences.npy")) 
			and os.path.isfile(os.path.join("data", "target_cornell_sequences.npy"))
		else gen_responses(conversations)
	)

	print "===== Loaded Cornell Sources And Targets ====="

	dictloc = gen_vocab(targets)

	print "===== Loaded Vocabulary ====="

	model = gen_model()

	print "===== Loaded Model ====="

	go_train(sources, targets, model, dictloc)

	print "===== Finished Training ====="
