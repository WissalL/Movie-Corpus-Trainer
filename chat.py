import numpy as np
import os
import cPickle as pkl
from skipthoughts import skipthoughts
from skipthoughts.decoding import tools


class ChatBot(object):

	def __init__(self):
		with open("data/dictionary_cornell.pkl", "rb") as f:
			cornell = pkl.load(f)
		with open("data/dictionary_fry.pkl", "rb") as f:
			fry = pkl.load(f)
		cornell.update(fry)
		with open("data/dictionary_combined.pkl", "wb") as f:
			pkl.dump(cornell, f)

		self.trmodel = tools.load_model("data/trainer.npz", "data/dictionary_combined.pkl")
		print "===== Loaded Trained Model ====="
		self.stmodel = skipthoughts.load_model()
		print "===== Loaded Skipthoughts Model ====="

	def gen_response(self, message):
		return tools.run_sampler(
			self.trmodel, 
			skipthoughts.encode(self.stmodel, [message], use_norm=True, verbose=False)
		)


if __name__ == "__main__":

	bot = ChatBot()
	
	while True:
		message = raw_input("Your message: ")
		if len(message) == 0:
			break

		print bot.gen_response(message)[0]

