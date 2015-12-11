import numpy as np
import os
import cPickle as pkl
from skipthoughts import skipthoughts
from skipthoughts.decoding import tools


class ChatBot(object):

	def __init__(self):

		self.trmodel = tools.load_model("data/trainer.npz", "data/dictionary_fry.pkl")
		print "===== Loaded Trained Model ====="
		self.stmodel = skipthoughts.load_model()
		print "===== Loaded Skipthoughts Model ====="

	def gen_response(self, message):
		return tools.run_sampler(
			self.trmodel, 
			skipthoughts.encode(self.stmodel, [message], use_norm=True, verbose=False)
		)[0]


if __name__ == "__main__":

	bot = ChatBot()

	# credit: http://ascii.co.uk/art/futurama
	print """
    .  ."|
      /| /  |  _.----._ 
     . |/  |.-"        ".  /|
    /                    \/ |__
   |           _.-\"""/        /
   |       _.-"     /."|     /
    ".__.-"         "  |     \\
       |              |       |
       /_      _.._   | ___  /
     ."  ""-.-"    ". |/.-.\/
    |    0  |    0  |     / |
    \      /\_     _/    "_/ 
     "._ _/   "---"       |  
     /\"""                 |  
     \__.--                |_ 
       )          .        | ". 
      /        _.-"\        |  ".
     /     _.-"             |    ".  
    (_ _.-|                  |     |"-._.
      "    "--.             .J     _.-'
              /\        _.-" | _.-'
             /  \__..--"   _.-'
            /   |      _.-'         
           /| /\|  _.-'                     
          / |/ _.-'          
         /|_.-'                                   
       _.-'
	"""

	print
	print "====================================="
	print "=====        Done Loading       ====="
	print "=====  You can now talk to Fry  ====="
	print "====================================="
	print
	
	while True:
		message = raw_input("Your message: ")

		if len(message) == 0:
			print "===== You are done talking to Fry. ====="
			break

		response = bot.gen_response(message)
		print "Fry says:    ", response
		print

