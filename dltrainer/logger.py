import os
import logging
import time
import datetime
import os

class Logger(object):
	def __init__(self):
		if not os.path.exists("./log"):
		    os.makedirs("./log")
		fname = './log/log_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
		logging.basicConfig(filename=fname, filemode='w', level=logging.DEBUG)
		return

	def log(self, *args):
		print("LOG", args)
		for arg in args:
			logging.info(arg)
		return

	def err(self, *args):
		print("ERROR", args)
		for arg in args:
			logging.error(arg)
		return