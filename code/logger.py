import sys
from datetime import datetime


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log = open(self.run_id+"train_gedetModel_logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass