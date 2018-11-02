from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from future import standard_library
standard_library.install_aliases()
from builtins import range

from subprocess import PIPE, Popen, check_output
from io import open

import pdb
import gpustat
import json
import psutil
import os
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler

from tensorboardX import SummaryWriter
from datetime import datetime
from logger import Logger
import traceback

class Profiler(object):
    def __init__(self, logger, tensorboard_logger, node_id=0, freq=20):
        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.node_id = node_id
        self.freq = freq

        # Base string for tensorboard logger; it captures node id
        self.log_base_str = "profiler/" + str(self.node_id)

        self.timestamp = 0
        return

    def log_gpustat(self, timestamp):
        proc = Popen(['gpustat', '--json'], stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        gpustat = json.loads(out)

        for gpu in gpustat['gpus']:
            gpu_index = gpu['index']
            gpu_type = gpu['name']
            gpu_util = gpu['utilization.gpu']    
            mem_total = gpu['memory.total']
            mem_used = gpu['memory.used']
            mem_percent = 100*float(mem_used)/mem_total

            # Captures gpu index and gpu type
            gpu_id = "gpu" + str(gpu_index) + "_" + gpu_type.replace(" ", "")

            # Log GPU utilization and GPU memory used to TensorBoard
            self.tensorboard_logger.add_scalar(self.log_base_str + "/" + str(gpu_id) + "/" +"util", gpu_util, timestamp)
            self.tensorboard_logger.add_scalar(self.log_base_str + "/" + str(gpu_id) + "/" +"memory", mem_percent, timestamp)
        return

    def log_cpustat(self, timestamp):
        overall_cpu_util = psutil.cpu_percent()
        mem_util = psutil.virtual_memory().percent

        self.tensorboard_logger.add_scalar(self.log_base_str + "/cpu/" + "util", overall_cpu_util, timestamp)
        self.tensorboard_logger.add_scalar(self.log_base_str + "/cpu/" + "RAM", mem_util, timestamp)
        return

    def log_networkstat(self, timestamp):
        # Command to get current network throughput in Bytes
        # Calculates an average throughput from 20 instances captured 1s apart
        # Also, calculates across all interfaces
        cmd = "bwm-ng -T avg -u bytes -o csv -c 20 | tail -n 1"
        cmd_out = check_output(cmd, shell=True)
        cmd_out = cmd_out.strip().decode().split(';')

        # Output format
        # unix_timestamp interface bytes_out/s bytes_in/s bytes_total/s bytes_in bytes_out 
        # packets_out/s packets_in/s packets_total/s packets_in packets_out errors_out/s errors_in/s errors_in errors_out
        outgoing_bw = float(cmd_out[2])/1024 # in KBps
        incoming_bw = float(cmd_out[3])/1024 # in KBps

        self.tensorboard_logger.add_scalar(self.log_base_str + "/network/" + "download", incoming_bw, timestamp)
        self.tensorboard_logger.add_scalar(self.log_base_str + "/network/" + "upload", outgoing_bw, timestamp)
        return


    def log(self, log_gpu=True, log_cpu=True, log_network=True):
        self.timer = threading.Timer(self.freq, self.log)
        self.timer.start()

        self.timestamp = self.timestamp + 1
        try:
            if log_gpu:
                self.log_gpustat(self.timestamp)

            if log_cpu:
                self.log_cpustat(self.timestamp)

            if log_network:
                self.log_networkstat(self.timestamp)
                
        except Exception as e:
            self.logger.err("Exception during Profiling", "\ntimestamp:", self.timestamp, "\nError message:", e, traceback.print_exc())        
        
        return

    def stop(self):
        self.timer.cancel()
        return

if __name__ == '__main__':
    log_dir = os.path.join('log', 'test', datetime.now().isoformat())
    tb_logger = SummaryWriter(log_dir=log_dir)
    logger = Logger()

    profiler = Profiler(logger, tb_logger, freq=30)
    profiler.log()
