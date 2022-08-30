# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from time import time
from collections import defaultdict
from sys import exit

class Metricstats:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.min = 1000000
        self.max = 0

    def addtag(self, dur):
        self.total += dur
        self.count += 1
        if dur < self.min:
           self.min = dur
        if dur > self.max:
           self.max = dur

    def getstats(self):
        return self.total, self.count, self.min, self.max

    def getcount(self):
        return self.count


class ScaleoutBridgeBase(object):
    FWD_TIME = 'fwd_time'
    BWD_TIME = 'bwd_time'
    FBWD_TIME = 'fbwd_time'
    OPT_TIME = 'opt_time'
    LOAD_TIME = 'load_time'
    EVAL_TIME = 'eval_time'
    ITER_TIME = 'iter_time'
    EPOCH_TIME = 'epoch_time'

    def start_prof(self, tag):
        pass

    def stop_prof(self, tag):
        return 1

    def stop_start_prof(self, tag1, tag2):
        pass

    def start_epoch_prof(self):
        pass

    def stop_epoch_prof(self):
        pass

    def start_eval_prof(self):
        pass

    def stop_eval_prof(self):
        pass


class ScaleoutBridgeIterwise(ScaleoutBridgeBase):
    def __init__(self, qmax, time_tags, nvtx_flag, deviceid,
                 prof_handler, comm_handler,
                 mllogger):
        super().__init__()
        print("Scaleout performance bridge is running ...")
        self.qmax = qmax
        self.time_tags = time_tags
        self.nvtx_flag = nvtx_flag
        self.deviceid = deviceid
        self.bridgestats = defaultdict(Metricstats)
        self.start_epoch = 0
        self.start_eval = 0

        self.start_time = 0
        self.start_iter = 0

        self.comm_handler = comm_handler
        self.prof_handler = prof_handler
        self.mllogger = mllogger

    def print_tag(self, tag, dur):
        self.mllogger.event(key=tag, value={'r':self.deviceid, 't':dur}, unique=False)

    def add_tag(self, tag, dur):
        self.bridgestats[tag].addtag(dur)
        if tag == self.ITER_TIME:
            if self.bridgestats[tag].getcount() > self.qmax:
                self.printstats()
                return 0
        return 1

    def start_prof(self, tag):
        if self.time_tags:
            self.comm_handler.device_sync()
            if tag == self.ITER_TIME:
                self.start_iter = time()
            else:
                self.start_time = time()
        if self.nvtx_flag:
            self.prof_handler.push_nvtx(tag)

    def stop_prof(self, tag):
        if self.time_tags:
            self.comm_handler.device_sync()
            if tag == self.ITER_TIME:
                if not self.add_tag(tag, time()-self.start_iter):
                    self.printstats()
                    self.prof_handler.profiler_stop()
                    self.time_tags = 0
                    exit()
                self.start_iter = 0
            else:
                self.add_tag(tag, time()-self.start_time)
                self.start_time = 0

        if self.nvtx_flag:
            self.prof_handler.pop_nvtx()
        return self.time_tags

    def stop_start_prof(self, tag1, tag2):
        if self.time_tags:
            self.comm_handler.device_sync()
            new_start_time = time()
            if not self.add_tag(tag1, new_start_time-self.start_time):
                self.printstats()
                self.time_tags = 0
            self.start_time = new_start_time
        if self.nvtx_flag:
            self.prof_handler.pop_nvtx()
            self.prof_handler.push_nvtx(tag2)

    def start_epoch_prof(self):
        self.prof_handler.profiler_start()
        self.comm_handler.device_sync()
        self.start_epoch = time()

    def stop_epoch_prof(self):
        self.printstats()
        self.comm_handler.device_sync()
        self.print_tag(self.EPOCH_TIME, time()-self.start_epoch)
        self.prof_handler.profiler_stop()
        exit()

    def start_eval_prof(self):
        self.comm_handler.device_sync()
        self.start_eval = time()

    def stop_eval_prof(self):
        self.printstats()
        self.comm_handler.device_sync()
        self.print_tag(self.EVAL_TIME, time()-self.start_eval)

    def printstats(self):
        if not self.time_tags:
            return
        for tag in self.bridgestats:
            self.printstat(tag)
        self.bridgestats.clear()

    def printstat(self, tag):
        total, count, minimum, maximum = self.bridgestats[tag].getstats()
        self.mllogger.event(key=tag+'_total', value={'r':self.deviceid, 't':total}, unique=False)
        self.mllogger.event(key=tag+'_count', value={'r':self.deviceid, 't':count}, unique=False)
        self.mllogger.event(key=tag+'_min', value={'r':self.deviceid, 't':minimum}, unique=False)
        self.mllogger.event(key=tag+'_max', value={'r':self.deviceid, 't':maximum}, unique=False)


class ScaleoutBridgeEpochwise(ScaleoutBridgeBase):
    def __init__(self, deviceid, comm_handler, mllogger):
        super().__init__()
        print("Scaleout performance bridge-epoch only is running ...")
        self.start_time = 0
        self.deviceid = deviceid
        self.comm_handler = comm_handler
        self.mllogger = mllogger

    def start_epoch_prof(self):
        self.comm_handler.device_sync()
        self.start_time = time()

    def stop_epoch_prof(self):
        self.comm_handler.device_sync()
        self.mllogger.event(key='epoch_time', value={'r':self.deviceid, 't':time()-self.start_time}, unique=False)


def init_bridge(prof_handler, comm_handler, mllogger):
    deviceid = comm_handler.global_rank()
    time_tags = int(os.getenv('TIME_TAGS', 0))
    nvtx_flag = int(os.getenv('NVTX_FLAG', 0))
    epoch_only = int(os.getenv('EPOCH_PROF', 0))
    sbridge = ScaleoutBridgeBase()
    if time_tags or nvtx_flag:
        sbridge = ScaleoutBridgeIterwise(1000, time_tags, nvtx_flag, deviceid,
                                         prof_handler, comm_handler, mllogger)
    elif epoch_only:
        sbridge = ScaleoutBridgeEpochwise(deviceid, comm_handler, mllogger)
    return sbridge
