# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from mlperf_logging import mllog
from mlperf_logging.mllog import constants


class MLLoggerWrapper:
    def __init__(self, comm_handler, mllogger=None, default_stack_offset=3, **default_log_kwargs):
        """
        Args:
            comm_handler (CommunicationHandler): framework specific handler for barrier
              and determining process rank
            mllogger: MLPerf official logger instance
            default_stack_offset: number of stack calls added by this wrapper and the wrapped logger
              when using `start`, `end` and `event` methods. Other helper methods automatically
              increment stack_offset by 1.
            **default_log_kwargs: kwargs appended to each logging call
        """
        self.comm_handler = comm_handler
        if mllogger is None:
            mllogger = mllog.get_mllogger()
        mllogger.default_stack_offset = default_stack_offset
        self.default_stack_offset = default_stack_offset
        self.mllogger = mllogger
        self.default_log_kwargs = default_log_kwargs

        # reducing imports convenience
        self.constants = constants

    def start(self, *args, **kwargs):
        """ Primitive for logging a time interval start. """
        self._log(self.mllogger.start, *args, **kwargs)

    def end(self, *args, **kwargs):
        """ Primitive for logging a time interval end. """
        self._log(self.mllogger.end, *args, **kwargs)

    def event(self, *args, **kwargs):
        """ Primitive for logging a point in time event. """
        self._log(self.mllogger.event, *args, **kwargs)

    def _log(self, log_fn, *args, unique=True, unique_log_rank=0,
             sync=False, sync_group=None, internal_call=False, **kwargs):
        """ Main logging function.

        Args:
            log_fn: actual logging function
            *args: passed to `log_fn`
            unique: if True, only `unique_log_rank` process will log
            unique_log_rank: which process should log a unique log
            sync: calls a barrier *before* logging
            sync_group: group passed to barrier function
            internal_call: set to True for logs made by this wrapper.
              Increments default stack_offset to correctly trace log call location.
            **kwargs: passed to `log_fn` along with `self.default_log_kwargs`
        """
        kwargs = {**self.default_log_kwargs, **kwargs}
        if kwargs.get('stack_offset') is None and internal_call:
            kwargs['stack_offset'] = self.default_stack_offset + 1

        if sync:
            self.comm_handler.barrier(sync_group)

        if not unique or self.comm_handler.global_rank() == unique_log_rank:
            log_fn(*args, **kwargs)

    def log_init_stop_run_start(self):
        """ Logs INIT_STOP and RUN_START with correct synchronization across workers. """
        self.end(key=constants.INIT_STOP, unique=True, sync=True, internal_call=True)
        self.start(key=constants.RUN_START, unique=True, sync=True, internal_call=True)
        self.comm_handler.barrier()

    def log_run_stop(self, status, sync=True, unique=True, unique_log_rank=0, **kwargs):
        """ Logs RUN_STOP with correct synchronization across workers. """
        self.end(key=constants.RUN_STOP, unique=unique, unique_log_rank=unique_log_rank, sync=sync,
                 metadata=dict(status=status, **kwargs), internal_call=True)

    def log_weights_initialization(self, tensor_name):
        """ Helper for weights initialization. """
        self.event(key=constants.WEIGHTS_INITIALIZATION,
                   metadata=dict(tensor=tensor_name), internal_call=True)

    def mlperf_submission_log(self, benchmark, num_nodes=None, org=None,
                              platform=None):
        """ Helper for logging submission entry. """
        if num_nodes is None:
            num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

        if org is None:
            org = os.environ.get('MLPERF_SUBMISSION_ORG',
                                 'SUBMISSION_ORG_PLACEHOLDER')

        if platform is None:
            platform = os.environ.get('MLPERF_SUBMISSION_PLATFORM',
                                      'SUBMISSION_PLATFORM_PLACEHOLDER')

        self.event(
            key=constants.SUBMISSION_BENCHMARK,
            value=benchmark,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_ORG,
            value=org,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_DIVISION,
            value=constants.CLOSED,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_STATUS,
            value=constants.ONPREM,
            internal_call=True)

        self.event(
            key=constants.SUBMISSION_PLATFORM,
            value=f'{num_nodes}x{platform}',
            internal_call=True)

