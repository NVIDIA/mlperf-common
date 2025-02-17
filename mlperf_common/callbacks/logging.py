# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only

import os
import time

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from mlperf_logging.mllog import constants

mllogger_instance = None


def get_mllogger():
    global mllogger_instance
    if mllogger_instance is None:
        mllogger_instance = MLLoggerWrapper(PyTCommunicationHandler())
    return mllogger_instance


class DeltaTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


class LoggingCallback(pl.Callback):
    def __init__(self, force_success: bool | None = None):
        if force_success is None:
            force_success = os.getenv("FORCE_SUCCESS_STATUS", "0") == "1"

        self.force_success = force_success

        self.train_current_block = 0
        self.train_block_started = False

        self.mllogger = get_mllogger()
        self.timer = DeltaTimer()
        self.previous_step = 0

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not self.train_block_started:
            self.mllogger.log_init_stop_run_start()
            self._start_train_block(
                trainer.global_step,
                trainer.train_dataloader.batch_sampler.global_batch_size,
            )
            self.timer.reset()

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if hasattr(trainer, "run_stop_logged") and not trainer.run_stop_logged:
            train_batch_size = trainer.train_dataloader.batch_sampler.global_batch_size
            self.mllogger.end(
                constants.RUN_STOP,
                metadata={
                    "samples_count": self.train_current_block * train_batch_size,
                    "status": "aborted" if not self.force_success else "success",
                },
            )

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not trainer.warmup:
            train_batch_size = trainer.train_dataloader.batch_sampler.global_batch_size
            self._log_training_throughput(
                trainer.global_step,
                train_batch_size,
            )

            if self.train_block_started:
                self._end_train_block(train_batch_size)

            self.mllogger.start(
                constants.EVAL_START,
                metadata={"step": self.train_current_block * train_batch_size},
            )

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not trainer.warmup:
            train_batch_size = trainer.train_dataloader.batch_sampler.global_batch_size
            validation_samples = (
                trainer.val_dataloaders.batch_sampler.global_batch_size
                * len(trainer.val_dataloaders)
            )

            self._log_validation_throughput(
                trainer.global_step,
                train_batch_size,
                validation_samples,
            )

            self.mllogger.end(
                constants.EVAL_STOP,
                metadata={"step": self.train_current_block * train_batch_size},
            )

            if not trainer.should_stop:
                self._start_train_block(
                    trainer.global_step,
                    train_batch_size,
                )

    def __deepcopy__(self, memo):
        return LoggingCallback(self.force_success)

    def _start_train_block(self, trainer_step: int, train_batch_size: int) -> None:
        self.train_block_started = True
        self.train_current_block = trainer_step
        self.mllogger.start(
            constants.BLOCK_START,
            metadata={"step": self.train_current_block * train_batch_size},
        )

    def _end_train_block(self, train_batch_size: int) -> None:
        self.mllogger.end(
            constants.BLOCK_STOP,
            metadata={"step": self.train_current_block * train_batch_size},
        )
        self.train_block_started = False

    def _log_training_throughput(
        self,
        trainer_step: int,
        train_batch_size: int,
    ) -> None:
        delta_t = self.timer.get_delta()
        throughput = ((trainer_step - self.previous_step) * train_batch_size) / delta_t
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": trainer_step * train_batch_size},
            value={
                "throughput": throughput,
                "train_step_time": delta_t / (trainer_step - self.previous_step),
            },
        )

        self.previous_step = trainer_step

    def _log_validation_throughput(
        self,
        trainer_step: int,
        train_batch_size: int,
        validation_samples: int,
    ) -> None:
        delta_t = self.timer.get_delta()
        throughput = (validation_samples) / delta_t
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": trainer_step * train_batch_size},
            value={"validation_throughput": throughput},
        )
