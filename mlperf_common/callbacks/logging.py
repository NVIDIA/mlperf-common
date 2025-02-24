# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import Logger
    from pytorch_lightning.utilities import rank_zero_only

import os
import time
from typing import Any, Callable, Literal

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper

mllogger = MLLoggerWrapper(PyTCommunicationHandler())


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

        self.timer = DeltaTimer()
        self.previous_step = 0
        self.mllogger = mllogger

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not hasattr(trainer, "warmup") or trainer.warmup:
            self.warmup(trainer, pl_module)
            trainer.warmup = False
            self.log_custom_timedelta("warmup_time", trainer.global_step)

        if not self.train_block_started:
            self.log_custom_timedelta("init_finished", trainer.global_step)
            mllogger.log_init_stop_run_start()
            self._start_train_block(
                trainer.global_step,
                self.get_train_step_samples_count(trainer, pl_module),
            )

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.train_block_started:
            self._end_train_block(self.get_train_step_samples_count(trainer, pl_module))

        if hasattr(trainer, "run_stop_logged") and not trainer.run_stop_logged:
            train_batch_size = self.get_train_step_samples_count(trainer, pl_module)
            if (
                hasattr(trainer, "target_reached") and trainer.target_reached
            ) or self.force_success:
                status = mllogger.constants.SUCCESS
            else:
                status = mllogger.constants.ABORTED

            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: self.train_current_block
                    * train_batch_size,
                    "status": status,
                },
            )

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not trainer.warmup:
            train_batch_size = self.get_train_step_samples_count(trainer, pl_module)
            self._log_training_throughput(
                trainer.global_step,
                train_batch_size,
            )

            if self.train_block_started:
                self._end_train_block(train_batch_size)

            mllogger.start(
                mllogger.constants.EVAL_START,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * train_batch_size
                },
            )

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not trainer.warmup:
            train_batch_size = self.get_train_step_samples_count(trainer, pl_module)
            validation_samples = self.get_validation_samples_count(trainer, pl_module)

            self._log_validation_throughput(
                trainer.global_step,
                train_batch_size,
                validation_samples,
            )

            mllogger.end(
                mllogger.constants.EVAL_STOP,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * train_batch_size
                },
            )

            if not trainer.should_stop:
                self._start_train_block(
                    trainer.global_step,
                    train_batch_size,
                )

    def log_custom_timedelta(self, value_key, step: int = 0):
        mllogger.event(
            key="tracked_stats",
            metadata={"step": step},
            value={value_key: self.timer.get_delta()},
        )

    def warmup(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    def get_train_step_samples_count(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> int:
        return trainer.train_dataloader.batch_sampler.global_batch_size

    def get_validation_samples_count(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> int:
        return trainer.val_dataloaders.batch_sampler.global_batch_size * len(
            trainer.val_dataloaders
        )

    def __deepcopy__(self, memo):
        return LoggingCallback(self.force_success)

    def _start_train_block(self, trainer_step: int, train_batch_size: int) -> None:
        self.train_block_started = True
        self.train_current_block = trainer_step
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.train_current_block
                * train_batch_size
            },
        )

    def _end_train_block(self, train_batch_size: int) -> None:
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.train_current_block
                * train_batch_size
            },
        )
        self.train_block_started = False

    def _log_training_throughput(
        self,
        trainer_step: int,
        train_batch_size: int,
    ) -> None:
        delta_t = self.timer.get_delta()
        delta_step = trainer_step - self.previous_step
        throughput = (delta_step * train_batch_size) / delta_t
        mllogger.event(
            key="tracked_stats",
            metadata={"step": self.train_current_block * train_batch_size},
            value={
                "throughput": throughput,
                "train_step_time": delta_t / delta_step,
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
        mllogger.event(
            key="tracked_stats",
            metadata={"step": trainer_step * train_batch_size},
            value={"validation_throughput": throughput},
        )


class MLPerfLogger(Logger):
    def __init__(
        self,
        callback_cls: type[LoggingCallback],
        model: pl.LightningModule,
        validation_metric: str,
        target_validation_metric: float,
        target_validation_metric_type: Literal["min", "max"],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.trainer = None
        self.model = model
        self.validation_metric = validation_metric
        self.target_validation_metric = target_validation_metric
        self.target_validation_metric_type = target_validation_metric_type

        self.custom_callback = self._build_custom_callback(
            callback_cls, *args, **kwargs
        )

        self.callback_arguments = (args, kwargs)
        self.mllogger = mllogger

    def __deepcopy__(self, memo):
        output = MLPerfLogger(
            self.callback_cls,
            self.model,
            self.validation_metric,
            self.target_validation_metric,
            self.target_validation_metric_type,  # type: ignore
            *self.callback_arguments[0],
            **self.callback_arguments[1],
        )

        output.trainer = self.trainer
        return output

    def set_trainer(self, trainer: pl.Trainer):
        self.trainer = trainer
        trainer.run_stop_logged = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        if self.validation_metric in metrics:
            computed_metric = self.compute_validation_metric(metrics)
            mllogger.event(
                key=mllogger.constants.EVAL_ACCURACY,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: self.trainer.global_step
                    * self.custom_callback.get_train_step_samples_count(
                        self.trainer, self.model
                    )
                },
                value=computed_metric,
            )

            if (
                self.target_validation_metric_type == "min"
                and computed_metric <= self.target_validation_metric
            ) or (
                self.target_validation_metric_type == "max"
                and computed_metric >= self.target_validation_metric
            ):
                if self.trainer is not None:
                    self.trainer.target_reached = True
                    self.trainer.should_stop = True

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | None = None, *args, **kwargs):
        for key, value in self.compute_hyperparams(params, *args, **kwargs).items():
            mllogger.event(key=key, value=value, sync=False)

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics[self.validation_metric]

    def compute_hyperparams(
        self, params: dict[str, Any] | None, *args, **kwargs
    ) -> dict[str, Any]:
        return {}

    @property
    def name(self) -> str:
        return "mlperf-metrics"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def callback(self) -> LoggingCallback:
        return self.custom_callback

    def _build_custom_callback(
        self,
        cls: type[LoggingCallback],
        *args,
        **kwargs,
    ) -> LoggingCallback:
        return cls(*args, **kwargs)
