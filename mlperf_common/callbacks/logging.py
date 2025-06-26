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


import json
import math
import os
import re
import time
from functools import partial
from typing import Any, Dict, Literal, Optional

import numpy as np

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import Logger
    from pytorch_lightning.utilities import rank_zero_only

import torch
import torch.distributed as dist

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
                trainer,
                self.get_train_step_samples_count(trainer, pl_module),
            )

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.train_block_started:
            self._end_train_block(trainer, self.get_train_step_samples_count(trainer, pl_module))

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
                    'step': trainer.global_step,
                    mllogger.constants.SAMPLES_COUNT: trainer.val_check_interval * train_batch_size,
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

            self._log_train_step_time(
                trainer.global_step,
                train_batch_size,
            )

            if self.train_block_started:
                self._end_train_block(trainer, train_batch_size)

            mllogger.start(
                mllogger.constants.EVAL_START,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: trainer.val_check_interval * train_batch_size,
                    'step': trainer.global_step
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

            mllogger.end(
                mllogger.constants.EVAL_STOP,
                metadata={
                    mllogger.constants.SAMPLES_COUNT: trainer.val_check_interval * train_batch_size,
                    'step': trainer.global_step
                },
            )

            if not trainer.should_stop:
                self._start_train_block(
                    trainer,
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

    def _start_train_block(self, trainer, train_batch_size: int) -> None:
        self.train_block_started = True
        self.train_current_block = trainer.global_step
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: trainer.val_check_interval * train_batch_size,
                'step': trainer.global_step,
            },
        )

    def _end_train_block(self, trainer, train_batch_size: int) -> None:
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: trainer.val_check_interval * train_batch_size,
                'step': trainer.global_step,
            },
        )
        self.train_block_started = False

    def _log_train_step_time(
        self,
        trainer_step: int,
        train_batch_size: int,
    ) -> None:
        delta_t = self.timer.get_delta()
        delta_step = trainer_step - self.previous_step
        mllogger.event(
            key="tracked_stats",
            metadata={mllogger.constants.SAMPLES_COUNT: delta_step * train_batch_size},
            value={
                "train_step_time": delta_t / delta_step,
            },
        )

        self.previous_step = trainer_step


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
                    'step': self.trainer.global_step,
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



class StatsLogCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        from megatron.core import parallel_state
        self.parallel_state = parallel_state
        self.logs = {
            "train_loss_batch": [],
            "val_loss_batch": [],
            "val_loss_epoch": [],
            "grad_stats": {},
            "activation_stats": {},
            "learning_rates": [],
            "weights_stats": {},
        }
        self.activation_hooks = []
        self.grad_hooks = []
        self.current_grad_stats = {}
        self.current_weights_stats = {}
        self.current_activation_stats = {}
        self.reduce_tp = os.environ.get("REDUCE_TP", "True").lower() in ("true", "1", "t")
        self.log_every_n_steps = int(os.environ.get("LOG_EVERY_N_BATCHES", "1"))
        self.current_batch_idx = 0
        self.run_n_iters = int(os.environ.get("RUN_N_ITERS", "0"))
        self.enabled = True

    @staticmethod
    def _compute_tensor_stats(
        tensor: torch.Tensor,
        per_batch: bool = True,
        batch_dim: int = 0
    ) -> Dict[str, Any]:
        try:
            if per_batch:
                dims = list(range(tensor.ndim))
                dims.remove(batch_dim)
                return {
                    "mean": tensor.mean(dim=dims).tolist(),
                    "var": tensor.var(dim=dims).tolist(),
                    "min": tensor.amin(dim=dims).tolist(),
                    "max": tensor.amax(dim=dims).tolist(),
                    "norm": torch.norm(tensor, p=2, dim=dims).tolist(),
                }
            else:
                return {
                    "mean": tensor.mean().item(),
                    "var": tensor.var().item(),
                    "min": tensor.amin().item(),
                    "max": tensor.amax().item(),
                    "norm": torch.norm(tensor, p=2).item(),
                }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _get_empty_stats() -> Dict[str, list]:
        return {"mean": [], "var": [], "min": [], "max": [], "norm": []}

    def _gather_group_stats(self, stats: Dict[str, Any], group) -> Dict[str, Any]:
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats, group=group)

        if isinstance(gathered_stats[0]["mean"], list):
            result = {key: [d[key][0] for d in gathered_stats] for key in gathered_stats[0]}
        else:
            result = {key: [d[key] for d in gathered_stats] for key in gathered_stats[0]}
        return result

    def _collect_pp_dicts(self, stats_dict: Dict[str, Any]) -> None:
        def alphanum_key(s: str):
            return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

        group = self.parallel_state.get_pipeline_model_parallel_group()
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats_dict, group=group)
        result = {k: v for d in gathered_stats for k, v in d.items()}
        stats_dict = dict(sorted(result.items(), key=lambda x: alphanum_key(x[0])))
        return stats_dict

    def _collect_tp_dicts(self, stats_dict: Dict[str, Any]) -> None:
        def alphanum_key(s: str):
            return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

        group = self.parallel_state.get_tensor_model_parallel_group()
        group_size = dist.get_world_size(group=group)
        gathered_stats = [None for _ in range(group_size)]
        dist.all_gather_object(gathered_stats, stats_dict, group=group)
        result = {k: v for d in gathered_stats for k, v in d.items()}
        stats_dict = dict(sorted(result.items(), key=lambda x: alphanum_key(x[0])))
        return stats_dict


    def _collect_group_stats(
        self,
        stats: Dict[str, Any],
        use_tp_group: bool = False,
        use_dp_group: bool = True,
    ) -> Dict[str, float]:
        if not self.reduce_tp:
            group = self.parallel_state.get_context_parallel_group()
        elif use_tp_group:
            group = self.parallel_state.get_tensor_model_parallel_group()
        else:
            group = self.parallel_state.get_tensor_and_context_parallel_group()
        result = self._gather_group_stats(stats, group=group)
        result = self._aggregate_local_stats(result)

        if use_dp_group:
            dp_group = self.parallel_state.get_data_parallel_group()
            result = self._gather_group_stats(result, group=dp_group)
            result = self._aggregate_local_stats(result, avg_variance=True)

        return result

    def _aggregate_local_stats(self, stats: Dict[str, list], avg_variance: bool = False) -> Dict[str, float]:
        global_mean = np.mean(stats["mean"])
        global_min = np.min(stats["min"])
        global_max = np.max(stats["max"])

        if avg_variance:
            global_variance = np.mean(stats["var"])
            global_norm = np.mean(stats["norm"])
        else:
            means_sq = [m * m for m in stats["mean"]]
            alpha = [v + ms for (v, ms) in zip(stats["var"], means_sq)]
            avg_alpha = sum(alpha) / len(alpha)
            global_variance = avg_alpha - global_mean ** 2

            sum_sq = sum((norm ** 2) for norm in stats["norm"])
            global_norm = math.sqrt(sum_sq)

        return {
            "mean": global_mean,
            "var": global_variance,
            "min": global_min,
            "max": global_max,
            "norm": global_norm,
        }

    def record_stats(
        self,
        module_name: str,
        stats: Dict[str, Any],
        log_key: Optional[str],
        logs: Dict[str, Any],
    ) -> None:
        target_logs = logs[log_key] if log_key else logs
        if module_name not in target_logs:
            target_logs[module_name] = self._get_empty_stats()

        for stat_name, stat_value in stats.items():
            target_logs[module_name][stat_name].append(stat_value)

    def _get_tensor(self, x: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(x):
            return x
        elif isinstance(x, (list, tuple)):
            return self._get_tensor(x[0])
        return None

    def _activation_hook(self, name: str, module: torch.nn.Module, inp, out):
        if not module.training or not self.enabled:
            return
        if self.current_batch_idx % self.log_every_n_steps != 0:
            return

        act = self._get_tensor(out)
        if act is None:
            return

        stats = self._compute_tensor_stats(act)
        if self.parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            stats = self._collect_group_stats(stats)

        self.record_stats(name, stats, None, self.current_activation_stats)

    def _grad_hook(self, name: str, module: torch.nn.Module, grad_in, grad_out):
        if not self.enabled or self.current_batch_idx % self.log_every_n_steps != 0:
            return

        grad = self._get_tensor(grad_out)
        if grad is None:
            return

        stats = self._compute_tensor_stats(grad, batch_dim=1)
        if self.parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            stats = self._collect_group_stats(stats)

        self.record_stats(name, stats, None, self.current_grad_stats)

    def _log_weights_stats(self, pl_module, chunk_idx=None) -> None:
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                stats = self._compute_tensor_stats(param.data, per_batch=False)
                if self.parallel_state.get_tensor_model_parallel_world_size() > 1:
                    stats = self._collect_group_stats(stats, use_tp_group=True, use_dp_group=False)

                if self.use_vp:
                    name = self.set_layer_vp_chunk(name, chunk_idx)
                if not self.reduce_tp:
                    name = f"{name}_tp{self.parallel_state.get_tensor_model_parallel_rank()}"
                self.current_weights_stats[name] = stats

    def extract_loss(self, outputs: Any) -> Optional[torch.Tensor]:
        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif torch.is_tensor(outputs):
            if outputs.numel() == 1:
                loss = outputs
            elif outputs.numel() == 2:
                loss = outputs[0] / outputs[1]
            else:
                loss = outputs

        if loss is not None:
            loss = loss.cpu()

        if self.parallel_state.get_pipeline_model_parallel_world_size() > 1:
            group = self.parallel_state.get_pipeline_model_parallel_group()
            gathered_losses = [None for _ in range(dist.get_world_size(group=group))]
            dist.all_gather_object(gathered_losses, loss, group=group)
            gathered_losses = [t.mean() for t in gathered_losses if isinstance(t, torch.Tensor)]
            loss = torch.stack(gathered_losses).sum()
        return loss

    def set_layer_vp_chunk(self, name, vp_chunk):
        pattern = r'\.(\d+)\.'
        match = re.search(pattern, name)

        if not match:
            return name

        layer_idx = int(match.group(1))
        pp_rank = self.parallel_state.get_pipeline_model_parallel_rank()
        orig_layer_idx = pp_rank * self.layers_per_pipeline + vp_chunk * self.layers_per_vchunk + layer_idx
        orig_name = re.sub(rf'\.{layer_idx}\.', f'.{orig_layer_idx}.', name, count=1)
        return orig_name

    def register_hooks(self, modules, chunk_idx=None):
        for name, module in modules.named_modules():
            if hasattr(module, "weight") or ".fused_attention" in name:
                if self.use_vp:
                    name = self.set_layer_vp_chunk(name, chunk_idx)
                if not self.reduce_tp:
                    name = f"{name}_tp{self.parallel_state.get_tensor_model_parallel_rank()}"
                self.activation_hooks.append(
                    module.register_forward_hook(partial(self._activation_hook, name))
                )
                self.grad_hooks.append(
                    module.register_full_backward_hook(partial(self._grad_hook, name))
                )
    #
    # PYTORCH LIGHTNING HOOKS
    #

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        vp_is_none = self.parallel_state.get_virtual_pipeline_model_parallel_world_size() is None
        self.use_vp = not vp_is_none and self.parallel_state.get_virtual_pipeline_model_parallel_world_size() > 0

        if self.use_vp:
            config = trainer.model[0].config
            num_layers = config.num_layers
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1
            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            self.layers_per_pipeline = num_layers // self.parallel_state.get_pipeline_model_parallel_world_size()
            self.layers_per_vchunk = self.layers_per_pipeline // self.parallel_state.get_virtual_pipeline_model_parallel_world_size()
            self.pp_size = self.parallel_state.get_pipeline_model_parallel_world_size()
            self.vp_size = self.parallel_state.get_virtual_pipeline_model_parallel_world_size()
            for chunk_idx, model_chunk in enumerate(trainer.model):
                self.register_hooks(model_chunk, chunk_idx)
                self._log_weights_stats(model_chunk, chunk_idx)
        else:
            self.register_hooks(pl_module)
            self._log_weights_stats(pl_module)

        if self.parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.current_weights_stats = self._collect_pp_dicts(self.current_weights_stats)
        if not self.reduce_tp:
            self.current_weights_stats = self._collect_tp_dicts(self.current_weights_stats)
        for module_name, stats in self.current_weights_stats.items():
            self.record_stats(module_name, stats, "weights_stats", self.logs)

        super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ):
        optimizer = trainer.optimizers[0] if isinstance(trainer.optimizers, list) else trainer.optimizers
        lr = optimizer.param_groups[0]["lr"]
        self.logs["learning_rates"].append(lr)
        self.current_grad_stats = {}
        self.current_weights_stats = {}
        self.current_activation_stats = {}
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ):

        self.current_batch_idx += 1

        if self.run_n_iters > 0 and self.current_batch_idx > self.run_n_iters and self.enabled:
            self.enabled = False
            for hook in self.activation_hooks:
                hook.remove()
            for hook in self.grad_hooks:
                hook.remove()
            self.activation_hooks = []
            self.grad_hooks = []

        if not self.enabled or (self.current_batch_idx - 1) % self.log_every_n_steps != 0:
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        loss = self.extract_loss(outputs)
        if loss is not None:
            self.logs["train_loss_batch"].append(loss.item())

        if self.parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.current_grad_stats = self._collect_pp_dicts(self.current_grad_stats)
            self.current_activation_stats = self._collect_pp_dicts(self.current_activation_stats)

        if not self.reduce_tp:
            self.current_grad_stats = self._collect_tp_dicts(self.current_grad_stats)
            self.current_activation_stats = self._collect_tp_dicts(self.current_activation_stats)

        for module_name, stats in self.current_grad_stats.items():
            stats_agg = self._aggregate_local_stats(stats, avg_variance=True)
            self.record_stats(module_name, stats_agg, "grad_stats", self.logs)

        for module_name, stats in self.current_activation_stats.items():
            stats_agg = self._aggregate_local_stats(stats, avg_variance=True)
            self.record_stats(module_name, stats_agg, "activation_stats", self.logs)

        if self.use_vp:
            for chunk_idx, model_chunk in enumerate(trainer.model):
                self._log_weights_stats(model_chunk, chunk_idx)
        else:
            self._log_weights_stats(pl_module)

        if self.parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.current_weights_stats = self._collect_pp_dicts(self.current_weights_stats)
        for module_name, stats in self.current_weights_stats.items():
            self.record_stats(module_name, stats, "weights_stats", self.logs)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.enabled:
            return super().on_validation_start(trainer, pl_module)

        self.logs["val_loss_batch"].append([])
        return super().on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if not self.enabled:
            return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        if self.parallel_state.get_pipeline_model_parallel_world_size() > 1:
            group = self.parallel_state.get_pipeline_model_parallel_group()
            group_size = dist.get_world_size(group=group)
            gathered_stats = [None for _ in range(group_size)]
            dist.all_gather_object(gathered_stats, outputs, group=group)

            if all(torch.is_tensor(t) for t in gathered_stats):
                outputs = torch.stack([t.cpu() for t in gathered_stats]).sum(dim=0)

        loss = self.extract_loss(outputs)
        if loss is not None:
            self.logs["val_loss_batch"][-1].append(loss.item())

        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.enabled:
            return super().on_validation_epoch_end(trainer, pl_module)

        if self.logs["val_loss_batch"] and self.logs["val_loss_batch"][-1]:
            losses = self.logs["val_loss_batch"][-1]
            epoch_loss = sum(losses) / len(losses)
            self.logs["val_loss_epoch"].append(epoch_loss)

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "is_global_zero", True):
            try:
                tp = self.parallel_state.get_tensor_model_parallel_world_size()
                pp = self.parallel_state.get_pipeline_model_parallel_world_size()
                cp = self.parallel_state.get_context_parallel_world_size()
                filepath = f"/results/stats_tp{tp}_pp{pp}_cp{cp}_seed{os.getenv('SEED', '1')}.json"
                with open(filepath, "w") as f:
                    json.dump(self.logs, f, indent=4)
            except Exception as e:
                print(f"Error saving debugging info: {e}")

        return super().on_train_end(trainer, pl_module)
