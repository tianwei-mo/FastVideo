# SPDX-License-Identifier: Apache-2.0
"""LoRA finetuning of a distilled PDD student on its own sampling grid.

The served FastH3 OmniRef student samples eight fused blocks of its PDD fine
grid. Training a LoRA on the base model and applying it to that student lands
attenuated, so this method trains the adapter on the student itself, at the
states and with the fused heads its sampler uses:

* real target latents are noised to an aligned block start (the student's
  sampling nodes, ``block_size_min`` apart);
* one fused forward over that block (``predict_pdd_fused``, the sampler's
  step) predicts the block's mean native velocity ``eps - x0``;
* the data term regresses it onto the straight-path velocity
  ``noise - real``;
* an anchor term pulls it toward the same forward with every LoRA layer
  disabled, weighted from ``anchor_weight_max`` at the noisiest block down to
  ``anchor_weight_min`` at the cleanest. Plain regression is ill-posed at high
  noise (its optimum is the posterior mean, which blurs), so the anchor keeps
  the distilled behaviour there and lets the data win near the clean end.

No teacher or critic role is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from fastvideo.dataset.sequence_capacity import sequence_capacity_metrics
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.trajectory_matching.pdd import PDDMethodMixin
from fastvideo.train.methods.trajectory_matching.pdd_core import (
    aligned_block_starts,
    head_losses,
    latent_loss_valid_mask,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler


@contextmanager
def lora_disabled(transformer: torch.nn.Module) -> Iterator[int]:
    """Run the wrapped forwards with every LoRA layer bypassed; yields the layer count."""
    layers = [module for module in transformer.modules() if isinstance(module, BaseLayerWithLoRA)]
    previous = [layer.disable_lora for layer in layers]
    for layer in layers:
        layer.disable_lora = True
    try:
        yield len(layers)
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer.disable_lora = value


def anchor_weight(block_index: int, num_blocks: int, *, high: float, low: float) -> float:
    """Linear anchor weight: ``high`` at block 0 (noisiest) to ``low`` at the last (cleanest)."""
    if num_blocks == 1:
        return high
    return high + (low - high) * block_index / (num_blocks - 1)


class PDDLoraFinetuneMethod(PDDMethodMixin, TrainingMethod):
    """Data plus anchor regression of a LoRA on a PDD student's fused sampling blocks."""

    # Only the adapters train; their small updates still need FP32 storage.
    requires_fp32_master_weights = True

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if set(role_models) != {"student"}:
            raise ValueError(f"PDDLoraFinetuneMethod uses only role 'student', got {sorted(role_models)}")
        if not self.student._trainable:
            raise ValueError("PDDLoraFinetuneMethod requires a trainable student")
        if not any(isinstance(module, BaseLayerWithLoRA) for module in self.student.transformer.modules()):
            raise ValueError("PDDLoraFinetuneMethod requires a LoRA student (models.student.lora.enable: true)")
        self._cfg_uncond = None
        self._init_pdd(carried=False)
        if self.pdd_config.block_size_min != self.pdd_config.block_size_max:
            raise ValueError("PDDLoraFinetuneMethod trains the sampler's fixed blocks: set "
                             "method.block_size_min equal to method.block_size_max")
        if self.pdd_steps % self.pdd_config.block_size_min:
            raise ValueError(f"block size {self.pdd_config.block_size_min} must divide pdd_steps={self.pdd_steps}")
        self.num_blocks = self.pdd_steps // self.pdd_config.block_size_min
        self.data_weight = float(self.method_config.get("data_weight", 1.0))
        self.anchor_weight_max = float(self.method_config.get("anchor_weight_max", 1.0))
        self.anchor_weight_min = float(self.method_config.get("anchor_weight_min", 0.0))
        for name in ("data_weight", "anchor_weight_max", "anchor_weight_min"):
            if getattr(self, name) < 0:
                raise ValueError(f"method.{name} must be non-negative")

        self.student.init_preprocessors(self.training_config)
        tc = self.training_config
        params = [p for p in self.student.transformer.parameters() if p.requires_grad]
        self._student_optimizer, self._student_lr_scheduler = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(tc.optimizer.learning_rate),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    # -- optimizer --------------------------------------------------------------

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._student_optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._student_lr_scheduler}

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        return [self._student_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        return [self._student_lr_scheduler]

    def apply_configured_lrs(self) -> None:
        lr = float(self.training_config.optimizer.learning_rate)
        for group in self._student_optimizer.param_groups:
            group["lr"] = lr
            if "initial_lr" in group:
                group["initial_lr"] = lr
        if hasattr(self._student_lr_scheduler, "base_lrs"):
            self._student_lr_scheduler.base_lrs = [lr] * len(self._student_lr_scheduler.base_lrs)

    # -- training ---------------------------------------------------------------

    def block_inputs(
        self,
        batch: dict[str, Any],
        *,
        block_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Noise real latents to one block start: ``(batch, state, target, block_index, fine_grid)``.

        ``block_index`` fixes the block (held-out evaluation); otherwise it is
        sampled. ``target`` is the straight-path native velocity ``noise - real``.
        """
        generator = self.cuda_generator if generator is None else generator
        training_batch = self.student.prepare_batch(batch, generator=generator, latents_source="data")
        real = training_batch.latents
        device = real.device
        fine_grid = self._pdd_fine_grid(device)
        block = self.pdd_config.block_size_min
        if block_index is None:
            step_idx = aligned_block_starts(real.shape[0],
                                            pdd_steps=self.pdd_steps,
                                            block_size_min=block,
                                            device=device,
                                            generator=generator)
            if bool((step_idx != step_idx[0]).any()):
                raise ValueError("one packed document per batch must share its block")
            block_index = int(step_idx[0]) // block
        if not 0 <= block_index < self.num_blocks:
            raise ValueError(f"block_index must be in [0, {self.num_blocks}), got {block_index}")
        step_idx = torch.full((real.shape[0], ), block_index * block, device=device, dtype=torch.long)
        noise = torch.randn(real.shape, device=device, dtype=real.dtype, generator=generator)
        state = self.student.add_noise_for_batch(real, noise, self.student.pdd_timestep(fine_grid[step_idx]),
                                                 training_batch)
        target = noise.to(torch.float32) - real.to(torch.float32)
        return training_batch, state, target, block_index, fine_grid

    def fused_prediction(self, state: torch.Tensor, block_index: int, fine_grid: torch.Tensor,
                         batch: Any) -> torch.Tensor:
        """The sampler's fused forward for one block: packed mean native ``eps - x0``."""
        start = block_index * self.pdd_config.block_size_min
        end = start + self.pdd_config.block_size_min
        timestep = self.student.pdd_timestep(fine_grid[start].expand(state.shape[0]))
        return self.student.predict_pdd_fused(
            state.to(self._pdd_model_dtype),
            timestep,
            start,
            end,
            fine_grid,
            batch,
            precision_decoding=self.pdd_config.precision_decoding,
            conditional=True,
            cfg_uncond=None,
            attn_kind="vsa",
        )

    def block_losses(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        anchor: torch.Tensor,
        block_index: int,
        batch: Any,
    ) -> dict[str, torch.Tensor]:
        """Weighted data and anchor terms, plus their per-modality parts."""
        blocks = self._pdd_blocks(batch)
        valid = latent_loss_valid_mask(batch)
        data = head_losses(prediction, target, blocks, valid=valid)
        anchor_terms = head_losses(prediction, anchor, blocks, valid=valid)
        weight = anchor_weight(block_index, self.num_blocks, high=self.anchor_weight_max, low=self.anchor_weight_min)
        data_loss = sum(data.values())
        anchor_loss = sum(anchor_terms.values())
        return {
            "total_loss": self.data_weight * data_loss + weight * anchor_loss,
            "data_loss": data_loss,
            "anchor_loss": anchor_loss,
            **{f"data_loss_{name}": value
               for name, value in data.items()},
            **{f"anchor_loss_{name}": value
               for name, value in anchor_terms.items()},
        }

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        training_batch, state, target, block_index, fine_grid = self.block_inputs(batch)
        # The anchor runs first: the student forward below records the
        # timestep / VSA metadata its backward replays.
        with torch.no_grad(), lora_disabled(self.student.transformer):
            anchor = self.fused_prediction(state, block_index, fine_grid, training_batch).to(torch.float32)
        prediction = self.fused_prediction(state, block_index, fine_grid, training_batch)
        student_ctx = (training_batch.timesteps, training_batch.attn_metadata_vsa)
        loss_map = self.block_losses(prediction, target, anchor, block_index, training_batch)
        weight = anchor_weight(block_index, self.num_blocks, high=self.anchor_weight_max, low=self.anchor_weight_min)
        with torch.no_grad():
            # Sign/scale diagnostic: a sane student predicts the straight-path
            # velocity with clearly positive cosine at every block.
            cosine = torch.nn.functional.cosine_similarity(prediction.detach().float().flatten(),
                                                           target.flatten(),
                                                           dim=0)
        return loss_map, {
            "_fv_backward": {
                "student_ctx": student_ctx
            }
        }, {
            "pdd_block": float(block_index),
            "anchor_weight": weight,
            "prediction_target_cosine": float(cosine),
            **sequence_capacity_metrics(training_batch),
        }

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        self.student.backward(
            loss_map["total_loss"],
            outputs["_fv_backward"]["student_ctx"],
            grad_accum_rounds=max(1, int(grad_accum_rounds)),
        )


__all__ = ["PDDLoraFinetuneMethod", "anchor_weight", "lora_disabled"]
