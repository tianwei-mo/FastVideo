# SPDX-License-Identifier: Apache-2.0
"""Held-out per-block evaluation of a PDD LoRA finetune run.

For every eval row and every sampling block, the row's latents are noised to
the block start with a seed fixed by (row index, block), and the student's
fused prediction is scored against the straight-path velocity. With a LoRA
checkpoint the same inputs are scored with the adapters on and off, so the
per-block effect of training is measured at matched timesteps. Predictions
can be saved and compared against another run's (for example another VSA
tile size or sparsity) as per-block relative errors.

Usage::

    torchrun --nproc_per_node=4 -m fastvideo.train.entrypoint.pdd_block_eval \
        --config run.yaml --data-path eval/encoded --out results.jsonl \
        [--checkpoint runs/x/checkpoints/checkpoint-120] \
        [--save-predictions preds/] [--compare-to other-preds/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _relative_error(prediction: torch.Tensor, reference: torch.Tensor, blocks: Any) -> dict[str, float]:
    errors = {}
    for block in blocks:
        p = prediction[..., block.columns].float()
        r = reference[..., block.columns].float()
        errors[block.name] = float((p - r).norm() / r.norm().clamp_min(1e-12))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", help="DCP training checkpoint whose LoRA is evaluated")
    parser.add_argument("--save-predictions", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.methods.fine_tuning.pdd_lora_finetune import lora_disabled
    from fastvideo.train.methods.trajectory_matching.pdd_core import head_losses, latent_loss_valid_mask
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.checkpoint import CheckpointConfig, CheckpointManager
    from fastvideo.train.utils.config import load_run_config

    overrides = ["--training.data.data_path", args.data_path, "--training.checkpoint.resume_from_checkpoint", ""]
    cfg = load_run_config(args.config, overrides=overrides)
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    _, method, dataloader, _ = build_from_config(cfg)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    method.on_train_start()
    if args.checkpoint:
        # Restore weights only: the training dataloader position would skip eval rows.
        manager = CheckpointManager(
            method=method,
            dataloader=None,
            output_dir=tc.checkpoint.output_dir,
            config=CheckpointConfig(
                save_steps=0,
                keep_last=0,
                start_step=0,
                save_inference_on_validation=False,
                inference_role=tc.checkpoint.inference_checkpoint_role,
                inference_dtype=tc.checkpoint.inference_checkpoint_dtype,
                inference_checkpoints_total_limit=0,
                require_complete_training_checkpoint=False,
            ),
        )
        step = manager.maybe_resume(resume_from_checkpoint=args.checkpoint)
        logger.info("Evaluating LoRA from %s (step %s)", args.checkpoint, step)
    transformer = method.student.transformer
    transformer.eval()
    device = method.student.device
    if args.save_predictions and rank == 0:
        args.save_predictions.mkdir(parents=True, exist_ok=True)

    # One pass: as many batches as there are encoded rows (one row per shard file).
    num_rows = sum(1 for path in Path(args.data_path).rglob("*.parquet") if "map_style_cache" not in path.parts)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= num_rows:
                break
            for block in range(method.num_blocks):
                generator = torch.Generator(device=device).manual_seed(args.seed + 1000 * index + block)
                training_batch, state, target, block_index, fine_grid = method.block_inputs(
                    batch, block_index=block, generator=generator)
                blocks = method._pdd_blocks(training_batch)
                valid = latent_loss_valid_mask(training_batch)
                prediction = method.fused_prediction(state, block_index, fine_grid, training_batch).float()
                row: dict[str, Any] = {"index": index, "block": block}
                row.update({f"data_loss_{k}": float(v)
                            for k, v in head_losses(prediction, target, blocks, valid=valid).items()})
                if args.checkpoint:
                    with lora_disabled(transformer):
                        base = method.fused_prediction(state, block_index, fine_grid, training_batch).float()
                    row.update({f"base_data_loss_{k}": float(v)
                                for k, v in head_losses(base, target, blocks, valid=valid).items()})
                    row.update({f"lora_delta_rel_{k}": v for k, v in _relative_error(prediction, base, blocks).items()})
                if args.save_predictions and rank == 0:
                    torch.save(prediction.to(torch.float16).cpu(), args.save_predictions / f"{index:04d}-{block}.pt")
                if args.compare_to:
                    reference = torch.load(args.compare_to / f"{index:04d}-{block}.pt").to(device)
                    row.update({f"rel_err_{k}": v for k, v in _relative_error(prediction, reference, blocks).items()})
                rows.append(row)
            if rank == 0:
                logger.info("block eval row %d done", index)
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
        logger.info("Wrote %d block evaluations to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
