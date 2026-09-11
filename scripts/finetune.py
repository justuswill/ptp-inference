"""
Fine-tune a frozen PTP checkpoint's new P-head: predicts p per call from the
model's own AR context representation, trained via a censored Geometric NLL
against the observed #correct-per-call (see src/ptp/p_head.py). All existing
model weights (base + LoRA + u_embed) are frozen; only the head trains.

Usage
-----
    uv run scripts/finetune.py <experiment_dir> [options]

Example
-------
    uv run scripts/finetune.py \\
        /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna \\
        --max-steps 2000 --lr 1e-3
"""
from __future__ import annotations

import contextlib
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

# Vicuna v1.1-style chat template, matching EXPERIMENT_CONFIGS["vicuna"]["chat_template"]
# in scripts/inference.py — used as a fallback when the checkpoint's tokenizer has none.
VICUNA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + ' ' }}"
    "{% set messages = messages[1:] %}{% else %}"
    "{{ 'A chat between a curious user and an artificial intelligence assistant."
    " The assistant gives helpful, detailed, and polite answers to the user\\'s questions. ' }}"
    "{% endif %}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)


HEAD_TARGETS = {
    "p": "ptp.p_head.PHeadLightningModule",
    "c": "ptp.p_head.CHeadLightningModule",
}


def load_lit_model(experiment_dir: Path, checkpoint: Path | None, head_type: str = "p",
                    self_mode: bool = False, nucleus_threshold: float | None = None):
    from omegaconf import OmegaConf
    from ptp.cli.generate import find_best_checkpoint
    from ptp.utils import instantiate

    with open(experiment_dir / "train.yaml") as f:
        config = DictConfig(yaml.safe_load(f))

    ckpt_dir = Path(config["training"].get("ckpt_dir", experiment_dir))
    if checkpoint is not None and not checkpoint.is_absolute() and not checkpoint.exists():
        checkpoint = ckpt_dir / checkpoint
    ckpt_path = checkpoint or find_best_checkpoint(ckpt_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    # Keep the default attn_implementation (flex_attention): unlike scripts/inference.py's
    # single-document inference batches, ChatDataModule(mode="full") packs multiple
    # conversations per sequence and needs the doc-isolation BlockMask, which sdpa can't
    # consume (TypeError: attn_mask must be Tensor, not BlockMask).

    # to_object recursively converts nested DictConfigs to plain dicts, matching what
    # instantiate() does internally — needed so the nested `model:` (MixedTransformerModel)
    # sub-config is still recognized (isinstance(..., dict)) after we edit the top level.
    model_cfg = OmegaConf.to_object(config["model"])
    model_cfg["_target_"] = HEAD_TARGETS[head_type]
    model_cfg["completion_loss_weight"] = 0.0
    model_cfg["self_mode"] = self_mode
    model_cfg["nucleus_threshold"] = nucleus_threshold
    lit_model = instantiate(model_cfg)
    lit_model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit_model.load_state_dict(ckpt["state_dict"])

    if head_type == "p":
        lit_model.add_p_head()
    else:
        lit_model.add_c_head()
    lit_model.freeze_base()

    precision = config["training"].get("precision", "32-true")
    if "bf16" in str(precision):
        autocast_dtype = torch.bfloat16
    elif "16" in str(precision):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    return lit_model, config, autocast_dtype, ckpt_path


def build_datamodule(args, tokenizer_id: str):
    from ptp.data.chat import ChatDataModule

    return ChatDataModule(
        dataset_name=args.dataset,
        tokenizer=tokenizer_id,
        max_sequence_length=args.max_sequence_length,
        mode="full",
        chat_template=args.chat_template,
        conversation_keys=args.conversation_keys,
        num_completions=args.num_completions,
        train_completion_len=args.train_completion_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def check_gradient_isolation(lit_model, head_attr: str) -> None:
    prefix = f"{head_attr}."
    bad = [n for n, p in lit_model.named_parameters()
           if not n.startswith(prefix) and p.grad is not None]
    if bad:
        raise RuntimeError(f"Gradients leaked into frozen parameters: {bad[:5]}"
                            f"{' ...' if len(bad) > 5 else ''}")
    n_head_grad = sum(1 for n, p in lit_model.named_parameters()
                       if n.startswith(prefix) and p.grad is not None)
    tqdm.write(f"Gradient isolation OK: only {n_head_grad} {head_attr} parameter(s) have gradients.")


def _f(x) -> float:
    return x.item() if isinstance(x, torch.Tensor) else float(x)


def main(args):
    head_attr = "p_head" if args.head_type == "p" else "c_head"
    loss_key = "p_loss" if args.head_type == "p" else "c_loss"
    ckpt_key = f"{head_attr}_state_dict"
    variant_tag = ("_self" if args.self_mode else "") + \
        (f"_nt{args.nucleus_threshold}" if args.nucleus_threshold is not None else "")
    tag = f"{args.head_type}head_ft{variant_tag}"

    lit_model, config, autocast_dtype, ckpt_path = load_lit_model(
        args.experiment_dir, args.checkpoint, head_type=args.head_type,
        self_mode=args.self_mode, nucleus_threshold=args.nucleus_threshold)
    if args.head_type == "p":
        lit_model.p_loss_weight = args.p_loss_weight
    else:
        lit_model.c_loss_weight = args.p_loss_weight
    head = getattr(lit_model, head_attr)

    if args.resume is not None:
        sidecar = torch.load(args.resume, map_location="cpu")
        head.load_state_dict(sidecar[ckpt_key])
        print(f"Resumed {head_attr} from {args.resume} (step {sidecar.get('step')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.train()

    tokenizer_id = config["data"]["tokenizer"] if "data" in config and "tokenizer" in config["data"] \
        else config["model"]["model"]["model_id"]
    datamodule = build_datamodule(args, tokenizer_id)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    if args.output_dir is None:
        args.output_dir = ckpt_path.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = ckpt_path.stem

    autocast_ctx = (
        torch.autocast(device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )

    base_wandb_project = config["training"].get("wandb_project", "ptp")
    is_offline_project = base_wandb_project == "offline"
    wandb_project = args.wandb_project or (base_wandb_project if is_offline_project
                                            else f"{base_wandb_project}-p-head")
    variant_suffix = ("-self" if args.self_mode else "") + \
        (f"-nt{args.nucleus_threshold}" if args.nucleus_threshold is not None else "")
    run_id = (f"{args.experiment_dir.name}-{args.head_type}head-"
              f"b{args.batch_size}-c{args.num_completions}-s{args.max_steps}{variant_suffix}")
    wandb.init(
        project=wandb_project,
        id=run_id,
        resume="allow",
        name=run_id,
        mode="offline" if (args.offline or is_offline_project
                           or os.environ.get("WANDB_MODE") == "offline") else "online",
        config=dict({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    base_checkpoint=str(ckpt_path)),
    )

    def save_sidecar(path: Path, step: int):
        torch.save({
            ckpt_key: head.state_dict(),
            "base_checkpoint": str(ckpt_path),
            "completion_length": args.train_completion_len,
            "step": step,
        }, path)

    train_iter = iter(train_loader)
    pbar = tqdm(range(args.max_steps), desc=f"finetune {head_attr}")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch(batch, device)

        optimizer.zero_grad()
        with autocast_ctx:
            metrics = lit_model.forward(batch, step)
        metrics["loss"].backward()

        if step == 0:
            check_gradient_isolation(lit_model, head_attr)

        optimizer.step()

        if step % args.log_every == 0:
            log = {
                "train/loss": _f(metrics["loss"]),
                f"train/{loss_key}": _f(metrics.get(loss_key, float("nan"))),
                "train/correct": _f(metrics.get("correct", float("nan"))),
            }
            if args.head_type == "p":
                log["train/p_mean"] = _f(metrics.get("p_mean", float("nan")))
                log["train/p_valid_frac"] = _f(metrics.get("p_valid_frac", float("nan")))
            wandb.log(log, step=step)
            pbar.set_postfix(loss=log["train/loss"])

        if val_loader is not None and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            lit_model.eval()
            with torch.no_grad():
                val_batch = move_batch(next(iter(val_loader)), device)
                with autocast_ctx:
                    val_metrics = lit_model.forward(val_batch, step, eval=True)
                wandb.log({
                    "val/loss": _f(val_metrics["loss"]),
                    f"val/{loss_key}": _f(val_metrics.get(loss_key, float("nan"))),
                }, step=step)
            lit_model.train()

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_sidecar(args.output_dir / f"{ckpt_stem}_{tag}_step{step}.ckpt", step)

    final_path = args.output_dir / f"{ckpt_stem}_{tag}.ckpt"
    save_sidecar(final_path, args.max_steps)
    print(f"Done. Saved final {head_attr} to {final_path}")
    wandb.finish()


def _parse_args():
    parser = ArgumentParser(
        description="Fine-tune a frozen PTP checkpoint's reward head: PHead (--head-type p, "
                    "default) predicts a scalar p for the 1 + Geometric(p) model; CHead "
                    "(--head-type c) predicts a full non-parametric distribution over #correct."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="RyokoAI/ShareGPT52K",
                        help="HF dataset id or local .json path (default: RyokoAI/ShareGPT52K)")
    parser.add_argument("--conversation-keys", type=str, nargs="+", default=["conversations", "data"])
    parser.add_argument("--chat-template", type=str, default=None,
                        help="Jinja2 chat template; defaults to the Vicuna v1.1 template if the "
                             "checkpoint's tokenizer has none")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--num-completions", type=int, default=128)
    parser.add_argument("--train-completion-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-type", choices=["p", "c"], default="p",
                        help="'p' (default): PHead predicts a scalar p for the 1+Geometric(p) "
                             "model. 'c': CHead predicts a full non-parametric categorical "
                             "distribution over #correct in {0,...,20} directly.")
    parser.add_argument("--p-loss-weight", type=float, default=1.0,
                        help="Weight on the head's loss term (applies to either head type).")
    parser.add_argument("--self-mode", action="store_true", default=False,
                        help="Apply LoRA to the context/backlog tokens too when computing the "
                             "AR pass (ar_forward) that feeds both the bin edges and the "
                             "p_head/c_head's context hidden state, not just the aux completion "
                             "window -- self-speculative training, matching inference's "
                             "_full_lora_mode / '...-self' variants (verify against the model's "
                             "own LoRA-adapted prediction instead of a plain base-model pass).")
    parser.add_argument("--nucleus-threshold", type=float, default=None,
                        help="If set, a completion token counts as 'correct' (feeding "
                             "correct_counts, and therefore the geometric/categorical head "
                             "loss) when it falls within the model's own top-p nucleus at that "
                             "position, not just on exact argmax match -- mirrors inference's "
                             "nucleus-acceptance ('...-top-p-...') variants.")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save the head sidecar checkpoint(s); defaults to the "
                             "loaded checkpoint's own directory, named "
                             "<ckpt_stem>_{p,c}head_ft.ckpt depending on --head-type")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Defaults to '<train.yaml wandb_project>-p-head'")
    parser.add_argument("--offline", action="store_true", default=False,
                        help="Disable wandb online logging.")
    args = parser.parse_args()

    args.experiment_dir = args.experiment_dir.resolve()
    if not args.experiment_dir.exists():
        parser.error(f"Experiment directory not found: {args.experiment_dir}")
    if not (args.experiment_dir / "train.yaml").exists():
        parser.error(f"train.yaml not found in {args.experiment_dir}")
    if args.chat_template is None:
        args.chat_template = VICUNA_CHAT_TEMPLATE
    # args.output_dir defaults to the checkpoint's own directory; resolved in main()
    # once the checkpoint path is known (see load_lit_model / find_best_checkpoint).

    main(args)


if __name__ == "__main__":
    _parse_args()
