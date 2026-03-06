# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LIRF training script on top of the current SiT codebase.

Implements iterative latent refinement:
1) Generation (from EMA snapshot)
2) Correction (cosine-kNN + spherical contraction)
3) Augmentation (accept if ||z - C(z)|| <= tau)
"""

import argparse
import logging
import math
import os
import re
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from augment import AugmentPipe
from download import find_model
from models import SiT_models
from train_utils import parse_transport_args, pachify
from transport import Sampler, create_transport

try:
    import swanlab
except Exception:
    swanlab = None

# A100 speedup flag (same as train.py).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint(ckpt_path):
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        checkpoint = find_model(ckpt_path)
    if isinstance(checkpoint, dict) and "model" in checkpoint and "ema" in checkpoint:
        return checkpoint
    return {"model": checkpoint, "ema": checkpoint, "opt": None, "args": None}


def parse_checkpoint_step(ckpt_path):
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else 0


def resolve_experiment_dirs(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    if os.path.basename(ckpt_dir) == "checkpoints":
        return os.path.dirname(ckpt_dir), ckpt_dir
    return ckpt_dir, ckpt_dir


def resolve_latent_settings(args):
    if args.train_mode == "dino":
        if args.vae_path is None:
            args.vae_path = "Dino_VAE"
        if args.latent_scale is None:
            args.latent_scale = 1.0
    else:
        if args.vae_path is None:
            args.vae_path = "vae"
        if args.latent_scale is None:
            args.latent_scale = 0.18215
    return args


def load_anchor_bank(path, latent_scale, are_scaled):
    latents = torch.load(path, map_location="cpu")
    if not torch.is_tensor(latents):
        raise ValueError(f"Expected tensor at {path}, got {type(latents)}")
    latents = latents.float().contiguous()
    if not are_scaled:
        latents = latents * latent_scale
    return latents


def sample_anchor_batch(anchor_bank_cpu, batch_size, device):
    idx = torch.randint(0, anchor_bank_cpu.shape[0], (batch_size,))
    x = anchor_bank_cpu[idx].to(device, non_blocking=True)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)
    return x, y


def batch_slerp_unit(a, b, lam):
    # a,b: [B,D], unit vectors.
    dot = (a * b).sum(dim=1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6

    s0 = torch.sin((1 - lam) * theta) / (sin_theta + 1e-8)
    s1 = torch.sin(lam * theta) / (sin_theta + 1e-8)
    out = s0 * a + s1 * b
    out = torch.where(small, ((1 - lam) * a + lam * b), out)
    out = F.normalize(out, dim=1, eps=1e-8)
    return out


@torch.no_grad()
def generate_candidate_latents(
    ema_model,
    sampler,
    num_candidates,
    latent_size,
    augment_dim,
    device,
    sampling_method,
    num_sampling_steps,
    atol,
    rtol,
):
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=num_sampling_steps,
        atol=atol,
        rtol=rtol,
        reverse=False,
    )

    z0 = torch.randn(num_candidates, 4, latent_size, latent_size, device=device)
    y = torch.zeros(num_candidates, dtype=torch.long, device=device)
    _, coords = pachify(z0, latent_size, latent_size)
    model_kwargs = dict(y=y, coords=coords)
    if augment_dim > 0:
        model_kwargs["aug_label"] = torch.zeros(num_candidates, augment_dim, device=device)
    return sample_fn(z0, ema_model.forward, **model_kwargs)[-1]


@torch.no_grad()
def correct_and_filter_candidates(
    candidates,
    anchor_bank_cpu,
    k_neighbors,
    lam,
    tau,
    device,
):
    # candidates: [M,C,H,W], already scaled latents.
    # anchor_bank_cpu: [N,C,H,W], scaled latents.
    bank = anchor_bank_cpu.to(device, non_blocking=True)
    m = candidates.shape[0]
    n = bank.shape[0]
    k = min(max(1, k_neighbors), n)

    cand_flat = candidates.view(m, -1)
    bank_flat = bank.view(n, -1)
    cand_norm = F.normalize(cand_flat, dim=1, eps=1e-8)
    bank_norm = F.normalize(bank_flat, dim=1, eps=1e-8)

    sim = torch.matmul(cand_norm, bank_norm.t())  # [M,N]
    topk_vals, topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
    weights = torch.softmax(topk_vals, dim=1)  # [M,K]

    neighbors = bank[topk_idx]  # [M,K,C,H,W]
    weighted_neighbors = (neighbors * weights[:, :, None, None, None]).sum(dim=1)
    u = F.normalize(weighted_neighbors.view(m, -1), dim=1, eps=1e-8)

    neighbor_norms = neighbors.view(m, k, -1).norm(dim=2)
    r_bar = (weights * neighbor_norms).sum(dim=1)  # [M]

    z_norm = cand_flat.norm(dim=1)  # [M]
    z_hat = F.normalize(cand_flat, dim=1, eps=1e-8)
    dir_corr = batch_slerp_unit(z_hat, u, lam)
    radial = (1 - lam) * z_norm + lam * r_bar
    corrected = (dir_corr * radial[:, None]).view_as(candidates)

    delta = (candidates - corrected).view(m, -1).norm(dim=1)
    accept_mask = delta <= tau
    accepted = corrected[accept_mask].detach().cpu()

    stats = {
        "num_candidates": int(m),
        "num_accepted": int(accept_mask.sum().item()),
        "accept_rate": float(accept_mask.float().mean().item()),
        "delta_mean": float(delta.mean().item()),
        "delta_max": float(delta.max().item()),
    }
    return accepted, stats


def lambda_schedule(round_idx, total_rounds, start_lam, end_lam):
    if total_rounds <= 1:
        return float(start_lam)
    progress = (round_idx - 1) / (total_rounds - 1)
    return float(start_lam + progress * (end_lam - start_lam))


def main(args):
    assert torch.cuda.is_available(), "train_lirf.py requires CUDA."
    if args.refine_every <= 0:
        raise ValueError("--refine-every must be > 0")
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be > 0")
    args = resolve_latent_settings(args)

    # DDP init.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    assert args.global_batch_size % world_size == 0, "global batch must be divisible by world size."
    local_batch_size = args.global_batch_size // world_size
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Resume handling.
    is_resume = args.resume
    resume_state = None
    train_steps = 0
    refinement_round = 0
    if is_resume:
        if args.ckpt is None:
            raise ValueError("--resume requires --ckpt.")
        resume_state = load_checkpoint(args.ckpt)
        train_steps = int(resume_state.get("train_steps", parse_checkpoint_step(args.ckpt)))
        refinement_round = int(resume_state.get("refinement_round", train_steps // args.refine_every))

    # Experiment dirs.
    if rank == 0:
        if is_resume:
            experiment_dir, checkpoint_dir = resolve_experiment_dirs(args.ckpt)
            experiment_name = os.path.basename(experiment_dir)
            os.makedirs(experiment_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            vis_dir = os.path.join(experiment_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming from {args.ckpt} at step {train_steps}")
        else:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_name = args.model.replace("/", "-")
            experiment_name = f"{experiment_index:03d}-{model_name}-LIRF-{args.path_type}-{args.prediction}-{args.loss_weight}"
            experiment_dir = f"{args.results_dir}/{experiment_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            vis_dir = os.path.join(experiment_dir, "vis")
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        if args.swanlab:
            if swanlab is None:
                logger.info("[swanlab] package not found, disabling logging.")
                args.swanlab = False
        if args.swanlab:
            entity = os.environ.get("ENTITY")
            project = os.environ.get("PROJECT", "SiT_LIRF")
            try:
                init_kwargs = dict(
                    project=project,
                    experiment_name=experiment_name,
                    config=vars(args),
                )
                if entity:
                    init_kwargs["workspace"] = entity
                swanlab.init(**init_kwargs)
            except Exception as exc:
                logger.info(f"[swanlab] init failed, disabling logging: {exc}")
                args.swanlab = False
    else:
        logger = create_logger(None)
        args.swanlab = False

    assert args.image_size % 8 == 0, "image size must be divisible by 8."
    latent_size = args.image_size // 8

    # Anchor bank.
    if (
        is_resume
        and resume_state is not None
        and isinstance(resume_state, dict)
        and "anchor_bank" in resume_state
        and torch.is_tensor(resume_state["anchor_bank"])
    ):
        anchor_bank_cpu = resume_state["anchor_bank"].float().cpu().contiguous()
        logger.info(f"Loaded anchor bank from checkpoint: {anchor_bank_cpu.shape[0]} samples")
    else:
        anchor_bank_cpu = load_anchor_bank(
            args.anchor_latents_path,
            latent_scale=args.latent_scale,
            are_scaled=args.anchor_latents_are_scaled,
        )
        logger.info(f"Loaded initial anchor bank: {anchor_bank_cpu.shape[0]} samples from {args.anchor_latents_path}")
    assert anchor_bank_cpu.ndim == 4 and anchor_bank_cpu.shape[1] == 4, "anchor latents must be [N,4,H,W]."
    assert anchor_bank_cpu.shape[2] == latent_size and anchor_bank_cpu.shape[3] == latent_size, (
        f"anchor latent size mismatch: got {anchor_bank_cpu.shape[2:]} but expected {(latent_size, latent_size)}"
    )

    # Augmentation.
    augment_pipe = None
    augment_dim = 0
    if args.augment > 0:
        augment_pipe = AugmentPipe(
            p=args.augment,
            xflip=0,
            yflip=1,
            scale=1,
            rotate_frac=1,
            aniso=1,
            translate_frac=1,
        )
        augment_dim = 8

    # Model / EMA / optimizer / transport.
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        augment_dim=augment_dim,
    )
    ema = deepcopy(model).to(device)
    model = DDP(model.to(device), device_ids=[device])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    transport_sampler = Sampler(transport)

    # Optional initialization ckpt (non-resume) or resume restore.
    if args.ckpt is not None:
        state = resume_state if is_resume else load_checkpoint(args.ckpt)
        if isinstance(state, dict) and "model" in state:
            model.module.load_state_dict(state["model"])
            ema.load_state_dict(state["ema"])
            if is_resume and state.get("opt") is not None:
                opt.load_state_dict(state["opt"])
                for g in opt.param_groups:
                    g["weight_decay"] = 0.05
        else:
            model.module.load_state_dict(state)
            ema.load_state_dict(state)
    else:
        update_ema(ema, model.module, decay=0.0)

    requires_grad(ema, False)
    model.train()
    ema.eval()

    vae = None
    vis_z = None
    vis_model_kwargs = None
    vis_model_fn = ema.forward
    if args.sample_every > 0:
        try:
            from diffusers.models import AutoencoderKL
        except Exception as exc:
            raise RuntimeError(
                "sample_every > 0 requires diffusers. Install it or set --sample-every 0."
            ) from exc

        # VAE only for periodic visualization decoding.
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
        requires_grad(vae, False)

        # Fixed sampling noise for periodic visuals.
        vis_n = min(16, local_batch_size)
        vis_y = torch.zeros(vis_n, dtype=torch.long, device=device)
        vis_z = torch.randn(vis_n, 4, latent_size, latent_size, device=device)
        _, vis_coords = pachify(vis_z, latent_size, latent_size)
        vis_model_kwargs = dict(y=vis_y, coords=vis_coords)
        if augment_dim > 0:
            vis_model_kwargs["aug_label"] = torch.zeros(vis_n, augment_dim, device=device)

    # Total steps.
    if args.max_steps is not None and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = math.ceil(anchor_bank_cpu.shape[0] / args.global_batch_size)
        total_steps = max(1, args.epochs * steps_per_epoch)
    total_refine_rounds = max(1, total_steps // args.refine_every)

    logger.info(f"Training until step {total_steps}.")
    logger.info(
        "LIRF config: refine_every=%d, candidates=%d, k=%d, lambda:[%.3f->%.3f], tau=%.4f",
        args.refine_every,
        args.num_candidates,
        args.k_neighbors,
        args.lambda_start,
        args.lambda_end,
        args.tau,
    )

    running_loss = 0.0
    log_steps = 0
    start_time = time()

    if train_steps >= total_steps:
        logger.info(f"Checkpoint step {train_steps} already meets total_steps={total_steps}.")
        cleanup()
        return

    # Patch training setup (same behavior as current train.py defaults).
    p_list = np.array([0.5, 0.5])
    patch_list = np.array([latent_size // 2, latent_size])

    while train_steps < total_steps:
        step_start = time()

        if rank == 0:
            patch_tensor = torch.tensor(
                [int(np.random.choice(patch_list, p=p_list))],
                device=device,
                dtype=torch.int64,
            )
        else:
            patch_tensor = torch.zeros(1, device=device, dtype=torch.int64)
        dist.broadcast(patch_tensor, src=0)
        patch_size = int(patch_tensor.item())

        batch_mul = 2 if patch_size < latent_size else 1
        x_list, y_list = [], []
        for _ in range(batch_mul):
            xb, yb = sample_anchor_batch(anchor_bank_cpu, local_batch_size, device)
            x_list.append(xb)
            y_list.append(yb)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)

        aug_label = None
        if augment_pipe is not None:
            x, aug_label = augment_pipe(x)

        x, coords = pachify(x, patch_size, full_size=latent_size)
        model_kwargs = dict(y=y, coords=coords)
        if aug_label is not None:
            model_kwargs["aug_label"] = aug_label

        loss_dict = transport.training_losses(model, x, model_kwargs)
        loss = loss_dict["loss"]
        if args.patch_loss == "edm":
            loss = loss * (patch_size * patch_size)
        loss = loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model.module, decay=args.ema_decay)

        train_steps += 1
        running_loss += loss.item()
        log_steps += 1

        if args.swanlab and rank == 0:
            step_time = time() - step_start
            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/pix_loss": loss.item() / (patch_size * patch_size),
                    "train/steps_per_sec": 1.0 / max(step_time, 1e-8),
                    "train/anchor_size": int(anchor_bank_cpu.shape[0]),
                },
                step=train_steps,
            )

        if train_steps % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = max(time() - start_time, 1e-8)
            steps_per_sec = log_steps / elapsed
            avg_loss = torch.tensor(running_loss / max(log_steps, 1), device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size
            logger.info(
                "(step=%07d) loss=%.4f, steps/sec=%.2f, anchors=%d",
                train_steps,
                avg_loss,
                steps_per_sec,
                anchor_bank_cpu.shape[0],
            )
            running_loss = 0.0
            log_steps = 0
            start_time = time()

        # LIRF refinement round.
        if train_steps % args.refine_every == 0:
            refinement_round += 1
            lam = lambda_schedule(
                refinement_round,
                total_refine_rounds,
                args.lambda_start,
                args.lambda_end,
            )

            if rank == 0:
                logger.info(
                    "[LIRF] round=%d (step=%07d), lambda=%.4f, anchor_size_before=%d",
                    refinement_round,
                    train_steps,
                    lam,
                    anchor_bank_cpu.shape[0],
                )
                candidates = generate_candidate_latents(
                    ema_model=ema,
                    sampler=transport_sampler,
                    num_candidates=args.num_candidates,
                    latent_size=latent_size,
                    augment_dim=augment_dim,
                    device=device,
                    sampling_method=args.sampling_method,
                    num_sampling_steps=args.num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                accepted_cpu, corr_stats = correct_and_filter_candidates(
                    candidates=candidates,
                    anchor_bank_cpu=anchor_bank_cpu,
                    k_neighbors=args.k_neighbors,
                    lam=lam,
                    tau=args.tau,
                    device=device,
                )
                accepted_count = accepted_cpu.shape[0]
            else:
                accepted_cpu = None
                corr_stats = None
                accepted_count = 0

            accepted_count_tensor = torch.tensor([accepted_count], device=device, dtype=torch.int64)
            dist.broadcast(accepted_count_tensor, src=0)
            accepted_count = int(accepted_count_tensor.item())

            if accepted_count > 0:
                if rank == 0:
                    accepted_device = accepted_cpu.to(device, non_blocking=True)
                else:
                    accepted_device = torch.empty(accepted_count, 4, latent_size, latent_size, device=device)
                dist.broadcast(accepted_device, src=0)
                anchor_bank_cpu = torch.cat([anchor_bank_cpu, accepted_device.cpu()], dim=0)

            if rank == 0:
                logger.info(
                    "[LIRF] candidates=%d accepted=%d rate=%.2f%% delta_mean=%.5f tau=%.4f anchor_size_after=%d",
                    corr_stats["num_candidates"],
                    corr_stats["num_accepted"],
                    corr_stats["accept_rate"] * 100.0,
                    corr_stats["delta_mean"],
                    args.tau,
                    anchor_bank_cpu.shape[0],
                )
                if args.swanlab:
                    swanlab.log(
                        {
                            "lirf/round": refinement_round,
                            "lirf/lambda": lam,
                            "lirf/candidates": corr_stats["num_candidates"],
                            "lirf/accepted": corr_stats["num_accepted"],
                            "lirf/accept_rate": corr_stats["accept_rate"],
                            "lirf/delta_mean": corr_stats["delta_mean"],
                            "lirf/delta_max": corr_stats["delta_max"],
                            "lirf/anchor_size": int(anchor_bank_cpu.shape[0]),
                        },
                        step=train_steps,
                    )

            dist.barrier()

        # Save checkpoint.
        if train_steps % args.ckpt_every == 0:
            if rank == 0:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                    "refinement_round": refinement_round,
                    "anchor_bank": anchor_bank_cpu if args.save_anchor_in_ckpt else None,
                }
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(ckpt, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
            dist.barrier()

        # Save periodic decoded samples from EMA.
        if args.sample_every > 0 and train_steps % args.sample_every == 0:
            sample_fn = transport_sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=False,
            )
            samples = sample_fn(vis_z, vis_model_fn, **vis_model_kwargs)[-1]
            decoded = vae.decode(samples / args.latent_scale).sample

            if world_size > 1:
                gathered_shape = list(decoded.shape)
                gathered_shape[0] *= world_size
                gathered = torch.zeros(gathered_shape, device=device, dtype=decoded.dtype)
                dist.all_gather_into_tensor(gathered, decoded)
            else:
                gathered = decoded

            if rank == 0:
                save_path = os.path.join(vis_dir, f"step_{train_steps:07d}.png")
                nrow = max(1, int(gathered.shape[0] ** 0.5))
                vutils.save_image(
                    gathered.float().cpu(),
                    save_path,
                    nrow=nrow,
                    normalize=True,
                    value_range=(-1, 1),
                )
                logger.info(f"Saved samples: {save_path}")
                if args.swanlab:
                    swanlab.log(
                        {"train/samples": [swanlab.Image(str(save_path), caption=f"Step {train_steps}")]},
                        step=train_steps,
                    )
            dist.barrier()

    logger.info("Done.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SiT with LIRF latent refinement.")

    # Experiment / model.
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)

    # Train mode / latent settings.
    parser.add_argument("--train-mode", type=str, choices=["base", "dino"], default="dino")
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument(
        "--anchor-latents-path",
        type=str,
        default="data/latents/train_latents_xflip_100.pt",
    )
    parser.add_argument(
        "--anchor-latents-are-scaled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-anchor-in-ckpt", action=argparse.BooleanOptionalAction, default=True)

    # Logging / checkpoint.
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    # Patch + augmentation (keep same behavior as current train.py).
    parser.add_argument("--augment", type=float, default=0.12)
    parser.add_argument("--patch-loss", type=str, choices=["mean", "edm"], default="edm")

    # Transport / sampling.
    parse_transport_args(parser)
    parser.add_argument("--sampling-method", type=str, default="dopri5")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)

    # LIRF controls (paper-aligned defaults from your request).
    parser.add_argument("--refine-every", type=int, default=10000)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--lambda-start", type=float, default=0.8)
    parser.add_argument("--lambda-end", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.1)

    parser.set_defaults(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=0.01,
        sample_eps=0.01,
    )
    main(parser.parse_args())
