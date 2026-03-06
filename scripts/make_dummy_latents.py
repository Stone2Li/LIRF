#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Create dummy latent anchors for smoke tests.")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--out", type=str, default="data/latents/train_latents_xflip_100.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    z = torch.randn(args.num, 4, args.latent_size, args.latent_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, out_path)
    print(f"Saved dummy anchors to {out_path} with shape={tuple(z.shape)}")


if __name__ == "__main__":
    main()
