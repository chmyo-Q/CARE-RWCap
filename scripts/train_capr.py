#!/usr/bin/env python3
"""Train the CAPR adapter with a frozen DeepRWCap Poisson anchor."""
import argparse
import csv
import json
from pathlib import Path
import random

from common import ROOT, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=ROOT / "models/checkpoints/deeprwcap/PoissonPredictor_best.pt",
    )
    parser.add_argument("--output", type=Path, required=True, help="New training directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-samples", type=int, default=10000)
    parser.add_argument("--expected-samples", type=int, default=100000)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from capr.data import PoissonDataset, normalize_input, split_indices
    from capr.model import build_model
    from capr.loss import capr_loss, components

    recipe = json.loads((ROOT / "configs/capr_training_recipe.json").read_text())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = PoissonDataset(args.dataset)
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {len(dataset)}")
    train_ids, val_ids = split_indices(
        len(dataset), args.validation_samples, recipe["dataset"]["split_seed"]
    )
    device = torch.device(args.device)
    model = build_model(args.anchor).to(device)
    model.max_log_dose = recipe["loss"]["max_log_dose"]
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=recipe["optimizer"]["learning_rate"],
        weight_decay=recipe["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=recipe["scheduler"]["eta_min"]
    )
    validation = DataLoader(
        Subset(dataset, val_ids),
        batch_size=recipe["validation_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    dump(
        args.output / "split.json",
        {
            "seed": recipe["dataset"]["split_seed"],
            "splits": {"train": train_ids, "validation": val_ids},
        },
    )
    dump(
        args.output / "configuration.json",
        {
            "recipe": recipe,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "torch": torch.__version__,
        },
    )
    best, best_epoch = float("inf"), None
    history = []
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        training = DataLoader(
            Subset(dataset, train_ids),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model.train()
        total, count = 0.0, 0
        for x, target in training:
            x, target = normalize_input(x.to(device)), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            anchor, _, _ = model.components(x)
            loss = capr_loss(pred, target, anchor)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), recipe["gradient_norm_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        scheduler.step()
        model.eval()
        metrics = [[], [], []]
        with torch.no_grad():
            for x, target in validation:
                x, target = normalize_input(x.to(device)), target.to(device)
                pred = model(x)
                anchor, _, _ = model.components(x)
                for values, metric in zip(metrics, components(pred, target, anchor)):
                    values.append(metric.cpu())
        kl, action, trust = [torch.cat(values) for values in metrics]
        # Match the original FP32 aggregation order for checkpoint selection.
        score = float(kl.mean()) + 10 * float(action.mean()) + 2 * float(trust.mean())
        score += 0.5 * (float(torch.quantile(kl, 0.95)) + 10 * float(torch.quantile(action, 0.95)))
        if not np.isfinite(score):
            raise RuntimeError("Nonfinite validation score")
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "validation_kl": float(kl.mean()),
            "validation_action": float(action.mean()),
            "validation_trust": float(trust.mean()),
            "selection_score": score,
        }
        history.append(row)
        if score < best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), args.output / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            args.output / "last.pt",
        )
        with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        dump(
            args.output / "summary.json",
            {"completed_epochs": epoch, "best_epoch": best_epoch, "best_validation_score": best},
        )
        print(f"Epoch {epoch}/{args.epochs}: validation score={score:.8g}", flush=True)


if __name__ == "__main__":
    main()
