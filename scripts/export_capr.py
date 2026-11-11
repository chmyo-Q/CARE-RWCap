#!/usr/bin/env python3
"""Export CAPR FP32 TorchScript and build FP16 engines for the current GPU."""
import argparse
from pathlib import Path
from common import ROOT, dump


def check_output(torch, source, candidate, x, rtol, atol):
    expected, actual = source(x), candidate(x)
    if not torch.isfinite(actual).all():
        raise RuntimeError("Nonfinite exported output")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return float((actual - expected).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/checkpoints/capr/best.pt")
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    parser.add_argument(
        "--torchscript-only", action="store_true", help="Export and check FP32 on CPU"
    )
    parser.add_argument(
        "--all-models", action="store_true", help="Also rebuild the five DeepRWCap engines"
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=512)
    parser.add_argument("--max-batch", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("Expected 1 <= min-batch <= opt-batch <= max-batch")
    import torch
    from capr.model import CAPRPredictor

    device = "cpu" if args.torchscript_only else "cuda:0"
    if not args.torchscript_only and not torch.cuda.is_available():
        parser.error("A CUDA GPU is needed to build TensorRT engines")
    model = CAPRPredictor().eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(2029)
    scripted = torch.jit.script(model)
    with (args.output / "PoissonPredictor_fp32.jit").open("wb") as stream:
        torch.jit.save(scripted, stream)
    results = {"torch": torch.__version__, "checkpoint": str(args.checkpoint), "fp32_checks": []}
    with torch.inference_mode():
        for batch in sorted({args.min_batch, min(args.opt_batch, 8)}):
            x = torch.rand(batch, 1, 1, 23, 23, 23, device=device)
            results["fp32_checks"].append(check_output(torch, model, scripted, x, 1e-5, 1e-7))
    if not args.torchscript_only:
        import torch_tensorrt
        import tensorrt

        results.update(
            gpu=torch.cuda.get_device_name(),
            torch_tensorrt=torch_tensorrt.__version__,
            tensorrt=tensorrt.__version__,
            batch_range=[args.min_batch, args.opt_batch, args.max_batch],
        )
        sources = {"capr/PoissonPredictor": scripted}
        if args.all_models:
            from common import NAMES

            for name in NAMES:
                with (ROOT / "models/checkpoints/deeprwcap" / f"{name}_best.jit").open(
                    "rb"
                ) as stream:
                    sources["deeprwcap/" + name] = torch.jit.load(
                        stream, map_location=device
                    ).eval()
        results["engines"] = {}
        with torch.inference_mode():
            for name, source in sources.items():
                compiled = torch_tensorrt.compile(
                    source,
                    ir="ts",
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(args.min_batch, 1, 1, 23, 23, 23),
                            opt_shape=(args.opt_batch, 1, 1, 23, 23, 23),
                            max_shape=(args.max_batch, 1, 1, 23, 23, 23),
                            dtype=torch.float32,
                        )
                    ],
                    enabled_precisions={torch.float16},
                    require_full_compilation=True,
                    workspace_size=2 << 30,
                    min_block_size=1,
                    allow_shape_tensors=False,
                    truncate_long_and_double=True,
                )
                path = args.output / (name + "_tensorrt_fp16.jit")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.jit.save(compiled, str(path))
                reloaded = torch.jit.load(str(path), map_location=device).eval()
                gaps = []
                for batch in sorted({args.min_batch, args.opt_batch, args.max_batch}):
                    x = torch.rand(batch, 1, 1, 23, 23, 23, device=device)
                    gaps.append(check_output(torch, source, reloaded, x, 0.05, 5e-4))
                results["engines"][name] = {"file": str(path), "max_abs_gaps": gaps}
    dump(args.output / "export.json", results)
    print(f"Exported and checked models in {args.output}")


if __name__ == "__main__":
    main()
