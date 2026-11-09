import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from utils import init_env  # noqa: F401 - registers the bundled diffusion package
from utils.import_utils import get_obj_from_str, instantiate_from_config, recurse_instantiate_from_config
from utils.init_utils import add_args
from utils.train_utils import set_random_seed


class DenoiserStep(torch.nn.Module):
    """Expose one fixed-shape denoising step to the FLOP analyzer."""

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, noisy_mask, log_snr, image):
        return self.denoiser.sample_unet(noisy_mask, log_snr, image)


def scaled_dot_product_attention_macs(inputs, outputs):
    """Count QK^T and attention-value matrix multiplications (one MAC = one op)."""

    query_shape = inputs[0].type().sizes()
    key_shape = inputs[1].type().sizes()
    value_shape = inputs[2].type().sizes()
    if not query_shape or not key_shape or not value_shape:
        return 0
    batch_heads = math.prod(query_shape[:-2])
    query_tokens = query_shape[-2]
    key_tokens = key_shape[-2]
    return batch_heads * query_tokens * key_tokens * (query_shape[-1] + value_shape[-1])


def build_model(cfg, checkpoint_path, device):
    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass),
    )
    denoiser = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
    model = instantiate_from_config(cfg.diffusion_model, model=denoiser)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")

    return model.eval().to(device)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def analyze_flops(model, image_size, sample_steps, device):
    denoiser = DenoiserStep(model.model).eval()
    noisy_mask = torch.randn(1, 1, image_size, image_size, device=device)
    log_snr = torch.zeros(1, device=device)
    image = torch.randn(1, 3, image_size, image_size, device=device)

    analysis = FlopCountAnalysis(denoiser, (noisy_mask, log_snr, image))
    analysis.set_op_handle("aten::scaled_dot_product_attention", scaled_dot_product_attention_macs)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    one_step_macs = int(analysis.total())
    unsupported = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}

    # fvcore counts a fused multiply-add as one operation. Both conventions are
    # reported because papers often call either MACs or 2 * MACs "FLOPs".
    return {
        "one_step_gmacs": one_step_macs / 1e9,
        "one_step_gflops_2x": 2.0 * one_step_macs / 1e9,
        "full_sample_gmacs": sample_steps * one_step_macs / 1e9,
        "full_sample_gflops_2x": 2.0 * sample_steps * one_step_macs / 1e9,
        "unsupported_operators": unsupported,
        "note": "Full-sample FLOPs are one denoiser step multiplied by sampling steps. Attention matrix multiplications are included; minor scheduler, activation and FFT operations are excluded.",
    }


def benchmark_fps(model, image_size, batch_size, warmup, iterations, repeats, device, fp16):
    image = torch.randn(batch_size, 3, image_size, image_size, device=device)
    amp_enabled = fp16 and device.type == "cuda"

    def infer():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            return model.sample(image, verbose=False)

    with torch.inference_mode():
        for _ in range(warmup):
            infer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        elapsed_runs = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    infer()
                end.record()
                torch.cuda.synchronize(device)
                elapsed_runs.append(start.elapsed_time(end) / 1000.0)
            else:
                start_time = time.perf_counter()
                for _ in range(iterations):
                    infer()
                elapsed_runs.append(time.perf_counter() - start_time)

    elapsed = statistics.median(elapsed_runs)
    batch_latency_ms = elapsed * 1000.0 / iterations
    result = {
        "precision": "fp16" if amp_enabled else "fp32",
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_iterations_per_repeat": iterations,
        "repeats": repeats,
        "batch_latency_ms_median": batch_latency_ms,
        "fps_images_per_second": batch_size * iterations / elapsed,
        "repeat_elapsed_seconds": elapsed_runs,
    }
    if device.type == "cuda":
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark DiffIRSTD parameters, GFLOPs and FPS")
    parser.add_argument(
        "--checkpoint",
        default="results_irstd/small_v2_512_fp32/model-best.pt",
        help="Checkpoint used for FPS measurement; pass an empty string to skip loading.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_sample_steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_flops", action="store_true")
    parser.add_argument("--output", default="results_irstd/small_v2_512_fp32/benchmark.json")
    cfg = add_args(parser)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or run in a CUDA-enabled environment.")
    if cfg.batch_size < 1 or cfg.warmup < 0 or cfg.iterations < 1 or cfg.repeats < 1:
        raise ValueError("batch_size, iterations and repeats must be positive; warmup cannot be negative")

    set_random_seed(7)
    device = torch.device(cfg.device)
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps
    image_size = int(cfg.diffusion_model.params.image_size)
    sample_steps = int(cfg.diffusion_model.params.num_sample_steps)

    model = build_model(cfg, cfg.checkpoint, device)
    total_params, trainable_params = count_parameters(model)
    result = {
        "config": str(cfg.config),
        "checkpoint": str(cfg.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "image_size": image_size,
        "sampling_steps": sample_steps,
        "parameters": {
            "total": total_params,
            "total_millions": total_params / 1e6,
            "trainable": trainable_params,
            "trainable_millions": trainable_params / 1e6,
        },
    }

    if not cfg.skip_flops:
        result["complexity"] = analyze_flops(model, image_size, sample_steps, device)
    result["speed"] = benchmark_fps(
        model,
        image_size,
        int(cfg.batch_size),
        int(cfg.warmup),
        int(cfg.iterations),
        int(cfg.repeats),
        device,
        bool(cfg.fp16),
    )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
