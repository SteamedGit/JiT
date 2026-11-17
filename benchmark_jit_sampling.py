"""Time compiled JiT sampling batches with randomly initialized weights."""

import argparse
from contextlib import nullcontext
import statistics
import time

import torch

import model_jit
from denoiser import Denoiser


MODEL_CONFIGS = {
    name.removeprefix('JiT_'): factory
    for name, factory in vars(model_jit).items()
    if name.startswith('JiT_') and callable(factory)
}


def config_name(value):
    # Accept B_4, JiT_B_4, and JiT-B/4.
    return value.removeprefix('JiT_').removeprefix('JiT-').replace('/', '_')


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=config_name, choices=MODEL_CONFIGS, default='B_16')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2,
                        help='Untimed full sampling batches (default: 2)')
    parser.add_argument('--num_sampling_steps', type=int, default=50)
    parser.add_argument('--sampling_method', choices=['heun', 'euler'], default='heun')
    parser.add_argument('--cfg', type=float, default=3.0)
    parser.add_argument('--interval_min', type=float, default=0.1)
    parser.add_argument('--interval_max', type=float, default=1.0)
    parser.add_argument('--noise_scale', type=float, default=None,
                        help='Default: img_size / 256')
    parser.add_argument('--t_eps', type=float, default=0.05)
    parser.add_argument('--class_num', type=int, default=1000)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='fp32')
    # Retain compatibility with earlier commands; compilation is always enabled.
    parser.add_argument('--compile', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=0)
    # Denoiser requires these training settings, but sampling does not use them.
    parser.set_defaults(attn_dropout=0.0, proj_dropout=0.0, label_drop_prob=0.1,
                        P_mean=-0.8, P_std=0.8, ema_decay1=0.9999, ema_decay2=0.9996)
    return parser


@torch.no_grad()
def benchmark(model, args):
    device = next(model.parameters()).device
    labels = torch.randint(args.class_num, (args.batch_size,), device=device)
    # Use the upstream torch.compile decorators in model_jit unconditionally.
    generate = model.generate

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def sample_batch():
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if args.precision == 'bf16' else nullcontext())
        with amp:
            return generate(labels)

    for i in range(args.warmup):
        images = sample_batch()
        synchronize()
        del images
        print(f'Warm-up {i + 1}/{args.warmup} complete', flush=True)

    timings = []
    for i in range(args.runs):
        synchronize()
        start = time.perf_counter()
        images = sample_batch()
        synchronize()
        elapsed = time.perf_counter() - start
        del images
        timings.append(elapsed)
        print(f'Run {i + 1}/{args.runs}: {elapsed:.4f} s', flush=True)

    mean = statistics.mean(timings)
    std = statistics.stdev(timings)
    print(f'Mean: {mean:.4f} s/batch')
    print(f'Std:  {std:.4f} s/batch (sample standard deviation)')
    print(f'Throughput: {args.batch_size / mean:.2f} images/s')
    return timings


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    for name in ('batch_size', 'img_size', 'num_sampling_steps', 'class_num'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.runs < 2:
        parser.error('--runs must be at least 2 to compute a sample standard deviation')
    if args.warmup < 1:
        parser.error('--warmup must be at least 1 to exclude first-call startup costs')
    if args.img_size % int(args.model.rsplit('_', 1)[1]) != 0:
        parser.error('--img_size must be divisible by the model patch size')
    if not 0 <= args.interval_min <= args.interval_max <= 1:
        parser.error('CFG interval must satisfy 0 <= interval_min <= interval_max <= 1')
    if args.noise_scale is None:
        args.noise_scale = args.img_size / 256
    if args.t_eps <= 0 or args.noise_scale <= 0:
        parser.error('--t_eps and --noise_scale must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA is required by the JiT attention implementation')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support bfloat16; use --precision fp32')

    torch.manual_seed(args.seed)
    # Include config functions that have not been added to the upstream registry.
    model_jit.JiT_models[args.model] = MODEL_CONFIGS[args.model]
    model = Denoiser(args).to('cuda').eval()
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Model: JiT_{args.model} (random weights), image size: {args.img_size}, '
          f'batch size: {args.batch_size}')
    print(f'Sampling: {args.sampling_method}, {args.num_sampling_steps} steps, '
          f'CFG {args.cfg}, interval ({args.interval_min}, {args.interval_max}), '
          f'precision: {args.precision}, compile: True')
    print(f'{args.warmup} warm-up batches, then {args.runs} timed batches. '
          'Timing includes noise generation and sampling; no image saving.', flush=True)
    benchmark(model, args)


if __name__ == '__main__':
    main()
