"""Time exact JiT likelihood evaluation with random weights and random inputs."""

import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from benchmark_jit_sampling import MODEL_CONFIGS, config_name
from likelihood_jit import compute_log_likelihood, conditional_velocity


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=config_name, choices=MODEL_CONFIGS, default='B_4')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2,
                        help='Untimed full likelihood batches (default: 2)')
    parser.add_argument('--ode_method', choices=['euler', 'rk4', 'dopri5'], default='euler')
    parser.add_argument('--ode_steps', type=int, default=100,
                        help='Step count for fixed-step Euler or RK4')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--trace_chunk_size', type=int, default=1)
    parser.add_argument('--noise_scale', type=float, default=None,
                        help='Default: img_size / 256')
    parser.add_argument('--t_eps', type=float, default=0.05)
    parser.add_argument('--class_num', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def benchmark(model, args):
    device = next(model.net.parameters()).device
    # Reuse one batch so adaptive solves do the same work across timed runs.
    images = torch.rand(args.batch_size, 3, args.img_size, args.img_size,
                        device=device, dtype=torch.float32) * 2 - 1
    labels = torch.randint(args.class_num, (args.batch_size,), device=device)
    velocity = conditional_velocity(model, labels)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def evaluate_batch():
        return compute_log_likelihood(
            velocity, images, noise_scale=args.noise_scale,
            chunk_size=args.trace_chunk_size, method=args.ode_method,
            steps=args.ode_steps, atol=args.atol, rtol=args.rtol,
            breakpoints=(1.0 - args.t_eps,) if 0 < args.t_eps < 1 else (),
        )

    for i in range(args.warmup):
        result = evaluate_batch()
        synchronize()
        del result
        print(f'Warm-up {i + 1}/{args.warmup} complete', flush=True)

    timings = []
    for i in range(args.runs):
        synchronize()
        start = time.perf_counter()
        result = evaluate_batch()
        synchronize()
        elapsed = time.perf_counter() - start
        nfe = result[2]
        del result
        timings.append(elapsed)
        print(f'Run {i + 1}/{args.runs}: {elapsed:.4f} s, {nfe} ODE evaluations', flush=True)

    mean = statistics.mean(timings)
    print(f'Mean: {mean:.4f} s/batch')
    print(f'Std:  {statistics.stdev(timings):.4f} s/batch (sample standard deviation)')
    print(f'Throughput: {args.batch_size / mean:.2f} images/s')
    return timings


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    for name in ('batch_size', 'img_size', 'class_num', 'ode_steps', 'trace_chunk_size'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.runs < 2:
        parser.error('--runs must be at least 2 to compute a sample standard deviation')
    if args.warmup < 1:
        parser.error('--warmup must be at least 1 to exclude first-call startup costs')
    if args.img_size % int(args.model.rsplit('_', 1)[1]) != 0:
        parser.error('--img_size must be divisible by the model patch size')
    if args.noise_scale is None:
        args.noise_scale = args.img_size / 256
    if min(args.t_eps, args.noise_scale, args.atol, args.rtol) <= 0:
        parser.error('--t_eps, --noise_scale, --atol, and --rtol must be positive')
    if not torch.cuda.is_available():
        parser.error('CUDA is required by the JiT attention implementation')

    torch.manual_seed(args.seed)
    net = MODEL_CONFIGS[args.model](
        input_size=args.img_size, in_channels=3, num_classes=args.class_num,
        attn_drop=0.0, proj_drop=0.0,
    ).to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)
    model = SimpleNamespace(net=net, t_eps=args.t_eps)
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Model: JiT_{args.model} (random weights), image size: {args.img_size}, '
          f'batch size: {args.batch_size}')
    solver = (f'dopri5, atol={args.atol}, rtol={args.rtol}' if args.ode_method == 'dopri5'
              else f'{args.ode_method}, {args.ode_steps} steps')
    print(f'Likelihood: {solver}, exact trace, chunk size: {args.trace_chunk_size}, '
          'CFG 1, precision: fp32, compile: False (exact-trace backward compatibility)')
    print(f'{args.warmup} warm-up batches, then {args.runs} timed batches. '
          'Timing covers likelihood only; model and input creation are excluded.', flush=True)
    benchmark(model, args)


if __name__ == '__main__':
    main()
