"""Generate PNGs from a JiT checkpoint on a single GPU."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

from denoiser import Denoiser
from model_jit import JiT_models


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('samples'))
    parser.add_argument('--model', choices=JiT_models, default='JiT-B/16')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_images', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--class_id', type=int, default=None,
                        help='ImageNet class index; default: random classes')
    parser.add_argument('--class_num', type=int, default=1000)
    parser.add_argument('--weights', choices=['model_ema1', 'model_ema2', 'model'],
                        default='model_ema1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sampling_method', choices=['heun', 'euler'], default='heun')
    parser.add_argument('--num_sampling_steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=3.0,
                        help='Default matches pretrained JiT-B')
    parser.add_argument('--interval_min', type=float, default=0.1)
    parser.add_argument('--interval_max', type=float, default=1.0)
    parser.add_argument('--noise_scale', type=float, default=None,
                        help='Default: img_size / 256 (1 at 256, 2 at 512)')
    parser.add_argument('--t_eps', type=float, default=0.05)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--compile', action='store_true',
                        help='Enable upstream torch.compile; adds startup time')
    # Required by Denoiser's constructor, but unused during sampling in eval mode.
    parser.set_defaults(attn_dropout=0.0, proj_dropout=0.0, label_drop_prob=0.1,
                        P_mean=-0.8, P_std=0.8, ema_decay1=0.9999, ema_decay2=0.9996)
    return parser


def load_model(args, device):
    if args.noise_scale is None:
        args.noise_scale = args.img_size / 256
    # Training checkpoints contain an argparse.Namespace alongside the tensors.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.weights not in checkpoint:
        raise ValueError(f'Checkpoint does not contain {args.weights!r}. '
                         'Choose --weights model for a checkpoint without EMA weights.')
    model = Denoiser(args)
    model.load_state_dict(checkpoint[args.weights], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def generate_images(model, args):
    device = next(model.parameters()).device
    torch.manual_seed(args.seed)
    # Choose labels before sampling, so changing batch size does not change them.
    if args.class_id is None:
        labels = torch.randint(args.class_num, (args.num_images,), device=device)
    else:
        labels = torch.full((args.num_images,), args.class_id, device=device, dtype=torch.long)
    generate = model.generate if args.compile else torch.compiler.disable(model.generate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, args.num_images, args.batch_size):
        batch_labels = labels[start:start + args.batch_size]
        amp = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if args.precision == 'bf16' else nullcontext())
        with amp:
            images = generate(batch_labels)
        # Match engine_jit's [-1, 1] -> rounded uint8 conversion, in RGB for PIL.
        pixels = ((images.float() + 1) * 127.5).clamp(0, 255).round().to(torch.uint8)
        pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
        for offset, (pixels_i, label) in enumerate(zip(pixels, batch_labels.tolist())):
            filename = f'{start + offset:05d}-class{label:04d}.png'
            Image.fromarray(pixels_i).save(args.output_dir / filename)
        print(f'Saved {start + len(batch_labels)}/{args.num_images}', flush=True)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    for name in ('num_images', 'batch_size', 'num_sampling_steps', 'class_num', 'img_size'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.img_size % int(args.model.split('/')[1]) != 0:
        parser.error('--img_size must be divisible by the model patch size')
    if args.class_id is not None and not 0 <= args.class_id < args.class_num:
        parser.error('--class_id must be in [0, class_num)')
    if not 0 <= args.interval_min <= args.interval_max <= 1:
        parser.error('CFG interval must satisfy 0 <= interval_min <= interval_max <= 1')
    if args.t_eps <= 0 or (args.noise_scale is not None and args.noise_scale <= 0):
        parser.error('--t_eps and --noise_scale must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('--output_dir must be empty; choose a new directory for each run')
    if not torch.cuda.is_available():
        parser.error('CUDA is required by the upstream JiT attention implementation')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support bfloat16; use --precision fp32')
    model = load_model(args, torch.device('cuda'))
    print(f'Loaded {args.weights} from {args.checkpoint}', flush=True)
    generate_images(model, args)
    print(f'Images saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
