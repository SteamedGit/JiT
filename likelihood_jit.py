"""Exact-divergence conditional likelihoods and a five-class JiT validation.

Reference approach:
https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py

Integrate (x, integral(div v)) from t=1 to t=0, then add the integral to
log p0(x0). Exact means an exact autograd trace, not exact numerical integration.
This is evaluation only: returned likelihoods are detached.
"""

import argparse
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path

import torch
from torchdiffeq import odeint


# Torchvision/ImageNet-1K label order. "Church" is a building class; ImageNet
# does not have a generic "house" class. "Sorrel" is the horse class.
VALIDATION_CLASSES = [(817, 'sports car'), (339, 'sorrel (horse)'),
                      (497, 'church'), (1, 'goldfish'), (402, 'acoustic guitar')]


def gaussian_log_prob(x, noise_scale=1.0):
    """Log density of N(0, noise_scale**2 I), one value per example."""
    if noise_scale <= 0:
        raise ValueError('noise_scale must be positive')
    return -0.5 * ((x / noise_scale).square() + math.log(2 * math.pi)
                   + 2 * math.log(noise_scale)).flatten(1).sum(1)


def exact_divergence(value, x, chunk_size=1):
    """Trace dv/dx per example, assuming no coupling across batch examples.

    Uses coordinate VJPs, optionally batched in chunks. Never forms the full
    D-by-D Jacobian and never uses random trace probes. A chunk of K still
    computes K coordinate derivatives; it trades more memory for batching.
    Consumes the autograd graph for value.
    """
    if value.shape != x.shape or x.ndim < 2:
        raise ValueError('value and x must have identical [batch, ...] shapes')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    flat = value.flatten(1)
    batch, dimension = flat.shape
    # Accumulate in float64 to reduce cancellation across many coordinates.
    trace = torch.zeros(batch, device=x.device, dtype=torch.float64)
    if not value.requires_grad:
        return trace.to(x.dtype)
    for start in range(0, dimension, chunk_size):
        stop = min(start + chunk_size, dimension)
        width = stop - start
        if width == 1:
            derivative = torch.autograd.grad(
                flat[:, start].sum(), x, retain_graph=stop < dimension,
                create_graph=False, allow_unused=True,
            )[0]
            if derivative is not None:
                trace += derivative.flatten(1)[:, start].double()
        else:
            basis = torch.eye(width, dtype=x.dtype, device=x.device)
            basis = basis[:, None, :].expand(width, batch, width)
            derivatives = torch.autograd.grad(
                flat[:, start:stop], x, grad_outputs=basis,
                is_grads_batched=True, retain_graph=stop < dimension,
                create_graph=False, allow_unused=True,
            )[0]
            if derivatives is not None:
                derivatives = derivatives.reshape(width, batch, dimension)
                index = torch.arange(width, device=x.device)
                diagonal = derivatives[index, :, start + index]
                trace += diagonal.double().sum(0)
    return trace.to(x.dtype)


@torch.no_grad()
def compute_log_likelihood(velocity, x1, *, noise_scale=1.0, chunk_size=1,
                           method='dopri5', steps=100, atol=1e-5, rtol=1e-5,
                           breakpoints=(), verbose=False):
    """Return (x0, log p1(x1), number of ODE field evaluations).

    velocity(x, t) must be differentiable in x and independent across examples.
    The base distribution is N(0, noise_scale**2 I). dopri5 is adaptive;
    rk4 and euler use 1/steps as their step size. Do not call inside inference_mode().
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('Likelihood needs input gradients; do not use inference_mode')
    if x1.ndim < 2 or not x1.is_floating_point():
        raise ValueError('x1 must be a floating tensor shaped [batch, ...]')
    if chunk_size < 1 or steps < 1 or atol <= 0 or rtol <= 0 or noise_scale <= 0:
        raise ValueError('chunk_size, steps, tolerances, and noise_scale must be positive')
    if method not in ('dopri5', 'rk4', 'euler'):
        raise ValueError('method must be dopri5, rk4, or euler')
    nfe = 0

    def augmented(t, state):
        nonlocal nfe
        nfe += 1
        if verbose:
            print(f'  ODE evaluation {nfe}: t={t.item():.6f}; exact trace', flush=True)
        with torch.enable_grad():
            x = state[0].detach().requires_grad_(True)
            v = velocity(x, t)
            divergence = exact_divergence(v, x, chunk_size)
        if not torch.isfinite(v).all() or not torch.isfinite(divergence).all():
            raise FloatingPointError(f'Non-finite velocity or divergence at t={t.item()}')
        return v.detach(), divergence.detach()

    options = ({'step_size': 1.0 / steps} if method in ('rk4', 'euler')
               else {'max_num_steps': 10000})
    if breakpoints and method == 'dopri5':
        # Force adaptive integration to handle the clamp's kink explicitly.
        options['jump_t'] = x1.new_tensor(breakpoints)
    trajectory, integral = odeint(
        augmented, (x1.detach(), x1.new_zeros(x1.shape[0])),
        x1.new_tensor([1.0, 0.0]), method=method, options=options,
        atol=atol, rtol=rtol,
    )
    x0 = trajectory[-1]
    # Reverse integration makes integral = - integral_0^1 div(v) dt.
    log_p1 = gaussian_log_prob(x0.double(), noise_scale) + integral[-1].double()
    if not torch.isfinite(log_p1).all():
        raise FloatingPointError('Non-finite final log likelihood')
    return x0, log_p1, nfe


@contextmanager
def eager_layer_forwards(net):
    """Temporarily bypass explicit torch.compile decorators on layer forwards.

    Disabling only the outer module does not reliably bypass nested compiled
    wrappers in PyTorch 2.7. Exact traces retain the graph for many VJPs, which
    conflicts with compiled backward buffer donation (and batched VJPs).
    Restore every layer afterward so sampling and benchmarks remain compiled.
    This scoped mutation assumes the model is not used concurrently.
    """
    missing = object()
    saved = []
    try:
        for module in net.modules():
            forward = module.forward
            original = forward
            while hasattr(original, '_torchdynamo_orig_callable'):
                original = original._torchdynamo_orig_callable
            if original is forward:
                continue
            if inspect.ismethod(forward) and not inspect.ismethod(original):
                original = original.__get__(module, type(module))
            saved.append((module, module.__dict__.get('forward', missing)))
            module.forward = original
        yield
    finally:
        for module, forward in reversed(saved):
            if forward is missing:
                del module.forward
            else:
                module.forward = forward


def conditional_velocity(model, labels):
    """JiT's class-conditional field (CFG=1), preserving its denominator clamp.

    Call net directly: Denoiser's sampling helpers are decorated with no_grad.
    Disable upstream compilation here for exact batched autograd VJPs.
    """
    net = torch.compiler.disable(model.net)

    def velocity(x, t):
        times = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0])
        with eager_layer_forwards(model.net):
            clean = net(x, times, labels)
        return (clean - x) / (1.0 - t).clamp_min(model.t_eps)

    return velocity


def run_self_tests():
    """Small analytic tests: trace, inversion, base scale, sign, and ranking."""
    dtype = torch.float64
    matrix = torch.tensor([[0.2, 0.35], [-0.15, -0.1]], dtype=dtype)
    x = torch.randn(2, 1, 1, 2, dtype=dtype)
    for chunk in (1, 2, 3):
        with torch.enable_grad():
            leaf = x.clone().requires_grad_(True)
            value = (leaf.flatten(1) @ matrix.T).reshape_as(leaf)
            actual = exact_divergence(value, leaf, chunk)
        torch.testing.assert_close(actual, matrix.trace().expand(2))

    # Check the JiT adapter, including the -x derivative and endpoint clamp.
    from types import SimpleNamespace
    class CleanPrediction(torch.nn.Module):
        def forward(self, x, t, labels):
            return 0.2 * x + labels.reshape(-1, 1, 1, 1)
    mock_model = SimpleNamespace(net=CleanPrediction(), t_eps=0.05)
    field = conditional_velocity(mock_model, torch.tensor([0, 1]))
    for t in (0.0, 0.5, 0.99, 1.0):
        leaf = x.clone().requires_grad_(True)
        actual = exact_divergence(field(leaf, leaf.new_tensor(t)), leaf, 2)
        expected_trace = 2 * (0.2 - 1) / max(1 - t, mock_model.t_eps)
        torch.testing.assert_close(actual, torch.full_like(actual, expected_trace))

    # Regression: nested compiled layers must not create compiled backward graphs
    # during exact traces. AOT eager exercises compiled autograd without a GPU.
    class CompiledLayer(torch.nn.Module):
        @torch.compile(backend='aot_eager')
        def forward(self, x):
            return x.sin() * x
    class NestedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = CompiledLayer()
        def forward(self, x, t, labels):
            return self.layer(x)
    compiled_model = SimpleNamespace(net=NestedNet(), t_eps=0.05)
    compiled_field = conditional_velocity(compiled_model, torch.tensor([0, 1]))
    with torch.no_grad():
        compiled_model.net(x, x.new_tensor(0.5), torch.tensor([0, 1]))
    for chunk in (1, 2):
        leaf = x.clone().requires_grad_(True)
        value = compiled_field(leaf, leaf.new_tensor(0.5))
        pending, visited = [value.grad_fn], set()
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            assert 'CompiledFunction' not in type(node).__name__
            pending.extend(parent for parent, _ in node.next_functions)
        actual = exact_divergence(value, leaf, chunk)
        expected_trace = ((leaf.sin() + leaf * leaf.cos() - 1) / 0.5).flatten(1).sum(1)
        torch.testing.assert_close(actual, expected_trace)
        assert 'forward' not in compiled_model.net.layer.__dict__
        assert hasattr(compiled_model.net.layer.forward, '_torchdynamo_orig_callable')
    # Restoration must also happen if a model forward raises.
    try:
        with eager_layer_forwards(compiled_model.net):
            raise RuntimeError('test restoration')
    except RuntimeError:
        pass
    assert 'forward' not in compiled_model.net.layer.__dict__

    sigma = 1.7
    z = torch.tensor([[0.3, -0.7], [1.1, 0.5]], dtype=dtype)
    x1 = (z @ torch.matrix_exp(matrix).T).reshape(2, 1, 1, 2)
    expected = gaussian_log_prob(z, sigma) - matrix.trace()
    def linear_field(x, t):
        return (x.flatten(1) @ matrix.T).reshape_as(x)
    for method in ('dopri5', 'rk4'):
        for chunk in (1, 2):
            recovered, logp, _ = compute_log_likelihood(
                linear_field, x1, noise_scale=sigma, chunk_size=chunk,
                method=method, steps=50, atol=1e-10, rtol=1e-10,
            )
            torch.testing.assert_close(recovered.flatten(1), z, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(logp, expected, atol=1e-7, rtol=1e-7)

    # Euler is first order: doubling steps should approximately halve both
    # inversion and likelihood errors, with exactly one trace per step.
    for chunk in (1, 2):
        errors = []
        for steps in (20, 40):
            recovered, logp, nfe = compute_log_likelihood(
                linear_field, x1, noise_scale=sigma, chunk_size=chunk,
                method='euler', steps=steps,
            )
            assert nfe == steps
            errors.append(torch.stack([
                (recovered.flatten(1) - z).abs().max(),
                (logp - expected).abs().max(),
            ]))
        assert torch.all(errors[1] < 0.6 * errors[0]), errors

    # Class-specific translations yield known Gaussian densities and rankings.
    means = torch.tensor([-6., -3., 0., 3., 6.], dtype=dtype)
    for true_index, mean in enumerate(means):
        sample = (mean + 0.1).reshape(1, 1)
        scores = []
        for candidate in means:
            _, logp, _ = compute_log_likelihood(
                lambda x, t: torch.ones_like(x) * candidate,
                sample, atol=1e-10, rtol=1e-10,
            )
            expected = gaussian_log_prob(sample - candidate)
            torch.testing.assert_close(logp, expected, atol=1e-8, rtol=1e-8)
            scores.append(logp.item())
        assert max(range(5), key=scores.__getitem__) == true_index
    print('PASS: exact trace (scalar/chunked), nested compiled-layer regression, '
          'JiT input gradients and clamp, '
          'ODE inversion, Euler convergence and evaluation count, '
          'likelihood sign, Gaussian scale, constant field, '
          'and five-class analytic rankings.')


def validate_checkpoint(args):
    """Generate five full-precision samples, then score all 25 image/class pairs."""
    from PIL import Image
    from sample_jit import get_args_parser as sampling_parser, load_model

    if not torch.cuda.is_available():
        raise RuntimeError('The pretrained JiT validation requires CUDA')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Choose an empty --output_dir to preserve previous results')
    sampling_args = sampling_parser().parse_args([
        '--checkpoint', str(args.checkpoint), '--model', 'JiT-B/16',
        '--img_size', '256', '--precision', 'fp32', '--cfg', str(args.cfg),
        '--interval_min', str(args.interval_min), '--interval_max', str(args.interval_max),
        '--num_sampling_steps', str(args.num_sampling_steps),
    ])
    torch.manual_seed(args.seed)
    model = load_model(sampling_args, torch.device('cuda')).requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Loaded B/16 EMA weights. Generation CFG:', args.cfg, flush=True)
    print('Scoring conditional density with CFG=1, FP32, and exact divergence. '
          'This is continuous density of the clamped ODE, not discrete image BPD.', flush=True)
    print('196,608 trace coordinates per ODE evaluation, 25 ODE solves. '
          f'Trace chunk size: {args.trace_chunk_size}.', flush=True)

    samples = []
    for class_id, name in VALIDATION_CLASSES:
        print(f'Generating {name} (class {class_id})', flush=True)
        labels = torch.tensor([class_id], device='cuda')
        with torch.no_grad():
            # Retain the upstream compiled sampler for image generation.
            sample = model.generate(labels).float().cpu()
        samples.append(sample)
        # PNGs are previews only. Score the original, unclipped float tensors.
        pixels = ((sample[0] + 1) * 127.5).clamp(0, 255).round().byte()
        Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(
            args.output_dir / f'{class_id:04d}.png')
    torch.save({'images': torch.cat(samples), 'class_ids': [c[0] for c in VALIDATION_CLASSES]},
               args.output_dir / 'samples.pt')

    report = {'settings': {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
              'weights': 'model_ema1', 'likelihood_cfg': 1.0, 't_eps': model.t_eps,
              'noise_scale': model.noise_scale, 'units': 'nats per image',
              'divergence': 'exact', 'results': []}
    for sample, (true_id, true_name) in zip(samples, VALIDATION_CLASSES):
        print(f'\nSample generated as {true_name} (class {true_id})', flush=True)
        scores = []
        for candidate_id, candidate_name in VALIDATION_CLASSES:
            print(f'  Scoring under {candidate_name} ...', flush=True)
            labels = torch.tensor([candidate_id], device='cuda')
            _, logp, nfe = compute_log_likelihood(
                conditional_velocity(model, labels), sample.cuda(),
                noise_scale=model.noise_scale, chunk_size=args.trace_chunk_size,
                method=args.ode_method, steps=args.ode_steps, atol=args.atol, rtol=args.rtol,
                breakpoints=(1.0 - model.t_eps,), verbose=args.verbose,
            )
            scores.append({'class_id': candidate_id, 'class_name': candidate_name,
                           'nll': -logp.item(), 'ode_evaluations': nfe})
            print(f'    NLL = {-logp.item():.6f} nats ({nfe} ODE evaluations)', flush=True)
        ranked = sorted(scores, key=lambda row: row['nll'])
        for rank, row in enumerate(ranked, 1):
            row['rank'] = rank
            marker = ' [generating class]' if row['class_id'] == true_id else ''
            print(f"  {rank}. {row['class_name']:<18} NLL={row['nll']:.6f}{marker}")
        true_rank = next(row['rank'] for row in ranked if row['class_id'] == true_id)
        report['results'].append({'generating_class': true_id, 'name': true_name,
                                  'true_class_rank': true_rank, 'ranking': ranked})
        (args.output_dir / 'likelihoods.json').write_text(json.dumps(report, indent=2) + '\n')
    top1 = sum(row['true_class_rank'] == 1 for row in report['results'])
    print(f'\nGenerating class ranked first: {top1}/5. '
          'This is a diagnostic, not a guaranteed property or correctness test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self_test', action='store_true', help='Run small analytic tests on CPU and exit')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).resolve().parent / 'jit-b-16/checkpoint-last.pth')
    parser.add_argument('--output_dir', type=Path, default=Path('likelihood_validation'))
    parser.add_argument('--trace_chunk_size', type=int, default=1,
                        help='Exact coordinate VJPs per chunk; larger values need more GPU memory')
    parser.add_argument('--ode_method', choices=['dopri5', 'rk4', 'euler'], default='dopri5')
    parser.add_argument('--ode_steps', type=int, default=100, help='Step count for fixed-step RK4 or Euler')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--num_sampling_steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=3.0, help='Generation CFG; likelihood always uses CFG=1')
    parser.add_argument('--interval_min', type=float, default=0.1)
    parser.add_argument('--interval_max', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', help='Print each ODE evaluation')
    args = parser.parse_args()
    if min(args.trace_chunk_size, args.ode_steps, args.num_sampling_steps) < 1:
        parser.error('Chunk size and step counts must be positive')
    if args.atol <= 0 or args.rtol <= 0:
        parser.error('Tolerances must be positive')
    if not 0 <= args.interval_min <= args.interval_max <= 1:
        parser.error('CFG interval must satisfy 0 <= interval_min <= interval_max <= 1')
    if args.self_test:
        run_self_tests()
    else:
        validate_checkpoint(args)


if __name__ == '__main__':
    main()
