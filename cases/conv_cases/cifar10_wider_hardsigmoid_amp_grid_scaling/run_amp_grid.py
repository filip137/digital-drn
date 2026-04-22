from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))
CONTRIB_CASE_DIR = REPO_ROOT / 'cases' / 'cifar10_wider_hardsigmoid_contribution_breakdown'
if str(CONTRIB_CASE_DIR) not in sys.path:
    sys.path.append(str(CONTRIB_CASE_DIR))

from run_contribution_checks import _summarize_variant, _variant_config  # noqa: E402
from digital_drn.utils.misc import resolve_device, set_seed  # noqa: E402

DEFAULT_CONFIG_PATH = Path(
    '/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/'
    'runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/'
    '20260408-163101-nom-cool-2/experiment_config.json'
)
DEFAULT_CHECKPOINT_PATH = Path(
    '/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/'
    'runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/'
    '20260408-163101-nom-cool-2/checkpoint_best.pt'
)
DEFAULT_CASE_DIR = Path('/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_amp_grid_scaling')


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def _ratio(bp_norm: float, ep_norm: float) -> float:
    return float(ep_norm) / float(bp_norm) if float(bp_norm) > 0.0 else float('nan')


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _tensor_kind(name: str) -> str:
    if name.startswith('W'):
        return 'weight'
    if name.startswith('b'):
        return 'bias'
    return 'other'


def _stats(rows: list[dict]) -> dict:
    finite = [row for row in rows if _finite(row['cosine']) and _finite(row['ep_bp_norm_ratio'])]
    if not finite:
        return {
            'count': len(rows),
            'finite_count': 0,
            'mean_cosine': float('nan'),
            'min_cosine': float('nan'),
            'mean_ep_bp_norm_ratio': float('nan'),
            'min_ep_bp_norm_ratio': float('nan'),
            'max_ep_bp_norm_ratio': float('nan'),
        }
    ratios = [float(row['ep_bp_norm_ratio']) for row in finite]
    return {
        'count': len(rows),
        'finite_count': len(finite),
        'mean_cosine': sum(float(row['cosine']) for row in finite) / len(finite),
        'min_cosine': min(float(row['cosine']) for row in finite),
        'mean_ep_bp_norm_ratio': sum(ratios) / len(ratios),
        'min_ep_bp_norm_ratio': min(ratios),
        'max_ep_bp_norm_ratio': max(ratios),
    }


def _collect_tensor_rows(result: dict, *, voltage_amp: float, current_amp: float) -> list[dict]:
    rows = []
    for row in result['tensor_rows']:
        bp_norm = float(row['bp_norm'])
        ep_norm = float(row['ep_norm'])
        rows.append({
            'voltage_amp': float(voltage_amp),
            'current_amp': float(current_amp),
            'amp_ratio': float(voltage_amp) / float(current_amp),
            'group': row['group'],
            'tensor': row['name'],
            'kind': _tensor_kind(row['name']),
            'cosine': float(row['cosine']),
            'relative_error': float(row['relative_error']),
            'bp_norm': bp_norm,
            'ep_norm': ep_norm,
            'ep_bp_norm_ratio': _ratio(bp_norm, ep_norm),
        })
    return rows


def _path_ratios(result: dict) -> dict:
    path_rows = result['path_rows']
    output = {}
    for name in ('ff_all', 'drive_all', 'drn_all'):
        row = path_rows[name]
        output[name] = {
            'cosine': float(row['cosine']),
            'relative_error': float(row['relative_error']),
            'bp_norm': float(row['bp_norm']),
            'ep_norm': float(row['ep_norm']),
            'ep_bp_norm_ratio': _ratio(float(row['bp_norm']), float(row['ep_norm'])),
        }
    return output


def _fit_power_law(points: list[dict], value_key: str) -> dict:
    clean = [
        point for point in points
        if _finite(point[value_key]) and float(point[value_key]) > 0.0
        and float(point['voltage_amp']) > 0.0 and float(point['current_amp']) > 0.0
    ]
    if len(clean) < 2:
        return {'count': len(clean), 'log_intercept': float('nan'), 'ratio_exponent_k': float('nan'), 'r2': float('nan')}
    x = np.array([math.log(float(point['voltage_amp']) / float(point['current_amp'])) for point in clean], dtype=float)
    y = np.array([math.log(float(point[value_key])) for point in clean], dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    slope = float(coef[1])
    return {
        'count': len(clean),
        'log_intercept': float(coef[0]),
        'intercept_C': float(math.exp(float(coef[0]))),
        'log_slope_vs_A_over_B': slope,
        'ratio_exponent_k': -slope,
        'r2': 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float('nan'),
    }


def _fit_two_factor(points: list[dict], value_key: str) -> dict:
    clean = [
        point for point in points
        if _finite(point[value_key]) and float(point[value_key]) > 0.0
        and float(point['voltage_amp']) > 0.0 and float(point['current_amp']) > 0.0
    ]
    if len(clean) < 3:
        return {'count': len(clean), 'log_intercept': float('nan'), 'a_exponent': float('nan'), 'b_exponent': float('nan'), 'r2': float('nan')}
    log_a = np.array([math.log(float(point['voltage_amp'])) for point in clean], dtype=float)
    log_b = np.array([math.log(float(point['current_amp'])) for point in clean], dtype=float)
    y = np.array([math.log(float(point[value_key])) for point in clean], dtype=float)
    X = np.column_stack([np.ones_like(log_a), log_a, log_b])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        'count': len(clean),
        'log_intercept': float(coef[0]),
        'intercept_C': float(math.exp(float(coef[0]))),
        'a_exponent': float(coef[1]),
        'b_exponent': float(coef[2]),
        'r2': 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float('nan'),
    }


def _make_fit_rows(summary_rows: list[dict], tensor_rows: list[dict]) -> list[dict]:
    fit_rows = []
    path_points = []
    for row in summary_rows:
        base = {
            'voltage_amp': row['voltage_amp'],
            'current_amp': row['current_amp'],
            'amp_ratio': row['amp_ratio'],
        }
        for path_name, path in row['path_rows'].items():
            path_points.append({**base, 'name': path_name, 'ep_bp_norm_ratio': path['ep_bp_norm_ratio']})
    for name in sorted({point['name'] for point in path_points}):
        points = [point for point in path_points if point['name'] == name]
        fit_rows.append({
            'target': name,
            'target_type': 'path',
            'ratio_fit': _fit_power_law(points, 'ep_bp_norm_ratio'),
            'two_factor_fit': _fit_two_factor(points, 'ep_bp_norm_ratio'),
        })

    for group, tensor in sorted({(row['group'], row['tensor']) for row in tensor_rows if '/drn' in row['group']}):
        points = [row for row in tensor_rows if row['group'] == group and row['tensor'] == tensor]
        fit_rows.append({
            'target': f'{group}/{tensor}',
            'target_type': _tensor_kind(tensor),
            'ratio_fit': _fit_power_law(points, 'ep_bp_norm_ratio'),
            'two_factor_fit': _fit_two_factor(points, 'ep_bp_norm_ratio'),
        })
    return fit_rows


def _write_csvs(case_dir: Path, summary: dict) -> None:
    results_dir = case_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / 'summary.csv').open('w', newline='') as f:
        fieldnames = [
            'voltage_amp', 'current_amp', 'amp_ratio', 'beta',
            'overall_cosine', 'drn_flat_cosine', 'drn_ep_bp_norm_ratio',
            'drn_local_mean_cosine', 'drn_weight_mean_cosine', 'drn_bias_mean_cosine',
            'drn_weight_mean_ep_bp_norm_ratio', 'drn_bias_mean_ep_bp_norm_ratio',
            'drn_weight_min_ep_bp_norm_ratio', 'drn_weight_max_ep_bp_norm_ratio',
            'drn_bias_min_ep_bp_norm_ratio', 'drn_bias_max_ep_bp_norm_ratio',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary['rows']:
            writer.writerow({
                'voltage_amp': row['voltage_amp'],
                'current_amp': row['current_amp'],
                'amp_ratio': row['amp_ratio'],
                'beta': row['beta'],
                'overall_cosine': row['overall']['overall_cosine'],
                'drn_flat_cosine': row['path_rows']['drn_all']['cosine'],
                'drn_ep_bp_norm_ratio': row['path_rows']['drn_all']['ep_bp_norm_ratio'],
                'drn_local_mean_cosine': row['tensor_stats']['drn_all']['mean_cosine'],
                'drn_weight_mean_cosine': row['tensor_stats']['drn_weights']['mean_cosine'],
                'drn_bias_mean_cosine': row['tensor_stats']['drn_biases']['mean_cosine'],
                'drn_weight_mean_ep_bp_norm_ratio': row['tensor_stats']['drn_weights']['mean_ep_bp_norm_ratio'],
                'drn_bias_mean_ep_bp_norm_ratio': row['tensor_stats']['drn_biases']['mean_ep_bp_norm_ratio'],
                'drn_weight_min_ep_bp_norm_ratio': row['tensor_stats']['drn_weights']['min_ep_bp_norm_ratio'],
                'drn_weight_max_ep_bp_norm_ratio': row['tensor_stats']['drn_weights']['max_ep_bp_norm_ratio'],
                'drn_bias_min_ep_bp_norm_ratio': row['tensor_stats']['drn_biases']['min_ep_bp_norm_ratio'],
                'drn_bias_max_ep_bp_norm_ratio': row['tensor_stats']['drn_biases']['max_ep_bp_norm_ratio'],
            })

    with (results_dir / 'tensor_norm_ratios.csv').open('w', newline='') as f:
        fieldnames = ['voltage_amp', 'current_amp', 'amp_ratio', 'group', 'tensor', 'kind', 'cosine', 'relative_error', 'bp_norm', 'ep_norm', 'ep_bp_norm_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary['tensor_rows'])

    with (results_dir / 'scaling_fit.csv').open('w', newline='') as f:
        fieldnames = [
            'target', 'target_type', 'count', 'intercept_C', 'ratio_exponent_k', 'ratio_fit_r2',
            'a_exponent', 'b_exponent', 'two_factor_r2',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary['fit_rows']:
            ratio_fit = row['ratio_fit']
            two_fit = row['two_factor_fit']
            writer.writerow({
                'target': row['target'],
                'target_type': row['target_type'],
                'count': ratio_fit['count'],
                'intercept_C': ratio_fit.get('intercept_C'),
                'ratio_exponent_k': ratio_fit['ratio_exponent_k'],
                'ratio_fit_r2': ratio_fit['r2'],
                'a_exponent': two_fit['a_exponent'],
                'b_exponent': two_fit['b_exponent'],
                'two_factor_r2': two_fit['r2'],
            })


def _write_readme(case_dir: Path, summary: dict) -> None:
    rows = summary['rows']
    fit_lookup = {row['target']: row for row in summary['fit_rows']}
    key_targets = ['drn_all', 'block_0/drn/W0', 'block_0/drn/W1', 'block_0/drn/b0', 'block_0/drn/b1', 'block_0/drn/b2', 'block_1/drn/W0', 'block_1/drn/W1', 'block_1/drn/b0', 'block_1/drn/b1', 'block_1/drn/b2']
    lines = [
        '# Hard-Sigmoid A/B Amplification Scaling Grid',
        '',
        'Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.',
        '',
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- beta: `{summary['beta']}`",
        '- amp_gradient_compensation: `False`',
        f"- amplify_first_free_layer: `{summary['amplify_first_free_layer']}`",
        '',
        '## Grid Summary',
        '',
        '| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row['voltage_amp']:g} | {row['current_amp']:g} | {row['amp_ratio']:.4g} | "
            f"{row['path_rows']['drn_all']['cosine']:.4f} | {row['path_rows']['drn_all']['ep_bp_norm_ratio']:.4g} | "
            f"{row['tensor_stats']['drn_all']['mean_cosine']:.4f} | "
            f"{row['tensor_stats']['drn_weights']['mean_ep_bp_norm_ratio']:.4g} | "
            f"{row['tensor_stats']['drn_biases']['mean_ep_bp_norm_ratio']:.4g} |"
        )
    lines.extend([
        '',
        '## Fitted Scaling',
        '',
        'Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.',
        '',
        '| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for target in key_targets:
        if target not in fit_lookup:
            continue
        row = fit_lookup[target]
        rf = row['ratio_fit']
        tf = row['two_factor_fit']
        lines.append(
            f"| `{target}` | {rf.get('intercept_C', float('nan')):.4g} | {rf['ratio_exponent_k']:.3f} | {rf['r2']:.3f} | "
            f"{tf['a_exponent']:.3f} | {tf['b_exponent']:.3f} | {tf['r2']:.3f} |"
        )
    lines.extend([
        '',
        '## Interpretation',
        '',
        '- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.',
        '- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.',
        '- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.',
    ])
    (case_dir / 'README.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='A/B amplification grid to infer EP/BP gradient norm scaling.')
    parser.add_argument('--config-path', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--checkpoint-path', type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--voltage-amps', default='2,3,4')
    parser.add_argument('--current-amps', default='0.5,0.75,2')
    parser.add_argument('--beta', type=float, default=1.0e-2)
    parser.add_argument('--no-amplify-first-free-layer', dest='amplify_first_free_layer', action='store_false')
    parser.set_defaults(amplify_first_free_layer=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    base_cfg = json.loads(args.config_path.read_text())
    set_seed(int(base_cfg.get('config', {}).get('seed', 0)))
    device = resolve_device(args.device)
    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    results_dir = case_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    voltage_amps = _parse_floats(args.voltage_amps)
    current_amps = _parse_floats(args.current_amps)

    summary_rows = []
    all_tensor_rows = []
    for voltage_amp in voltage_amps:
        for current_amp in current_amps:
            print(f'[run] A={voltage_amp:g} B={current_amp:g} ratio={voltage_amp/current_amp:g}')
            cfg = _variant_config(
                base_cfg,
                voltage_amp=voltage_amp,
                current_amp=current_amp,
                amplify_first_free_layer=bool(args.amplify_first_free_layer),
            )
            cfg.setdefault('algorithm', {}).setdefault('config', {})['ad_hoc_amp_gradient_scale'] = False
            result = _summarize_variant(
                cfg,
                args.checkpoint_path,
                beta=args.beta,
                device=device,
                amp_gradient_compensation=False,
            )
            tensor_rows = _collect_tensor_rows(result, voltage_amp=voltage_amp, current_amp=current_amp)
            all_tensor_rows.extend(tensor_rows)
            drn_rows = [row for row in tensor_rows if '/drn' in row['group']]
            weight_rows = [row for row in drn_rows if row['kind'] == 'weight']
            bias_rows = [row for row in drn_rows if row['kind'] == 'bias']
            path_rows = _path_ratios(result)
            row = {
                'voltage_amp': float(voltage_amp),
                'current_amp': float(current_amp),
                'amp_ratio': float(voltage_amp / current_amp),
                'beta': float(args.beta),
                'overall': result['overall'],
                'path_rows': path_rows,
                'group_rows': result['group_rows'],
                'tensor_stats': {
                    'drn_all': _stats(drn_rows),
                    'drn_weights': _stats(weight_rows),
                    'drn_biases': _stats(bias_rows),
                },
                'displacement': result['displacement'],
            }
            summary_rows.append(row)
            print(
                '  '
                f"drn_epbp={path_rows['drn_all']['ep_bp_norm_ratio']:.4g} "
                f"local_cos={row['tensor_stats']['drn_all']['mean_cosine']:.4f} "
                f"w_mean_epbp={row['tensor_stats']['drn_weights']['mean_ep_bp_norm_ratio']:.4g}"
            )
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    fit_rows = _make_fit_rows(summary_rows, all_tensor_rows)
    summary = {
        'mode': 'amp_grid_scaling',
        'config_path': str(args.config_path),
        'checkpoint_path': str(args.checkpoint_path),
        'device': str(device),
        'beta': float(args.beta),
        'voltage_amps': voltage_amps,
        'current_amps': current_amps,
        'amp_gradient_compensation': False,
        'amplify_first_free_layer': bool(args.amplify_first_free_layer),
        'rows': summary_rows,
        'tensor_rows': all_tensor_rows,
        'fit_rows': fit_rows,
    }
    (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    _write_csvs(case_dir, summary)
    (results_dir / 'scaling_fit.json').write_text(json.dumps(fit_rows, indent=2))
    _write_readme(case_dir, summary)
    print(f'[summary] wrote {results_dir / "summary.json"}')
    print(f'[readme] wrote {case_dir / "README.md"}')


if __name__ == '__main__':
    main()
