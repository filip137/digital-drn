from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

from digital_drn.core.parameter import Bias  # noqa: E402
from digital_drn.training.experiment import build_dataloaders_from_config, build_model_from_config  # noqa: E402
from digital_drn.training.loaded_model_beta_sweep import (  # noqa: E402
    _analyze_batch,
    _cosine,
    _flatten,
    _group_metrics,
    _overall_metrics,
    _relative_error,
)
from digital_drn.training.trainer import build_criterion  # noqa: E402
from digital_drn.utils.misc import set_seed  # noqa: E402

DEFAULT_CONFIG_PATH = Path(
    '/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/'
    'runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/'
    '20260408-163101-nom-cool-2/experiment_config.json'
)
DEFAULT_CASE_DIR = Path('/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_size_sweep')


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(',') if item.strip()]


def _even(value: float) -> int:
    rounded = int(round(value))
    if rounded % 2:
        rounded += 1
    return max(2, rounded)


def _scale_config(base_cfg: dict, scale: float, *, batch_size: int) -> dict:
    cfg = deepcopy(base_cfg)
    cfg['algorithm']['config']['ad_hoc_amp_gradient_scale'] = False
    cfg['algorithm']['config']['beta'] = 1.0e-2
    cfg['data']['config']['batch_size'] = int(batch_size)
    cfg['model']['name'] = f"cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_size_scale_{scale:g}"
    cfg['model']['config']['voltage_amp'] = 4.0
    cfg['model']['config']['current_amp'] = 1.0
    cfg['model']['config']['amplify_first_free_layer'] = True

    block0_out = _even(64 * scale)
    block0_hidden = _even(32 * scale)
    block1_in = 2 * block0_out
    block1_hidden = _even(256 * scale)
    block1_out = _even(512 * scale)

    b0 = cfg['model']['blocks_config'][0]
    b0['drn']['state_shape'] = [6, 32, 32]
    b0['drn']['channels'] = [6, block0_hidden, block0_out]
    b0['drn']['voltage_amp'] = 4.0
    b0['drn']['current_amp'] = 1.0
    b0['drn']['amplify_first_free_layer'] = True

    b1 = cfg['model']['blocks_config'][1]
    b1['ff']['layers'][0]['num_features'] = block0_out
    b1['ff']['output_shape'] = [block0_out, 16, 16]
    b1['drn']['state_shape'] = [block1_in, 16, 16]
    b1['drn']['channels'] = [block1_in, block1_hidden, block1_out]
    b1['drn']['voltage_amp'] = 4.0
    b1['drn']['current_amp'] = 1.0
    b1['drn']['amplify_first_free_layer'] = True

    for layer in cfg['model']['head']['layers']:
        if layer.get('op') == 'linear' and layer.get('in_features') == 32768:
            layer['in_features'] = block1_out * 8 * 8
            break
    return cfg


def _tensor_names_for_group(model) -> OrderedDict[str, list[str]]:
    names: OrderedDict[str, list[str]] = OrderedDict()
    if getattr(model, 'head', None) is not None:
        names['head'] = [name for name, _ in model.head.named_parameters()]
    for block_idx, block in enumerate(model.blocks):
        names[f'block_{block_idx}/ff'] = [name for name, _ in block.ff.named_parameters()]
        names[f'block_{block_idx}/drive'] = ['drive_scale']
        drn_names: list[str] = []
        weight_index = 0
        bias_index = 0
        for param in block.resistive_params():
            if isinstance(param, Bias):
                drn_names.append(f'b{bias_index}')
                bias_index += 1
            else:
                drn_names.append(f'W{weight_index}')
                weight_index += 1
        names[f'block_{block_idx}/drn'] = drn_names
    return names


def _merge_suffix(bp_groups, ep_groups, suffix: str) -> dict:
    bp_tensors = []
    ep_tensors = []
    for group_name in bp_groups.keys():
        if group_name.endswith('/' + suffix):
            bp_tensors.extend(bp_groups[group_name])
            ep_tensors.extend(ep_groups[group_name])
    bp_flat = _flatten(bp_tensors)
    ep_flat = _flatten(ep_tensors)
    return {
        'cosine': _cosine(bp_flat, ep_flat),
        'relative_error': _relative_error(bp_flat, ep_flat),
        'bp_norm': float(bp_flat.norm().item()),
        'ep_norm': float(ep_flat.norm().item()),
    }


def _ratio(row: dict) -> float:
    bp = float(row['bp_norm'])
    return float(row['ep_norm']) / bp if bp > 0.0 else float('nan')


def _stats(rows: list[dict]) -> dict:
    finite_rows = [
        row for row in rows
        if math.isfinite(float(row['cosine'])) and math.isfinite(_ratio(row))
    ]
    if not finite_rows:
        return {
            'count': len(rows),
            'finite_count': 0,
            'mean_cosine': float('nan'),
            'min_cosine': float('nan'),
            'max_cosine': float('nan'),
            'mean_ep_bp_norm_ratio': float('nan'),
            'min_ep_bp_norm_ratio': float('nan'),
            'max_ep_bp_norm_ratio': float('nan'),
        }
    ratios = [_ratio(row) for row in finite_rows]
    return {
        'count': len(rows),
        'finite_count': len(finite_rows),
        'mean_cosine': sum(float(row['cosine']) for row in finite_rows) / len(finite_rows),
        'min_cosine': min(float(row['cosine']) for row in finite_rows),
        'max_cosine': max(float(row['cosine']) for row in finite_rows),
        'mean_ep_bp_norm_ratio': sum(ratios) / len(ratios),
        'min_ep_bp_norm_ratio': min(ratios),
        'max_ep_bp_norm_ratio': max(ratios),
    }


def _param_count(model) -> dict:
    torch_params = sum(param.numel() for param in model.parameters())
    drn_params = sum(param.state.numel() for param in model.resistive_params())
    return {'torch_param_count': int(torch_params), 'drn_param_count': int(drn_params)}


def _run_variant(cfg: dict, *, scale: float, beta: float, sample_indices: list[int], device: torch.device) -> dict:
    set_seed(0)
    model = build_model_from_config(cfg)
    model.set_device(device)
    model.enable_resistive_grad_()
    model.eval()
    model.detach_state_()

    _train_loader, eval_loader = build_dataloaders_from_config(cfg, download=False)
    subset = Subset(eval_loader.dataset, sample_indices)
    loader = DataLoader(subset, batch_size=len(sample_indices), shuffle=False, drop_last=False, num_workers=0)
    batch_inputs, batch_targets = next(iter(loader))
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)
    criterion = build_criterion(cfg.get('trainer', {}).get('criterion', 'cross_entropy')).to(device)

    bp_groups, ep_groups, displacement, _hybrid = _analyze_batch(
        model,
        batch_inputs,
        batch_targets,
        criterion,
        beta,
        None,
        amp_gradient_compensation=False,
    )
    group_rows = _group_metrics(bp_groups, ep_groups)
    overall = _overall_metrics(bp_groups, ep_groups)
    path_rows = {
        'ff_all': _merge_suffix(bp_groups, ep_groups, 'ff'),
        'drive_all': _merge_suffix(bp_groups, ep_groups, 'drive'),
        'drn_all': _merge_suffix(bp_groups, ep_groups, 'drn'),
    }

    tensor_names = _tensor_names_for_group(model)
    tensor_rows = []
    for group_name, names in tensor_names.items():
        for tensor_name, bp_tensor, ep_tensor in zip(names, bp_groups[group_name], ep_groups[group_name]):
            row = {
                'group': group_name,
                'name': tensor_name,
                'cosine': _cosine(bp_tensor, ep_tensor),
                'relative_error': _relative_error(bp_tensor, ep_tensor),
                'bp_norm': float(bp_tensor.norm().item()),
                'ep_norm': float(ep_tensor.norm().item()),
            }
            row['ep_bp_norm_ratio'] = _ratio(row)
            tensor_rows.append(row)

    drn_tensors = [row for row in tensor_rows if '/drn' in row['group']]
    drn_weights = [row for row in drn_tensors if row['name'].startswith('W')]
    drn_biases = [row for row in drn_tensors if row['name'].startswith('b')]

    result = {
        'scale': float(scale),
        'beta': float(beta),
        'sample_indices': sample_indices,
        'amp_gradient_compensation': False,
        'voltage_amp': 4.0,
        'current_amp': 1.0,
        'amplify_first_free_layer': True,
        **_param_count(model),
        'overall': overall,
        'path_rows': path_rows,
        'group_rows': group_rows,
        'displacement': displacement,
        'tensor_stats': {
            'drn_all': _stats(drn_tensors),
            'drn_weights': _stats(drn_weights),
            'drn_biases': _stats(drn_biases),
        },
        'tensor_rows': tensor_rows,
    }
    del model, batch_inputs, batch_targets, criterion
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def _write_outputs(case_dir: Path, summary: dict) -> None:
    results_dir = case_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    csv_path = results_dir / 'summary.csv'
    with csv_path.open('w', newline='') as f:
        fieldnames = [
            'scale', 'beta', 'torch_param_count', 'drn_param_count',
            'overall_cosine', 'flattened_drn_cosine', 'flattened_drn_relative_error',
            'drn_local_mean_cosine', 'drn_weight_mean_cosine', 'drn_bias_mean_cosine',
            'drn_finite_count', 'drn_weight_finite_count', 'drn_bias_finite_count',
            'drn_weight_min_ratio', 'drn_weight_max_ratio', 'drn_bias_min_ratio', 'drn_bias_max_ratio',
            'displacement_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary['rows']:
            writer.writerow({
                'scale': row['scale'],
                'beta': row['beta'],
                'torch_param_count': row['torch_param_count'],
                'drn_param_count': row['drn_param_count'],
                'overall_cosine': row['overall']['overall_cosine'],
                'flattened_drn_cosine': row['path_rows']['drn_all']['cosine'],
                'flattened_drn_relative_error': row['path_rows']['drn_all']['relative_error'],
                'drn_local_mean_cosine': row['tensor_stats']['drn_all']['mean_cosine'],
                'drn_weight_mean_cosine': row['tensor_stats']['drn_weights']['mean_cosine'],
                'drn_bias_mean_cosine': row['tensor_stats']['drn_biases']['mean_cosine'],
                'drn_finite_count': row['tensor_stats']['drn_all']['finite_count'],
                'drn_weight_finite_count': row['tensor_stats']['drn_weights']['finite_count'],
                'drn_bias_finite_count': row['tensor_stats']['drn_biases']['finite_count'],
                'drn_weight_min_ratio': row['tensor_stats']['drn_weights']['min_ep_bp_norm_ratio'],
                'drn_weight_max_ratio': row['tensor_stats']['drn_weights']['max_ep_bp_norm_ratio'],
                'drn_bias_min_ratio': row['tensor_stats']['drn_biases']['min_ep_bp_norm_ratio'],
                'drn_bias_max_ratio': row['tensor_stats']['drn_biases']['max_ep_bp_norm_ratio'],
                'displacement_mean': row['displacement']['mean_relative_disp'],
            })

    detail_path = results_dir / 'tensor_details.csv'
    with detail_path.open('w', newline='') as f:
        fieldnames = ['scale', 'beta', 'group', 'tensor', 'cosine', 'relative_error', 'bp_norm', 'ep_norm', 'ep_bp_norm_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary['rows']:
            for tensor in row['tensor_rows']:
                writer.writerow({
                    'scale': row['scale'],
                    'beta': row['beta'],
                    'group': tensor['group'],
                    'tensor': tensor['name'],
                    'cosine': tensor['cosine'],
                    'relative_error': tensor['relative_error'],
                    'bp_norm': tensor['bp_norm'],
                    'ep_norm': tensor['ep_norm'],
                    'ep_bp_norm_ratio': tensor['ep_bp_norm_ratio'],
                })

    md_lines = [
        '# Hard-Sigmoid Size Sweep',
        '',
        'This is a random-initialization diagnostic. No checkpoint was loaded, because wider networks do not match the existing checkpoint shapes.',
        '',
        '- voltage_amp: `4.0`',
        '- current_amp: `1.0`',
        '- amp_gradient_compensation: `False`',
        '- amplify_first_free_layer: `True`',
        f"- sample_indices: `{summary['sample_indices']}`",
        '',
        '| scale | beta | DRN params | flattened DRN | local DRN mean | finite tensors | weight mean | bias mean | weight EP/BP range | bias EP/BP range |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary['rows']:
        w = row['tensor_stats']['drn_weights']
        b = row['tensor_stats']['drn_biases']
        md_lines.append(
            f"| {row['scale']:g} | {row['beta']:g} | {row['drn_param_count']} | "
            f"{row['path_rows']['drn_all']['cosine']:.4f} | {row['tensor_stats']['drn_all']['mean_cosine']:.4f} | "
            f"{row['tensor_stats']['drn_all']['finite_count']}/{row['tensor_stats']['drn_all']['count']} | "
            f"{w['mean_cosine']:.4f} | {b['mean_cosine']:.4f} | "
            f"{w['min_ep_bp_norm_ratio']:.4g}-{w['max_ep_bp_norm_ratio']:.4g} | "
            f"{b['min_ep_bp_norm_ratio']:.4g}-{b['max_ep_bp_norm_ratio']:.4g} |"
        )
    md_lines.extend([
        '',
        'Interpretation should focus on the local tensor means and EP/BP norm-ratio ranges, not only the flattened cosine.',
    ])
    (case_dir / 'README.md').write_text('\n'.join(md_lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Width-size BP-vs-EP diagnostic for hard-sigmoid CIFAR networks.')
    parser.add_argument('--config-path', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--scales', default='1,2,4')
    parser.add_argument('--betas', default='1e-3,1e-2')
    parser.add_argument('--sample-indices', default='0,1')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    base_cfg = json.loads(args.config_path.read_text())
    scales = _parse_floats(args.scales)
    betas = _parse_floats(args.betas)
    sample_indices = _parse_ints(args.sample_indices)
    device = torch.device(args.device)

    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'mode': 'hard_sigmoid_size_sweep_random_init',
        'source_config_path': str(args.config_path),
        'device': str(device),
        'scales': scales,
        'betas': betas,
        'sample_indices': sample_indices,
        'batch_size': args.batch_size,
        'rows': [],
    }

    for scale in scales:
        cfg = _scale_config(base_cfg, scale, batch_size=args.batch_size)
        for beta in betas:
            print(f'[run] scale={scale:g} beta={beta:g}')
            row = _run_variant(cfg, scale=scale, beta=beta, sample_indices=sample_indices, device=device)
            summary['rows'].append(row)
            print(
                '  '
                f"flattened_drn={row['path_rows']['drn_all']['cosine']:.4f} "
                f"local_drn={row['tensor_stats']['drn_all']['mean_cosine']:.4f} "
                f"weights={row['tensor_stats']['drn_weights']['mean_cosine']:.4f}"
            )

    _write_outputs(case_dir, summary)
    print(f'[summary] wrote {case_dir / "results" / "summary.json"}')
    print(f'[readme] wrote {case_dir / "README.md"}')


if __name__ == '__main__':
    main()
