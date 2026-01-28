import csv
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend("agg")


_ADJ_STATS_KEYS = [
    "entropy_mean",
    "entropy_std",
    "entropy_p10",
    "entropy_p50",
    "entropy_p90",
    "topk_mass",
    "topk_overlap",
    "l1_adj_diff",
    "adj_mean",
    "adj_var",
    "adj_max",
    "diag_mean",
    "offdiag_mean",
]

_CONF_STATS_KEYS = [
    "conf_mean",
    "conf_std",
    "conf_p10",
    "conf_p50",
    "conf_p90",
]

_TENSOR_STATS_KEYS = [
    "mean",
    "std",
    "min",
    "max",
    "p10",
    "p50",
    "p90",
    "sat_low",
    "sat_high",
]


def _stack_adjs(adjs):
    if not adjs:
        return None
    if isinstance(adjs[0], torch.Tensor):
        return torch.stack(adjs, dim=0).float()
    return torch.from_numpy(np.stack(adjs, axis=0)).float()


def _stack_single_adj(adj):
    if adj is None:
        return None
    if isinstance(adj, torch.Tensor):
        tensor = adj.detach().cpu().float()
    else:
        tensor = torch.from_numpy(np.asarray(adj)).float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _quantiles(values, qs):
    if values is None:
        return [float("nan")] * len(qs)
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return [float("nan")] * len(qs)
    data = flat.detach().cpu().numpy()
    return [float(v) for v in np.quantile(data, qs)]


def _entropy_from_adj(stacked):
    eps = 1e-12
    return -(stacked * (stacked + eps).log()).sum(-1)


def _topk_overlap(stacked, k):
    if stacked is None:
        return float("nan")
    if stacked.shape[0] <= 1 or k <= 0:
        return 0.0
    k = min(int(k), stacked.shape[-1])
    scores = stacked.clone()
    diag = scores.diagonal(dim1=-2, dim2=-1)
    diag.fill_(-float("inf"))
    idx = torch.topk(scores, k, dim=-1).indices
    prev = idx[:-1]
    curr = idx[1:]
    matches = curr.unsqueeze(-1) == prev.unsqueeze(-2)
    overlap = matches.any(-1).float().mean(-1)
    return overlap.mean().item()


def _adj_stats(stacked, topk, prefix):
    stats = {f"{prefix}{key}": float("nan") for key in _ADJ_STATS_KEYS}
    if stacked is None:
        return stats
    entropy = _entropy_from_adj(stacked)
    entropy_mean = entropy.mean().item()
    entropy_std = entropy.std(unbiased=False).item()
    entropy_p10, entropy_p50, entropy_p90 = _quantiles(entropy, [0.1, 0.5, 0.9])

    k = int(topk)
    if k <= 0:
        topk_mass = 0.0
    else:
        k = min(k, stacked.shape[-1])
        topk_mass = torch.topk(stacked, k, dim=-1).values.sum(-1).mean().item()

    adj_mean = stacked.mean().item()
    adj_var = stacked.var(unbiased=False).item()
    adj_max = stacked.max().item()

    diag = stacked.diagonal(dim1=-2, dim2=-1)
    diag_mean = diag.mean().item()
    total_sum = stacked.sum()
    diag_sum = diag.sum()
    denom = max(1, stacked.numel() - diag.numel())
    offdiag_mean = ((total_sum - diag_sum) / denom).item()

    if stacked.shape[0] > 1:
        l1_diff = torch.abs(stacked[1:] - stacked[:-1]).mean().item()
    else:
        l1_diff = 0.0

    stats.update(
        {
            f"{prefix}entropy_mean": entropy_mean,
            f"{prefix}entropy_std": entropy_std,
            f"{prefix}entropy_p10": entropy_p10,
            f"{prefix}entropy_p50": entropy_p50,
            f"{prefix}entropy_p90": entropy_p90,
            f"{prefix}topk_mass": topk_mass,
            f"{prefix}topk_overlap": _topk_overlap(stacked, k),
            f"{prefix}l1_adj_diff": l1_diff,
            f"{prefix}adj_mean": adj_mean,
            f"{prefix}adj_var": adj_var,
            f"{prefix}adj_max": adj_max,
            f"{prefix}diag_mean": diag_mean,
            f"{prefix}offdiag_mean": offdiag_mean,
        }
    )
    return stats


def _conf_stats_from_entropy(entropy, num_vars):
    stats = {key: float("nan") for key in _CONF_STATS_KEYS}
    if entropy is None:
        return stats
    if num_vars <= 1:
        return stats
    denom = float(np.log(num_vars))
    if denom <= 0:
        return stats
    conf = 1.0 - (entropy / denom)
    conf = conf.clamp(min=0.0, max=1.0)
    conf_mean = conf.mean().item()
    conf_std = conf.std(unbiased=False).item()
    conf_p10, conf_p50, conf_p90 = _quantiles(conf, [0.1, 0.5, 0.9])
    stats.update(
        {
            "conf_mean": conf_mean,
            "conf_std": conf_std,
            "conf_p10": conf_p10,
            "conf_p50": conf_p50,
            "conf_p90": conf_p90,
        }
    )
    return stats


def compute_tensor_stats(tensor, prefix):
    stats = {f"{prefix}{key}": float("nan") for key in _TENSOR_STATS_KEYS}
    if tensor is None:
        return stats
    values = tensor.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return stats
    mean = values.mean().item()
    std = values.std(unbiased=False).item()
    min_val = values.min().item()
    max_val = values.max().item()
    p10, p50, p90 = _quantiles(values, [0.1, 0.5, 0.9])
    sat_low = (values < 0.05).float().mean().item()
    sat_high = (values > 0.95).float().mean().item()
    stats.update(
        {
            f"{prefix}mean": mean,
            f"{prefix}std": std,
            f"{prefix}min": min_val,
            f"{prefix}max": max_val,
            f"{prefix}p10": p10,
            f"{prefix}p50": p50,
            f"{prefix}p90": p90,
            f"{prefix}sat_low": sat_low,
            f"{prefix}sat_high": sat_high,
        }
    )
    return stats


def compute_graph_stats(adjs, topk=5, raw_adjs=None, base_adj=None):
    stacked = _stack_adjs(adjs)
    if stacked is None:
        return None
    stats = {}
    stats.update(_adj_stats(stacked, topk, prefix=""))

    raw_stacked = _stack_adjs(raw_adjs) if raw_adjs is not None else None
    stats.update(_adj_stats(raw_stacked, topk, prefix="raw_"))
    raw_entropy = _entropy_from_adj(raw_stacked) if raw_stacked is not None else None
    stats.update(_conf_stats_from_entropy(raw_entropy, int(stacked.shape[-1])))

    base_stacked = _stack_single_adj(base_adj)
    stats.update(_adj_stats(base_stacked, topk, prefix="base_"))

    stats["segments"] = int(stacked.shape[0])
    stats["num_vars"] = int(stacked.shape[-1])
    return stats


def append_graph_stats(csv_path, stats):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(stats)


def update_graph_metrics(csv_path, metrics, epoch=-1, step=-1):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    safe_metrics = {}
    for key, value in metrics.items():
        if value is None:
            safe_metrics[key] = float("nan")
        else:
            safe_metrics[key] = float(value)

    if not os.path.exists(csv_path):
        fieldnames = ["epoch", "step"] + list(safe_metrics.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"epoch": epoch, "step": step, **safe_metrics})
        return

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        fieldnames = ["epoch", "step"]
    for key in safe_metrics:
        if key not in fieldnames:
            fieldnames.append(key)

    if rows:
        target = rows[-1]
    else:
        target = {"epoch": epoch, "step": step}
        rows.append(target)
    target["epoch"] = epoch
    target["step"] = step
    for key, value in safe_metrics.items():
        target[key] = value

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _topk_neighbors(adj, topk):
    k = int(topk)
    if k <= 0:
        return {}
    c = adj.shape[0]
    k = min(k, c)
    scores = adj.copy()
    np.fill_diagonal(scores, -np.inf)
    idx = np.argsort(-scores, axis=1)[:, :k]
    vals = np.take_along_axis(adj, idx, axis=1)
    result = {}
    for i in range(c):
        result[str(i)] = [
            {"j": int(idx[i, j]), "w": float(vals[i, j])} for j in range(k)
        ]
    return result


def save_graph_visuals(adjs, out_dir, topk=5, num_segments=1):
    stacked = _stack_adjs(adjs)
    if stacked is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    mean_adjs = stacked.mean(dim=1).cpu().numpy()
    np.save(os.path.join(out_dir, "adj_segments_mean.npy"), mean_adjs)

    seg_count = mean_adjs.shape[0]
    num_segments = max(1, min(int(num_segments), seg_count))
    for seg_idx in range(num_segments):
        adj = mean_adjs[seg_idx]
        plt.figure(figsize=(4, 3))
        plt.imshow(adj, cmap="viridis")
        plt.colorbar()
        plt.title(f"Adjacency mean seg {seg_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"adj_heatmap_seg{seg_idx}.png"))
        plt.close()

        if topk > 0:
            neighbors = _topk_neighbors(adj, topk)
            with open(os.path.join(out_dir, f"topk_neighbors_seg{seg_idx}.json"), "w") as f:
                json.dump(neighbors, f, indent=2)
