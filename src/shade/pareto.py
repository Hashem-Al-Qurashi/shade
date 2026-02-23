"""Pareto frontier computation and visualization."""

from __future__ import annotations

import numpy as np


def compute_pareto_frontier_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D Pareto frontier for minimization of both objectives.

    O(n log n) sort-and-sweep algorithm.

    Args:
        points: Shape (n, 2) array. Both columns minimized.

    Returns:
        Boolean mask of shape (n,) where True = Pareto-optimal.
    """
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    pareto_mask = np.zeros(len(points), dtype=bool)
    min_second = np.inf

    for i in range(len(sorted_points)):
        if sorted_points[i, 1] < min_second:
            min_second = sorted_points[i, 1]
            pareto_mask[sorted_idx[i]] = True

    return pareto_mask


def extract_optuna_trial_data(checkpoint_path: str) -> dict:
    """Extract all completed trial data from a Shade Optuna checkpoint.

    Returns dict with keys: refusals, kl, trial_numbers.
    """
    from optuna.storages.journal import (
        JournalFileBackend, JournalFileOpenLock, JournalStorage,
    )
    from optuna.trial import TrialState
    import optuna

    lock = JournalFileOpenLock(checkpoint_path)
    backend = JournalFileBackend(checkpoint_path, lock_obj=lock)
    storage = JournalStorage(backend)
    study = optuna.load_study(study_name="shade", storage=storage)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    return {
        "refusals": [t.user_attrs["refusals"] for t in completed],
        "kl": [t.user_attrs["kl_divergence"] for t in completed],
        "trial_numbers": [t.number for t in completed],
    }


def run_steering_sweep(
    model: object,
    evaluator: object,
    steering_vector: object,
    multipliers: list[float] | None = None,
) -> dict:
    """Sweep steering multiplier values and collect metrics.

    Returns dict with keys: refusals, kl, multipliers.
    """
    import torch.nn.functional as F

    if multipliers is None:
        multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results: dict = {"refusals": [], "kl": [], "multipliers": multipliers}

    base_model = model.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    for mult in multipliers:
        if mult == 0.0:
            refusals = evaluator.count_refusals()
            logprobs = model.get_logprobs_batched(evaluator.good_prompts)
        else:
            with steering_vector.apply(base_model, multiplier=mult):
                refusals = evaluator.count_refusals()
                logprobs = model.get_logprobs_batched(evaluator.good_prompts)

        kl = F.kl_div(
            logprobs, evaluator.base_logprobs,
            reduction="batchmean", log_target=True,
        ).item()
        results["refusals"].append(refusals)
        results["kl"].append(kl)

    return results


def plot_pareto_frontier(
    abliteration_data: dict,
    steering_data: dict,
    total_prompts: int,
    model_name: str = "",
    save_path: str | None = None,
) -> object:
    """Plot abliteration vs steering Pareto frontier.

    Two panels: raw metrics (arXiv style) and normalized (0-1).
    """
    import matplotlib.pyplot as plt

    abl_kl = np.array(abliteration_data["kl"])
    abl_ref = np.array(abliteration_data["refusals"], dtype=float)
    steer_kl = np.array(steering_data["kl"])
    steer_ref = np.array(steering_data["refusals"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: raw metrics
    ax1.scatter(abl_kl, abl_ref, c="royalblue", alpha=0.25, s=30)
    pts = np.column_stack([abl_kl, abl_ref])
    mask = compute_pareto_frontier_2d(pts)
    ax1.scatter(abl_kl[mask], abl_ref[mask], c="royalblue", s=80,
                edgecolors="navy", lw=1.5, label="Abliteration Pareto")
    pidx = np.where(mask)[0][np.argsort(abl_kl[mask])]
    ax1.plot(abl_kl[pidx], abl_ref[pidx], c="royalblue", lw=2)

    ax1.scatter(steer_kl, steer_ref, c="darkorange", s=80,
                edgecolors="saddlebrown", lw=1.5, label="Steering")
    sidx = np.argsort(steer_kl)
    ax1.plot(steer_kl[sidx], steer_ref[sidx], c="darkorange", lw=2, ls="--")
    for i, m in enumerate(steering_data["multipliers"]):
        ax1.annotate(f"x{m}", (steer_kl[i], steer_ref[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax1.set_xlabel("KL Divergence")
    ax1.set_ylabel(f"Refusals (out of {total_prompts})")
    ax1.set_title("Raw Metrics (lower-left = optimal)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: normalized
    abl_x = 1.0 - abl_ref / total_prompts
    abl_y = 1.0 / (1.0 + abl_kl)
    steer_x = 1.0 - steer_ref / total_prompts
    steer_y = 1.0 / (1.0 + steer_kl)

    ax2.scatter(abl_x, abl_y, c="royalblue", alpha=0.25, s=30)
    neg_pts = np.column_stack([-abl_x, -abl_y])
    mask2 = compute_pareto_frontier_2d(neg_pts)
    ax2.scatter(abl_x[mask2], abl_y[mask2], c="royalblue", s=80,
                edgecolors="navy", lw=1.5, label="Abliteration Pareto")
    ax2.scatter(steer_x, steer_y, c="darkorange", s=80,
                edgecolors="saddlebrown", lw=1.5, label="Steering")
    ax2.set_xlabel("Refusal Removal Rate")
    ax2.set_ylabel("Capability Preservation")
    ax2.set_title("Normalized (upper-right = optimal)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Abliteration vs Steering{' — ' + model_name if model_name else ''}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
