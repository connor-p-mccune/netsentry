"""Online expert advice: the Hedge/fixed-share updates, the regret bound, and tracking.

The report's guarantees are properties of the online algorithms, provable on synthetic loss
streams without any model: Hedge's regret against the best fixed expert stays under the
theoretical bound, and fixed-share tracks a best expert that *switches* mid-stream where plain
Hedge cannot. These pin exactly that.
"""

from __future__ import annotations

import numpy as np

from netsentry.monitoring.experts import (
    best_expert_per_segment,
    log_loss_stream,
    run_online,
)


def test_log_loss_stream_matches_hand_values_and_clips() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([1, 0])
    loss = log_loss_stream(probs, y, clip=10.0)
    # Expert 0: -log(0.9), -log(0.8); expert 1: -log(0.1), -log(0.2).
    assert np.allclose(loss[:, 0], [-np.log(0.9), -np.log(0.8)])
    assert np.allclose(loss[:, 1], [-np.log(0.1), -np.log(0.2)])
    # A confident wrong prediction is capped at the clip.
    capped = log_loss_stream(np.array([[1e-9]]), np.array([1]), clip=3.0)
    assert capped[0, 0] == 3.0


def test_hedge_concentrates_on_the_best_fixed_expert() -> None:
    # Expert 0 is uniformly better; Hedge weight must pile onto it.
    rng = np.random.default_rng(0)
    t = 2000
    losses = np.column_stack([rng.uniform(0, 0.3, t), rng.uniform(0.6, 1.0, t)])
    probs = np.tile([0.5, 0.5], (t, 1))
    y = rng.integers(0, 2, t)
    hedge, _, _ = run_online(losses, probs, y, eta=0.5, alpha=0.0, expert_names=["good", "bad"])
    assert hedge.final_weights["good"] > 0.99


def test_hedge_regret_stays_under_the_bound() -> None:
    # Realized Hedge regret vs the best fixed expert must respect sqrt((T/2) ln N).
    rng = np.random.default_rng(1)
    t, n = 3000, 4
    losses = rng.uniform(0, 1, (t, n))
    probs = np.tile(np.full(n, 0.5), (t, 1))
    y = rng.integers(0, 2, t)
    eta = float(np.sqrt(8 * np.log(n) / t))
    hedge, _, _ = run_online(losses, probs, y, eta, alpha=0.0, expert_names=list("abcd"))
    best_fixed = losses.sum(axis=0).min()
    regret = hedge.cumulative_loss - best_fixed
    bound = np.sqrt(0.5 * t * np.log(n))
    assert 0 <= regret <= bound


def test_fixed_share_tracks_a_switching_best_expert() -> None:
    # Expert 0 is best for the first half, expert 1 for the second. Fixed-share, which keeps
    # weight on both, must beat plain Hedge, which over-commits to the early leader.
    t = 4000
    half = t // 2
    losses = np.zeros((t, 2))
    losses[:half, 0], losses[:half, 1] = 0.0, 1.0
    losses[half:, 0], losses[half:, 1] = 1.0, 0.0
    probs = np.tile([0.5, 0.5], (t, 1))
    y = np.tile([0, 1], t // 2)  # both classes present so PR-AUC is well-defined
    eta = 1.0
    hedge, share, _ = run_online(losses, probs, y, eta, alpha=0.05, expert_names=["a", "b"])
    assert share.cumulative_loss < hedge.cumulative_loss
    # The best *sequence* loss is ~0; fixed-share must be far closer to it than Hedge.
    assert share.cumulative_loss < 0.25 * hedge.cumulative_loss


def test_fixed_share_weight_recovers_after_a_switch() -> None:
    # After the switch the previously-losing expert must climb back to dominate the weight,
    # which plain Hedge (alpha = 0) cannot do once it has collapsed onto expert 0.
    t = 4000
    half = t // 2
    losses = np.zeros((t, 2))
    losses[:half, 1] = 1.0
    losses[half:, 0] = 1.0
    probs = np.tile([0.5, 0.5], (t, 1))
    y = np.tile([0, 1], t // 2)  # both classes present so PR-AUC is well-defined
    _, _share, traj = run_online(losses, probs, y, eta=1.0, alpha=0.05, expert_names=["a", "b"])
    assert traj[half - 1, 0] > 0.9  # expert a dominates before the switch
    assert traj[-1, 1] > 0.9  # expert b has taken over by the end


def test_best_expert_per_segment_finds_the_shift() -> None:
    losses = np.zeros((6, 2))
    losses[:3, 1] = 1.0  # expert 0 best in segment A
    losses[3:, 0] = 1.0  # expert 1 best in segment B
    segments = np.array(["A", "A", "A", "B", "B", "B"])
    result = best_expert_per_segment(losses, segments, ["m0", "m1"])
    assert result == [("A", "m0"), ("B", "m1")]
