"""
Learning algorithms for RGB color matching — Python port of server/learning.ts
Supports: Gradient Descent, Bayesian Optimization, Evolutionary Strategy, Thompson Sampling
"""

import math
import random
from typing import Callable, Optional

# ─────────────────────────────────────────── types ──

RGB = tuple[float, float, float]   # values 0.0–1.0
History = list[dict]               # [{"rgb": RGB, "score": float}]


# ─────────────────────────────────────── utilities ──

def color_distance(a: RGB, b: RGB) -> float:
    """Euclidean distance in RGB space (normalised 0–1)."""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def calculate_score(distance: float) -> float:
    """Convert distance to 0–100 score. Score ≥ 90 = success."""
    max_dist = math.sqrt(3)
    return 100.0 * (1.0 - distance / max_dist)


def normalized_rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert normalised RGB (0–1) to CSS hex string."""
    ri = max(0, min(255, round(r * 255)))
    gi = max(0, min(255, round(g * 255)))
    bi = max(0, min(255, round(b * 255)))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def normalize_to_sum(rgb: RGB) -> RGB:
    """Normalize so components sum to 1.0; falls back to equal if sum≈0."""
    total = sum(rgb)
    if total < 1e-9:
        return (1/3, 1/3, 1/3)
    return (rgb[0]/total, rgb[1]/total, rgb[2]/total)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def is_duplicate(guess: RGB, history: History, tol: float = 0.05) -> bool:
    for h in history:
        if color_distance(guess, h["rgb"]) < tol:
            return True
    return False


def deduplicate_guess(guess: RGB, history: History) -> RGB:
    """Perturb guess slightly until it is not a duplicate."""
    g = list(guess)
    for _ in range(20):
        if not is_duplicate(tuple(g), history):
            return normalize_to_sum(tuple(g))
        g[0] = clamp(g[0] + random.uniform(-0.05, 0.05))
        g[1] = clamp(g[1] + random.uniform(-0.05, 0.05))
        g[2] = clamp(g[2] + random.uniform(-0.05, 0.05))
    return normalize_to_sum(tuple(g))


# ──────────────────────────────── Gradient Descent ──

class GradientDescentState:
    def __init__(self):
        self.name = "Gradient Descent"


def _gradient_descent_initial(target: RGB) -> RGB:
    # Start with a slight random perturbation near the target
    r = clamp(target[0] + random.uniform(-0.15, 0.15))
    g = clamp(target[1] + random.uniform(-0.15, 0.15))
    b = clamp(target[2] + random.uniform(-0.15, 0.15))
    return normalize_to_sum((r, g, b))


def _gradient_descent_next(history: History, target: RGB,
                            log: Callable[[str], None]) -> RGB:
    if len(history) < 2:
        best = max(history, key=lambda h: h["score"])
        r, g, b = best["rgb"]
        delta = 0.2
        nr = clamp(r + (target[0] - r) * delta + random.uniform(-0.05, 0.05))
        ng = clamp(g + (target[1] - g) * delta + random.uniform(-0.05, 0.05))
        nb = clamp(b + (target[2] - b) * delta + random.uniform(-0.05, 0.05))
        guess = normalize_to_sum((nr, ng, nb))
        log(f"Gradient step toward target — adjusting by {delta*100:.0f}%")
        return deduplicate_guess(guess, history)

    # Sort by score descending
    sorted_h = sorted(history, key=lambda h: h["score"], reverse=True)
    best = sorted_h[0]
    second = sorted_h[1]

    br, bg, bb = best["rgb"]
    sr_rgb, sg, sb = second["rgb"]

    # Gradient: direction from second-best to best
    grad_r = br - sr_rgb
    grad_g = bg - sg
    grad_b = bb - sb

    # Also pull toward target
    target_pull = 0.35
    step = 0.35
    nr = clamp(br + step * grad_r + target_pull * (target[0] - br))
    ng = clamp(bg + step * grad_g + target_pull * (target[1] - bg))
    nb = clamp(bb + step * grad_b + target_pull * (target[2] - bb))

    guess = normalize_to_sum((nr, ng, nb))
    log(
        f"Gradient descent: best score {best['score']:.1f}, "
        f"stepping R{grad_r:+.2f} G{grad_g:+.2f} B{grad_b:+.2f}"
    )
    return deduplicate_guess(guess, history)


# ──────────────────────────── Bayesian Optimization ──

GRID_SIZE = 8  # 8×8×8 = 512 cells


class BayesianState:
    def __init__(self):
        self.name = "Bayesian Optimization"
        n = GRID_SIZE ** 3
        self.mean = [50.0] * n      # prior mean (middle of score range)
        self.variance = [50.0] * n  # prior variance (high uncertainty)

    def cell_index(self, r: float, g: float, b: float) -> int:
        ri = min(GRID_SIZE - 1, int(r * GRID_SIZE))
        gi = min(GRID_SIZE - 1, int(g * GRID_SIZE))
        bi = min(GRID_SIZE - 1, int(b * GRID_SIZE))
        return ri * GRID_SIZE * GRID_SIZE + gi * GRID_SIZE + bi

    def cell_rgb(self, idx: int) -> RGB:
        bi = idx % GRID_SIZE
        gi = (idx // GRID_SIZE) % GRID_SIZE
        ri = idx // (GRID_SIZE * GRID_SIZE)
        return (
            (ri + 0.5) / GRID_SIZE,
            (gi + 0.5) / GRID_SIZE,
            (bi + 0.5) / GRID_SIZE,
        )

    def update(self, rgb: RGB, score: float):
        idx = self.cell_index(*rgb)
        # Simple Gaussian update: blend toward observed score
        obs_var = 5.0
        k = self.variance[idx] / (self.variance[idx] + obs_var)
        self.mean[idx] += k * (score - self.mean[idx])
        self.variance[idx] *= (1 - k)

        # Smooth neighbors (propagate information)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ri = min(GRID_SIZE - 1, max(0, (idx // (GRID_SIZE * GRID_SIZE)) + di))
                    gi = min(GRID_SIZE - 1, max(0, (idx // GRID_SIZE % GRID_SIZE) + dj))
                    bi = min(GRID_SIZE - 1, max(0, (idx % GRID_SIZE) + dk))
                    n_idx = ri * GRID_SIZE * GRID_SIZE + gi * GRID_SIZE + bi
                    blend = 0.1
                    self.mean[n_idx] = (1 - blend) * self.mean[n_idx] + blend * score


def _bayesian_initial(target: RGB) -> RGB:
    r = clamp(target[0] + random.uniform(-0.1, 0.1))
    g = clamp(target[1] + random.uniform(-0.1, 0.1))
    b = clamp(target[2] + random.uniform(-0.1, 0.1))
    return normalize_to_sum((r, g, b))


def _bayesian_next(state: BayesianState, history: History, target: RGB,
                   log: Callable[[str], None]) -> RGB:
    # Update GP from history
    for h in history:
        state.update(h["rgb"], h["score"])

    # UCB acquisition: score = mean + kappa * sqrt(variance)
    kappa = 2.0
    best_ucb = -1.0
    best_idx = 0

    # Sample 200 random cells for efficiency (skip if small grid)
    indices = random.sample(range(GRID_SIZE ** 3), min(200, GRID_SIZE ** 3))
    for idx in indices:
        ucb = state.mean[idx] + kappa * math.sqrt(max(0, state.variance[idx]))
        if ucb > best_ucb:
            best_ucb = ucb
            best_idx = idx

    guess_raw = state.cell_rgb(best_idx)
    # Add small noise to avoid always picking exact cell centers
    nr = clamp(guess_raw[0] + random.uniform(-0.04, 0.04))
    ng = clamp(guess_raw[1] + random.uniform(-0.04, 0.04))
    nb = clamp(guess_raw[2] + random.uniform(-0.04, 0.04))
    guess = normalize_to_sum((nr, ng, nb))

    best_score = max(h["score"] for h in history) if history else 0
    log(
        f"Bayesian UCB acquisition: best known score {best_score:.1f}, "
        f"exploring cell {best_idx} (UCB={best_ucb:.1f})"
    )
    return deduplicate_guess(guess, history)


# ──────────────────────────── Evolutionary Strategy ──

class EvolutionaryState:
    def __init__(self):
        self.name = "Evolutionary Strategy"
        self.population: list[RGB] = []
        self.sigma = 0.15  # mutation spread


def _evolutionary_initial(target: RGB) -> RGB:
    r = clamp(target[0] + random.uniform(-0.2, 0.2))
    g = clamp(target[1] + random.uniform(-0.2, 0.2))
    b = clamp(target[2] + random.uniform(-0.2, 0.2))
    return normalize_to_sum((r, g, b))


def _evolutionary_next(state: EvolutionaryState, history: History, target: RGB,
                        log: Callable[[str], None]) -> RGB:
    # Build population from best historical results
    sorted_h = sorted(history, key=lambda h: h["score"], reverse=True)
    elites = sorted_h[:2]

    if len(elites) < 2:
        parent = elites[0]["rgb"]
        nr = clamp(parent[0] + random.gauss(0, state.sigma))
        ng = clamp(parent[1] + random.gauss(0, state.sigma))
        nb = clamp(parent[2] + random.gauss(0, state.sigma))
        guess = normalize_to_sum((nr, ng, nb))
        log(f"Evolutionary: mutating best result (score {elites[0]['score']:.1f})")
        return deduplicate_guess(guess, history)

    # Crossover two elites + mutation
    p1 = elites[0]["rgb"]
    p2 = elites[1]["rgb"]
    alpha = random.uniform(0.3, 0.7)

    nr = clamp(alpha * p1[0] + (1-alpha) * p2[0] + random.gauss(0, state.sigma))
    ng = clamp(alpha * p1[1] + (1-alpha) * p2[1] + random.gauss(0, state.sigma))
    nb = clamp(alpha * p1[2] + (1-alpha) * p2[2] + random.gauss(0, state.sigma))
    guess = normalize_to_sum((nr, ng, nb))

    # Adapt sigma: shrink when improving
    score_diff = elites[0]["score"] - sorted_h[-1]["score"] if len(sorted_h) > 1 else 0
    state.sigma = max(0.03, min(0.25, state.sigma * (0.9 if score_diff > 10 else 1.1)))

    log(
        f"Evolutionary: crossover elites (scores {elites[0]['score']:.1f}, "
        f"{elites[1]['score']:.1f}), σ={state.sigma:.3f}"
    )
    return deduplicate_guess(guess, history)


# ──────────────────────────── Thompson Sampling ──

class ThompsonState:
    def __init__(self):
        self.name = "Thompson Sampling"
        # Alpha/beta for a simplified beta-like distribution per grid cell
        n = GRID_SIZE ** 3
        self.alpha = [1.0] * n
        self.beta_param = [1.0] * n

    def cell_index(self, r: float, g: float, b: float) -> int:
        ri = min(GRID_SIZE - 1, int(r * GRID_SIZE))
        gi = min(GRID_SIZE - 1, int(g * GRID_SIZE))
        bi = min(GRID_SIZE - 1, int(b * GRID_SIZE))
        return ri * GRID_SIZE * GRID_SIZE + gi * GRID_SIZE + bi

    def cell_rgb(self, idx: int) -> RGB:
        bi = idx % GRID_SIZE
        gi = (idx // GRID_SIZE) % GRID_SIZE
        ri = idx // (GRID_SIZE * GRID_SIZE)
        return (
            (ri + 0.5) / GRID_SIZE,
            (gi + 0.5) / GRID_SIZE,
            (bi + 0.5) / GRID_SIZE,
        )

    def update(self, rgb: RGB, score: float):
        idx = self.cell_index(*rgb)
        norm_score = score / 100.0
        self.alpha[idx] += norm_score
        self.beta_param[idx] += (1.0 - norm_score)


def _thompson_initial(target: RGB) -> RGB:
    r = clamp(target[0] + random.uniform(-0.1, 0.1))
    g = clamp(target[1] + random.uniform(-0.1, 0.1))
    b = clamp(target[2] + random.uniform(-0.1, 0.1))
    return normalize_to_sum((r, g, b))


def _thompson_next(state: ThompsonState, history: History, target: RGB,
                   log: Callable[[str], None]) -> RGB:
    for h in history:
        state.update(h["rgb"], h["score"])

    # Sample from beta distribution for each cell, pick argmax
    indices = random.sample(range(GRID_SIZE ** 3), min(200, GRID_SIZE ** 3))
    best_sample = -1.0
    best_idx = indices[0]

    for idx in indices:
        a = state.alpha[idx]
        b = state.beta_param[idx]
        # Sample from Beta(a, b) using gamma variates
        try:
            xa = random.gammavariate(a, 1)
            xb = random.gammavariate(b, 1)
            sample = xa / (xa + xb)
        except Exception:
            sample = a / (a + b)
        if sample > best_sample:
            best_sample = sample
            best_idx = idx

    guess_raw = state.cell_rgb(best_idx)
    nr = clamp(guess_raw[0] + random.uniform(-0.04, 0.04))
    ng = clamp(guess_raw[1] + random.uniform(-0.04, 0.04))
    nb = clamp(guess_raw[2] + random.uniform(-0.04, 0.04))
    guess = normalize_to_sum((nr, ng, nb))

    best_score = max(h["score"] for h in history) if history else 0
    log(
        f"Thompson Sampling: sampled cell {best_idx} (value={best_sample:.3f}), "
        f"best known score {best_score:.1f}"
    )
    return deduplicate_guess(guess, history)


# ─────────────────────────── algorithm registry ──

ALGORITHMS = {
    "Gradient Descent": {
        "description": "AI-guided gradient steps, adjusts direction toward target",
    },
    "Bayesian Optimization": {
        "description": "Models color space as Gaussian process with UCB acquisition",
    },
    "Evolutionary Strategy": {
        "description": "Population-based crossover and mutation of best candidates",
    },
    "Thompson Sampling": {
        "description": "Probabilistic multi-armed bandit — samples from posteriors",
    },
}


def create_algorithm_state(name: str):
    if name == "Gradient Descent":
        return GradientDescentState()
    elif name == "Bayesian Optimization":
        return BayesianState()
    elif name == "Evolutionary Strategy":
        return EvolutionaryState()
    elif name == "Thompson Sampling":
        return ThompsonState()
    else:
        return GradientDescentState()


def get_algorithm_initial_guess(name: str, state, target: RGB,
                                 target_name: str, discoveries: list) -> RGB:
    if name == "Gradient Descent":
        return _gradient_descent_initial(target)
    elif name == "Bayesian Optimization":
        return _bayesian_initial(target)
    elif name == "Evolutionary Strategy":
        return _evolutionary_initial(target)
    elif name == "Thompson Sampling":
        return _thompson_initial(target)
    return _gradient_descent_initial(target)


def get_algorithm_next_guess(name: str, state, history: History, target: RGB,
                              log: Callable[[str], None]) -> RGB:
    if name == "Gradient Descent":
        return _gradient_descent_next(history, target, log)
    elif name == "Bayesian Optimization":
        return _bayesian_next(state, history, target, log)
    elif name == "Evolutionary Strategy":
        return _evolutionary_next(state, history, target, log)
    elif name == "Thompson Sampling":
        return _thompson_next(state, history, target, log)
    return _gradient_descent_next(history, target, log)
