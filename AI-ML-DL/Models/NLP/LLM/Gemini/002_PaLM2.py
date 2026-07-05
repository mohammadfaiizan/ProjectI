"""
PaLM 2 (Google, 2023) -- this file does not implement a PaLM-2-specific
architecture, because Google never disclosed one (see 002_PaLM2.md,
Section 2/11). What CAN be concretely demonstrated in code is the concept
PaLM 2 is the industry case study of: Chinchilla-style compute-optimal
scaling, i.e. given a fixed training compute budget C (in FLOPs), there is
a joint optimum over model parameters N and training tokens D, and it is
NOT optimal to maximize N alone (the pre-Chinchilla, Kaplan-era intuition).

We implement:
  1. The standard compute/params/tokens relationship used throughout the
     scaling-law literature: C ~= 6 * N * D.
  2. Chinchilla's power-law allocation result: for compute budget C, the
     compute-optimal parameter count and token count scale as
         N_opt(C) = G * (C / 6) ** a
         D_opt(C) = (1 / G) * (C / 6) ** b
     with a + b = 1, using the fitted exponents reported in Hoffmann et al.
     (2022), Table 3 (approach 3): a ~= 0.49, b ~= 0.51, G chosen so the
     two curves cross Chinchilla's own (N=70B, D=1.4T) operating point.
  3. A concrete "PaLM vs. compute-optimal-at-PaLM's-compute" comparison,
     showing that PaLM's actual (N=540B, D=780B) allocation is
     over-parameterized / under-trained relative to what a Chinchilla-style
     fit would recommend at the same total training compute -- which is
     the precise sense in which PaLM 2 (smaller, more tokens) is a
     real-world correction of that allocation.
"""
from __future__ import annotations

from dataclasses import dataclass


# Chinchilla (Hoffmann et al., 2022) fitted exponents for the compute-optimal
# frontier N_opt(C) ~ C**a, D_opt(C) ~ C**b, with a + b == 1.
CHINCHILLA_A = 0.49  # exponent on compute for optimal parameter count
CHINCHILLA_B = 1.0 - CHINCHILLA_A  # exponent on compute for optimal token count

# Anchor point: Chinchilla itself, used to fix the proportionality constants.
CHINCHILLA_N = 70e9
CHINCHILLA_D = 1.4e12
CHINCHILLA_C = 6 * CHINCHILLA_N * CHINCHILLA_D  # ~= 5.88e23 FLOPs


def training_compute(n_params: float, n_tokens: float) -> float:
    """The standard approximation: forward+backward FLOPs ~= 6 * N * D."""
    return 6.0 * n_params * n_tokens


def chinchilla_optimal_split(compute_budget: float) -> tuple[float, float]:
    """Given a compute budget C (FLOPs), return the Chinchilla-optimal
    (N_opt, D_opt) -- compute-optimal parameter count and token count.

    Derivation: anchor the power laws N_opt(C) = k_n * C**a and
    D_opt(C) = k_d * C**b at Chinchilla's own reported operating point,
    then evaluate at the requested budget. This reproduces the qualitative
    shape of the Chinchilla frontier; the exact constants Google used
    internally for PaLM 2 are not published, so this is a pedagogical
    reconstruction, not a claim about PaLM 2's literal training recipe.
    """
    k_n = CHINCHILLA_N / (CHINCHILLA_C ** CHINCHILLA_A)
    k_d = CHINCHILLA_D / (CHINCHILLA_C ** CHINCHILLA_B)

    n_opt = k_n * (compute_budget ** CHINCHILLA_A)
    d_opt = k_d * (compute_budget ** CHINCHILLA_B)
    return n_opt, d_opt


@dataclass
class ScalingComparison:
    label: str
    n_params: float
    n_tokens: float
    compute: float

    def describe(self) -> str:
        return (
            f"{self.label:<28} N={self.n_params/1e9:>8.1f}B  "
            f"D={self.n_tokens/1e9:>9.1f}B tok  "
            f"C={self.compute:.3e} FLOPs"
        )


def compare_actual_vs_optimal(n_params: float, n_tokens: float, label: str) -> None:
    """Given an actual (N, D) allocation, compute its implied compute budget,
    then show what a Chinchilla-optimal allocation at that SAME compute
    budget would look like -- this is exactly the comparison that explains
    why "PaLM 2 is smaller but better than PaLM" is not a paradox: at fixed
    (or larger) compute, redistributing budget from parameters to tokens is
    the compute-optimal move if the original allocation was
    over-parameterized relative to Chinchilla's fitted frontier.
    """
    c = training_compute(n_params, n_tokens)
    n_opt, d_opt = chinchilla_optimal_split(c)

    actual = ScalingComparison(f"{label} (actual)", n_params, n_tokens, c)
    optimal = ScalingComparison(f"{label} (Chinchilla-optimal)", n_opt, d_opt, c)

    print(actual.describe())
    print(optimal.describe())
    ratio = n_params / n_opt
    verdict = "OVER-parameterized" if ratio > 1.1 else (
        "under-parameterized" if ratio < 0.9 else "close to optimal"
    )
    print(f"  -> actual N is {ratio:.2f}x the compute-optimal N at this budget "
          f"=> {verdict} relative to a Chinchilla-style fit\n")


if __name__ == "__main__":
    print("=== Chinchilla compute-optimal frontier: worked examples ===\n")

    # 1. Sanity-check the anchor point reproduces Chinchilla itself.
    n_opt, d_opt = chinchilla_optimal_split(CHINCHILLA_C)
    print(f"Chinchilla anchor check: N_opt={n_opt/1e9:.1f}B (actual 70B), "
          f"D_opt={d_opt/1e9:.1f}B tokens (actual 1400B)\n")

    # 2. PaLM's actual allocation (540B params, 780B tokens) vs. what a
    #    Chinchilla-optimal split at PaLM's own compute budget would be.
    #    This is the concrete illustration of the "PaLM was over-parameterized
    #    relative to its data budget" argument that PaLM 2's positioning
    #    (smaller model, more/better data, better results) is the industry
    #    response to.
    compare_actual_vs_optimal(n_params=540e9, n_tokens=780e9, label="PaLM-540B")

    # 3. GPT-3 (175B params, ~300B tokens) as a second, independent
    #    illustration of the same pre-Chinchilla-era pattern.
    compare_actual_vs_optimal(n_params=175e9, n_tokens=300e9, label="GPT-3-175B")

    # 4. Show the frontier itself at a range of compute budgets, to make the
    #    monotonic joint growth of N_opt and D_opt with compute concrete.
    print("=== Compute-optimal frontier across budgets ===")
    for exponent in range(21, 26):
        c = 10.0 ** exponent
        n_opt, d_opt = chinchilla_optimal_split(c)
        print(f"C=1e{exponent:<3} FLOPs  ->  N_opt={n_opt/1e9:9.2f}B params, "
              f"D_opt={d_opt/1e9:11.2f}B tokens")
