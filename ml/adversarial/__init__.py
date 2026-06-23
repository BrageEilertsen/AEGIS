"""Adversarial robustness subsystem (spec §7.8) — the showpiece.

Threat model: an adaptive launderer perturbing graph structure under realistic constraints
(split transfers / structuring, inject pass-through mule accounts, rewire flows; cannot change
labels, limited edit budget, must preserve net flow).

- Attacks: >=1 structural evasion (Nettack-style targeted edge perturbation and/or node injection).
- Defenses: >=1 (adversarial training, robust/median aggregation, structural regularization).
- Artifact: a reproducible before/after (naïve fooled -> hardened holds) the live app can trigger.

Phase 0: placeholder. Implemented in Phase 5.
"""
