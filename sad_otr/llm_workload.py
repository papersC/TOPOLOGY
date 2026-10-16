"""Trace-derived synthetic workload from published model dimensions (final-check revision).

Model dimensions are those published for DeepSeek-V3 (arXiv:2412.19437):
61 layers (first 3 dense, 58 MoE), hidden size 7168, 256 routed experts of
intermediate size 2048 per MoE layer, 8 experts activated per token, and
node-limited routing to at most 4 of 8 expert groups (nodes), 671B parameters.
Everything else (the deployment, batch sizes, routing statistics) is our
assumption and is set in LLM_CONFIG.

Deployment (assumed): N = PP * EP = 4 * 8 = 32 racks, one 8-GPU node per rack;
tensor parallelism stays inside a node (NVLink) and generates no fabric
traffic.  Rack index r = s * EP + e for pipeline stage s and EP/DP rank e.
Each stage's 8 racks form one expert-parallel group holding the 256 routed
experts (32 per rack, i.e. one expert group per rack) and one data-parallel
ring for the non-expert parameters.

Phase types
  EPD  expert dispatch in all 4 stage groups: every sampled token goes once to
       each distinct remote node among its (<= 4) selected groups;
       volume per token and node = hidden * bytes_act.
  EPC  expert combine: transpose of the preceding dispatch (tokens return).
  PP   1F1B steady state: activations s -> s+1 and gradients s+1 -> s,
       tokens_per_microbatch * hidden * bytes_act each.
  DP   ring all-reduce of the non-expert gradients of one stage within its
       EP/DP group: each rack sends 2(E-1)/E * P_stage * bytes_grad to its ring
       successor.  P_dense = 671e9 - 58*256*3*7168*2048 (SwiGLU experts).
Routing model (assumed): per phase (layer) expert popularity logits
~ N(0, sigma_pop^2); per token scores = popularity + Gumbel noise; groups
ranked by the sum of their top-2 expert scores, top-4 groups kept, top-8
experts selected inside them (the DeepSeek-V3 node-limited rule).  Token
counts are estimated from `sample_tokens` sampled tokens per rack and scaled.
"""
from __future__ import annotations

import numpy as np

LLM_CONFIG = {
    "model": "DeepSeek-V3 dimensions (arXiv:2412.19437)",
    "n_layers": 61, "n_dense_layers": 3, "hidden": 7168,
    "n_routed_experts": 256, "expert_inter": 2048, "top_k": 8,
    "n_groups": 8, "max_groups_per_token": 4, "total_params": 671e9,
    "PP": 4, "EP": 8,
    "tokens_per_microbatch_per_rack": 65536,   # 16 sequences x 4096 tokens
    "bytes_act": 2, "bytes_grad": 2,            # BF16
    "sigma_pop": 1.0, "sample_tokens": 4096,
    "phase_pattern": ["EPD", "EPC", "PP", "EPD", "EPC", "PP",
                      "EPD", "EPC", "PP", "EPD", "EPC", "DP"],
}


def dense_params(cfg=LLM_CONFIG) -> float:
    n_moe = cfg["n_layers"] - cfg["n_dense_layers"]
    routed = n_moe * cfg["n_routed_experts"] * 3 * cfg["hidden"] * cfg["expert_inter"]
    return cfg["total_params"] - routed


def _dispatch_counts(rng, cfg) -> np.ndarray:
    """Return C[e_src, g] = sampled tokens of rank e_src sent to group g."""
    E, G, X = cfg["EP"], cfg["n_groups"], cfg["n_routed_experts"]
    per = X // G
    S = cfg["sample_tokens"]
    pop = rng.normal(0.0, cfg["sigma_pop"], size=X)
    C = np.zeros((E, G))
    for e in range(E):
        sc = pop[None, :] + rng.gumbel(size=(S, X))
        grp = np.sort(sc.reshape(S, G, per), axis=2)[:, :, -2:].sum(axis=2)
        top_g = np.argsort(-grp, axis=1)[:, :cfg["max_groups_per_token"]]
        mask = np.full((S, G), -np.inf)
        np.put_along_axis(mask, top_g, 0.0, axis=1)
        sc_m = sc + np.repeat(mask, per, axis=1)
        top_e = np.argpartition(-sc_m, cfg["top_k"], axis=1)[:, :cfg["top_k"]]
        used = np.zeros((S, G), dtype=bool)
        np.put_along_axis(used, top_e // per, True, axis=1)
        C[e] = used.sum(axis=0)
    return C


def make_llm_phase_sequence(seed: int = 0, cfg=LLM_CONFIG) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    P, E = cfg["PP"], cfg["EP"]
    N = P * E
    h, ba = cfg["hidden"], cfg["bytes_act"]
    tok = cfg["tokens_per_microbatch_per_rack"]
    scale = tok / cfg["sample_tokens"]
    gb = 8e-9  # bytes -> Gb
    phases = []
    last_disp = None
    for kind in cfg["phase_pattern"]:
        D = np.zeros((N, N))
        if kind == "EPD":
            for s in range(P):
                C = _dispatch_counts(rng, cfg)
                for e in range(E):
                    for g in range(E):
                        if g != e:
                            D[s * E + e, s * E + g] = C[e, g] * scale * h * ba * gb
            last_disp = D.copy()
        elif kind == "EPC":
            D = last_disp.T.copy()
        elif kind == "PP":
            v = tok * h * ba * gb
            for s in range(P - 1):
                for e in range(E):
                    D[s * E + e, (s + 1) * E + e] = v
                    D[(s + 1) * E + e, s * E + e] = v
        elif kind == "DP":
            v = 2 * (E - 1) / E * dense_params(cfg) / P * cfg["bytes_grad"] * gb
            for s in range(P):
                for e in range(E):
                    D[s * E + e, s * E + (e + 1) % E] = v
        phases.append((kind, D))
    return phases
