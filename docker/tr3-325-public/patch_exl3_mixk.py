#!/usr/bin/env python3
"""patch_exl3_mixk.py (v6) — mixed-K (per-expert 3/4 bpw) rank-sliced EXL3 MoE
for the installed vLLM exl3.py (verdictai glm52-exl3-sparkinfer v20final).

v6 over v5:
  * removes the fused path's duplicate global [E,3I] rotation table; the
    cooperative kernel resolves global expert ids through the immutable
    tier-local map and reads the existing per-tier rotation tables directly.

v5 over v4:
  * optional exact one-pack/one-launch mixed-K decode for M<=8 through the
    public SparkInfer planned hybrid API; larger decode and all prefill retain
    the byte-identical serial tier path;
  * preserves the trained EXL3 projection rotations and constructs a per-layer
    global->tier-local descriptor map.

v4 over v3:
  * persistently plans the BF16 result buffer beside the FP32 accumulator;
    the terminal FP32->BF16 conversion writes into that buffer instead of
    allocating a new 36 MiB tensor at MNBT=3072 after the KV pool is live.

v3 over v2:
  * runtime cache mirrors the uniform path: module-global, keyed on dims and
    tier signature, shared across all target layers (v2 keyed per layer and
    allocated per-layer scratch -> OOM during the profile forward);
  * one decode arena + one prefill arena aliased across tiers via exact-shape
    views (tiers execute sequentially); net scratch ~= uniform + accumulator;
  * env knobs identical to the uniform path: VLLM_EXL3_TRELLIS_MIN_M/MAX_M,
    VLLM_EXL3_TRELLIS_BLOCK_M, VLLM_EXL3_PREFILL_TRELLIS/PREFILL_BLOCK_M;
    max_batched from layer.exl3_max_num_batched_tokens like uniform;
  * keeps the v2 mapped top-k-sum fix (route map passed as output_expert_map).

Idempotent + upgradeable: v6 marker short-circuits; any older mixk install is
rebuilt in place from the v1 marker. Backup at exl3.py.orig on first apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER_V1 = "# === EXL3-MIXK-PATCH v1 ==="
MARKER_V6 = (
    "# === EXL3-MIXK-PATCH v6 "
    "(exact fused M<=8, tier-local rotations) ==="
)


def find_exl3() -> Path:
    import vllm
    import os
    return Path(os.path.dirname(vllm.__file__)) / "model_executor/layers/quantization/exl3.py"


E1_OLD = '''        self.rank_sliced_metadata = dict(metadata)
        self.bits = float(metadata["bits"])
        self.codebook = str(metadata["codebook"])'''
E1_NEW = '''        self.rank_sliced_metadata = dict(metadata)
        bits_field = metadata["bits"]
        if isinstance(bits_field, str) and bits_field.strip().lower() == "mixed":
            k_values = sorted(int(k) for k in metadata.get("k_values", ()))
            if not k_values or any(k not in (3, 4, 5, 6) for k in k_values):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires k_values within 3..6, got "
                    f"{metadata.get('k_values')!r}"
                )
            self.bits = None
            self.mixed_k_values = tuple(k_values)
        else:
            self.bits = float(bits_field)
            self.mixed_k_values = None
        self.codebook = str(metadata["codebook"])'''

E2_OLD = 'preallocate=rank_sliced and suffix in {"suh", "svh", "trellis"},'
E2_NEW = ('preallocate=rank_sliced and suffix in '
          '({"suh", "svh"} if getattr(self.quant_config, "mixed_k_values", None) '
          'else {"suh", "svh", "trellis"}),')

E3_OLD = '''    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        api = _load_sparkinfer_trellis()'''
E3_NEW = '''    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        if getattr(self.quant_config, "mixed_k_values", None):
            return self._prepare_rank_sliced_weights_mixk(layer)
        api = _load_sparkinfer_trellis()'''

E5_OLD = '''        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])'''
E5_NEW = '''        if getattr(layer, "exl3_mixk", None) is not None:
            return self._apply_rank_sliced_mixk(layer, x, topk_weights, topk_ids)
        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        m = int(x.shape[0])'''


APPEND = MARKER_V1 + "\n" + MARKER_V6 + '''
# Mixed-K rank-sliced EXL3: experts partitioned by native trellis width into
# K-homogeneous tiers; one sparkinfer trellis_moe Weights per tier per layer;
# global routing with per-tier route/output expert maps (-1 filtered by the
# w4a16 route pack and its mapped top-k sum). Plans and scratch are shared
# across all layers with the same tier signature, exactly like the uniform
# runtime; the two tier arenas alias one allocation since tiers execute
# sequentially within a layer.

_MIXK_RUNTIMES: dict[tuple, dict] = {}


def _mixk_prepare_rank_sliced_weights(self, layer) -> None:
    api = _load_sparkinfer_trellis()
    num_experts = int(layer.local_num_experts)
    hidden_size = int(layer.exl3_hidden_size)
    intermediate_size = int(layer.exl3_intermediate_size_per_partition)
    allowed = set(self.quant_config.mixed_k_values)

    w13_p = layer.w13_trellis
    w2_p = layer.w2_trellis
    sids = list(w13_p.exl3_shard_ids)
    if len(sids) != 2 or len(w2_p.exl3_shard_ids) != 1:
        raise ValueError("mixed EXL3 expects two w13 shards and one w2 shard")
    w2_sid = w2_p.exl3_shard_ids[0]

    k_of = []
    for e in range(num_experts):
        widths = {
            int(w13_p.exl3_tensors[(e, sids[0])].shape[-1]),
            int(w13_p.exl3_tensors[(e, sids[1])].shape[-1]),
            int(w2_p.exl3_tensors[(e, w2_sid)].shape[-1]),
        }
        if len(widths) != 1:
            raise ValueError(f"expert {e}: inconsistent trellis widths {widths}")
        width = widths.pop()
        if width % 16 or width // 16 not in allowed:
            raise ValueError(
                f"expert {e}: trellis width {width} outside declared k_values "
                f"{sorted(allowed)}"
            )
        k_of.append(width // 16)
    tiers = {}
    for e, k in enumerate(k_of):
        tiers.setdefault(k, []).append(e)

    gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
    gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
    down_suh = self._rank_sliced_backing(layer, "w2_suh")
    down_svh = self._rank_sliced_backing(layer, "w2_svh")
    tile_config = self._trellis_tile_config(hidden_size, intermediate_size)
    marker = next(iter(layer.w13_mcg.exl3_tensors.values()))
    device = gate_suh.device

    tier_entries = []
    for k in sorted(tiers):
        experts = tiers[k]
        idx = torch.tensor(experts, dtype=torch.int64, device=device)
        w13_k = torch.stack([
            torch.stack([w13_p.exl3_tensors[(e, sid)] for e in experts])
            for sid in sids
        ]).contiguous()
        w2_k = torch.stack(
            [w2_p.exl3_tensors[(e, w2_sid)] for e in experts]
        ).contiguous()
        expected_w13 = (2, len(experts), hidden_size // 16,
                        intermediate_size // 16, 16 * k)
        expected_w2 = (len(experts), intermediate_size // 16,
                       hidden_size // 16, 16 * k)
        if tuple(w13_k.shape) != expected_w13 or tuple(w2_k.shape) != expected_w2:
            raise ValueError(
                f"mixed EXL3 tier K={k} slab geometry mismatch: "
                f"w13={tuple(w13_k.shape)}, w2={tuple(w2_k.shape)}"
            )
        g_suh = gate_suh.index_select(0, idx).contiguous()
        u_suh = up_suh.index_select(0, idx).contiguous()
        d_svh = down_svh.index_select(0, idx).contiguous()
        rotations = torch.cat(
            (
                gate_svh.index_select(0, idx),
                up_svh.index_select(0, idx),
                down_suh.index_select(0, idx),
            ),
            dim=1,
        ).contiguous()
        weights = api.prepare_weights(
            w13_k, w2_k,
            gate_suh=g_suh, up_suh=u_suh,
            intermediate_rotations=rotations,
            down_svh=d_svh,
            codebook="mcg", mcg=marker,
            tile_config=tile_config,
        )
        route_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        route_map[idx] = torch.arange(len(experts), dtype=torch.int32, device=device)
        tier_entries.append({
            "k": k,
            "num_experts": len(experts),
            "global_expert_ids": tuple(experts),
            "weights": weights,
            "route_expert_map": route_map,
            "_slabs": (w13_k, w2_k, g_suh, u_suh, rotations, d_svh),
        })
        logger.info(
            "EXL3 mixk %s: tier K=%d holds %d/%d experts",
            getattr(layer, "prefix", "moe"), k, len(experts), num_experts,
        )

    w13_p.exl3_tensors.clear()
    w2_p.exl3_tensors.clear()
    layer.exl3_mixk = {
        "tiers": tier_entries,
        "signature": tuple((t["k"], t["num_experts"]) for t in tier_entries),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "global_gate_suh": gate_suh,
        "global_up_suh": up_suh,
        "global_down_svh": down_svh,
        "tier_local_map": None,
    }
    layer.exl3_trellis_tile_config = tile_config


def _mixk_scratch_view(backing: torch.Tensor, spec) -> torch.Tensor:
    nbytes = 1
    for dim in spec.shape:
        nbytes *= int(dim)
    nbytes *= torch.empty((), dtype=spec.dtype).element_size()
    return backing.narrow(0, 0, nbytes).view(spec.dtype).view(tuple(spec.shape))


def _mixk_runtime(self, layer, x, topk_ids):
    mix = layer.exl3_mixk
    min_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MIN_M", 4)
    max_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)
    block_m = _positive_env_int("VLLM_EXL3_TRELLIS_BLOCK_M", 8)
    prefill_trellis = os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") == "1"
    prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)
    try:
        fused_max_m = int(os.environ.get("VLLM_EXL3_MIXK_FUSED_MAX_M", "0"))
    except ValueError as exc:
        raise ValueError(
            "VLLM_EXL3_MIXK_FUSED_MAX_M must be an integer in [0,8]"
        ) from exc
    if fused_max_m < 0 or fused_max_m > 8:
        raise ValueError(
            "VLLM_EXL3_MIXK_FUSED_MAX_M must be in [0,8]; the measured "
            "production crossover does not admit larger shapes"
        )
    if min_trellis_m > max_trellis_m:
        raise ValueError("VLLM_EXL3_TRELLIS_MIN_M cannot exceed VLLM_EXL3_TRELLIS_MAX_M")
    if not prefill_trellis:
        raise ValueError(
            "mixed-K EXL3 requires the planned prefill path "
            "(VLLM_EXL3_PREFILL_TRELLIS=1); the eager parity fallback has no "
            "mixed-K support"
        )
    max_batched_tokens = max(
        int(layer.exl3_max_num_batched_tokens),
        int(x.shape[0]),
    )
    topk = int(topk_ids.shape[1])
    key = (
        _runtime_scope_id(self.quant_config),
        x.device.index,
        x.dtype,
        mix["hidden_size"],
        mix["intermediate_size"],
        mix["num_experts"],
        mix["signature"],
        topk,
        max_batched_tokens,
        min_trellis_m,
        max_trellis_m,
        block_m,
        prefill_block_m,
        fused_max_m,
        layer.exl3_trellis_tile_config,
    )
    runtime = _MIXK_RUNTIMES.get(key)
    if runtime is not None:
        return runtime
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "mixed-K EXL3 runtime must be planned during the eager profile "
            "pass before CUDA graph capture"
        )
    api = _load_sparkinfer_trellis()

    def make_plan(k, tier_experts, plan_max_tokens, plan_block_m):
        caps = api.Caps(
            max_tokens=plan_max_tokens,
            num_topk=topk,
            num_experts=tier_experts,
            hidden_size=mix["hidden_size"],
            intermediate_size=mix["intermediate_size"],
            route_num_experts=mix["num_experts"],
            block_size_m=plan_block_m,
            trellis_bits=k,
            tile_config=layer.exl3_trellis_tile_config,
            input_dtype=x.dtype,
            device=x.device,
        )
        return api.plan(caps)

    tiers_rt = []
    for k, tier_experts in mix["signature"]:
        decode_plan = make_plan(k, tier_experts, max_trellis_m, block_m)
        prefill_plan = (
            make_plan(k, tier_experts, max_batched_tokens, prefill_block_m)
            if max_batched_tokens > max_trellis_m else None
        )
        tiers_rt.append({"k": k, "num_experts": tier_experts,
                         "decode_plan": decode_plan, "prefill_plan": prefill_plan})
    hybrid_plan = None
    fused_signature = (
        len(tiers_rt) == 2
        and tuple(t["k"] for t in tiers_rt) == (3, 4)
    )
    if fused_max_m and fused_signature:
        hybrid_plan = api.plan_hybrid(
            tiers_rt[0]["decode_plan"],
            tiers_rt[1]["decode_plan"],
        )
    elif fused_max_m:
        # The target transformer layers are K3/K4 mixed, while the MTP draft
        # layer is K3-only.  Keep unsupported signatures on the unchanged
        # serial per-tier path instead of broadening the fused-kernel contract.
        logger.info(
            "EXL3 mixk fused decode not eligible for tiers %s; using exact "
            "serial tier path",
            mix["signature"],
        )

    def arena_for(plans):
        specs = [p.scratch_specs()[0] for p in plans if p is not None]
        if not specs:
            return None
        nbytes = 0
        for spec in specs:
            b = 1
            for dim in spec.shape:
                b *= int(dim)
            b *= torch.empty((), dtype=spec.dtype).element_size()
            nbytes = max(nbytes, b)
        return torch.empty(nbytes, dtype=torch.uint8, device=x.device)

    decode_arena = arena_for(
        [t["decode_plan"] for t in tiers_rt] + [hybrid_plan]
    )
    prefill_arena = arena_for([t["prefill_plan"] for t in tiers_rt])
    accum = torch.zeros(
        (max(max_batched_tokens, max_trellis_m), mix["hidden_size"]),
        dtype=torch.float32, device=x.device,
    )
    # Plan the terminal conversion target before KV-cache sizing. Allocating
    # it lazily with accum.to(x.dtype) costs 36 MiB at MNBT=3072/hidden=6144
    # and can fail after the KV pool consumes the remaining contiguous VRAM.
    # The buffer is safe to share across target layers for the same reason as
    # accum: tiers and layers execute serially, and the copy is issued only
    # after every tier has finished consuming x.
    output = torch.empty(
        (max(max_batched_tokens, max_trellis_m), mix["hidden_size"]),
        dtype=x.dtype, device=x.device,
    )
    runtime = {
        "api": api,
        "tiers": tiers_rt,
        "hybrid_plan": hybrid_plan,
        "fused_max_m": fused_max_m,
        "decode_arena": decode_arena,
        "prefill_arena": prefill_arena,
        "accum": accum,
        "output": output,
        "min_trellis_m": min_trellis_m,
        "max_trellis_m": max_trellis_m,
        "max_batched_tokens": max_batched_tokens,
    }
    _MIXK_RUNTIMES[key] = runtime
    arena_mib = sum(
        a.numel() / (1 << 20) for a in (decode_arena, prefill_arena) if a is not None
    )
    logger.info(
        "EXL3 mixk runtime: tiers %s, decode window [%d, %d], "
        "fused_max_m=%d, prefill_m=%d, shared arenas %.1f MiB",
        mix["signature"], min_trellis_m, max_trellis_m, fused_max_m,
        max_batched_tokens, arena_mib,
    )
    return runtime


def _mixk_apply(self, layer, x, topk_weights, topk_ids):
    runtime = self._mixk_runtime(layer, x, topk_ids)
    m = int(x.shape[0])
    if m > runtime["max_batched_tokens"]:
        raise ValueError(
            f"mixed-K EXL3 batch exceeds planned capacity: m={m}, "
            f"capacity={runtime['max_batched_tokens']}"
        )
    decode = runtime["min_trellis_m"] <= m <= runtime["max_trellis_m"]
    arena = runtime["decode_arena"] if decode else runtime["prefill_arena"]
    accum = runtime["accum"][:m]
    output = runtime["output"][:m]
    mix_tiers = layer.exl3_mixk["tiers"]
    for rt_tier, mix_tier in zip(runtime["tiers"], mix_tiers):
        if (rt_tier["k"], rt_tier["num_experts"]) != (
            mix_tier["k"], mix_tier["num_experts"]
        ):
            raise RuntimeError(
                "mixed-K tier signature drifted between runtime and layer weights"
            )

    hybrid_plan = runtime["hybrid_plan"]
    if hybrid_plan is not None and 1 <= m <= runtime["fused_max_m"]:
        if layer.exl3_mixk["tier_local_map"] is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "mixed-K fused tier map must be built during eager profile"
                )
            layer.exl3_mixk["tier_local_map"] = (
                runtime["api"].build_tier_local_map(
                    hybrid_plan,
                    mix_tiers[0]["global_expert_ids"],
                    mix_tiers[1]["global_expert_ids"],
                )
            )
        scratch = _mixk_scratch_view(
            runtime["decode_arena"],
            hybrid_plan.scratch_specs()[0],
        )
        hybrid_output = runtime["accum"][: runtime["max_trellis_m"]]
        binding = runtime["api"].bind_hybrid(
            hybrid_plan,
            scratch=scratch,
            a=x,
            tier0_weights=mix_tiers[0]["weights"],
            tier1_weights=mix_tiers[1]["weights"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            tier_local_map=layer.exl3_mixk["tier_local_map"],
            global_gate_suh=layer.exl3_mixk["global_gate_suh"],
            global_up_suh=layer.exl3_mixk["global_up_suh"],
            global_down_svh=layer.exl3_mixk["global_down_svh"],
            output=hybrid_output,
        )
        fused = runtime["api"].run_hybrid(binding=binding)
        output.copy_(fused)
        return output

    first = True
    for rt_tier, mix_tier in zip(runtime["tiers"], mix_tiers):
        plan = rt_tier["decode_plan"] if decode else rt_tier["prefill_plan"]
        if plan is None:
            raise RuntimeError("mixed-K EXL3 prefill plan missing for large batch")
        scratch = _mixk_scratch_view(arena, plan.scratch_specs()[0])
        binding = runtime["api"].bind(
            plan,
            scratch=scratch,
            a=x,
            weights=mix_tier["weights"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=mix_tier["route_expert_map"],
            # the w4a16 top-k sum resolves sum_expert_map from
            # output_expert_map first; bind keys its mapped launch variant on
            # output_expert_map only, so pass the same map to align the
            # preplanned launch with run's requested contract.
            output_expert_map=mix_tier["route_expert_map"],
            output=accum if first else None,
        )
        out = runtime["api"].run(binding=binding)
        if not first:
            accum.add_(out)
        first = False
    output.copy_(accum)
    return output


Exl3MoEMethod._prepare_rank_sliced_weights_mixk = _mixk_prepare_rank_sliced_weights
Exl3MoEMethod._mixk_runtime = _mixk_runtime
Exl3MoEMethod._apply_rank_sliced_mixk = _mixk_apply
'''


def main() -> None:
    if "--print-path" in sys.argv:
        print(find_exl3())
        return
    path = find_exl3()
    src = path.read_text()
    if MARKER_V6 in src:
        print(f"already patched (v6): {path}")
        return
    if MARKER_V1 in src:
        # rebuild any older mixk install: keep the in-place E-edits, replace
        # the appended block wholesale from the v1 marker onward
        src = src[: src.index(MARKER_V1)].rstrip() + "\n\n" + APPEND + "\n"
        path.write_text(src)
        import py_compile
        py_compile.compile(str(path), doraise=True)
        print(f"upgraded to v6: {path}")
        return
    missing = [name for name, old in (("E1", E1_OLD), ("E2", E2_OLD),
                                      ("E3", E3_OLD), ("E5", E5_OLD))
               if old not in src]
    if missing:
        sys.exit(f"ANCHOR MISMATCH {missing} in {path}; image lineage differs "
                 "from v20final — do not apply blindly.")
    for old, new in ((E1_OLD, E1_NEW), (E2_OLD, E2_NEW),
                     (E3_OLD, E3_NEW), (E5_OLD, E5_NEW)):
        if src.count(old) != 1:
            sys.exit(f"anchor not unique ({src.count(old)}x): {old[:60]!r}")
        src = src.replace(old, new)
    src = src + "\n\n" + APPEND + "\n"
    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(src)
    import py_compile
    py_compile.compile(str(path), doraise=True)
    print(f"patched {path} (v6, backup at {backup})")


if __name__ == "__main__":
    main()
