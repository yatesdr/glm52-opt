# v20 Boot 10 — CUDA_LAUNCH_BLOCKING localization: FAILING OPERATION NAMED

Date: 2026-07-23 · Boot 10 · Operator: Fable · **Ordered by Derek** (out-of-band vs Sol's plan)
Image: `…-int8-nvme-bt1876-w4a16coop-ckvreset`, `sha256:5f0c7b43daaa…` (Boot 8 image, unchanged)
Compose: Boot 8 **+ one line** — `CUDA_LAUNCH_BLOCKING=1`. GMU 0.980.

**Result: FAILED — and for the first time the traceback names the actual operation.**

## 1. The failing operation

```python
torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
```

`vllm/model_executor/layers/attention/mla_attention.py:1214`, in `forward_impl`.

This is the **MLA query-absorption batched matmul** (`q_nope @ W_UK^T`) — inside the **MTP draft
model**, during production decode-speculator capture.

## 2. Full call path (synchronous launcher)

```text
gpu_worker.py:804          compile_or_warm_up_model
model_runner.py:1013       capture_model
  speculator.py:125        self.speculator.capture()
  autoregressive/cudagraph_utils.py:74   super().capture(create_forward_fn, ...)
  cudagraph_utils.py:531   forward_fn(CUDAGraphMode.NONE)          ← eager, uncaptured
  speculator.py:567        _generate_draft → last_hidden_states, hidden_states = self._run_model(
  speculator.py:438        _run_model  → ret_hidden_states = self.model(**model_inputs)
  compilation/decorators.py:520 / caching.py:217 / aot_compile.py:240
  deepseek_mtp.py:676      forward                                  ← MTP DRAFT MODEL
  mla_attention.py:2017    unified_mla_attention_with_output
  torch/_ops.py:1275       self._op(*args, **kwargs)
  kv_transfer_utils.py:48  wrapper
  mla_attention.py:1214    layer.forward_impl(...)
     → torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
→ CUDA error: an illegal memory access was encountered
```

Every prior boot terminated the stack at `input_batch.py:151 make_dummy` — a downstream
synchronizing H2D copy. With `CUDA_LAUNCH_BLOCKING=1` the report lands on the true launch site.

## 3. This does not match the current hypothesis

**No MoE frames appear anywhere in the log.** A search for `sparkinfer|fused_moe|shared_experts|
moe_runner|nvfp4|grid188|w4a16` frames returns **zero** matches.

The fault is in **MLA attention**, specifically the `mqa_*` (multi-query absorption) path of the
**draft** model — not in the shared-expert/resident-grid overlap that `bf1b32cf` targets.

Stated plainly so it isn't over-read: this does not prove the aux-stream overlap is benign. An
illegal access caused by a concurrent kernel violating exclusive residency could surface in
whatever kernel happens to touch bad state next, and `CUDA_LAUNCH_BLOCKING=1` **serializes
launches, which can itself suppress or relocate a concurrency fault**. But it is now established
that the faulting *launch* is a `bmm` in MLA, and no MoE kernel is on the stack at fault time.

The `mqa` naming is worth noting: `VLLM_DCP_A2A_MAX_TOKENS` gates on `num_mqa_tokens`, the same
quantity this bmm operates on. The DCP/MQA shard geometry and this bmm's operand shapes are
plausibly related — that is a lead, not a finding.

## 4. Progress advanced again

Production decode-speculator capture bar at fault:

| Boot | Config | Bar at fault |
|---|---|---|
| 7 | CG_DIAG on | 3/16 (CG_DIAG: failed at descriptor 8, **M=9**) |
| 8 | W4A16 coop + CKV reset | 6/16 |
| **10** | same image + launch-blocking | **7/16** |

Profiling round again completed **16/16**.

## 5. Memory note — launch-blocking changed the pool

| | Boot 8 | **Boot 10** |
|---|---|---|
| Available KV | 4.14 GiB | 4.14 GiB |
| **KV pool** | 555,520 | **544,000** |

Same available memory, **−11,520 tokens**. Launch-blocking alters allocation/free timing during
profiling, so this is an artifact of the diagnostic, not a property of the build. Do not carry it
into any acceptance record.

## 6. Timeline

```text
01:53:40  boot
02:01:10  weights loaded (305.15 s)
02:06:40  Available KV 4.14 GiB · pool 544,000
02:11:29  production speculator capture
02:11:31  illegal memory access — torch.bmm in MLA forward_impl
```

## 7. Evidence — three independent captures

- `v20-boot10-lb-gmu980-STREAM.log` — **766 KB**, continuous `docker logs -f --timestamps`, started
  mid-boot, immune to teardown races and to the `json-file` 50 MB × 3 rotation policy
- `v20-boot10-lb-gmu980.log` — 693 KB snapshot, written **before** `compose down`
- `v20-boot10-lb-gmu980-inspect.json`

Container `de3ef56013d4…`, StartedAt `2026-07-23T01:53:40Z`.
CN3 clear: 0 containers, 0 CUDA procs, 0 stale `/dev/shm` mmaps.
**NVMe acceptance namespace still pristine (0 entries)** across all ten boots.

## 8. Bearing on Sol's Boot 11 (his "Boot 9") plan

His procedure is staged and pre-verified — patch `90b28e8b…` matches, and all four input bytes in
the Boot 8 image match his pins (`b8d3f53f`, `531a271b`, `5d1cc315`, `a8b4e19c`). It can build
immediately.

Recommendation: **hand him this traceback first.** His diagnosis was reasoned from source history
and from Boot 8 progressing farther; this is the first direct evidence of what actually faults, and
it points at MLA/MQA in the draft model rather than at the MoE shared-expert stream. He may want to
revise before spending a boot — or he may have a reason the overlap manifests exactly here, which
would make his patch the right fix anyway.

Also note his Gate 0 pytest (`test_aux_stream_respects_expert_kernel_capability`,
`test_hybrid_moe_rejects_shared_expert_aux_stream`) **cannot run in-image** — the runtime image
carries no repository tests. Everything else in his Gate 0 is executable.
