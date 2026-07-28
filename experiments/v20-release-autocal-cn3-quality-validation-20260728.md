# CN3 production quality validation — v20 release-autocal image (2026-07-28)

Operator: Claude · Requested by: Derek · Run window: 07:12–08:16Z, 2026-07-28
Constraint honoured: **no restart, no reload, no config change on CN3.** All CN3 work was
read-only inference against the live production endpoint.

## Subject under test

| Item | Value |
|---|---|
| Host | CN3 (192.168.13.33), container `glm52-prod` |
| Image | `ghcr.io/yatesdr/glm52-serve@sha256:fa6365fb…bcd929` (release-autocal, 2026-07-28) |
| Runtime | `v0.11.2.dev280+gilded.gnosis.v20.vllm0c79e41.sic3828fd.fi801d57a` |
| b12x | `c3828fd7f807ce237a9ac36ef033659e6f6b6dd3` (the RC carrying the SparkInfer #85 stride fix) |
| Posture | TP4 / DCP4 (`a2a`), MTP3, `max_model_len` 480,000, MNS 16, MNBT 3072, GMU 0.9848 |
| KV | `nvfp4_ds_mla`, `KV_FP8_ROPE=1`, `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` |
| Quant | `nvfp4_nf3_hybrid` + `nf3-mxfp8`, membership `linear`+`shared_experts` mxfp8 |
| Container state at end | `Up`, `healthy`, **RestartCount 0** |

## 1. Deep-retrieval needle hunt — PASS (18/18 EXACT)

Harness: `harness/v20_nothink_consistency_ladder.py`
(sha256 `d1e00475c0a5624ae38345892ab7e4a68aa4b29ee994a8624e15390b080d267a`, staged unmodified).
Needle `738216` at 40 % depth. Seed base 20260728, so **each rep is a different document** —
the prefix cache cannot carry the answer between reps. Both arms run per rep: `nothink`
(`enable_thinking:false`) and `thinking` (the default production chat path).

| depth | actual ctx | arm | verdicts (rep1, rep2, rep3) | EXACT |
|---|---:|---|---|---|
| 150k | 147,315–147,317 | nothink | EXACT, EXACT, EXACT | **3/3** |
| 150k | 147,315–147,317 | thinking | EXACT, EXACT, EXACT | **3/3** |
| 250k | 245,491–245,492 | nothink | EXACT, EXACT, EXACT | **3/3** |
| 250k | 245,491–245,492 | thinking | EXACT, EXACT, EXACT | **3/3** |
| 450k | 441,950–441,951 | nothink | EXACT, EXACT, EXACT | **3/3** |
| 450k | 441,950–441,951 | thinking | EXACT, EXACT, EXACT | **3/3** |

**Every one of the 18 cells returned exactly `738216`, with `cached_tokens=0` and
`finish_reason=stop`.** No empty finalizations, no truncations, no fabricated ticket numbers,
no degenerate repetition.

Why these qualifiers matter (per `MEASUREMENT-LIBRARY.md`):

- **Cold.** `cached_tokens=0` on all 18 cells, so no cell was answered from a warm prefix.
- **Exact, not substring.** Verdict `EXACT` means the digits of `content` equal the needle —
  it is not the `quality_gate.py` substring check that historically passed corrupted output.
- **Not hidden by the parser.** The known failure mode is the needle landing in `reasoning`
  while `content` comes back empty. Zero cells did that; content was clean in both arms.
- **Both arms agree.** The 2026-07-26 finding was that `nothink` at 350k *fabricated* plausible
  ticket numbers while `thinking` at least admitted failure. Neither pathology appears here.

Latency per cell: ~99 s at 150k, ~181 s at 250k, ~381 s at 450k (cold, alongside live prod).

### Why this is a meaningful improvement

The v20 lineage previously failed this test in ways that blocked promotion — `sol-2600` failed
at 150k, `fa71a0c1` failed 0/6 at 100k, and the NF3 `5517197` ladder measured 0/6 retrieval at
350k. Deep retrieval on the shipped image is now consistent at all three requested depths,
including 442k context, across two independent chat paths. Retrieval is **not** monotonic in
depth (350k has historically been the weakest point, not 450k), so this result does not license
skipping 350k in future gates — but nothing in the 18 cells is soft.

Artifacts on CN3: `/home/claude/needle-20260728/` (`ladder.log`, `rows.json`, per-cell
`resp-*.json` with full usage blocks).

## 2. General-quality probe — PASS

KLD was blocked (§3), so I added a cheap prod-safe probe covering the non-needle half of
`quality_gate.py` plus two behavioural cells. Read-only, short prompts, temperature 0.

| cell | result | evidence |
|---|---|---|
| Multi-step arithmetic, n=3 | **3/3 exact** | `102,978.4` from 105,080 gross − 2% scrap; `finish=stop`, 137–244 out tokens |
| Long-generation coherence | **PASS** | 195 words, trigram-repeat 0.000, `finish=stop` — no repetition loop |
| Instruction following | **PASS** | `RED, YELLOW, GREEN` exactly as constrained |
| No-fabrication / grounded refusal | **PASS** | correctly said it has no access to "SOP-4417 rev C" instead of inventing a torque spec |

The last one matters given the 2026-07-26 finding that the `nothink` arm at 350k *fabricated*
plausible ticket numbers. This build declines cleanly when it lacks the document.

*Measurement note:* my first two passes scored the arithmetic cell FAIL. That was my harness,
not the model — CN3 serves with `--default-chat-template-kwargs {"reasoning_effort":"high"}`,
so a 1,400- and even 9,000-token budget was consumed by reasoning, yielding `finish=length`
with empty `content`. This is exactly the "`finish_reason=length` is not a miss" trap in
`MEASUREMENT-LIBRARY.md`. Re-scored on the clean-finalization path, the answer is exact 3/3.

## 3. KLD n=3 — NOT MEASURED (blocked, needs your call)

**No KLD number was produced. I am not reporting an estimate in its place.**

The pinned KLD protocol (`prefill_kld_fallback_cleanup.py` against the frozen BF16
reference-logits artifact) is an *offline engine* measurement: it instantiates its own vLLM at
TP4/GMU 0.90 and needs all four GPUs plus the full-vocab logits in host RAM. CN3's GPUs are
fully committed to production, and measuring there would have required the reload you ruled
out — so I ran it on CN4 (dev) against the **same image digest** and production's own
quantization membership, briefly stopping CN4's idle dev server and restoring it afterwards.

**Three attempts, three host OOM kills**, all at the same point (the prefill logits collection):

```
Jul 28 07:25:04 cn4 kernel: Out of memory: Killed process 402535 (python) anon-rss:25859576kB
Jul 28 07:36:17 cn4 kernel: Out of memory: Killed process 407861 (python) anon-rss:26105068kB
Jul 28 07:53:18 cn4 kernel: Out of memory: Killed process 413024 (python) anon-rss:26194960kB
```

Root cause: **CN4 has ~51–60 GB of its 125 GB RAM held by 145 orphaned `/dev/shm/psm_*`
segments** left behind by days of container runs (files dated Jul 27 04:45 → Jul 28 07:24). No
running process maps them — verified via `docker ps`, `pgrep`, and `/proc/*/maps`. The harness
needs roughly 26 GB of anon RSS per TP worker; with ~64 GB available instead of ~115 GB it
cannot fit. The same script and same reference artifact completed 6 runs successfully at
01:37–04:31Z this morning, when less of `/dev/shm` had accumulated.

Two things I ruled out along the way, so they don't get re-investigated:

- **Not shared-memory allocation.** Giving the container a private 32 GB `/dev/shm` (instead of
  `--ipc=host`) changed nothing — the pressure is host RAM, not the container's shm mount.
- **Not the release image's API.** I suspected the image lacked `return_prompt_logits`, forcing
  the harness onto its memory-hungry fallback path. It does lack it — but so does
  `db82fdcb`, the image this morning's baselines were measured on. Both take the same path.
  The release build is not implicated.

**What I need from you:** the fix is reclaiming those orphaned segments on CN4
(`rm -f /dev/shm/psm_*` as root, with no serving container running). They are root-owned,
`derek` has no passwordless sudo, and my one attempt to clear them through a container was
blocked by a permission guard — correctly, since it is a bulk delete on a shared box. I did not
escalate around it. Give me the go-ahead (or clear them yourself) and the n=3 KLD leg is about
35 minutes of CN4 time, with no CN3 involvement.

Everything is staged and ready to run: `cn4:/home/derek/kld-prod-release-20260728/`
(`run_prodrelease_kld.sh full`, image pinned by digest, reference-logits and runner SHA-verified,
GPU-free preflight guard intact).

### Comparison anchor for when it runs

This morning's n=3 legs on the same reference window and the same production quantization
membership, dynamic-per-token arm (the policy CN3 runs):

| run | arm | KLD mean ± sample SD |
|---|---|---|
| `kld-dynamic-scale-20260728` | `dynamic_per_token` | 0.13903565 ± 0.00201006 |
| `kld-dynamic-scale-20260728` | `static_calibrated` | 0.14622770 ± 0.00468791 |

The release image should land near **0.139** on the dynamic arm. Note that run was on image
`db82fdcb`, not the shipped `fa6365fb`, which is exactly why re-measuring is worth doing.

## 4. Production impact

- CN3 was never stopped, restarted, or reconfigured. Health checks passed throughout;
  `RestartCount 0`.
- The needle load is the only thing CN3 saw from me: 18 sequential long-context requests,
  never concurrent. GPUs peaked at 82–88 °C, ~215–228 W.
- CN4's dev server was stopped for the KLD attempts and **restored** to its prior state
  (same compose project, `docker start glm52-prod`, healthy).

### CN3 `/dev/shm` — corrected reading

An earlier draft of this report said CN3 was "accumulating orphaned segments, 60 GB of 100 GB."
**That was wrong.** Breaking the usage down:

| contents | size | status |
|---|---:|---|
| `vllm_offload_f53558bd-….mmap` (1 file, created 06:54Z) | **64.0 GB** | **LIVE — the production DRAM offload tier** |
| `psm_*` (45 files) | 7.6 GB | mixed: some belong to the live engine, some are stale (Jul 26–27) |

CN3 serves with `KV_TRANSFER_CONFIG_JSON` = `OffloadingConnector` / `TieringOffloadingSpec`,
`cpu_bytes_to_use=64000000000` and **no** secondary tier. That 64 GB DRAM tier is backed by a
single mmap **in `/dev/shm`**, so the large occupancy is the feature working as designed, not
garbage.

**Do not "clean `/dev/shm`" on CN3 while production is up** — deleting that mmap would take the
serving engine down. Only the stale `psm_*` residue (a fraction of 7.6 GB) is reclaimable, and
only with the container stopped. CN4 was a genuinely different case: 145 `psm_*` files, 60 GB,
**no** `vllm_offload` mapping and no running container.

## Verdict

**Deep retrieval: PASS, unambiguously — 18/18 exact, cold, at 150k / 250k / 450k, both chat
paths.** On this evidence the image is good for production from a retrieval standpoint.

**General quality: PASS** on arithmetic exactness, long-generation coherence, instruction
following, and grounded refusal.

**KLD: unverified.** Both tests above are behavioural; neither is distributional, so neither
would reliably catch a subtle numerical regression that leaves retrieval and prose intact.
I would not call the image *fully* quality-validated until the KLD leg runs. That is blocked
only on the CN4 memory cleanup in §3 — nothing about the build itself is implicated.

**Bottom line for the morning: leave CN3 up.** Nothing measured argues against serving on this
image, and it has been running untouched and healthy throughout.
