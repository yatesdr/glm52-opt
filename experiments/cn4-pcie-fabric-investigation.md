# CN4 PCIe fabric investigation — 2026-07-25

## Outcome

CN4's hardware links are healthy, but the production Compose selected the
wrong NCCL transport policy for its two-IIO-domain topology.

`NCCL_P2P_LEVEL=SYS` permits direct GPU P2P across the two PEX8747 switches.
On CN4 those switches terminate in separate Skylake-W IIO domains. NCCL
models the path as a normal same-CPU `PHB` route, but the measured direct path
is strongly asymmetric and concurrent transfers collapse. The packed-CKV
prefill gather therefore ran at only about 0.65 GB/s.

`NCCL_P2P_LEVEL=PXB` retains direct P2P between GPUs behind the same PEX
switch and selects NCCL shared-memory transport between the two switches. It
improves the exact packed-CKV all-gather by 10.1–11.3x without changing the
custom CE-DMA `i8_ring` path.

The production-candidate Compose now uses `PXB`. The byte-pinned v20
qualification boot and end-to-end throughput gate passed on CN4.

## Physical topology

```text
GPU0 + GPU1 -> PEX8747 #1 -> Gen3 x16 -> CPU IIO domain 0000:16
GPU2 + GPU3 -> PEX8747 #2 -> Gen3 x16 -> CPU IIO domain 0000:64
```

All GPU downstream links and both PEX upstream links reach Gen3 x16 under
load. Idle Gen1 x16 is normal link power management.

Measured no-model peer bandwidth:

| Path | Result |
|---|---:|
| Same PEX, one direction | 14.1–14.2 GB/s |
| Cross IIO, upper to lower | about 10.1 GB/s |
| Cross IIO, lower to upper | about 4.5–5.5 GB/s |
| Two simultaneous cross edges, forward aggregate | about 10.1 GB/s |
| Two simultaneous cross edges, reverse aggregate | about 4.9 GB/s |
| Four-edge bidirectional cross traffic, aggregate | about 6.5–6.9 GB/s |

The asymmetry follows the IIO path rather than a GPU or slot.

## Exact packed-CKV gather proof

The v20 source gathers head-independent packed CKV records through PyNCCL.
With `KV_FP8_ROPE=1`, each record is 368 bytes. The proof uses the runtime's
actual rank-concatenated all-gather layout and checks output correctness.

| Local tokens/rank | Total context equivalent | `SYS` | `PXB` | Speedup |
|---:|---:|---:|---:|---:|
| 13,750 | 55,000 | 22.382 ms | 2.213 ms | 10.1x |
| 120,000 | 480,000 | 210.339 ms | 18.608 ms | 11.3x |

The same `PXB` results reproduced inside the exact production candidate
`fa71a0c1e06e…`:

```text
55k:  2.211472 ms, 6.864 GB/s
480k: 18.603696 ms, 7.121 GB/s
```

## All-reduce interaction

`PXB` improves NCCL all-reduce fallback by roughly 8–11x on production-shaped
sizes. The custom block-INT8 `i8_ring` timing is effectively unchanged,
because it uses CUDA IPC/copy engines rather than NCCL's P2P-level routing.

For the 44,040,192-byte cell:

| Path | `SYS` | `PXB` |
|---|---:|---:|
| NCCL fallback | 102.316 ms | 9.176 ms |
| custom `i8_ring` | about 8.45 ms | about 8.47 ms |

Changing CUDA rank order from `0,1,2,3` to `0,1,3,2` did not provide a
repeatable gain in the exact candidate with `PXB`, so the production Compose
keeps the original rank order.

## Controls ruled out

- Locking uncore at 2.4 GHz did not improve the matrices.
- Disabling PEX Relaxed Ordering did not improve them; original state restored.
- Disabling Intel root-port Extended Tags did not improve the production-shaped
  collective; original state restored.
- ACS is disabled, ASPM is disabled, VT-d is disabled, and MPS/MRRS values are
  sane.
- PEX AER correctable status was cleared, remained clear during more than
  85 GiB of traffic, and no active link retry/error signature appeared.
- Alternate logical ring order produced only a small raw microbenchmark change
  and no repeatable exact-image improvement.

## PEX8747 errata review

Broadcom document `DB05-000346-03` covers the PEX 87xx family. CN4 uses
PEX8747 revision CA.

- The read-only completion starvation and Gen3 x8 LCRC items are marked for
  revision BA and do not match CN4's CA/x16 devices.
- The PAT auto-load item could matter only if the board supplied invalid PAT
  entries. Active virtual-channel/arbitration state matches the stable CN3
  system, so there is no evidence for that condition.
- A stale Advisory Non-Fatal correctable-status bit was observed but masked,
  cleared normally, and did not re-latch under load.

The errata is useful for eliminating known mechanisms, but it does not explain
CN4's measured slowdown.

## Production change

Only this transport policy changed:

```yaml
NCCL_P2P_LEVEL: PXB
```

The image, CUDA rank order, model configuration, `i8_ring` wire mode, memory
settings, and all source bytes remain unchanged.

## End-to-end v20 qualification

Image and process:

```text
image:       sha256:fa71a0c1e06e29db88364dcaa047c09c37662fda105551a988a2d09e54fdec86
container:   a5982029f831149d7a8f19276c79a3bf9a88be0cd7f48b0dfa1c593a755d6dd2
started:     2026-07-25T13:48:13.965085908Z
restart:     no / RestartCount=0
host boot:   189fc5c2-65e3-46fa-bbd3-8bbfb67b2f3c
```

All production graph captures passed, including PIECEWISE 7/7, target FULL
16/16, and the MTP speculator capture that exposed the earlier MLA query-BMM
fault. The API became healthy and returned a finalized arithmetic answer.

The live transport selectors were:

```text
SPARKINFER_PCIE_DMA_FP8=i8_ring -> normalized i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
NCCL_P2P_LEVEL=PXB
TP dispatch: B12X_PCIE_ONESHOT_DMA, then PYNCCL
DCP CKV prefetch group: PYNCCL
```

The GPU KV pool was 501,504 tokens (3.79 GiB), only 1,504 tokens above the
500k acceptance floor. It passed, but the capacity margin is thin.

Fresh-prefix prefill results:

| Cell | Prompt tokens | Server prefill | Cache evidence |
|---|---:|---:|---|
| 8k counted | 7,919 | 1,364 tok/s | unique first block |
| 55k counted | 54,210 | 1,115 tok/s | unique first block |
| 55k telemetry repeat | 54,208 | 1,221 tok/s | unique first block |

Aggregate cache counters after the cells were 141,533 prefix queries, zero
hits, and zero cached prompt tokens. This is a 3.2--3.5x recovery from CN4's
previous roughly 300--353 tok/s long-prefill ceiling.

Decode on the same process (256 generated tokens/request, `ignore_eos`):

| Concurrency | Aggregate tok/s | Median per-user | MTP acceptance |
|---:|---:|---:|---:|
| 1 | 53.64 | 53.64 | 48.4% |
| 4 | 110.68 | 28.10 | 61.0% |
| 8 | 133.04 | 16.92 | 55.8% |
| 16 | 171.35 | 11.02 | 56.8% |

Every request completed, client/server token counters agreed, requested
concurrency was achieved, and container/host identity remained unchanged.

## Residual prefill gap is not PCIe starvation

The final 55k request was sampled at 250 ms. All GPUs held Gen3 x16 during
load and showed:

| GPU | Average SM utilization | Average power | Average SM clock | Peak temp |
|---:|---:|---:|---:|---:|
| 0 | 98.1% | 196.3 W | 2,510 MHz | 44 C |
| 1 | 98.2% | 193.7 W | 2,495 MHz | 49 C |
| 2 | 98.3% | 198.2 W | 2,516 MHz | 57 C |
| 3 | 98.3% | 213.1 W | 2,501 MHz | 66 C |

Only 1% of samples were below 50% SM utilization. PXB therefore removed the
inter-GPU starvation mechanism. The remaining gap from CN3's 1.5--1.8k class
results is now in the compute/kernel/software lane, not the PCIe-routing lane.

## CN4 stability posture

The first end-to-end boot was invalidated by a silent host reset because the
post-reboot service restored only the 300 W cap and left graphics clocks
unlocked to 3,090 MHz. There was no CUDA, Xid, AER, MCE, OOM, panic, or
application failure trail.

The corrected posture was proved without loading a model:

```text
power limit:       300 W on all four GPUs
graphics ceiling: 0--2600 MHz (actual load clock 2587 MHz)
burn:              four synchronized 8192x8192 BF16 matmuls
duration:          180 s / 28,821 iterations
result:            PASS, unchanged boot id
```

`nvidia-powercap.service` now persists `-pm 1`, `-pl 300`, and
`-lgc 0,2600`. The successful v20 qualification ran under that policy.
