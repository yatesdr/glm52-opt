# CN4 evidence archive — 2026-07-26 (v20 NF3 `5517197` deep-context session)

Pulled from `cn4:/home/derek/` on 2026-07-26. **CN4 is the dev box and gets rebooted and rebuilt
freely — these are the only durable copies.** Analysis and interpretation live in
[`design/v20-nf3-350k-findings-handoff-20260726.md`](../../../design/v20-nf3-350k-findings-handoff-20260726.md)
and [`MEASUREMENT-LIBRARY.md`](../../../MEASUREMENT-LIBRARY.md).

**Image for every result here:** `glm52-serve:v20-5517197-pxb-20260725`
(id `2566f905f132`, base `e7a8a854…`, vLLM `5517197`, SparkInfer `be0edcaa`, FlashInfer `801d57a`).
Stock bytes plus one launcher-script layer making `NCCL_P2P_LEVEL` overridable. **No source patch.**
One live container throughout, started `2026-07-25T23:23:59Z`, `restarts=0`, `OOMKilled=false`.
Needle `738216` at 40% depth.

| path | what it is | headline |
|---|---|---|
| `nothink-consistency-20260726T0125Z/` | Consistency ladder, 12 cells, `nothink` vs `thinking`, 3 reps × 350k/450k, **all `cached_tokens=0`** | **350k retrieved 0/6, 450k retrieved 5/6** |
| `v20-chat-token-identity-250k.tar.gz` | Sol's `proofs#133` token-identity probe at 250k (5 response bodies + both prompt-token-id dumps + metadata + summary). 18 MB raw → 132 K gz | Same input IDs → different streams; `154842` (`</think>`) emission varies |
| `thinkmode-250k-20260726T0105Z/` | Thinking-mode finalization cells at 250k: `{}`, `enable_thinking:false`, `thinking:false`, untemplated raw | `{}` → `content=None` cold; `enable_thinking:false` → EXACT cold |
| `deepraw-350000-20260726T002335Z/` | 350k raw vs chat. **`raw-raw.json` is the untemplated `/v1/completions` result** (`raw-` prefix = raw response *body*) | raw EXACT `' \n\n738216'`; **`cached=97,792`, NOT cold** |
| `deepraw-450000-20260726T002335Z/` | 450k raw vs chat | chat `content=None` w/ correct answer in `reasoning`; **`cached=137,216`, NOT cold** |
| `repro-350k-20260726T0320Z/` | Exact-prompt reproducibility probe — **INCOMPLETE, 1 of 9 cells**, stopped deliberately | `rep1 seed=20260725 ABSENT cached=0` |
| `verify_prompt.py` | Prompt-structure verifier | 350k passing/failing prompts identical: ctx 343,721, needle ×1 at token 137,496, 40.0% |
| `causal-gate-freeze-manifest.json` + `causal-gate-freeze-20260726.tar.gz` | Frozen gate inputs: prompt sha256, chat-rendered input-id sha256, sampling fields, recorded stock verdicts | 3× 350k FAIL + 1× 250k PASS control, all reproduce stock ctx |
| `causal-gate-legacy-summary.json` + `causal-gate-legacy-projection-20260726.tar.gz` | **Causal gate 1** — legacy MLA projection. Responses, gate log, full boot log | **REFUTED 0/3**; control PASS; pool 491,520 |
| `stock-5517197-boot-20260726.log.gz`, `stock-5517197-inspect-20260726.json` | Stock baseline boot log (3,232 lines) + container inspect, captured before the window | stock pool 559,616; logs the MXFP8-pack line at `mla_attention.py:1744` |
| `chain-*.log`, `deepraw-*.log` | Driver logs for the above runs | `chain-20260726T0110Z.log` is the self-matching-pgrep deadlock (105 bytes, never ran) |

## Reading traps recorded with the data

- **`raw-<variant>.json` means raw response *body*, not the raw API.** The untemplated
  `/v1/completions` result is `raw-raw.json` (`object=text_completion`, `choices[0].text`, no
  `message`). Sol misread this once; it is a bad name I chose.
- **The `deepraw-*` cells are not cold** (`cached_tokens` 97,792 / 137,216). The ladder and
  thinkmode cells *are* (`cached_tokens=0`). Always check, never assume.
- **`nothink-consistency` summary counts only `EXACT`**, so it prints "450k nothink 0/3 EXACT" when
  those three cells were `IN_CONTENT` — retrieval **succeeded**, inside a sentence also containing
  "Facility 27" so digit-only equality failed. Score `reasoning`+`content` together.
- **Cells labelled "low" effort ran at `max`.** `chat_template.jinja:2` maps everything except the
  literal `'high'` to `'max'`.

## Harnesses that produced these (in `harness/`)

| script | sha256 |
|---|---|
| `v20_nothink_consistency_ladder.py` | — (see git) |
| `v20_thinking_mode_finalization_fix.py` | `35d408572b4075417b056707c6a0943d2b21bf03afda4d17796ef69db8848d6f` |
| `v20_chat_token_identity_probe.py` (Sol) | `d3f5f90d1104a1c83626dce5a8c57b3fedab4522c9a887329abc25f3941b0a0e` |
| `v20_prompt_reproducibility_probe.py` | `3476fb7a1cd1779d06c5378e95cefdc38f0954518ca548e4330d64f9ed914f11` |
| `v20_finalization_discriminator.py` | — (see git) |

Token decode used throughout: `22,100919,122250` = `738216`; `154841` = `<think>`;
`154842` = `</think>`; `154827` = eos.
