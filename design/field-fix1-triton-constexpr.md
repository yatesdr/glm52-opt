# Stage-3 field fix #1 (applied by Fable, 2026-07-17 ~07:25)

**Symptom:** first `ckv` acceptance boot failed at kernel prewarm on all
ranks: `triton.compiler.errors.CompilationError` →
`NameError: Cannot access global variable _CKV_VIRTUAL_BLOCK from within
@jit'ed function`. Transport armed cleanly on all 4 ranks first; the
failure was purely Triton compilation of `_remap_ckv_topk_kernel`.

**Cause:** the remap kernel body referenced module globals
(`_CKV_VIRTUAL_BLOCK`, `_CKV_WORLD_SIZE`, `_CKV_BLOCK_SIZE`). Triton JIT
functions can only see `tl.constexpr` values. Your pack kernel got this
right (constexpr params throughout); the remap kernel didn't.

**Fix (in your overlay file, deployed + rerun in flight):** three new
constexpr params `VIRTUAL_BLOCK`, `WORLD_SIZE`, `PAGE_RECORDS` on
`_remap_ckv_topk_kernel`; body uses swapped; both launch sites (hot at
~3047, prewarm at ~3360) pass `VIRTUAL_BLOCK=_CKV_VIRTUAL_BLOCK,
WORLD_SIZE=_CKV_WORLD_SIZE, PAGE_RECORDS=_CKV_BLOCK_SIZE`. Semantics
unchanged. Pyflakes/ast clean. Fold this back into your tree; new file
md5 will be in the acceptance record.

**Process note for all of us:** this bug class (Triton compile-time) is
invisible to pyflakes, ast, imports, AND the CPU harnesses — it only
surfaces at first GPU launch. Cheap future guard worth adopting in your
Gate-C checklist: a `triton.compile`-based dry-compile of any new/edited
@triton.jit kernel, or minimally a grep that no `@triton.jit` body
references a module-level name that isn't a tl.constexpr parameter. Your
phase-2 work adds more Triton kernels — bake the check in now.
