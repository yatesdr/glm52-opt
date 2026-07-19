# Challenge: fix the module load-order cycle

A small, self-contained reproduction of a real load-time bug: a low-level module
needs to call an accessor that lives in a higher-level module which **already
imports the low-level module**. The module-level back-edge closes an import
cycle, so the package fails to load.

This is distilled from a real defect in a serving stack (a per-phase profiler
hook added to a low-level attention-ops module needed a singleton accessor from
the backend module that imports those ops). All domain specifics are stripped;
the import topology is preserved exactly.

## The setup (3 modules)

```
pkg/
  phase_profiler.py   # a trivial profiler: records named wire-span starts
  sparse_backend.py   # HIGHER-level: imports the ops module AND owns _get_phase_prof()
  ops_common.py       # LOWER-level: the reduce-scatter op wants to time itself
                      #   -> needs _get_phase_prof() from sparse_backend
```

`ops_common` imports the profiler accessor from `sparse_backend` at module
scope. But `sparse_backend` imports `ops_common` at module scope. So:

```
import pkg.sparse_backend
  -> sparse_backend.py: `from .ops_common import cp_lse_ag_out_rs_into`
     -> ops_common.py:  `from .sparse_backend import _get_phase_prof`
        -> sparse_backend is only half-initialized; _get_phase_prof not defined yet
        -> ImportError: cannot import name '_get_phase_prof' from partially
           initialized module 'pkg.sparse_backend' (circular import)
```

## Reproduce the failure

```bash
cd challenges/circular-import-load-order
python reproduce.py        # prints the ImportError traceback, exits 1
python -m unittest test_load_order   # tests ERROR out on import
```

## Your task

Make the package load and the profiler hook work, **without**:

- removing the profiling call from `ops_common` (the wire span must still be
  timed at the point where the work happens),
- moving `_get_phase_prof` out of `sparse_backend`,
- reordering or merging the two modules, or
- introducing a fourth module just to hold the accessor.

The op must still record exactly one `"rs"` span when it runs.

## Acceptance

Both must pass:

```bash
python reproduce.py                    # -> "LOAD OK ... profiler spans = ['rs']", exit 0
python -m unittest test_load_order     # -> OK (2 tests)
```

## Constraints

- Standard library only; no third-party packages (no `torch`, no `pytest`).
- Keep the change minimal and local — ideally confined to one module.
- The fix must be robust: if the profiler is ever unavailable, serving (i.e.
  the op returning its result) must not break.

A reference solution and the rationale are in `SOLUTION.md`. Delete that file
before handing this off if you want a blind test.
