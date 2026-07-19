# Reference solution

> **Delete this file before handing the challenge off if you want a blind test.**

## Fix: defer the back-edge import to call time (lazy import), fail-closed

The cycle only exists at **module-load** time. The op does not actually need the
profiler until it **runs**, and by the time it runs, `sparse_backend` is fully
initialized. So move the `from .sparse_backend import _get_phase_prof` off the
module top and into the function body, and treat any failure as "no profiler."

`pkg/ops_common.py` becomes:

```python
"""Low-level ops module."""


def cp_lse_ag_out_rs_into(values):
    """Simplified reduce-scatter op; times its wire span via the phase profiler."""
    # Lazy import dodges the load-order cycle: at call time both modules are
    # fully loaded. Any failure permanently opts out -- profiling must never
    # break serving.
    prof = None
    try:
        from .sparse_backend import _get_phase_prof
        prof = _get_phase_prof()
    except Exception:
        prof = None
    if prof is not None:
        prof.start("rs")

    # ... real reduce-scatter work would happen here ...
    return sum(values)
```

That is the entire change — one module, no reordering, no new module, the timing
call stays where the work is.

## Why it works

At import time, `ops_common` no longer reaches back into `sparse_backend`, so the
edge `ops_common -> sparse_backend` disappears and the graph is acyclic:

```
sparse_backend -> ops_common          (only edge left at load time)
```

The reverse edge is created lazily, on first call, when `sparse_backend` has
finished initializing and `_get_phase_prof` is defined.

## Why the try/except matters

The import is instrumentation, not core function. Wrapping it so any failure
leaves `prof = None` guarantees the op still returns its result even if the
profiler module is missing, renamed, or broken. Instrumentation can degrade to a
no-op; it must never take down the request path.

## The general rule

> When a low-level module needs a symbol from a higher-level module that already
> depends on it, import it lazily inside the function and treat failure as
> opt-out — never add the back-edge at module scope.

## Acceptance (after applying the fix)

```
$ python reproduce.py
LOAD OK. run_backend() -> 6.0; profiler spans = ['rs']

$ python -m unittest test_load_order
..
----------------------------------------------------------------------
Ran 2 tests in 0.00s

OK
```

## Anti-patterns to reject

- Deleting the profiler call from `ops_common` (loses the measurement — the span
  must be timed where the work happens).
- Moving `_get_phase_prof` into a new/third module (works, but the challenge asks
  for a minimal local fix; in the real codebase the accessor genuinely belongs
  with the backend that owns the profiler singleton).
- Importing `sparse_backend` as a module object at top and reaching through it at
  call time (`import pkg.sparse_backend as sb` then `sb._get_phase_prof()`): this
  still leaves a module-level back-edge and can still fail depending on which
  module is imported first. The import must be deferred into the function body.
