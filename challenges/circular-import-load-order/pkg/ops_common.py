"""Low-level ops module  --  BROKEN as shipped.

The reduce-scatter wire span is timed HERE, at the point where the work happens.
To time it, the op needs the profiler accessor `_get_phase_prof` -- but that
accessor lives in `sparse_backend`, which already imports THIS module. The
module-level import below therefore closes a load-order cycle:

    sparse_backend  ->  ops_common  ->  sparse_backend  ->  ...

so `import pkg.sparse_backend` raises:

    ImportError: cannot import name '_get_phase_prof' from partially
    initialized module 'pkg.sparse_backend' (most likely due to a circular import)

Fix this file so the package loads and the "rs" span is still recorded when the
op runs. (See README.md for the rules; SOLUTION.md for the reference fix.)
"""

from .sparse_backend import _get_phase_prof  # <-- back-edge that breaks module load


def cp_lse_ag_out_rs_into(values):
    """Simplified reduce-scatter op; times its wire span via the phase profiler."""
    prof = _get_phase_prof()
    prof.start("rs")
    # ... real reduce-scatter work would happen here ...
    return sum(values)
