"""Higher-level backend module.

It depends on the low-level ops module (imports it at module scope) AND owns the
process-wide phase-profiler accessor that the ops module wants to call. This is
one half of the load-order cycle. Do NOT change this file to solve the
challenge -- the fix belongs in the module that reaches back into this one.
"""

from .ops_common import cp_lse_ag_out_rs_into  # module-level dependency on the ops

from .phase_profiler import PhaseProfiler

_PHASE_PROF = None


def _get_phase_prof():
    """Return the process-wide profiler, created on first use."""
    global _PHASE_PROF
    if _PHASE_PROF is None:
        _PHASE_PROF = PhaseProfiler()
    return _PHASE_PROF


def run_backend():
    """Entry point that exercises the op (so the profiler hook fires)."""
    return cp_lse_ag_out_rs_into([1.0, 2.0, 3.0])
