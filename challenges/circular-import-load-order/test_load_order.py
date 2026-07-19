"""Acceptance test (stdlib unittest, no third-party deps).

Broken state: importing pkg.sparse_backend raises ImportError -> both tests error.
Fixed state: both tests pass.

Run from this directory:  python -m unittest test_load_order
"""

import unittest


class LoadOrderTest(unittest.TestCase):
    def test_backend_imports_without_cycle(self):
        # Must not raise ImportError from a partially-initialized module.
        import pkg.sparse_backend as backend

        self.assertTrue(hasattr(backend, "run_backend"))

    def test_profiler_hook_fires_at_call_time(self):
        import pkg.sparse_backend as backend

        # Fresh profiler state for this assertion.
        backend._PHASE_PROF = None
        backend.run_backend()
        prof = backend._get_phase_prof()
        self.assertEqual(prof.spans, ["rs"])


if __name__ == "__main__":
    unittest.main()
