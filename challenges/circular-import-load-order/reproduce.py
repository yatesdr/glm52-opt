"""Reproduce / verify the load-order cycle.

Exit 0 = package loads and the op records exactly the ['rs'] span (fixed).
Exit 1 = circular-import ImportError at load (broken, as shipped).
Exit 2 = loaded, but the profiler hook did not fire as expected.

Run from this directory:  python reproduce.py
"""

import sys
import traceback

try:
    import pkg.sparse_backend as backend
except Exception:
    print("LOAD FAILED (circular import):\n")
    traceback.print_exc()
    sys.exit(1)

result = backend.run_backend()
prof = backend._get_phase_prof()
print(f"LOAD OK. run_backend() -> {result}; profiler spans = {prof.spans}")
sys.exit(0 if prof.spans == ["rs"] else 2)
