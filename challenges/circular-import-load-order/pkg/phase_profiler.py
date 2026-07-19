"""A trivial stand-in for a runtime phase profiler.

The real one records CUDA-event spans around communication phases. Here it just
appends the span name to a list so the test can assert the hook fired.
"""


class PhaseProfiler:
    def __init__(self):
        self.spans = []

    def start(self, name):
        self.spans.append(name)
