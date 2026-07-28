### Latest-v20 randomized cold ladder: 6/6 PASS through 475k

The same current-v20 process from the preceding causal-gate update has now
passed the complete randomized cold ladder:

| Target | Actual prompt tokens | Cached | Completion | Elapsed | Verdict |
|---:|---:|---:|---:|---:|---|
| 50k | 49,097 | 0 | 93 | 33.60 s | PASS |
| 150k | 147,275 | 0 | 99 | 107.33 s | PASS |
| 250k | 245,506 | 0 | 66 | 193.73 s | PASS |
| 300k | 294,620 | 0 | 66 | 241.03 s | PASS |
| 350k | 343,735 | 0 | 64 | 292.17 s | PASS |
| 475k | 466,493 | 0 | 71 | 432.71 s | PASS |

Every request:

- used a unique natural-language prefix and reported `cached_tokens=0`;
- returned exact finalized content `738216`;
- stopped normally rather than exhausting its token budget;
- passed arithmetic, coherence, and degeneration side checks.

The server remained healthy on the original container with zero restarts. Its
480,000-token maximum exposed a 545,280-token KV pool.

Pinned summary:

```text
b855f1febae880a6ae146797fbf37707e3ea02bccd213578d41ec5ba19ae6268
```

This closes the current-base long-context retrieval gate for the clean
`oldest_boundary` patch. KLD/general quality and matched prefill/decode remain
promotion gates; they are no longer being used to infer whether deep retrieval
works.
