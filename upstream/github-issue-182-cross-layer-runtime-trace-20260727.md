### Runtime cross-layer trace: the single-layer repair proof does not reproduce

The bounds-safe `segmented_exact` experiment was correct at the operator level
but failed the frozen end-to-end 350k gate. I have now captured the actual final
logical top-k consumed by sparse attention at all 21 GLM indexer-compute layers
for that same frozen request.

#### Frozen request and trace integrity

- prompt: `fail-350k-r1`
- prompt SHA-256:
  `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5`
- rendered prompt tokens: `343727`
- final prefill chunk: `2735`
- final absolute query position: `343726`
- prefix-cache hits: `0`
- trace coverage: `21/21` expected F layers
- each row: exactly `2048` unique valid logical token IDs
- tracer source SHA-256:
  `23e910a5aa88644f3817ba62d15bdd69a6b89f4191c41c3555deb84c2d1f6254`
- trace image:
  `sha256:6a3edc097955ff77aedb42bfc2656e9da68fce26f9bbc481d20ff735edcc4e3d`
- report SHA-256:
  `8db269bbed4cd9b80dc95f7e5750e22f55c83a7001e0a6e0b92bf0a3ddcdb7f8`

The diagnostic is an explicitly armed opaque custom op after the real
`indexer_op`. It copies the already-produced top-k row to CPU and does not
change query/key arithmetic, quantization, selection, or attention inputs.

#### Result

Using the needle-local logical-token window `137472..137520`:

| Layers | Runtime result |
|---|---|
| 0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38 | no needle-local token selected |
| 42, 46, 50, 54, 58 | token `137485` selected |
| 62 | `137485`, `137488`, `137492` selected |
| 66 | no needle-local token selected |
| 70 | token `137485` selected |
| 74 | 14 needle-local tokens selected |

The important discriminator is layer 34. The earlier isolated replay used
stock-captured layer-34 activations and predicted that `segmented_exact` would
retain `137483`, `137485`, `137493`, and `137503`. In the real multi-layer
patched run, layer 34 retained none of them.

Therefore:

1. the standalone selector proof was valid for its captured inputs;
2. those inputs are no longer the live layer-34 inputs once earlier sparse
   layers use the new policy;
3. selector changes feed back through the hidden-state trajectory;
4. a single-layer activation replay cannot establish an end-to-end repair.

This narrows the failure to a model-trajectory/selection interaction rather
than a simple one-call top-k bug. It does **not** yet prove that downstream DCP
logical-to-physical mapping is correct at the later layers where needle tokens
do appear.

#### Next discriminators

1. Run this identical 21-layer tracer on the already-qualified, v20
   `bounded_compat` configuration that returns `738216` for the same cold
   prompt. Compare exact selected sets, chronological distribution, and the
   first working-vs-failing layer divergence.
2. If later-layer logical selections are sufficient in the working reference,
   trace layer 74 after DCP filtering/gather mapping and independently
   reconstruct the physical attention slots.

`bounded_compat` remains a diagnostic reference, not the proposed permanent
v20 fix.
