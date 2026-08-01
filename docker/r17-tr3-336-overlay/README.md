# Exact r17 TR3-3.36 overlay build context

This directory mirrors the build context used for the published production
candidate. The base image is immutable, and `Dockerfile` verifies the five
runtime files against `r17-tr3-336-installed-files.sha256` before completing.

Build from this directory:

```bash
cd docker/r17-tr3-336-overlay
sha256sum -c CONTEXT.sha256
docker build \
  -t glm52-v20-r17-tr3-336-i8hier-pairedfc2:local \
  .
```

The source review commits are:

- i8_hier: `ce92f13a6256fdd86d52276955e9f760aaf3516d`
- paired-M8 SparkInfer: `1c2b052c426ea861c140f0981a7ea78709c9fdde`
- one-grid vLLM planner: `bde767ae23194c884a48b256cd4719c532f42512`

The five runtime files in those clean review branches are byte-identical to
this directory and to the final qualified image manifest.
