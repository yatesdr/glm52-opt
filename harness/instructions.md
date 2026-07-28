# Raja's NVMe KV-cache eviction acceptance test

1. Copy `nvme_kv_eviction_acceptance.py` onto the Linux host.

2. Verify the script:

   ```bash
   sha256sum nvme_kv_eviction_acceptance.py
   ```

   Expected SHA-256:

   ```text
   7df4b95926816fac96139531c1bc25be4b1e9a4a569360073ef2aea80b9b595f
   ```

3. Identify the host path that maps to
   `/nvme-kv/glm52-eviction-test` inside the container. Confirm that the host
   path is on NVMe:

   ```bash
   findmnt -T /HOST/NVME/PATH/glm52-eviction-test
   ```

4. Wait for vLLM to become healthy:

   ```bash
   curl -fsS http://127.0.0.1:5001/health
   ```

5. Run the fill and eviction phase. The state directory must not already
   exist, and a fresh/empty cache namespace is preferred:

   ```bash
   uv run --no-project --python 3.12 \
     ./nvme_kv_eviction_acceptance.py fill \
     --cache-root /HOST/NVME/PATH/glm52-eviction-test \
     --state-dir /tmp/nvme-kv-acceptance
   ```

6. After the fill phase reports `PASS`, restart vLLM without deleting or
   changing the NVMe cache directory:

   ```bash
   docker restart glm52-prod
   ```

7. Wait for the health endpoint to succeed again:

   ```bash
   curl -fsS http://127.0.0.1:5001/health
   ```

8. Run the persisted-NVMe replay phase:

   ```bash
   uv run --no-project --python 3.12 \
     ./nvme_kv_eviction_acceptance.py replay \
     --cache-root /HOST/NVME/PATH/glm52-eviction-test \
     --state-dir /tmp/nvme-kv-acceptance
   ```

9. Return these artifacts for review:

   ```text
   /tmp/nvme-kv-acceptance/fill-report.json
   /tmp/nvme-kv-acceptance/replay-report.json
   /tmp/nvme-kv-acceptance/capacity-samples.csv
   /tmp/nvme-kv-acceptance/capacity-samples-replay.csv
   ```

If the API uses authentication, export `VLLM_API_KEY` before running the
script. If the model name, container name, or API address differs, pass
`--model`, `--container`, or `--base-url` respectively.
