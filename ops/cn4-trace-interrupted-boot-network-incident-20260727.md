# CN4 cross-layer trace: interrupted boot and route incident

Date: 2026-07-27

## Trace boot identity

- Container: `glm52-v20-indexer-segmented-exact-selection-trace-20260727`
- Container ID: `9fce94d5452fc30fb9ee736f385c998b684f7b5dc00a560cc7dd00f4ae77752c`
- Image ID: `sha256:6a3edc097955ff77aedb42bfc2656e9da68fce26f9bbc481d20ff735edcc4e3d`
- Started: `2026-07-27T01:40:44.934568264Z`
- Finished: `2026-07-27T02:45:52.926376239Z`
- Exit: `255`
- `OOMKilled`: `false`
- Available KV memory: `7.26 GiB`
- KV pool: `984,562` tokens
- Maximum concurrency at 360,000 tokens: `2.73x`
- API reached `Application startup complete` and repeatedly returned HTTP 200 from `/health`.
- The trace directory was empty and the `ARM` file was absent; no trace request ran.

The host journal explains the exit:

```text
Jul 27 02:45:11 cn4 systemd-logind: Power key pressed short.
Jul 27 02:45:11 cn4 systemd-logind: Powering off...
```

This was an orderly host shutdown, not a model crash or OOM.

## Route incident

Docker automatically allocated `192.168.32.0/20` to the test Compose
network. The operator Mac was `192.168.36.16`, inside that prefix, so CN4
incorrectly routed SSH replies to the Docker bridge:

```text
192.168.36.16 dev br-2070074dafe0 src 192.168.32.1
```

The network contained no containers after the host returned. Removing
that exact empty test network restored the physical route immediately:

```text
192.168.36.16 via 192.168.13.1 dev enp180s0f0 src 192.168.13.34
```

No host NIC, firewall, or persistent route setting was changed.

The corrected Compose pins its isolated test network to
`10.253.27.0/24`. Before the restart, CN4 resolved both
`192.168.36.16` and `10.253.27.1` through `192.168.13.1`; therefore the
new subnet did not overlap an existing local route.

Corrected Compose SHA-256:

```text
0651135fa20b9e48f36f876c9532123cc59e3dbf96094e5165051eacc6850326
```
