# Legacy Python daemons — DEPRECATED

These files were the original Python NPU daemon implementations, kept for reference.

They have been fully replaced by the pure C++ stack:

| Legacy Python | C++ Replacement |
|---------------|----------------|
| `1bit-npu-server.py.legacy` (375 lines) | `zaya_server` + `onebitd` |
| `npu-cppd.py.legacy` (270 lines) | `tools/onebitd.cpp` |
| `npu-gpu-cpud.py.legacy` (869 lines) | `tools/onebitd.cpp` + `tools/unified_router.cpp` |

The Rust crates in `daemon/npu-cppd/`, `daemon/npu-gpu-cpud/`, and
`daemon/1bit-npu-server/` are similarly deprecated — see `rust/DEPRECATED.md`.

The `npu-daemon.service` systemd unit should be updated to point at
`onebitd` instead of the legacy Python daemon.
