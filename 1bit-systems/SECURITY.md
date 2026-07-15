# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | ✅ |
| Older releases | ❌ — upgrade to latest |

## Reporting a Vulnerability

This is a small open-source project. If you find a security issue:

1. **Do not open a public GitHub issue** — email `admin@1bit.systems`
2. Include:
   - Description of the issue
   - Steps to reproduce (hardware, XRT version, command)
   - Potential impact (RCE, DOS, data exposure, etc.)
   - Suggested fix if you have one

## Scope

The following are in scope:
- The NPU engine (`engine/npu/src/`)
- The HTTP API server (`packaging/binary/server.cpp`)
- The GPU engine (`engine/gpu/`)
- Build-time toolchain (XRT, Chess compiler)

Out of scope:
- Third-party models loaded onto the NPU
- FastFlowLM (FLM proxy) — report to AMD
- XRT runtime vulnerabilities — report to AMD/Xilinx

## Response

I aim to acknowledge reports within 48 hours. For validated vulnerabilities, a fix and release will follow.
