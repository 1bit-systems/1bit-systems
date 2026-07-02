# SMU Init Order Fix

## Problem

On Strix Point/Halo (NPU4, PCI DID `0x17f0` rev `0x10`), the NPU's SMU
(System Management Unit) cannot initialize during driver probe because
it requires firmware loaded by PSP (Platform Security Processor).

The current init order in `aie2_hw_start()` at `aie2_pci.c:374-382` is:

```
1. aie_smu_init()       ← FAILS — firmware not loaded yet
2. aie_psp_start()      ← loads firmware via PSP
```

## Fix

Swap the two calls so PSP loads the firmware before SMU tries to use it:

```
1. aie_psp_start()      ← loads firmware via PSP
2. aie_smu_init()       ← SMU now finds firmware ready
```

Also added `stop_psp_no_smu` cleanup label: when SMU init fails after PSP
succeeds, we must stop PSP but skip SMU fini (since SMU never initialized).

## Patch

`patches/0001-amdxdna-fix-smu-init-order-strix-halo.patch`

## Status

Applied to `amd/xdna-driver` local clone. Needs to be submitted upstream
to the amd-gfx mailing list and/or the `lemonade-sdk/amdxdna-dkms` package.
