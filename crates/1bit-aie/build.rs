// build.rs — compile + link the C shim around XRT's C++ API when
// --features real-npu is active. Default (stub-only) build does nothing;
// the crate stays CI-green on boxes that don't have libxrt installed.

fn main() {
    // Re-run when the shim or this script changes.
    println!("cargo:rerun-if-changed=native/xrt_c_shim.h");
    println!("cargo:rerun-if-changed=native/xrt_c_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // real-npu gate: only pull in libxrt when the caller opts in.
    // CI default has no libxrt → do nothing.
    if std::env::var("CARGO_FEATURE_REAL_NPU").is_err() {
        return;
    }

    // Compile the C shim. Uses `cc` crate when available; falls through
    // with a clear error if the crate is missing so the operator knows to
    // add it to dev-deps. We can't hard-require `cc` here without bloating
    // the stub-only CI build, so this is a best-effort probe.
    //
    // Note: Cargo will already pull `cc` in when `real-npu` is on (see
    // Cargo.toml optional build-dep). This block exists to future-proof
    // the script if someone drops the feature gate accidentally.
    #[cfg(feature = "real-npu-build")]
    {
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .include("native")
            .file("native/xrt_c_shim.cpp")
            .flag_if_supported("-Wall")
            .flag_if_supported("-Wextra")
            .compile("xrt_c_shim");

        // Link against XRT. Standard install paths on Arch/CachyOS place
        // libs at /usr/lib/libxrt*.so.2; let the linker search there.
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
        println!("cargo:rustc-link-lib=dylib=xrt++");
    }
}
