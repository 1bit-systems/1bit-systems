// build.rs — compile + link the C shim around XRT's C++ API when
// --features real-npu is active. Default (stub-only) build does nothing;
// the crate stays CI-green on boxes that don't have libxrt installed.

fn main() {
    // Re-run when the shim or this script changes.
    println!("cargo:rerun-if-changed=native/xrt_c_shim.h");
    println!("cargo:rerun-if-changed=native/xrt_c_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // real-npu gate: only pull in libxrt when the caller opts in.
    // CI default has no libxrt → no work to do.
    if std::env::var("CARGO_FEATURE_REAL_NPU").is_ok() {
        // Compile the C shim + link XRT. Enabled only when the feature is
        // on AND the `cc` build-dep is available in Cargo.toml. Gated on
        // the `cc` cfg so the crate still lints clean when neither is set.
        #[cfg(feature = "real-npu")]
        compile_shim();
    }
}

#[cfg(feature = "real-npu")]
fn compile_shim() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include("native")
        .file("native/xrt_c_shim.cpp")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("xrt_c_shim");

    // Link against XRT. Standard install paths on Arch/CachyOS place libs
    // at /usr/lib/libxrt*.so.2; let the linker search there.
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=xrt++");
}
