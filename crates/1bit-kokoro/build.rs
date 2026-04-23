//! build.rs — compile the C++ shim and link against kokoro.cpp +
//! onnxruntime, but **only** when the `real-kokoro` feature is active.
//!
//! With the default `stub` feature we emit zero link directives: the
//! crate is pure-Rust and compiles on any host without kokoro.cpp /
//! onnxruntime installed.
//!
//! When `real-kokoro` is on we:
//!   1. Compile `cpp/shim.cpp` into a static library
//!      `libonebit_kokoro_shim.a` via the `cc` crate. The shim exposes
//!      the C-linkage `onebit_kokoro_*` surface declared in
//!      `cpp/shim.h`, on top of the single-header kokoro.cpp + its
//!      onnxruntime dependency.
//!   2. Emit `cargo:rustc-link-lib=onnxruntime` so the final binary
//!      resolves the ONNX symbols the shim ends up pulling in
//!      transitively through kokoro.cpp's header.
//!
//! By default we assume kokoro.hpp + onnxruntime headers are on the
//! compiler include path. Set `ONEBIT_KOKORO_PREFIX=/path/to/install`
//! to point at a non-default install — used on sliger where the
//! onnxruntime build lives under `/opt/onnxruntime-openvino/` for the
//! Arc B580 voice pipeline. The `vulkan` / `openvino` Cargo features
//! are signals to downstream services; the actual backend is chosen at
//! onnxruntime build time.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REAL_KOKORO");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VULKAN");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENVINO");
    println!("cargo:rerun-if-env-changed=ONEBIT_KOKORO_PREFIX");

    let real_kokoro = env::var_os("CARGO_FEATURE_REAL_KOKORO").is_some();
    if !real_kokoro {
        // Stub mode: nothing to compile, nothing to link.
        return;
    }

    // Optional install-prefix override for non-default kokoro / onnxruntime
    // installs (e.g. an OpenVINO-enabled build under
    // `/opt/onnxruntime-openvino` on sliger).
    let prefix = env::var_os("ONEBIT_KOKORO_PREFIX").map(std::path::PathBuf::from);

    // 1. Build the C-linkage shim over kokoro.hpp + onnxruntime_cxx_api.h.
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("cpp/shim.cpp")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wno-unused-parameter");
    if let Some(p) = &prefix {
        build.include(p.join("include"));
    }
    build.compile("onebit_kokoro_shim");

    // 2. Link against system onnxruntime. /usr/lib is on the default linker
    //    search path. If a prefix was provided we add its lib dir too.
    if let Some(p) = &prefix {
        println!("cargo:rustc-link-search=native={}", p.join("lib").display());
    }
    println!("cargo:rustc-link-lib=onnxruntime");
}
