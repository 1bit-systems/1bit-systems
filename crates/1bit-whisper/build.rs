//! build.rs — compile the C++ shim and link against libwhisper, but **only**
//! when the `real-whisper` feature is active.
//!
//! With the default `stub` feature we emit zero link directives: the crate
//! is pure-Rust and compiles on any host without whisper.cpp installed.
//!
//! When `real-whisper` is on we:
//!   1. Compile `cpp/shim.cpp` into a static library `libonebit_whisper_shim.a`
//!      via the `cc` crate. The shim exposes the C-linkage `onebit_whisper_*`
//!      surface declared in `cpp/shim.h`, on top of `whisper.h`.
//!   2. Emit `cargo:rustc-link-lib=whisper` so the final binary resolves the
//!      `whisper_*` symbols our shim calls into. The system whisper.cpp 1.8.x
//!      install on this box provides `/usr/lib/libwhisper.so.1.8.3` plus
//!      `/usr/include/whisper.h`.
//!
//! By default we assume whisper.h is on the compiler include path
//! (`/usr/include`). Set `ONEBIT_WHISPER_PREFIX=/path/to/install` to point
//! at a non-default install — used on sliger where whisper.cpp is built
//! with `-DGGML_VULKAN=ON` into `/opt/whisper-vulkan/` for the Arc B580
//! audio pipeline. The `vulkan` Cargo feature is a signal to downstream
//! services; the actual backend is chosen at whisper.cpp build time.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/shim.h");
    println!("cargo:rerun-if-changed=cpp/segment_ring.hpp");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REAL_WHISPER");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VULKAN");
    println!("cargo:rerun-if-env-changed=ONEBIT_WHISPER_PREFIX");

    let real_whisper = env::var_os("CARGO_FEATURE_REAL_WHISPER").is_some();
    if !real_whisper {
        // Stub mode: nothing to compile, nothing to link.
        return;
    }

    // Optional install-prefix override for non-default whisper.cpp installs
    // (e.g. a Vulkan-enabled build at `/opt/whisper-vulkan` on sliger).
    let prefix = env::var_os("ONEBIT_WHISPER_PREFIX").map(std::path::PathBuf::from);

    // 1. Build the C-linkage shim over whisper.h.
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
    build.compile("onebit_whisper_shim");

    // 2. Link against system libwhisper. /usr/lib is on the default linker
    //    search path. If a prefix was provided we add its lib dir too.
    if let Some(p) = &prefix {
        println!("cargo:rustc-link-search=native={}", p.join("lib").display());
    }
    println!("cargo:rustc-link-lib=whisper");
}
