//! build.rs — wire `1bit-cpu` against the already-built static lib
//! `libcpu_avx2_ternary.a` sitting in rocm-cpp/cpu-avx2/build/.
//!
//! We do not compile C++ here — the kernel has its own CMake invocation.
//! All we emit are link directives so the Rust linker can resolve the
//! two `halo_cpu_ternary_gemv_tq2*` symbols declared in src/ffi.rs.
//!
//! Resolution order for the static lib's directory:
//!   1. `HALO_CPU_AVX2_LIB_DIR` env override
//!   2. `<workspace>/rocm-cpp/cpu-avx2/build`                 (canonical)
//!   3. `/usr/local/lib`                                      (system-install)
//!
//! With the `link-cpu-avx2` feature off (default), we emit nothing —
//! the crate compiles as stubs-only and every kernel call returns
//! [`CpuError::Unsupported`].

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=HALO_CPU_AVX2_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let link_cpu = env::var_os("CARGO_FEATURE_LINK_CPU_AVX2").is_some();
    if !link_cpu {
        println!(
            "cargo:warning=1bit-cpu built without `link-cpu-avx2` — \
             every kernel call will return CpuError::Unsupported."
        );
        return;
    }

    let dir = find_lib_dir();
    match &dir {
        Some(p) => {
            println!("cargo:rustc-link-search=native={}", p.display());
            println!("cargo:cpu_avx2_lib_dir={}", p.display());
            println!(
                "cargo:rerun-if-changed={}/libcpu_avx2_ternary.a",
                p.display()
            );
        }
        None => {
            println!(
                "cargo:warning=1bit-cpu: could not locate libcpu_avx2_ternary.a \
                 (set HALO_CPU_AVX2_LIB_DIR=/path/to/rocm-cpp/cpu-avx2/build to override)"
            );
        }
    }

    // Static archive built with OpenMP + libstdc++ — we need both on the
    // link line. `gomp` is the GNU runtime (matches the CMake
    // `OpenMP::OpenMP_CXX` target when compiled with GCC/clang-with-libgomp).
    // We also pin libm explicitly — the scalar fp16 path uses F16C intrinsics
    // but any future fallback could need libm symbols.
    println!("cargo:rustc-link-lib=static=cpu_avx2_ternary");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");
}

fn find_lib_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("HALO_CPU_AVX2_LIB_DIR") {
        let p = PathBuf::from(dir);
        if contains_static(&p) {
            return Some(p);
        }
    }

    // Canonical in-tree path. build.rs sits at
    // <workspace>/crates/1bit-cpu/build.rs, so the workspace root is two
    // `..` up from CARGO_MANIFEST_DIR.
    if let Some(manifest_dir) = env::var_os("CARGO_MANIFEST_DIR") {
        let in_tree = PathBuf::from(manifest_dir)
            .join("../../rocm-cpp/cpu-avx2/build")
            .canonicalize()
            .ok();
        if let Some(p) = in_tree
            && contains_static(&p)
        {
            return Some(p);
        }
    }

    for candidate in ["/usr/local/lib", "/usr/lib"] {
        let p = PathBuf::from(candidate);
        if contains_static(&p) {
            return Some(p);
        }
    }
    None
}

fn contains_static(dir: &Path) -> bool {
    dir.join("libcpu_avx2_ternary.a").exists()
}
