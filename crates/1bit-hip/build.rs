//! build.rs — wire `1bit-hip` up to the already-built
//! `librocm_cpp.so` that lives next to the rocm-cpp checkout.
//!
//! We do NOT recompile rocm-cpp; we only emit cargo link directives so
//! the Rust linker can resolve the `rcpp_*` symbols declared in src/ffi.rs.
//!
//! Resolution order for the .so directory:
//!   1. `ROCM_CPP_LIB_DIR` environment variable (packaging / CI override)
//!   2. `<workspace>/rocm-cpp/build`                      (canonical, post-fold)
//!   3. `$HOME/repos/rocm-cpp/build`                      (legacy-clone fallback)
//!   4. `/usr/local/lib`, `/usr/lib`                      (system-install)
//!
//! If the `link-rocm` feature is off, we emit NO link directives — the crate
//! compiles as a stubs-only library for CI hosts without an AMD GPU.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Re-run if the env override changes, or if the build.rs itself changes.
    println!("cargo:rerun-if-env-changed=ROCM_CPP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // If the `link-rocm` feature was disabled, skip every link directive.
    // The crate will build against the stub implementation in src/ffi_stub.rs.
    let link_rocm = env::var_os("CARGO_FEATURE_LINK_ROCM").is_some();
    if !link_rocm {
        println!(
            "cargo:warning=1bit-hip built without `link-rocm` — \
             every kernel call will return RcppStatus::Unsupported."
        );
        return;
    }

    // 1. Find librocm_cpp.so's directory.
    let rocm_cpp_dir = find_rocm_cpp_dir();
    match &rocm_cpp_dir {
        Some(p) => {
            println!("cargo:rustc-link-search=native={}", p.display());
            // Surface for downstream runtime (rpath-like hint via cargo metadata).
            println!("cargo:rocm_cpp_lib_dir={}", p.display());
            println!("cargo:rerun-if-changed={}/librocm_cpp.so", p.display());
        }
        None => {
            // We don't hard-fail: on a misconfigured host the user may still
            // be running `cargo check` just to exercise the bindings.
            // `cargo build` / `cargo test` will fail at the link step with a
            // readable "cannot find -lrocm_cpp" message.
            println!(
                "cargo:warning=1bit-hip: could not locate librocm_cpp.so \
                 (set ROCM_CPP_LIB_DIR=/path/to/rocm-cpp/build to override)"
            );
        }
    }

    // 2. Find HIP runtime (libamdhip64.so). rocm-cpp's .so lists amdhip64 as
    //    NEEDED, so a dynamic linker will pull it in at load time; but on the
    //    Rust link step some toolchains still need an explicit -L so the .so
    //    itself can be verified. Canonical location on this box is /opt/rocm/lib.
    let hip_dir = find_hip_dir();
    if let Some(p) = hip_dir {
        println!("cargo:rustc-link-search=native={}", p.display());
    }

    // 3. Emit the link names. dylib = dynamic — we explicitly do not want
    //    static. amdhip64 gets linked so the Rust linker can resolve
    //    librocm_cpp's undefined refs during link verification.
    println!("cargo:rustc-link-lib=dylib=rocm_cpp");
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // 4. Encode an rpath pointing at the found rocm-cpp dir so a developer
    //    running `cargo test` from the workspace doesn't have to set
    //    LD_LIBRARY_PATH by hand. Packaged builds should strip rpaths and
    //    rely on system loader config.
    //
    // NOTE: `rustc-link-arg` from a library build script applies only to
    // this rlib's own link step (effectively a no-op since rlibs are
    // archives). Downstream consumers need their OWN build.rs to emit
    // `rustc-link-arg-bins` / `-tests`, because those variants are only
    // honoured from crates that have bin / test targets. 1bit-server and
    // 1bit-router carry matching build.rs files to re-export the rpaths.
    if let Some(p) = &rocm_cpp_dir {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p.display());
        // Expose the resolved path to downstream build.rs via metadata.
        println!("cargo:rocm_cpp_lib_dir={}", p.display());
    }
    // /opt/rocm/lib is canonical for libamdhip64 on this host.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
}

fn find_rocm_cpp_dir() -> Option<PathBuf> {
    // 1. Explicit env override.
    if let Some(dir) = env::var_os("ROCM_CPP_LIB_DIR") {
        let p = PathBuf::from(dir);
        if contains_rocm_cpp_so(&p) {
            return Some(p);
        }
    }

    // 2. Canonical in-tree path — <workspace>/rocm-cpp/build (post-fold).
    //    build.rs lives at <workspace>/crates/1bit-hip/build.rs so the
    //    workspace root is two `..` up from CARGO_MANIFEST_DIR.
    if let Some(manifest_dir) = env::var_os("CARGO_MANIFEST_DIR") {
        let in_tree = PathBuf::from(manifest_dir)
            .join("../../rocm-cpp/build")
            .canonicalize()
            .ok();
        if let Some(p) = in_tree
            && contains_rocm_cpp_so(&p)
        {
            return Some(p);
        }
    }

    // 3. Legacy-clone fallback — $HOME/repos/rocm-cpp/build.
    if let Some(home) = env::var_os("HOME") {
        let canonical = PathBuf::from(home).join("repos/rocm-cpp/build");
        if contains_rocm_cpp_so(&canonical) {
            return Some(canonical);
        }
    }

    // 4. System-install fallbacks.
    for candidate in ["/usr/local/lib", "/usr/lib"] {
        let p = PathBuf::from(candidate);
        if contains_rocm_cpp_so(&p) {
            return Some(p);
        }
    }

    None
}

fn find_hip_dir() -> Option<PathBuf> {
    // Respect an explicit ROCM_PATH if set.
    if let Some(root) = env::var_os("ROCM_PATH") {
        let p = PathBuf::from(root).join("lib");
        if p.join("libamdhip64.so").exists() {
            return Some(p);
        }
    }
    for candidate in ["/opt/rocm/lib", "/usr/lib"] {
        let p = PathBuf::from(candidate);
        if p.join("libamdhip64.so").exists() {
            return Some(p);
        }
    }
    None
}

fn contains_rocm_cpp_so(dir: &Path) -> bool {
    dir.join("librocm_cpp.so").exists()
}
