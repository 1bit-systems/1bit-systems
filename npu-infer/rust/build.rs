/// build.rs — Compile the NPU engine C/C++ code with XRT + AIEBu for Rust FFI
use std::path::PathBuf;

fn main() {
    // Source root is one level up from the Cargo.toml (rust/ → npu-infer/)
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("src");
    let inc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("include");

    // Compile C files
    cc::Build::new()
        .std("c11")
        .opt_level(3)
        .include(&inc)
        .include("/usr/include")
        .file(src.join("model.c"))
        .compile("npu_model_c");

    // Compile C++ files
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .opt_level(3)
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mavx512f")
        .flag("-ffast-math")
        .flag("-march=native")
        .flag("-funroll-loops")
        .flag("-fno-math-errno")
        .flag("-fvect-cost-model=unlimited")
        .include(&inc)
        .include("/usr/include")
        .include("/usr/include/xrt")
        .file(src.join("engine.cpp"))
        .file(src.join("ffi_bridge.cpp"))
        .compile("npu_engine_ffi");

    // Tell cargo to link the system libraries
    println!("cargo:rustc-link-lib=static=aiebu");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=xrt_core");
    println!("cargo:rustc-link-lib=dylib=uuid");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");

    // Re-run if headers change
    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rerun-if-changed=src/engine.cpp");
    println!("cargo:rerun-if-changed=src/ffi_bridge.cpp");
}
