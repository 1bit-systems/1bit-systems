fn main() {
    let build_dir = std::path::Path::new("../build");

    // Check for librocm_cpp.so or librocm_cpp.dylib
    let lib_found =
        build_dir.join("librocm_cpp.so").exists() || build_dir.join("librocm_cpp.dylib").exists();

    if lib_found {
        println!("cargo:rustc-link-search=native=../build/");
        println!("cargo:rustc-link-lib=dylib=rocm_cpp");
    } else {
        println!(
            "cargo:warning=librocm_cpp.so not found in ../build/. \
            Run `cmake -B build && ninja -C build rocm_cpp` to build it. \
            (Library is only needed at runtime when bitnet_decode is spawned)"
        );
    }
}
