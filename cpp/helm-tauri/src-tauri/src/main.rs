// helm-tauri — dynamic-tiling desktop shell entry point.
//
// All host-side work is in lib.rs so `cargo build --lib` can drive a
// headless smoke build without spawning the windowing layer.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    helm_tauri_lib::run();
}
