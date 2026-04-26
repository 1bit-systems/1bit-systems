# RPM spec for 1bit-systems — Fedora / RHEL / openSUSE.
#
# Build:
#   rpmbuild -ba 1bit-systems.spec
# Or via copr:
#   copr-cli build bong-water-water-bong/1bit-systems 1bit-systems.spec
#
# Installs only the portable C++23 orchestration binaries. The GPU
# kernels (rocm-cpp / librocm_cpp.so) are skipped because they need
# ROCm at build time; end users bring ROCm separately and wire model
# weights via `1bit install <model-id>` on first run.

Name:           1bit-systems
Version:        0.1.8
Release:        1%{?dist}
Summary:        Local AI inference on Strix Halo — LLM, TTS, STT, image, video, NPU

License:        MIT
URL:            https://1bit.systems
Source0:        https://github.com/bong-water-water-bong/1bit-systems/archive/refs/tags/v%{version}.tar.gz

BuildRequires:  cmake >= 3.27
BuildRequires:  ninja-build
BuildRequires:  gcc-c++ >= 14
BuildRequires:  pkgconfig
BuildRequires:  systemd-rpm-macros

Requires:       glibc
Requires:       vulkan-loader
Recommends:     rocm-hip
Recommends:     xrt
Recommends:     huggingface-hub
Suggests:       rocm-hip-devel

%description
1bit-systems is a local AI inference stack targeting AMD Strix Halo
(Ryzen AI MAX+ 395, gfx1151). It ships:

  * BitNet-1.58 ternary LLM via native HIP kernels
  * Qwen3-TTS text-to-speech via ggml Vulkan
  * whisper.cpp STT
  * stable-diffusion.cpp (SDXL + Wan 2.2 TI2V-5B) via native HIP
  * IRON + MLIR-AIE authoring lane for XDNA2 NPU kernels

All media engines AND the orchestration tower are pure C++23. No
Python runs at serving time. This package installs only the
orchestration binaries; model weights download via
`1bit install <model-id>` on first run.

%prep
%autosetup -n 1bit-systems-%{version}

%build
cd cpp
cmake --preset release-strix
cmake --build --preset release-strix

%check
cd cpp
ctest --preset release-strix

%install
install -Dm755 cpp/build/strix/cli/1bit               %{buildroot}%{_bindir}/1bit
install -Dm755 cpp/build/strix/power/1bit-power       %{buildroot}%{_bindir}/1bit-power
install -Dm755 cpp/build/strix/watchdog/1bit-watchdog %{buildroot}%{_bindir}/1bit-watchdog
install -Dm755 cpp/build/strix/helm/1bit-helm         %{buildroot}%{_bindir}/1bit-helm
install -Dm755 cpp/build/strix/voice/1bit-voice       %{buildroot}%{_bindir}/1bit-voice
install -Dm755 cpp/build/strix/echo/1bit-echo         %{buildroot}%{_bindir}/1bit-echo
install -Dm755 cpp/build/strix/mcp/1bit-mcp           %{buildroot}%{_bindir}/1bit-mcp

for unit in strixhalo/systemd/user/1bit-*.service strixhalo/systemd/user/1bit-*.timer; do
    [ -f "$unit" ] && install -Dm644 "$unit" %{buildroot}%{_userunitdir}/$(basename "$unit")
done

install -Dm644 README.md %{buildroot}%{_docdir}/%{name}/README.md
install -Dm644 LICENSE   %{buildroot}%{_licensedir}/%{name}/LICENSE

%files
%license LICENSE
%doc README.md
%{_bindir}/1bit
%{_bindir}/1bit-power
%{_bindir}/1bit-watchdog
%{_bindir}/1bit-helm
%{_bindir}/1bit-voice
%{_bindir}/1bit-echo
%{_bindir}/1bit-mcp
%{_userunitdir}/1bit-*.service
%{_userunitdir}/1bit-*.timer

%changelog
* Wed Apr 23 2026 bong-water-water-bong <277547417+bong-water-water-bong@users.noreply.github.com> - 0.1.8-1
- NPU toolchain verified on npu5 (IRON axpy 160/160 on Strix Halo).
- Five media lanes live; 1bit install all meta-target.
- AppImage + Flatpak + Homebrew + AUR + deb + rpm channels.
