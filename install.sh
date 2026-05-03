#!/usr/bin/env bash
# 1bit-systems installer — Strix Halo (gfx1151)
#
# Lean 1-bit inference engine: Lemonade Server (ROCm + Vulkan llama.cpp)
# and FastFlowLM (XDNA 2 NPU) behind one OpenAI-compat endpoint.
#
# Idempotent. Safe to re-run. Targets CachyOS / Arch (pacman).
#
# Usage:
#   ./install.sh             # actually install
#   ./install.sh --dry-run   # preview every action without mutating anything
#   ./install.sh --help      # show this help

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

DRY_RUN=0

say()  { printf '%b▸%b %s\n' "$CYAN" "$NC" "$*"; }
ok()   { printf '%b✓%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b!%b %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '%b✗%b %s\n' "$RED" "$NC" "$*" >&2; exit 1; }
dry()  { printf '%b≈%b %bDRY%b %s\n' "$YELLOW" "$NC" "$BOLD" "$NC" "$*"; }

# Run a command, or just show what would have run when DRY_RUN=1.
# Read-only probes (command -v, [[ -f ... ]], grep, flm version) bypass this.
run() {
    if (( DRY_RUN )); then
        dry "$*"
    else
        "$@"
    fi
}

usage() {
    cat <<EOF
1bit-systems installer

Usage:
  ./install.sh             actually install (idempotent)
  ./install.sh --dry-run   preview every action; mutate nothing
  ./install.sh -h | --help show this help

What runs in dry-run mode:
  - Read-only probes: pacman presence, hardware / service state,
    config presence, model-cache check, and local port health.
  - Every mutating command (pacman -S, sudo tee/sed/install,
    systemctl start, curl download) is printed prefixed with "DRY"
    instead of executing.

Targets CachyOS / Arch. Run on the box you want to host the stack.
EOF
}

banner() {
    local tag=""
    if (( DRY_RUN )); then tag=" — DRY-RUN"; fi
    printf '\n'
    printf '%b╔═══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║%b  %b1bit-systems%b — strix-halo 1-bit inference engine%s%b║%b\n' "$CYAN" "$NC" "$BOLD" "$NC" "$tag" "$CYAN" "$NC"
    printf '%b║%b  %bgfx1151 ROCm + Vulkan + XDNA 2 NPU%b                       %b║%b\n' "$CYAN" "$NC" "$DIM" "$NC" "$CYAN" "$NC"
    printf '%b╚═══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
    # Don't end the function on a `(( ... )) && cmd` line — under `set -e`
    # the arithmetic returns 1 when DRY_RUN=0, the && chain returns 1, the
    # function returns 1, and `set -e` exits the whole script silently
    # right after the banner.
    if (( DRY_RUN )); then
        warn "Dry-run mode — no commands will mutate the system."
    fi
}

require_pacman() {
    command -v pacman >/dev/null 2>&1 || die "pacman not found — this installer targets CachyOS / Arch."
    # NB: don't `pacman --version | head -1` here — that SIGPIPEs pacman, and
    # with `set -o pipefail` the whole script silently exits after the banner.
    ok "pacman found"
}

# Detect if this box has the AMD XDNA NPU (Strix Halo / Strix Point / Kraken Point).
# Returns 0 if NPU is present, 1 otherwise. Used to decide whether to install the
# xrt / xrt-plugin-amdxdna / fastflowlm trio (no point on a dGPU-only machine).
has_xdna_npu() {
    # Hardware ID 1022:17f0 (XDNA 2 — Strix Halo / Strix Point) or
    #             1022:1502 (XDNA 1 — Phoenix / Hawk Point — not supported by FLM).
    if command -v lspci >/dev/null 2>&1; then
        if lspci -n 2>/dev/null | grep -qE '1022:(17f0|1502)'; then
            return 0
        fi
    fi
    [[ -e /dev/accel/accel0 ]] && return 0
    lsmod 2>/dev/null | grep -q '^amdxdna ' && return 0
    return 1
}

install_packages() {
    # 1bit.systems policy: NO upstream application packages. Lemonade Server and
    # FastFlowLM are built from forks under bong-water-water-bong (see
    # build_install_forks). Only kernel/driver-layer packages and build deps
    # come from pacman.
    local base_pkgs=(
        # Build toolchain — needed to compile lemonade + flm from forks
        cmake ninja base-devel rust nodejs npm github-cli git pkgconf
        # Lemonade C++ runtime deps (pkg-config visible)
        libwebsockets libcap libdrm openssl curl
        # FLM optional engines (ASR + embeddings) and transitive native deps
        boost ffmpeg fftw
        # GPU stack — kernel/HIP layer only; iGPU lanes (rocm/vulkan llama.cpp)
        # come from cachyos-extra-znver4 separately
        rocm-hip-sdk
    )
    local xdna_pkgs=(xrt xrt-plugin-amdxdna)   # NPU driver/firmware only — flm itself comes from fork

    say "Installing build toolchain + driver packages (pacman)"
    run sudo pacman -S --needed --noconfirm "${base_pkgs[@]}"
    # Mark our build deps --asexplicit so a later -R cascade can't orphan them.
    run sudo pacman -D --asexplicit --noconfirm "${base_pkgs[@]}"

    if has_xdna_npu; then
        say "XDNA NPU detected — installing NPU driver layer (xrt + xrt-plugin-amdxdna)"
        run sudo pacman -S --needed --noconfirm "${xdna_pkgs[@]}"
        run sudo pacman -D --asexplicit --noconfirm "${xdna_pkgs[@]}"
    else
        warn "no XDNA NPU detected — skipping xrt / xrt-plugin-amdxdna"
        warn "(this box runs the iGPU/dGPU lane only; flm:npu recipe will stay 'unsupported')"
    fi

    # If a previous install left lemonade-server (AUR) or fastflowlm (pacman) installed,
    # remove them — they conflict with the from-source builds at /opt/1bit/.
    if pacman -Qq lemonade-server >/dev/null 2>&1; then
        warn "Removing legacy AUR lemonade-server (replaced by /opt/1bit/lemonade source build)"
        run sudo pacman -R --noconfirm lemonade-server
    fi
    if pacman -Qq fastflowlm >/dev/null 2>&1; then
        warn "Removing legacy pacman fastflowlm (replaced by /opt/1bit/flm source build)"
        run sudo pacman -R --noconfirm fastflowlm
    fi
    ok "Build deps installed; legacy app packages removed if present"
}

# Clone bong-water-water-bong/1bit-{lemonade,fastflowlm}, configure with
# Strix-Halo-tuned flags, build, install to /opt/1bit/<name>, symlink binaries
# into /usr/local/bin so the rest of install.sh and `1bit` CLI find them on
# PATH. Idempotent: cmake --build is a no-op when sources haven't changed.
build_install_forks() {
    local fork_owner=bong-water-water-bong
    local projects=$HOME/Projects
    local cflags="-O3 -march=znver4 -mtune=znver4 -DNDEBUG"

    run mkdir -p "$projects"
    if [[ ! -w /opt/1bit ]]; then
        say "Taking ownership of /opt/1bit ($(id -un))"
        run sudo install -d -o "$(id -un)" -g "$(id -gn)" /opt/1bit
    fi

    # ---- 1bit-lemonade ----
    local lemo="$projects/1bit-lemonade"
    if [[ ! -d "$lemo/.git" ]]; then
        say "Cloning $fork_owner/1bit-lemonade → $lemo"
        run git clone "https://github.com/$fork_owner/1bit-lemonade.git" "$lemo"
        run git -C "$lemo" remote add upstream https://github.com/lemonade-sdk/lemonade.git
    fi

    say "Configuring + building 1bit-lemonade (znver4 / -O3 / LTO)"
    run cmake -S "$lemo" -B "$lemo/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/1bit/lemonade \
        -DCMAKE_C_FLAGS_RELEASE="$cflags" \
        -DCMAKE_CXX_FLAGS_RELEASE="$cflags" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DBUILD_WEB_APP=ON \
        -DBUILD_TAURI_APP=OFF
    run cmake --build "$lemo/build" -j"$(nproc)"
    run cmake --install "$lemo/build" --prefix /opt/1bit/lemonade || \
        warn "cmake install hit non-/opt paths (systemd, /etc/lemonade) — we install those ourselves below"

    # lemond hardcodes its resource search to <bindir>/resources but cmake_install
    # puts them in <prefix>/share/lemonade-server/resources. Bridge with a symlink.
    if [[ ! -e /opt/1bit/lemonade/bin/resources ]]; then
        run ln -sfn /opt/1bit/lemonade/share/lemonade-server/resources /opt/1bit/lemonade/bin/resources
    fi

    # ---- 1bit-fastflowlm ----
    local flm="$projects/1bit-fastflowlm"
    if [[ ! -d "$flm/.git" ]]; then
        say "Cloning $fork_owner/1bit-fastflowlm → $flm"
        run git clone "https://github.com/$fork_owner/1bit-fastflowlm.git" "$flm"
        run git -C "$flm" remote add upstream https://github.com/FastFlowLM/FastFlowLM.git
    fi
    run git -C "$flm" submodule update --init --recursive

    say "Configuring + building 1bit-fastflowlm (znver4 / -O3 / LTO)"
    run cmake -S "$flm/src" -B "$flm/build" --preset linux-default \
        -DCMAKE_INSTALL_PREFIX=/opt/1bit/flm \
        -DCMAKE_XCLBIN_PREFIX=/opt/1bit/flm/share/flm \
        -DXRT_INCLUDE_DIR=/usr/include/xrt \
        -DXRT_LIB_DIR=/usr/lib \
        -DCMAKE_C_FLAGS_RELEASE="$cflags" \
        -DCMAKE_CXX_FLAGS_RELEASE="$cflags" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
    run cmake --build "$flm/build" -j"$(nproc)"
    run cmake --install "$flm/build" --prefix /opt/1bit/flm || \
        warn "cmake install hit non-/opt paths (e.g. /usr/local/bin symlink) — we install those ourselves below"

    # ---- PATH symlinks ----
    say "Symlinking binaries into /usr/local/bin"
    for bin in /opt/1bit/lemonade/bin/* /opt/1bit/flm/bin/*; do
        [[ -f "$bin" && -x "$bin" ]] || continue
        local link
        link="/usr/local/bin/$(basename "$bin")"
        if [[ -L "$link" && "$(readlink -f "$link")" == "$(readlink -f "$bin")" ]]; then
            continue
        fi
        run sudo ln -sfn "$bin" "$link"
    done

    # ---- Systemd unit ----
    if [[ -f /opt/1bit/lemonade/lib/systemd/system/lemond.service ]]; then
        run sudo install -m 0644 /opt/1bit/lemonade/lib/systemd/system/lemond.service \
            /etc/systemd/system/lemond.service
        run sudo systemctl daemon-reload
    fi
    # Lemonade's cmake_install.cmake tries to create /etc/lemonade/conf.d/zz-secrets.conf
    # which fails without sudo. Install it ourselves if missing.
    if [[ ! -f /etc/lemonade/conf.d/zz-secrets.conf && -f "$lemo/data/secrets.conf" ]]; then
        run sudo install -d /etc/lemonade/conf.d
        run sudo install -m 0640 "$lemo/data/secrets.conf" /etc/lemonade/conf.d/zz-secrets.conf
    fi


    ok "Forks built and installed under /opt/1bit; binaries symlinked into /usr/local/bin"
}

write_memlock_limits() {
    local conf=/etc/security/limits.d/99-1bit-systems.conf
    if [[ -f "$conf" ]] && grep -q 'memlock     unlimited' "$conf"; then
        ok "memlock limits already set ($conf)"
        return
    fi
    say "Writing memlock limits to $conf"
    if (( DRY_RUN )); then
        dry "sudo tee $conf <<EOF"
        printf '       %b# 1bit-systems: NPU + GPU buffers need unlimited memlock.%b\n' "$DIM" "$NC"
        printf '       %b*       soft    memlock     unlimited%b\n' "$DIM" "$NC"
        printf '       %b*       hard    memlock     unlimited%b\n' "$DIM" "$NC"
        return
    fi
    sudo tee "$conf" >/dev/null <<'EOF'
# 1bit-systems: NPU + GPU buffers need unlimited memlock.
*       soft    memlock     unlimited
*       hard    memlock     unlimited
EOF
    ok "memlock limits written — re-login or reboot for them to apply"
}

# Patch ~/.cache/lemonade/config.json so lemond binds 0.0.0.0 even when started
# directly (without `1bit up`'s explicit --host flag). Idempotent; skips if the
# config doesn't exist yet (it's created on first lemond run).
configure_lemonade_lan_bind() {
    local cfg="$HOME/.cache/lemonade/config.json"
    if [[ ! -f "$cfg" ]]; then
        ok "lemonade config not yet created; --host 0.0.0.0 still applied via 1bit up"
        return
    fi
    if grep -q '"host": "0\.0\.0\.0"' "$cfg"; then
        ok "lemonade config already binds 0.0.0.0"
        return
    fi
    say "Patching lemonade config to bind 0.0.0.0 ($cfg)"
    if (( DRY_RUN )); then
        dry "sed -i 's/\"host\": \"[^\"]*\"/\"host\": \"0.0.0.0\"/' $cfg"
        return
    fi
    sed -i 's/"host": "[^"]*"/"host": "0.0.0.0"/' "$cfg"
    ok "lemonade now binds 0.0.0.0 (restart lemond for it to take effect)"
}

# If UFW is installed and active, add allow rules for the LAN-facing services
# from the local /24. Skips silently when UFW isn't present.
configure_ufw_lan() {
    command -v ufw >/dev/null 2>&1 || { ok "ufw not installed; skipping firewall rules"; return; }
    if ! sudo ufw status 2>/dev/null | head -1 | grep -q 'active'; then
        ok "ufw not active; skipping firewall rules"
        return
    fi
    # Detect the local /24 from the default-route source IP.
    local src cidr
    src=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}') || true
    if [[ -z "$src" ]]; then
        warn "Could not detect local IP — skipping UFW LAN rules"
        return
    fi
    cidr="$(echo "$src" | awk -F. '{print $1"."$2"."$3".0/24"}')"
    say "Adding UFW LAN rules for $cidr (lemond:13305+9000, 1bit-proxy:13306, open-webui:3000)"
    local port name
    for entry in "13305:lemonade LAN" "9000:lemonade websocket LAN" "13306:1bit-proxy LAN" "3000:open-webui LAN"; do
        port="${entry%%:*}"; name="${entry#*:}"
        if sudo ufw status | grep -qE "^${port}/tcp\b.*${cidr}"; then
            ok "ufw rule for ${port}/tcp from $cidr already present"
            continue
        fi
        run sudo ufw allow from "$cidr" to any port "$port" proto tcp comment "$name"
    done
}

configure_optc_mitigation() {
    # Strix Halo gfx1151 can hard-freeze in the display controller under mixed
    # compositor + GPU-compute load. Stage the tested dcdebugmask workaround on
    # Limine systems; it takes effect after the next reboot.
    if ! lspci 2>/dev/null | grep -qiE 'Strix|Radeon.*8060S|gfx1151|AMD.*Graphics'; then
        [[ -e /sys/class/drm/card1/device/power_dpm_force_performance_level ]] || {
            ok "OPTC mitigation: no Strix Halo-like amdgpu display device detected"
            return
        }
    fi

    local flag="amdgpu.dcdebugmask=0x410"
    if tr ' ' '\n' </proc/cmdline | grep -qx "$flag"; then
        ok "OPTC mitigation active in current boot ($flag)"
        return
    fi

    if [[ ! -f /etc/default/limine ]] || ! command -v limine-install >/dev/null 2>&1; then
        warn "OPTC mitigation not active; add $flag to your bootloader cmdline"
        return
    fi

    say "Staging OPTC mitigation in /etc/default/limine ($flag)"
    if (( DRY_RUN )); then
        dry "optional: snapper -c root create -d pre-1bit-optc-dcdebugmask"
        dry "backup /etc/default/limine, append $flag, run limine-install"
        return
    fi

    if command -v snapper >/dev/null 2>&1; then
        sudo snapper -c root create -d "pre-1bit-optc-dcdebugmask" || \
            warn "snapper snapshot failed; continuing with /etc/default/limine backup"
    fi
    sudo cp -a /etc/default/limine "/etc/default/limine.pre-1bit-optc.$(date +%Y%m%d%H%M%S)"
    if ! grep -q "$flag" /etc/default/limine; then
        sudo sed -i "/KERNEL_CMDLINE\\[default\\]/ s/\"$/ $flag\"/" /etc/default/limine
    fi
    sudo limine-install
    warn "OPTC mitigation staged; reboot required before $flag is active"
}

install_systemd_units() {
    say "Installing systemd units for the full 1bit.systems stack (auto-start at boot)"
    local install_user install_group install_home
    install_user="$(id -un)"
    install_group="$(id -gn)"
    install_home="$HOME"
    run sudo mkdir -p /var/log/1bit-systems
    run sudo chown "$install_user:$install_group" /var/log/1bit-systems

    if (( DRY_RUN )); then
        dry "write /etc/systemd/system/{lemond,flm,1bit-proxy,open-webui}.service + 1bit-stack.target"
        dry "systemctl daemon-reload && enable + start 1bit-stack.target"
        dry "write ~/.config/autostart/gaia-agent-ui.desktop"
        return
    fi

    sudo tee /etc/systemd/system/lemond.service >/dev/null <<UNIT
[Unit]
Description=Lemonade Server (1bit-systems / source-built)
After=network-online.target
Wants=network-online.target
PartOf=1bit-stack.target
[Service]
Type=simple
User=$install_user
Group=$install_group
WorkingDirectory=$install_home
EnvironmentFile=-/etc/lemonade/conf.d/zz-secrets.conf
ExecStart=/usr/local/bin/lemond --host 0.0.0.0
ExecReload=/bin/kill -HUP $MAINPID
KillSignal=SIGINT
Restart=on-failure
RestartSec=5s
LimitMEMLOCK=infinity
LimitNOFILE=65536
AmbientCapabilities=CAP_SYS_RESOURCE
[Install]
WantedBy=multi-user.target 1bit-stack.target
UNIT

    sudo tee /etc/systemd/system/flm.service >/dev/null <<UNIT
[Unit]
Description=FastFlowLM NPU server (1bit-systems / source-built)
After=network-online.target lemond.service
Wants=network-online.target
PartOf=1bit-stack.target
[Service]
Type=simple
User=$install_user
Group=$install_group
WorkingDirectory=$install_home
Environment=ONEBIT_NPU_MODEL=qwen3:1.7b
Environment="ONEBIT_FLM_FLAGS=--port 52625 --embed 1 --socket 20 --q-len 20"
ExecStart=/bin/bash -c '/usr/local/bin/flm serve "\${ONEBIT_NPU_MODEL}" \${ONEBIT_FLM_FLAGS}'
StandardOutput=append:/var/log/1bit-systems/flm.log
StandardError=append:/var/log/1bit-systems/flm.log
Restart=on-failure
RestartSec=5s
LimitMEMLOCK=infinity
[Install]
WantedBy=multi-user.target 1bit-stack.target
UNIT

    sudo tee /etc/systemd/system/1bit-proxy.service >/dev/null <<UNIT
[Unit]
Description=1bit-systems proxy (unifies lemond + flm on :13306)
After=lemond.service flm.service
Wants=lemond.service flm.service
PartOf=1bit-stack.target
[Service]
Type=simple
User=$install_user
Group=$install_group
WorkingDirectory=$install_home
ExecStart=/usr/bin/node /usr/local/share/1bit-systems/1bit-proxy.js
StandardOutput=append:/var/log/1bit-systems/1bit-proxy.log
StandardError=append:/var/log/1bit-systems/1bit-proxy.log
Restart=on-failure
RestartSec=5s
[Install]
WantedBy=multi-user.target 1bit-stack.target
UNIT

    sudo tee /etc/systemd/system/open-webui.service >/dev/null <<UNIT
[Unit]
Description=open-webui (1bit-systems / points at :13306 unified endpoint)
After=network-online.target 1bit-proxy.service
Wants=network-online.target 1bit-proxy.service
PartOf=1bit-stack.target
[Service]
Type=simple
User=$install_user
Group=$install_group
WorkingDirectory=$install_home
Environment=WEBUI_NAME=1bit.systems
Environment=OPENAI_API_BASE_URL=http://127.0.0.1:13306/v1
Environment=OPENAI_API_KEY=local-no-auth
Environment=ENABLE_OLLAMA_API=False
Environment=WEBUI_AUTH=False
ExecStart=$install_home/.local/bin/open-webui serve --host 0.0.0.0 --port 3000
StandardOutput=append:/var/log/1bit-systems/open-webui.log
StandardError=append:/var/log/1bit-systems/open-webui.log
Restart=on-failure
RestartSec=5s
[Install]
WantedBy=multi-user.target 1bit-stack.target
UNIT

    sudo tee /etc/systemd/system/1bit-stack.target >/dev/null <<'UNIT'
[Unit]
Description=1bit-systems full stack (lemond + flm + proxy + open-webui)
Wants=lemond.service flm.service 1bit-proxy.service open-webui.service
After=lemond.service flm.service 1bit-proxy.service open-webui.service
[Install]
WantedBy=multi-user.target
UNIT

    sudo tee /etc/systemd/system/1bit-optc-gpu-perf.service >/dev/null <<'UNIT'
[Unit]
Description=1bit Strix Halo OPTC workaround: pin amdgpu DPM level high
ConditionPathExistsGlob=/sys/class/drm/card*/device/power_dpm_force_performance_level
After=systemd-udev-settle.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do [ -e "$f" ] && echo high > "$f" || true; done'
RemainAfterExit=yes

[Install]
WantedBy=graphical.target
UNIT

    sudo systemctl daemon-reload
    sudo systemctl enable lemond.service flm.service 1bit-proxy.service open-webui.service 1bit-stack.target 1bit-optc-gpu-perf.service

    # XDG autostart for the GAIA Agent UI desktop AppImage (graphical login only).
    if [[ -x "$install_home/Applications/gaia-agent-ui.AppImage" ]]; then
        mkdir -p "$install_home/.config/autostart"
        cat > "$install_home/.config/autostart/gaia-agent-ui.desktop" <<XDG
[Desktop Entry]
Type=Application
Name=GAIA Agent UI
Comment=AMD GAIA agent UI - auto-launches on graphical login (1bit-systems)
Exec=$install_home/Applications/gaia-agent-ui.AppImage --no-sandbox
Icon=gaia
Terminal=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
Categories=Development;
XDG
        ok "GAIA Agent UI XDG autostart installed at ~/.config/autostart/gaia-agent-ui.desktop"
    else
        warn "GAIA AppImage not found at ~/Applications/gaia-agent-ui.AppImage — skipping autostart"
        warn "(install via: amd-gaia.ai installer; XDG entry will be added on next install.sh run)"
    fi

    ok "systemd units installed + enabled — stack will auto-start at boot"
}

start_stack() {
    say "Bringing up 1bit-stack.target (lemond + flm + proxy + open-webui)"
    if (( DRY_RUN )); then
        dry "sudo systemctl start 1bit-stack.target"
        return
    fi
    # Stop any nohup-launched legacy processes so systemd has clean ports.
    for name in lemond flm; do
        if pgrep -x "$name" >/dev/null 2>&1; then
            local actual
            actual=$(readlink "/proc/$(pgrep -x "$name" | head -1)/exe" 2>/dev/null || true)
            if [[ "$actual" != *"systemd"* && "$actual" != /opt/1bit/*/bin/* ]]; then
                warn "Killing legacy nohup-launched $name (pid $(pgrep -x "$name" | head -1))"
                sudo kill -9 $(pgrep -x "$name") 2>/dev/null || true
            fi
        fi
    done
    sudo systemctl start 1bit-stack.target
    sleep 4
    for unit in lemond flm 1bit-proxy open-webui; do
        if systemctl is-active --quiet "$unit.service"; then
            ok "$unit.service: active"
        else
            warn "$unit.service: $(systemctl is-active "$unit.service")"
        fi
    done
    if curl -s --max-time 3 http://127.0.0.1:13305/api/v1/health >/dev/null 2>&1; then
        ok "lemond responding on http://127.0.0.1:13305"
    else
        warn "lemond didn't respond — journalctl -u lemond.service for details"
    fi
}

pull_default_model() {
    local dest file
    dest=/home/$(id -un)/halo-ai/models/ternary-test/lily-bonsai-1.7b-rq
    file=Bonsai-1.7B-IQ1_S.gguf
    if [[ -f "$dest/$file" ]]; then
        ok "Default 1-bit model already present ($file)"
        return
    fi
    say "Pulling default 1-bit model: $file (385 MB)"
    run mkdir -p "$dest"
    if (( DRY_RUN )); then
        dry "curl -L --fail -o $dest/$file https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized/resolve/main/$file"
        return
    fi
    curl -L --fail \
        -o "$dest/$file" \
        "https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized/resolve/main/$file" \
        || die "Failed to fetch $file"
    ok "Default model cached at $dest/$file"
}

install_cli() {
    say "Installing /usr/local/bin/1bit + proxy + omni"
    run sudo install -m 0755 "$(dirname "$0")/scripts/1bit" /usr/local/bin/1bit
    run sudo install -m 0755 "$(dirname "$0")/scripts/optc-status" /usr/local/bin/1bit-optc-status
    run sudo install -m 0755 "$(dirname "$0")/scripts/optc_soak_2hr.sh" /usr/local/bin/1bit-optc-soak
    run sudo install -d /usr/local/share/1bit-systems
    run sudo install -d /usr/local/share/1bit-systems/omni-plugins
    run sudo install -m 0644 "$(dirname "$0")/scripts/1bit-home.html" /usr/local/share/1bit-systems/1bit-home.html
    run sudo install -m 0644 "$(dirname "$0")/scripts/1bit-proxy.js" /usr/local/share/1bit-systems/1bit-proxy.js
    run sudo install -m 0755 "$(dirname "$0")/scripts/1bit-omni.py" /usr/local/share/1bit-systems/1bit-omni.py
    for plugin in "$(dirname "$0")"/scripts/omni-plugins/*.json; do
        [[ -f "$plugin" ]] || continue
        run sudo install -m 0644 "$plugin" "/usr/local/share/1bit-systems/omni-plugins/$(basename "$plugin")"
    done
    ok "1bit CLI + proxy + omni installed — try: 1bit up   or   1bit omni \"...\""
}

parse_args() {
    while (( $# )); do
        case "$1" in
            -n|--dry-run) DRY_RUN=1 ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown arg: $1 (try --help)" ;;
        esac
        shift
    done
}

main() {
    parse_args "$@"
    banner
    require_pacman
    install_packages
    build_install_forks
    write_memlock_limits
    configure_lemonade_lan_bind
    configure_ufw_lan
    configure_optc_mitigation
    install_cli
    install_systemd_units
    start_stack
    pull_default_model
    echo
    if (( DRY_RUN )); then
        printf '%b≈%b %bDry-run complete.%b No system state was modified.\n' "$YELLOW" "$NC" "$BOLD" "$NC"
        printf '   Re-run without %b--dry-run%b to actually install.\n' "$BOLD" "$NC"
    else
        printf '%b✓%b Done. Run %b1bit up%b to launch lemond + flm + proxy.\n' "$GREEN" "$NC" "$BOLD" "$NC"
        printf '   %bUnified OpenAI endpoint:%b http://127.0.0.1:13306/v1/  (recommended — both lanes)\n' "$DIM" "$NC"
        printf '   %bGAIA users:%b https://amd-gaia.ai (.deb on Linux). Point it at http://127.0.0.1:13306\n' "$DIM" "$NC"
    fi
}

main "$@"
