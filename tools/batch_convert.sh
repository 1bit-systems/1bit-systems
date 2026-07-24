#!/bin/bash
# batch_convert.sh — Batch convert GGUF models to 1BP format
#
# Converts any GGUF model that maps to a supported architecture
# (llama, qwen2, qwen3, gemma, phi, mistral, falcon, olmo, etc.)
# into the 1BP format for the 1bit-systems inference engine.
#
# Usage:
#   bash tools/batch_convert.sh                    # Convert all local GGUF files
#   bash tools/batch_convert.sh --download         # Download + convert new models
#   bash tools/batch_convert.sh --list             # List convertable GGUF files
#   bash tools/batch_convert.sh model.gguf         # Convert one file
#
# Dependencies:
#   - build/gguf_to_onebp (built from tools/gguf_to_onebp.cpp)
#   - curl / wget (for --download)
#   - huggingface-cli (optional, for --download)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CONVERTER="./build/gguf_to_onebp"
MODELS_DIR="models"

# ─── Supported architecture handlers ────────────────────────────────
# These are the GGUF general.architecture values that rcpp_arch_from_string() handles.
# Any model with one of these archs can be converted to 1BP.
declare -A ARCH_MAP
ARCH_MAP=(
    ["qwen3"]="qwen3"
    ["qwen2"]="qwen2"
    ["llama"]="llama"
    ["mistral"]="mistral"
    ["gemma"]="gemma"
    ["gemma2"]="gemma"
    ["phi"]="phi"
    ["phi3"]="phi"
    ["falcon"]="falcon"
    ["starcoder"]="llama"
    ["starcoder2"]="llama"
    ["command-r"]="llama"
    ["dbrx"]="llama"
    ["jamba"]="llama"
    ["olmo"]="olmo"
    ["olmo2"]="olmo"
    ["granite"]="gemma"
    ["deepseek2"]="qwen2"
    ["deepseek3"]="qwen2"
    ["zaya"]="zaya"
)

# ─── Color output ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[info]${NC} $1"; }
ok()    { echo -e "${GREEN}[ok]${NC}   $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $1"; }
err()   { echo -e "${RED}[err]${NC}  $1"; }

# ─── Check prereqs ─────────────────────────────────────────────────
check_prereqs() {
    if [ ! -f "$CONVERTER" ]; then
        info "Building gguf_to_onebp..."
        cmake --build build --target gguf_to_onebp -j8 2>/dev/null || {
            err "Failed to build converter. Run cmake -B build first."
            return 1
        }
    fi
    ok "Converter: $CONVERTER"
}

# ─── Detect GGUF architecture ──────────────────────────────────────
detect_arch() {
    local gguf="$1"
    # Read general.architecture from GGUF metadata
    local arch
    arch=$(python3 -c "
import struct
with open('$gguf', 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF': exit(1)
    f.read(4)  # version
    f.read(8)  # tensor_count
    n_kvs = struct.unpack('Q', f.read(8))[0]
    for i in range(min(n_kvs, 100)):
        klen = struct.unpack('Q', f.read(8))[0]
        key = f.read(klen).decode('utf-8')
        vtype = struct.unpack('I', f.read(4))[0]
        if key == 'general.architecture':
            slen = struct.unpack('Q', f.read(8))[0]
            val = f.read(slen).decode('utf-8')
            print(val)
            exit(0)
        # skip value
        if vtype == 8:
            sl = struct.unpack('Q', f.read(8))[0]
            if sl > 10000: sl = 0
            f.read(sl)
        elif vtype == 9:
            at = struct.unpack('I', f.read(4))[0]
            an = struct.unpack('Q', f.read(8))[0]
            if an > 10000: an = 0
            for _ in range(an):
                if at == 8:
                    sl2 = struct.unpack('Q', f.read(8))[0]
                    if sl2 > 1000: sl2 = 0
                    f.read(sl2)
                else:
                    f.read(4)
        else:
            f.read(4)
" 2>/dev/null) || echo ""
    echo "$arch"
}

# ─── Convert GGUF to 1BP ──────────────────────────────────────────
convert_file() {
    local gguf="$1"
    local out_name="$2"
    local out_path="$MODELS_DIR/${out_name}.1bp"
    
    if [ -f "$out_path" ]; then
        warn "Already exists: $out_path (skipping)"
        return 0
    fi
    
    info "Converting: $(basename $gguf) → ${out_name}.1bp"
    info "  Source: $(ls -lh "$gguf" | awk '{print $5}')"
    
    local t0
    t0=$(date +%s%N)
    
    "$CONVERTER" "$gguf" "$out_path" 2>&1 | tail -3
    
    local rc=$?
    local t1
    t1=$(date +%s%N)
    local elapsed=$(( (t1 - t0) / 1000000 ))
    
    if [ $rc -eq 0 ] && [ -f "$out_path" ]; then
        ok "Converted: $(ls -lh "$out_path" | awk '{print $5}') in ${elapsed}ms"
        return 0
    else
        err "Conversion failed for $(basename $gguf)"
        return 1
    fi
}

# ─── Derive output name from GGUF path ────────────────────────────
derive_name() {
    local gguf="$1"
    local base
    base=$(basename "$gguf" .gguf)
    
    # Remove common suffixes
    base=$(echo "$base" | sed -E '
        s/-Q[0-9]_[A-Z0-9_]+$//;
        s/-q[0-9]_[a-z0-9_]+$//;
        s/-instruct(-v[0-9])?$//i;
        s/-GGUF$//i;
    ')
    
    # Check for special prefixes
    local arch
    arch=$(detect_arch "$gguf")
    
    # Prettify name
    case "$arch" in
        qwen2|qwen3)
            echo "Qwen${arch:4}-${base#*qwen*}"
            ;;
        llama)
            echo "${base^}"
            ;;
        *)
            echo "$base"
            ;;
    esac
}

# ─── List convertable files ──────────────────────────────────────
list_files() {
    echo "Convertable GGUF files in $MODELS_DIR/:"
    echo ""
    printf "  %-50s %-12s %-12s %s\n" "File" "Size" "Arch" "Status"
    printf "  %s\n" "$(printf '%.0s─' {1..85})"
    
    for gguf in "$MODELS_DIR"/*.gguf; do
        [ -f "$gguf" ] || continue
        local size
        size=$(ls -lh "$gguf" | awk '{print $5}')
        local arch
        arch=$(detect_arch "$gguf")
        local name
        name=$(derive_name "$gguf")
        local status
        
        if [ -f "$MODELS_DIR/${name}.1bp" ]; then
            status="${GREEN}converted${NC}"
        elif [ -n "$arch" ] && [ -n "${ARCH_MAP[$arch]:-}" ]; then
            status="${YELLOW}ready${NC}"
        else
            status="${RED}unsupported arch: ${arch:-unknown}${NC}"
        fi
        
        printf "  %-50s %-12s %-12s %b\n" "$(basename $gguf)" "$size" "${arch:-?}" "$status"
    done
}

# ─── Download new models from HuggingFace ─────────────────────────
download_and_convert() {
    local model_id="$1"
    local arch="$2"
    
    info "Downloading $model_id (arch: $arch)..."
    
    # Try huggingface-cli first
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$model_id" --local-dir "$MODELS_DIR" 2>&1 | tail -3
    else
        # Fallback: direct download
        local files
        files=$(curl -s "https://huggingface.co/api/models/$model_id" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d.get('siblings',[]):
    if f['rfilename'].endswith('.gguf'):
        print(f['rfilename'])
" 2>/dev/null) || files=""
        
        if [ -z "$files" ]; then
            err "No GGUF files found for $model_id"
            return 1
        fi
        
        local smallest
        smallest=$(echo "$files" | grep -i 'Q4_K_M\|Q4_0\|Q3_K' | head -1 || echo "$files" | head -1)
        
        info "  Downloading: $smallest"
        local url="https://huggingface.co/$model_id/resolve/main/$smallest"
        local out_path
        out_path="$MODELS_DIR/$(echo "$smallest" | sed 's/.*\///')"
        
        if command -v wget &>/dev/null; then
            wget -q --show-progress "$url" -O "$out_path"
        else
            curl -sL "$url" -o "$out_path"
        fi
        
        if [ -f "$out_path" ]; then
            ok "Downloaded: $(ls -lh "$out_path" | awk '{print $5}')"
            convert_file "$out_path" "$(derive_name "$out_path")"
        fi
    fi
}

# ─── Download catalog of known models ─────────────────────────────
download_catalog() {
    info "Downloading missing catalog models..."
    
    # Small models (< 5B params) that map to supported archs
    local models=(
        "CohereForAI/c4ai-command-r-v01-GGUF:command-r"
        "bigcode/starcoder2-3b:starcoder2"
        "BAAI/bge-small-en-v1.5-GGUF:bert"  # placeholder
    )
    
    for entry in "${models[@]}"; do
        local model_id="${entry%%:*}"
        local arch="${entry##*:}"
        
        if [ -n "${ARCH_MAP[$arch]:-}" ] || [ "$arch" = "bert" ]; then
            download_and_convert "$model_id" "$arch"
        else
            warn "Skipping $model_id: unsupported arch '$arch'"
        fi
    done
}

# ─── Main ─────────────────────────────────────────────────────────
main() {
    mkdir -p "$MODELS_DIR"
    check_prereqs
    
    case "${1:-}" in
        --list)
            list_files
            ;;
        --download)
            download_catalog
            ;;
        --all)
            # Convert all ready GGUF files
            for gguf in "$MODELS_DIR"/*.gguf; do
                [ -f "$gguf" ] || continue
                local arch
                arch=$(detect_arch "$gguf")
                if [ -n "$arch" ] && [ -n "${ARCH_MAP[$arch]:-}" ]; then
                    local name
                    name=$(derive_name "$gguf")
                    convert_file "$gguf" "$name"
                fi
            done
            ;;
        *.gguf)
            # Single file
            local arch
            arch=$(detect_arch "$1")
            if [ -z "$arch" ]; then
                err "Cannot detect architecture in $1"
                return 1
            fi
            if [ -z "${ARCH_MAP[$arch]:-}" ]; then
                warn "Architecture '$arch' has no explicit handler, but trying anyway..."
            fi
            convert_file "$1" "$(derive_name "$1")"
            ;;
        "")
            # Default: list + suggest
            list_files
            echo ""
            echo "Run with --all to convert all ready files, or --download to fetch new models."
            ;;
        *)
            echo "Usage: $0 [--list|--download|--all|model.gguf]"
            ;;
    esac
}

main "$@"
