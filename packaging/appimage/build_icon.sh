#!/usr/bin/env bash
# build_icon.sh — generate a 256×256 monochrome "1" glyph PNG.
#
# Build-time only (Rule A allows Python / ImageMagick on build hosts).
# Tries ImageMagick first (magick / convert), then falls back to a tiny
# Python/Pillow generator. The output lives next to this script:
#
#   packaging/appimage/1bit-systems.AppDir/1bit-systems.png
#
# Idempotent: skips work if the target already exists and --force wasn't
# passed.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/1bit-systems.AppDir/1bit-systems.png"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        -h|--help)
            printf 'usage: %s [--force]\n' "$0"
            exit 0
            ;;
    esac
done

if [[ -f "$OUT" && "$FORCE" -eq 0 ]]; then
    printf 'build_icon: already exists at %s (pass --force to rebuild)\n' "$OUT"
    exit 0
fi

mkdir -p "$(dirname "$OUT")"

# --- option 1: ImageMagick ------------------------------------------------
if command -v magick >/dev/null 2>&1; then
    MAGICK=magick
elif command -v convert >/dev/null 2>&1; then
    MAGICK=convert
else
    MAGICK=""
fi

if [[ -n "$MAGICK" ]]; then
    printf 'build_icon: using %s\n' "$MAGICK"
    # 256x256 black square with a centered white "1" glyph. DejaVu-Sans-Bold
    # is present in most Linux distros; fall back to default font if not.
    FONT=""
    for f in DejaVu-Sans-Bold DejaVu-Sans Liberation-Sans-Bold Liberation-Sans Helvetica Arial; do
        if "$MAGICK" -list font 2>/dev/null | grep -qi "$f"; then
            FONT="$f"
            break
        fi
    done

    if [[ "$MAGICK" == "magick" ]]; then
        # ImageMagick 7 syntax: `magick -size ... canvas:... -annotate ... out`
        if [[ -n "$FONT" ]]; then
            "$MAGICK" -size 256x256 canvas:black -fill white -font "$FONT" \
                -pointsize 220 -gravity center -annotate +0+0 '1' "$OUT"
        else
            "$MAGICK" -size 256x256 canvas:black -fill white \
                -pointsize 220 -gravity center -annotate +0+0 '1' "$OUT"
        fi
    else
        # ImageMagick 6: `convert -size ... xc:...`
        if [[ -n "$FONT" ]]; then
            "$MAGICK" -size 256x256 xc:black -fill white -font "$FONT" \
                -pointsize 220 -gravity center -annotate +0+0 '1' "$OUT"
        else
            "$MAGICK" -size 256x256 xc:black -fill white \
                -pointsize 220 -gravity center -annotate +0+0 '1' "$OUT"
        fi
    fi
    printf 'build_icon: wrote %s (ImageMagick)\n' "$OUT"
    exit 0
fi

# --- option 2: Python + Pillow -------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    printf 'build_icon: trying python3 + Pillow fallback\n'
    if python3 - "$OUT" <<'PY' ; then
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("build_icon: Pillow not installed; try `pip install --user Pillow`")
    sys.exit(2)

img = Image.new("RGB", (256, 256), "black")
drw = ImageDraw.Draw(img)

font = None
for candidate in (
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
):
    try:
        font = ImageFont.truetype(candidate, 220)
        break
    except OSError:
        continue
if font is None:
    font = ImageFont.load_default()

# Center the glyph.
bbox = drw.textbbox((0, 0), "1", font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (256 - w) // 2 - bbox[0]
y = (256 - h) // 2 - bbox[1]
drw.text((x, y), "1", fill="white", font=font)

img.save(out, format="PNG")
print(f"build_icon: wrote {out} (Pillow)")
PY
        exit 0
    fi
fi

# --- option 3: write a raw 1x1-scaled placeholder PNG ---------------------
# Last-resort: emit a tiny hand-built PNG so the AppImage still has an
# icon. Encodes a 1x1 black PNG (binary is base64-decoded inline). The
# build still succeeds; operators can regenerate with a real tool later.
printf 'build_icon: no ImageMagick or Pillow found — writing 1x1 placeholder\n' >&2
printf 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII=\n' \
    | base64 -d > "$OUT"
printf 'build_icon: wrote %s (1x1 fallback)\n' "$OUT"
