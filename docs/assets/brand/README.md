# Brand assets

Source files for external surfaces. Keep these in sync with `1bit-site/index.html` theme tokens.

## Theme tokens

| token | hex | usage |
|---|---|---|
| `--bg` | `#0a0e10` | page + canvas background |
| `--accent` (cyan) | `#00e5d1` | primary glyph, ring accents |
| `--magenta` | `#ff3d9a` | counter-cycle glow, tagline accent |
| `--text` | `#e6edef` | body text |
| `--text-subtle` | `#565d61` | muted tagline |

## Files

### `discord-server-icon-256.gif`

Animated server icon on the 1bit.systems Discord guild (`1488323665836642348`).

- 256×256, 24-frame loop (~1 s), ~48 KB
- Bold `1` (Fira Sans Condensed ExtraBold, 170 pt) + subscript `bs` (Noto Serif Regular, 85 pt — exactly half) with a ~20 px gap
- Glyphs pulse cyan↔magenta via sinusoidal interp; ring counter-cycles magenta↔cyan
- Regenerate: re-run the `/tmp/brand/frames.sh` builder or replicate with `magick` + a 24-frame loop; upload via `PATCH /guilds/{id}` with `icon: data:image/gif;base64,...`

Current upload hash on guild: `a_ddd473dc6d25e9c2035d54037adf23ef`.

### `discord-invite-splash-1920x1080.png`

Invite-link background on the same guild.

- 1920×1080 PNG, ~150 KB
- Big cyan `1bit.systems` wordmark, magenta tagline *"local ternary AI · bare metal · no cloud"*, muted accent credits line, cyan separator bar
- Regenerate: single `magick -size 1920x1080 xc:#0a0e10 ... -annotate ...` invocation; see `scripts/` parent for the one-liner if it gets factored out

Current upload hash on guild: `38ba2d6bfc73436c4330e31ec462f708`.

## Re-upload

Both uploaded via the echo bot (has `MANAGE_GUILD` perm on the guild). Token lives in
`~/.config/systemd/user/strix-watch-discord.service.d/token.conf`.

```bash
ECHO_TOKEN=...  # from the drop-in
GUILD=1488323665836642348

# build JSON body (needed because base64 blob overflows argv on some systems)
python3 -c '
import base64, json
with open("discord-server-icon-256.gif","rb") as f:
    d = base64.b64encode(f.read()).decode()
json.dump({"icon": f"data:image/gif;base64,{d}"}, open("/tmp/icon-body.json","w"))'

curl -X PATCH -H "Authorization: Bot $ECHO_TOKEN" -H "Content-Type: application/json" \
  --data-binary @/tmp/icon-body.json \
  "https://discord.com/api/v10/guilds/$GUILD"
```

## Future uses

- GitHub social preview (1280×640 derivative)
- Favicon derivatives (32, 48, 192, 512 PNG + ICO)
- Twitter/Bluesky banner if the stack ever gets a social presence
- YouTube channel art if we ever publish bench videos

None wired today — when any of those land, derive from these sources, keep the token palette intact.
