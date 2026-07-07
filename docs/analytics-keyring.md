# Analytics Keyring Setup

Analytics secrets (Cloudflare API token, Cloudflare account ID, zone ID, Web Analytics token) are stored in the system **GNOME Keyring** and accessed via `secret-tool` / `python-keyring`. This keeps secrets off disk and away from `.env` files.

## Current Secrets Inventory

| Secret | Keyring Service | Keyring Account/Key | Source |
|--------|----------------|---------------------|--------|
| Cloudflare API Token | `cloudflare-analytics` | `api-token` / `token` | Cloudflare Dashboard → Manage API Tokens |
| Cloudflare Account ID | `cloudflare-analytics` | `account-id` | Cloudflare Dashboard → Account Home |
| Cloudflare Zone ID (1bit.systems) | `cloudflare-analytics` | `zone` | Cloudflare Dashboard → 1bit.systems → Overview |
| Cloudflare Web Analytics Token | injected into HTML | — | Cloudflare Web Analytics → 1bit.systems |

> **Note:** The current API token has Zone-level permissions but **not** Cloudflare Pages scope.
> For Pages project/deployment data, use a token with `Cloudflare Pages:Read` permission.

## Prerequisites

GNOME Keyring and `secret-tool` are pre-installed on the system:

```bash
# Verify
dpkg -l | grep gnome-keyring   # gnome-keyring, libpam-gnome-keyring
which secret-tool               # /usr/bin/secret-tool
```

The Python keyring library is available in the `.gaia-venv`:

```bash
/home/bcloud/.gaia-venv/bin/python3 -c "import keyring; print(keyring.get_keyring().__class__.__name__)"
# Expected: Keyring (SecretService backend)
```

## Keyring Architecture

```
GNOME Keyring Daemon (gnome-keyring-daemon)
  └── SecretService DBus API (/run/user/1000/bus)
       ├── secret-tool (CLI)
       └── python-keyring (Python lib)
```

Files on disk (encrypted):
- `~/.local/share/keyrings/login.keyring` — user login keyring
- `~/.local/share/keyrings/user.keystore` — keystore metadata

The keyring is **unlocked at login** via PAM (`libpam-gnome-keyring`). No manual unlock needed.

## Storing Analytics Secrets

### Cloudflare API Token

```bash
secret-tool store --label="Cloudflare API Token" \
  service cloudflare-analytics \
  account api-token
# Paste the API token from Cloudflare Dashboard → Manage API Tokens
```

### Cloudflare Account ID

```bash
secret-tool store --label="Cloudflare Account ID" \
  service cloudflare-analytics \
  account account-id
# Paste the 32-char account ID from Cloudflare Dashboard → Account Home
```

### Cloudflare Zone ID (1bit.systems)

```bash
secret-tool store --label="Cloudflare Zone ID" \
  service cloudflare-analytics \
  key zone
# Paste the 32-char zone ID from Cloudflare Dashboard → 1bit.systems → Overview
```

### GitHub Token (optional, for local traffic stats)

```bash
secret-tool store --label="GitHub Token" \
  service github-analytics \
  account pat
# (paste GitHub PAT with repo:read + metadata:read)
```

## Verifying Stored Secrets

```bash
# List all analytics secrets
secret-tool search --all service cloudflare-analytics

# Look up a specific secret
secret-tool lookup service cloudflare-analytics account api-token
```

## Using in Scripts

### CLI (secret-tool)

```bash
CF_TOKEN=$(secret-tool lookup service cloudflare-analytics account api-token)
CF_ACCOUNT=$(secret-tool lookup service cloudflare-analytics account account-id)

curl -H "Authorization: Bearer $CF_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT/pages/projects"
```

### Python (keyring)

```python
import keyring

cf_token = keyring.get_password("cloudflare-analytics", "api-token")
cf_account = keyring.get_password("cloudflare-analytics", "account-id")
```

## Running Analytics Locally

Once secrets are stored in the keyring, use the helper script:

```bash
./scripts/analytics-keyring.sh
```

This loads secrets from the keyring and runs the full analytics collection pipeline locally, outputting `site/analytics-data.json`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `secret-tool: Cannot autolaunch D-Bus` | Ensure `$DBUS_SESSION_BUS_ADDRESS` is set (should auto-set at GUI login) |
| Keyring locked | Run `gnome-keyring-daemon --unlock` (should auto-unlock at login) |
| `secret-tool lookup` returns nothing | Secret not stored yet — use `secret-tool store` |
| `org.freedesktop.DBus.Error.UnknownMethod` | Keyring daemon restart needed: `gnome-keyring-daemon --replace --components=secrets` |

## Security Notes

- Keyring is **encrypted at rest** with your login password
- No secrets appear in environment variables when not in use
- Keyring auto-locks on screen lock / suspend
- Secrets are only accessible while your session is active
