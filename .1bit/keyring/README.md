# 🔑 1bit Keyring — Single Source of Truth

One ring to store them all. API keys, tokens, wallets, credentials —
never hunt for another key.

## Location

```
~/.1bit/keyring/
├── README.md         # This file
├── setup.sh          # First-time setup wizard
├── kr                # CLI: get/set/list/delete
├── _loader.sh        # Source this for script access
├── inventory.md      # What's stored and where it's used
├── services.json     # Registry of all services
└── .secrets/         # Encrypted secret store (locked)
```

## Quick Start

```bash
# First time: run the setup wizard
bash ~/.1bit/keyring/setup.sh

# Use the CLI
source ~/.1bit/keyring/kr
kr set github token ghp_xxxx        # Store
kr get github token                 # Retrieve
kr list                             # List all
kr delete github token              # Remove
```

## How It Works

**No custom vault.** Everything uses the **system keyring** that's already on your machine:
- Ubuntu/GNOME: `gnome-keyring` via D-Bus Secret Service
- KDE: `kwallet` via Secret Service adapter
- Headless: `keepassxc` or `secretservice`

The `kr` CLI is just a thin wrapper around `secret-tool` (libsecret).

- ✅ Encrypted at rest by the OS (unlocked at login)
- ✅ Survives reboots
- ✅ Same keyring Firefox, Chrome, and Git use
- ✅ No custom encryption, no new daemons, no plaintext files

Each service has a namespace. Common namespaces:

| Namespace | What's stored |
|-----------|---------------|
| `github` | Tokens for github.com accounts |
| `cloudflare` | API tokens, zone IDs, account IDs |
| `ai` | Provider API keys (DeepSeek, OpenAI, Anthropic, etc.) |
| `wallet` | Crypto wallet keys, seed phrases |
| `protonmail` | Email credentials |
| `huggingface` | HF API tokens |
| `docker` | Docker Hub / registry credentials |
| `ssh` | SSH key passwords |
| `server` | Server login credentials |

## Usage in Scripts

```bash
source ~/.1bit/keyring/_loader.sh

# Will try: keyring → env var → plaintext file
TOKEN=$(get_secret github token)
echo "Token: ${TOKEN:0:8}..."
```
