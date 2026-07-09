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

Keys are stored in the **system keyring** (GNOME Keyring / D-Bus Secret Service)
using `secret-tool`. This means:
- ✅ Encrypted at rest (your login session unlocks it)
- ✅ Survives reboots
- ✅ Accessible from scripts via `_loader.sh`
- ✅ No plaintext files with secrets

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
