# 🔑 1bit Keyring — Inventory

> Generated: 2026-07-08
> Managed at: `~/.1bit/keyring/`

## Current Keyring Contents

| # | Service | Key | Label | Status |
|---|---------|-----|-------|--------|
| 1 | `firefox` | `app_id` | Firefox key | ✅ Stored |
| 2 | `gh:github.com` | `bong-water-water-bong` | GitHub CLI token | ✅ Stored |
| 3 | `gh:github.com` | `` | GitHub fallback | ✅ Stored |
| 4 | `cloudflare-analytics` | `key=token` | CF API token | ✅ Stored |
| 5 | `cloudflare-analytics` | `key=zone` | CF zone ID | ✅ Stored |
| 6 | `cloudflare-analytics` | `account=api-token` | CF API Token | ✅ Stored |
| 7 | `cloudflare-analytics` | `account=account-id` | CF Account ID | ✅ Stored |
| 8 | `protonmail/bridge-v3/users/bridge-vault-key` | `bridge-vault-key` | ProtonMail bridge vault key | ✅ Stored |

## Services to Add

| Service | What | Priority | Setup Command |
|---------|------|----------|---------------|
| `ai/deepseek` | DeepSeek API key | 🔴 High | `kr set ai deepseek` |
| `ai/openai` | OpenAI API key | 🟡 Medium | `kr set ai openai` |
| `ai/anthropic` | Anthropic API key (Claude) | 🟡 Medium | `kr set ai anthropic` |
| `ai/mistral` | Mistral API key | 🟢 Low | `kr set ai mistral` |
| `wallet/eth` | Ethereum wallet key | 🟡 Medium | `kr set wallet eth` |
| `wallet/solana` | Solana wallet key | 🟢 Low | `kr set wallet solana` |
| `huggingface` | Hugging Face token | 🟡 Medium | `kr set huggingface token` |
| `docker` | Docker Hub token | 🟢 Low | `kr set docker token` |
| `server/bcloud` | Server SSH/root creds | 🟡 Medium | `kr set server bcloud` |
| `email/smtp` | SMTP credentials | 🟢 Low | `kr set email smtp` |

## Migration Plan

1. ✅ GitHub token → `gh:github.com` (already in keyring)
2. ✅ Cloudflare tokens → `cloudflare-analytics` (already in keyring)
3. ✅ ProtonMail bridge → `protonmail/bridge-v3` (already in keyring)
4. ☐ DeepSeek API → `ai/deepseek`
5. ☐ All provider keys → `ai/*`
6. ☐ Wallet keys → `wallet/*`
