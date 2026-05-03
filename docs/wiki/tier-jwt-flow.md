# Tier-JWT issuance flow — 1bit.systems Premium

Status: draft v0.1, 2026-04-23.

How a paying customer or machine client turns money into a `tier: premium`
JWT that unlocks `/v1/catalogs/:slug/lossless` on `1bit-stream`. The
original payment rail is BTCPay (Lightning); the JWT shape is open to
additional rails later without changing the stream-server side.

2026-05-03 side-project note: for future machine-to-machine inference/API
access, a stablecoin settlement rail is a serious alternative to batch token
purchases. This is not part of the current 1bit inference stack. The benefit
is not ideology; it is latency and operational shape. A machine can settle
small balances continuously or near-continuously, then receive a short-lived
JWT/capability without waiting for a human-sized prepaid credit batch to
clear, reconcile, and be reloaded into an account ledger.

The minting side is implemented by `cpp/tier-mint`, running
as `1bit-tier-mint.service` on `127.0.0.1:8151` (one port above
`1bit-stream`). The stream server verifies the JWT in-process — no
IPC between the two beyond a shared HMAC secret in env.

---

## 1. End-to-end flow

### 1a. BTCPay Lightning

```
                                 1bit-stream (8150)
                                       ▲
                                       │ Authorization: Bearer <jwt>
                                       │
  ┌──────────┐      ┌──────────────┐   │   ┌─────────────────┐
  │ customer │──1──▶│ storefront   │   │   │ 1bit-tier-mint  │
  │ browser  │      │ (static SPA) │   │   │    (8151)       │
  └──────────┘      └──────┬───────┘   │   └─────────────────┘
        │                  │ 2                       ▲   │
        │                  ▼                         │   │
        │           ┌───────────────┐    4 verify    │   │
        │           │ BTCPay Server │◀───────────────┘   │
        │           │  (sliger)     │                    │
        │           └──────┬────────┘                    │
        │                  │ 3 Lightning invoice         │
        │◀─────────────────┘                             │
        │                                                │
        │ 5 pays invoice (LN, seconds)                   │
        ├───────────────────────────────▶                │
        │                                                │
        │                              6 InvoiceSettled  │
        │                                 webhook        │
        │                             (BTCPay-Sig HMAC)  │
        │                              ───────────────▶  │
        │                                                │
        │  7 GET /tier/poll/:invoice_id                  │
        ├────────────────────────────────────────────────▶
        │  8 200 { jwt: "..." }                          │
        │◀───────────────────────────────────────────────┘
        │
        │  9 GET /v1/catalogs/<slug>/lossless
        │     Authorization: Bearer <jwt>
        ▼
   1bit-stream verifies → 200 .1bl residual bytes
```

1. Customer clicks "Go Premium" on the storefront (static SPA,
   served from the same origin as `1bit-stream` so no CORS).
2. Storefront `POST`s to BTCPay's `/invoices` API to open a new
   invoice, receives an invoice id.
3. BTCPay returns a BOLT11 Lightning invoice (QR + text).
4. Storefront displays the QR; begins polling
   `GET /tier/poll/:invoice_id` on tier-mint every ~1.5 s.
5. Customer scans → wallet pays → c-lightning on sliger receives.
6. BTCPay fires `InvoiceSettled` webhook at tier-mint. The request
   body is HMAC-SHA256-signed with the per-store webhook secret
   (`HALO_BTCPAY_WEBHOOK_SECRET`). Tier-mint verifies, mints a JWT,
   writes it to the poll cache keyed by invoice id, and returns the
   token in the response body (so BTCPay is the secondary delivery
   path — the poll cache is canonical because BTCPay's UI doesn't
   forward response bodies to the customer).
7. Storefront's next poll hits the cache.
8. Cache hit → 200 + `{ "jwt": "..." }`. Cache entry is removed on
   first read; repeat polls get 202.
9. Customer's player fetches `/v1/catalogs/<slug>/lossless` with
   the JWT; stream server verifies signature, tier, expiry, and the
   revocation list before streaming the residual section.

Lightning is final-on-confirm; there is no refund path. See §6.

### 1b. Stablecoin rail for machine clients

Same minting contract, different settlement backend:

```text
machine client -> payment rail -> tier-mint -> short-lived JWT -> API call
```

The point is to avoid the old "buy a batch of tokens, wait for account credit,
then spend down a local balance" workflow. For machine clients, that batch
workflow adds coordination latency and forces a prepaid inventory model. A
stablecoin rail can instead:

- settle per job, per session, or per small balance threshold;
- mint a scoped JWT as soon as settlement is observed;
- keep usage authorization close to the API call;
- avoid long-lived account balances when the caller is another machine.

This is not implemented in the current runtime. It is the preferred design
direction if we add paid machine-to-machine API access.

---

## 2. JWT shape

Algorithm: **HS256**. Symmetric HMAC shared between tier-mint and
stream server via env. No asymmetric keys — the stream server has to
trust tier-mint anyway (they run on the same box under the same user),
so ECDSA would just mean more key management for no threat-model win.

### Header

```json
{ "alg": "HS256", "typ": "JWT" }
```

### Payload

```json
{
  "sub":               "inv-01GXYZ...",        // invoice id
  "tier":              "premium",
  "iss":               "1bit.systems",
  "exp":               1745000000,              // unix, ~30 days from iat
  "iat":               1742500000,
  "jti":               "inv-01GXYZ...",        // = sub; used as revoke key
  "btcpay_invoice":    "inv-01GXYZ..."
}
```

Claims in the struct but omitted when empty via `skip_serializing_if`.
`btcpay_invoice` records the provenance of the mint so the stream
server (and the revoke endpoint) can attribute it back.

### Stream-server verify rules

- Signature valid under current or previous HMAC secret (dual-validation
  window; see §3).
- `exp > now`.
- `iss == "1bit.systems"`.
- `tier == "premium"`.
- `jti` is not in the revocation list.

All five must hold. Any failure → 401. Expired JWTs can be re-minted
via a future `/tier/refresh` endpoint that takes the original invoice
id; not in the MVP skeleton.

---

## 3. Key rotation

`HALO_TIER_HMAC_SECRET` rotates every **90 days** by default, or
immediately on any suspected compromise.

### Mechanics

Rotation is dual-key: during the overlap window the stream server
accepts JWTs signed under either the current or previous secret.

- Day 0: new secret `K_new` generated (256 bits from `/dev/urandom`).
  `HALO_TIER_HMAC_SECRET=K_new` on tier-mint. Stream server's
  env file holds `HALO_TIER_HMAC_SECRET=K_new` *and*
  `HALO_TIER_HMAC_SECRET_PREV=K_old` for a 30-day overlap.
- All newly minted JWTs use `K_new`.
- Day 30: `K_old` is removed from stream server env. Any JWT still
  in flight that was signed under `K_old` is now invalid — the
  customer's player retries, polls `/tier/poll/:invoice_id` (which
  re-mints off the original invoice id since it's still in the
  paid set), gets a fresh token.
- Days 31-89: single-key regime, steady state.

### Where to rotate

1. Generate `K_new`: `openssl rand -hex 32`.
2. Edit `~/.config/1bit/tier-mint.env`: set
   `HALO_TIER_HMAC_SECRET=<K_new>`.
3. Edit `~/.config/1bit/stream.env`: set
   `HALO_TIER_HMAC_SECRET=<K_new>` and
   `HALO_TIER_HMAC_SECRET_PREV=<K_old>`.
4. `systemctl --user restart 1bit-tier-mint.service 1bit-stream.service`.
5. Calendar-note to remove `_PREV` in 30 days.

### Why 90 days, not 30?

Balancing: shorter windows mean more rotation pain (manual), longer
means a stolen secret has more dwell time. 90 days is the same cadence
AWS IAM recommends for access keys. If we ever wire automated rotation
(systemd timer + Vault / file-watcher), drop to 30.

---

## 4. Webhook authentication

### BTCPay (`/btcpay/webhook`)

- Header: `BTCPay-Sig` (casing varies — we check multiple spellings).
- Algorithm: **HMAC-SHA256**, hex, prefixed `sha256=`.
- Secret: configured per-store in the BTCPay admin UI. Stored in our
  env as `HALO_BTCPAY_WEBHOOK_SECRET`.
- We compare using `hmac::Hmac<Sha256>::verify_slice` which is
  constant-time.

**Open question:** BTCPay's public docs call the header
`BTCPay-Sig` but some integrations see `BTCPAY-SIG` and newer
versions may use `btcpay-sig`. We check all three. The `sha256=`
prefix is also inconsistent across versions — we tolerate bare-hex.
Lock the exact spelling after the first real `InvoiceSettled` hits
staging.

---

## 5. Refund / chargeback / revoke

| Rail | Reversible? | How we handle it |
|------|-------------|------------------|
| BTCPay Lightning | **No.** Settled on-chain (well, off-chain final) in seconds. | Once minted, the JWT is good for its `exp`. No revocation path unless we manually hit `/tier/revoke`. |
| Admin override | — | `POST /tier/revoke { id, admin_token }`. For bad actors / abuse / leaked tokens. |

The revocation list is the single source of truth on the stream
server. In MVP it's in-memory + a TODO to persist to sqlite. Before
shipping to actual paying customers: persist the revoke list.

---

## 6. Storage model

We intentionally **do not store minted JWTs** after issuance:

- The webhook response carries the JWT once.
- The poll cache holds it for ≤ 10 minutes and deletes on first read.
- If the customer loses the token before getting it into their player,
  re-polling `/tier/poll/:invoice_id` after expiry returns 410. They
  contact support and we re-mint manually via a CLI that takes the
  invoice id.

What we **do** store:

- The **revocation list** (keyed by invoice id). Write-through
  in-memory now, sqlite on disk soon.
- BTCPay keeps its own invoice ledger; we never duplicate.

This keeps the service stateless enough that restarts are free and
there's no PII on our box beyond what BTCPay already has.

---

## 7. Threat model

| Threat | Mitigation |
|--------|------------|
| **Replay** — attacker records an old webhook and re-plays it. | BTCPay includes a timestamp and a unique delivery id; tier-mint is idempotent on `invoice_id` (duplicate mints overwrite but the revoke list is keyed the same, so nothing is gained). We could optionally track seen delivery ids in a short-TTL bloom filter — deferred. |
| **Forged webhook** — attacker POSTs a fake `InvoiceSettled` from the open internet. | HMAC verification rejects anything not signed under the shared secret. Caddy / cloudflared both front the service on 127.0.0.1 so unauthenticated traffic can't even reach it. |
| **Stolen JWT** — user shares their token on Discord. | Short-ish `exp` (30 days). Admin can `/tier/revoke` the `jti` which invalidates all existing JWTs from that invoice. Future: bind the JWT to a client fingerprint (UA + install id) — but that defeats the "just a bearer" simplicity. |
| **Leaked signing secret (`HALO_TIER_HMAC_SECRET`)** | Quarterly rotation (§3). Dual-key overlap means rotation is not load-bearing on customer uptime — worst case, player retries once and re-polls. |
| **Leaked BTCPay webhook secret** | Per-store; rotated via BTCPay UI. Attacker could forge webhooks, mint JWTs for arbitrary invoice ids, and collect Premium access. Not a money loss but a tier-fraud issue. Detection: webhook deliveries that don't match any real invoice in BTCPay's own ledger. |
| **Tier-mint goes down during paid flow** | BTCPay retries webhooks with exponential backoff for ~24 h. Customer's browser keeps polling `/tier/poll`. As long as tier-mint comes back before BTCPay gives up, the flow completes. If it doesn't, admin CLI can re-mint from the invoice id. |
| **Compromised storefront SPA** | Storefront is a static file served from the same origin as `1bit-stream`. Any XSS in the stream server is game-over anyway. Storefront touches no secrets. |

Not-mitigated-and-we-know-it:

- **Coerced customer.** If someone is held at gunpoint to buy Premium
  then hand over the JWT, nothing here helps.
- **Insider at sliger.** Root on sliger can read the BTCPay store key
  and forge invoices; shared physical trust with the host.

---

## 8. Secrets

All live in **`~/.config/1bit/tier-mint.env`** (mode 0600, not in
git). The systemd unit loads via `EnvironmentFile=`.

```sh
# ~/.config/1bit/tier-mint.env   (placeholder — real values never in repo)
HALO_TIER_HMAC_SECRET=REPLACE_WITH_OPENSSL_RAND_HEX_32
HALO_BTCPAY_WEBHOOK_SECRET=REPLACE_WITH_BTCPAY_STORE_SECRET
HALO_TIER_LISTEN=127.0.0.1:8151
RUST_LOG=info
```

Generation:

```bash
install -m 0700 -d ~/.config/1bit
umask 077
cat > ~/.config/1bit/tier-mint.env <<EOF
HALO_TIER_HMAC_SECRET=$(openssl rand -hex 32)
HALO_BTCPAY_WEBHOOK_SECRET=<paste from BTCPay admin>
HALO_TIER_LISTEN=127.0.0.1:8151
EOF
chmod 600 ~/.config/1bit/tier-mint.env
```

The stream server reads `HALO_TIER_HMAC_SECRET` out of its own env
file (`~/.config/1bit/stream.env`) — same value, duplicated. Keep
them in sync during rotation (§3).

Placeholders committed to the repo: only the unit file references
the env path. No real secret ever enters git; if one does, rotate
immediately and `git filter-repo` the history.

---

## 9. Open questions

1. **BTCPay signature header exact format.** Multiple spellings and
   prefix conventions seen in different BTCPay versions. Test with
   real staging webhook against the live sliger instance, lock the
   string.
2. **Idempotent webhook delivery.** BTCPay does retry on non-2xx;
   do duplicate `InvoiceSettled` events for the same invoice cause
   two mints? Current code: yes, but both produce the same JWT, and
   the poll cache just overwrites. Safe, but worth a bloom filter if
   we see it in practice.
3. **Refresh endpoint.** `POST /tier/refresh { jwt }` that re-mints
   against the same `jti` if the invoice is still not revoked. Not
   in MVP; customers currently re-poll off the original invoice.
4. **Admin auth.** `/tier/revoke` compares admin token against the
   JWT signing secret as a stopgap. Needs its own key soon.

---

## 10. Related

- `docs/wiki/1bl-container-spec.md` — the `.1bl` format that Premium
  unlocks the lossless residual of.
- `cpp/tier-mint/` — the implementation.
- `strixhalo/systemd/user/1bit-tier-mint.service` — unit file.
