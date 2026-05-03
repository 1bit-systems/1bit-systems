---
phase: historical
owner: scribe
---

# Historical: 1bit-landing

`1bit-landing` was the earlier LAN dashboard plan. Current public website work lives in `1bit-site/`, and current operational checks are done through:

```bash
1bit status
systemctl status lemond flm 1bit-proxy open-webui
```

Current runtime endpoints:

| Service | URL |
|---|---|
| `1bit-proxy` | `http://127.0.0.1:13306/v1` and `/api/v1` |
| Lemonade | `http://127.0.0.1:13305/api/v1` |
| FastFlowLM | `http://127.0.0.1:52625/v1` |
| Open WebUI | `http://127.0.0.1:3000` |

Do not treat the older `strix-landing` telemetry design as the live dashboard unless it is revived and rewired to the current services.
