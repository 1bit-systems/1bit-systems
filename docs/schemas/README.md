# 1bit-systems JSON Schemas

Canonical JSON Schema (Draft 2020-12) definitions for public 1bit-systems
data shapes. Hand-authored from the prose specs; update both when either
side changes.

## Files

| File | Describes |
|---|---|
| `1bl-manifest.schema.json` | CBOR-encoded manifest inside a `.1bl` catalog container. Logical shape only — CBOR ↔ JSON is isomorphic for this data. Source of truth: [`docs/wiki/1bl-container-spec.md`](../wiki/1bl-container-spec.md). |
| `catalog-index.schema.json` | `GET /v1/catalogs` response body served by `1bit-stream`. Array of catalog summary rows. |
| `examples/valid-kevin-macleod.json` | Passing fixture — lossy-only tier, no residual. |
| `examples/valid-classical-pack.json` | Passing fixture — premium tier with residual section. |

## Publishing

Schemas are served at the canonical `$id` URLs:

- `https://1bit.systems/schemas/1bl-manifest.schema.json`
- `https://1bit.systems/schemas/catalog-index.schema.json`

The 1bit.systems site serves the raw JSON directly from this directory so
`$id` resolves via HTTP GET (served with `Content-Type: application/schema+json`).

## Validating

Any Draft 2020-12 validator works. The fixtures in `examples/` are the
canonical smoke test.

### Python — `jsonschema`

```bash
pip install jsonschema
python3 - <<'PY'
import json, pathlib
from jsonschema import Draft202012Validator
root = pathlib.Path("docs/schemas")
schema = json.loads((root / "1bl-manifest.schema.json").read_text())
for ex in (root / "examples").glob("valid-*.json"):
    Draft202012Validator(schema).validate(json.loads(ex.read_text()))
    print(f"OK  {ex.name}")
PY
```

### Node — `ajv-cli`

```bash
npm i -g ajv-cli ajv-formats
ajv validate --spec=draft2020 -c ajv-formats \
  -s docs/schemas/1bl-manifest.schema.json \
  -d "docs/schemas/examples/valid-*.json"
```

### `check-jsonschema`

```bash
pipx install check-jsonschema
check-jsonschema \
  --schemafile docs/schemas/1bl-manifest.schema.json \
  docs/schemas/examples/valid-*.json
```

## License

The schemas themselves are licensed **CC BY 4.0** (standard for published
JSON Schema files). Implementations consuming them carry their own
license — this directory only governs the schema documents.
