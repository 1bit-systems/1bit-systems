### Task 1: Dynamic Model Discovery (`models.ts` rewrite)

**Files:**
- Modify: `/home/bcloud/npu-sandbox/npu-infer/1bit/src/models.ts`

**Interfaces:**
- Consumes: `~/.config/flm/models/*/config.json`, Q4NX header
- Produces: `ModelInfo[]` with all fields populated from config.json, no hardcoded model list

**Details:**
The current `KNOWN_MODELS` and `knownDim` dicts are the primary bottleneck. Replace with:

- Scan `~/.config/flm/models/*/` for directories
- For each, check for `model.q4nx` (or any `*.q4nx`)
- Read `config.json` in the same dir for:
  - `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `vocab_size`, `model_type`, `max_position_embeddings`, `rope_theta`, `rms_norm_eps`
- Generate a tag from `{model_type}_{hidden_size}` (e.g. `qwen3_1024`, `llama_4096`)
- Verify xclbin set exists for that tag
- If no xclbins match, try without tag suffix (generic `final_i8_QKV.xclbin` etc)
- Add a `needsXclbins` field so user knows what's missing

- [ ] **Step 1: Read the current `models.ts` file**

```bash
cat /home/bcloud/npu-sandbox/npu-infer/1bit/src/models.ts
```

- [ ] **Step 2: Write the new `discoverModels()` — dynamic directory scan + config.json reader**

```typescript
import { readFileSync, readdirSync, existsSync, openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";

export interface ModelInfo {
  id: string;            // model ID for OpenAI API
  name: string;          // display name
  modelPath: string;     // path to model.q4nx
  tokenizerPath: string; // path to dir containing tokenizer.json
  configPath: string;    // path to config.json
  modelType: string;     // "qwen2", "llama", "gemma2", etc.
  tag: string;           // derived tag for xclbin selection
  dimensions: {
    H: number; NC: number; NH: number; NKV: number;
    HD: number; IM: number; NV: number;
  };
  architecture: {
    q_norm: boolean; k_norm: boolean;
    rope_freqs: boolean; lm_head: boolean; gu_split: boolean;
  };
  status: "ready" | "no_xclbins" | "no_model_file" | "no_config";
  needsXclbins: string[];  // list of missing xclbin operation names
}

const MODELS_BASE = "/home/bcloud/.config/flm/models";
const MODELS_BASE_FALLBACK = "/home/bcloud/models";
const XCLBIN_DIR = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
const ENGINE = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt";

interface HfConfig {
  hidden_size?: number;
  intermediate_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  vocab_size?: number;
  model_type?: string;
  max_position_embeddings?: number;
  rope_theta?: number;
  rms_norm_eps?: number;
  // Qwen-specific fields:
  num_layers?: number;
  // Gemma-specific:
  head_dim?: number;
}

function readConfig(dir: string): HfConfig | null {
  try {
    const raw = readFileSync(join(dir, "config.json"), "utf-8");
    return JSON.parse(raw);
  } catch { return null; }
}

function findQ4NXFile(dir: string): string | null {
  try {
    const entries = readdirSync(dir);
    const q4nx = entries.find(e => e.endsWith(".q4nx"));
    return q4nx ? join(dir, q4nx) : null;
  } catch { return null; }
}

function deriveTag(modelType: string, H: number): string {
  // Known model dims get their existing tag for xclbin compatibility
  const known: Record<string, Record<number, string>> = {
    qwen2: { 1024: "qwen3_0_6b", 2560: "qwen3_vl_4b", 4096: "qwen3_8b" },
    llama: { 4096: "llama" },
    gemma2: { 1536: "gemma4_e2b", 2304: "gemma4_e2b" },
    qwen3: { 1024: "qwen3_0_6b", 2560: "qwen3_vl_4b", 4096: "qwen3_8b" },
  };
  return known[modelType]?.[H] || `${modelType}_${H}`;
}

function requiredOps(H: number, IM: number): string[] {
  const guSplit = (2 * IM > 14336);
  const base = ["QKV", "O", "D"];
  if (guSplit) return [...base, "G", "U"];
  return [...base, "GU"];
}

function checkXclbins(tag: string, ops: string[]): { status: "ready" | "no_xclbins"; missing: string[] } {
  const missing: string[] = [];
  for (const op of ops) {
    // Try tag-specific first, then generic
    const specificPath = join(XCLBIN_DIR, `final_i8_${op}_${tag}.xclbin`);
    const genericPath = join(XCLBIN_DIR, `final_i8_${op}.xclbin`);
    if (!existsSync(specificPath) && !existsSync(genericPath)) {
      missing.push(op);
    }
  }
  return { status: missing.length ? "no_xclbins" : "ready", missing };
}

function parseQ4NXHeader(filePath: string): { model_type: string; tensors: Record<string, any> } | null {
  try {
    const fd = openSync(filePath, "r");
    try {
      const hdrSizeBuf = Buffer.alloc(8);
      readSync(fd, hdrSizeBuf, 0, 8, 0);
      const hdrSize = Number(hdrSizeBuf.readBigUInt64LE(0));
      if (hdrSize <= 0 || hdrSize > 1024 * 1024) return null;
      const headerBuf = Buffer.alloc(hdrSize);
      readSync(fd, headerBuf, 0, hdrSize, 8);
      const header = headerBuf.toString("utf-8").replace(/\x00+$/, "");
      return JSON.parse(header);
    } finally { closeSync(fd); }
  } catch { return null; }
}

export function discoverModels(): ModelInfo[] {
  const modelDirs = new Map<string, string>();

  // Collect all model directories from primary and fallback paths
  for (const base of [MODELS_BASE, MODELS_BASE_FALLBACK]) {
    try {
      const entries = readdirSync(base);
      for (const entry of entries) {
        const fullPath = join(base, entry);
        if (!modelDirs.has(entry)) modelDirs.set(entry, fullPath);
      }
    } catch { /* path may not exist */ }
  }

  const models: ModelInfo[] = [];

  for (const [dirName, dirPath] of modelDirs) {
    const q4nxPath = findQ4NXFile(dirPath);
    const config = readConfig(dirPath);

    if (!q4nxPath) {
      models.push({
        id: dirName,
        name: dirName,
        modelPath: dirPath,
        tokenizerPath: dirPath,
        configPath: join(dirPath, "config.json"),
        modelType: config?.model_type || "unknown",
        tag: "unknown",
        dimensions: { H: 0, NC: 0, NH: 0, NKV: 0, HD: 128, IM: 0, NV: 0 },
        architecture: { q_norm: false, k_norm: false, rope_freqs: false, lm_head: false, gu_split: false },
        status: config ? "no_model_file" : "no_config",
        needsXclbins: [],
      });
      continue;
    }

    if (!config) {
      models.push({
        id: dirName,
        name: dirName,
        modelPath: q4nxPath,
        tokenizerPath: dirPath,
        configPath: join(dirPath, "config.json"),
        modelType: "unknown",
        tag: "unknown",
        dimensions: { H: 0, NC: 0, NH: 0, NKV: 0, HD: 128, IM: 0, NV: 0 },
        architecture: { q_norm: false, k_norm: false, rope_freqs: false, lm_head: false, gu_split: false },
        status: "no_config",
        needsXclbins: [],
      });
      continue;
    }

    const H = config.hidden_size ?? 0;
    const IM = config.intermediate_size ?? 0;
    const NC = config.num_hidden_layers ?? config.num_layers ?? 0;
    const NH = config.num_attention_heads ?? 0;
    const NKV = config.num_key_value_heads ?? NH;
    const HD = config.head_dim ?? (H / NH);
    const NV = config.vocab_size ?? 0;
    const modelType = config.model_type ?? "unknown";

    const tag = deriveTag(modelType, H);
    const ops = requiredOps(H, IM);
    const { status, missing } = checkXclbins(tag, ops);

    const hasQNorm = modelType === "qwen2" || modelType === "qwen3" || modelType === "gemma2";
    const hasKNorm = modelType === "qwen2" || modelType === "qwen3" || modelType === "gemma2";
    const hasRope = modelType === "llama" || modelType === "gemma2" || modelType === "mistral";
    const guSplit = (2 * IM > 14336);

    models.push({
      id: dirName,
      name: dirName,
      modelPath: q4nxPath,
      tokenizerPath: dirPath,
      configPath: join(dirPath, "config.json"),
      modelType,
      tag,
      dimensions: { H, NC, NH, NKV, HD, IM, NV },
      architecture: {
        q_norm: hasQNorm, k_norm: hasKNorm,
        rope_freqs: hasRope, lm_head: false, gu_split: guSplit,
      },
      status,
      needsXclbins: missing,
    });
  }

  return models;
}

export function findModel(id: string): ModelInfo | undefined {
  return discoverModels().find(m => m.id === id);
}

export function getModels(): ModelInfo[] {
  return discoverModels();
}
```

- [ ] **Step 3: Test the new discovery against the actual models on disk**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/1bit
node -e "
import { discoverModels } from './dist/models.js';
const models = discoverModels();
console.log(JSON.stringify(models.map(m => ({
  id: m.id, type: m.modelType, tag: m.tag, status: m.status,
  dims: m.dimensions, missing: m.needsXclbins
})), null, 2));
"
```

Expected: all 5 existing models found with `"status": "ready"` and correct dims.

- [ ] **Step 4: Build and verify**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/1bit
npm run build
node -e "
import { discoverModels } from './dist/models.js';
const models = discoverModels();
console.log('Models found:', models.length);
models.forEach(m => console.log(\`  \${m.id}: \${m.status} tag=\${m.tag} \${m.dimensions.H}x\${m.dimensions.IM}\`));
"
```

- [ ] **Step 5: Commit**

```bash
git -C /home/bcloud/npu-sandbox/npu-infer/1bit add src/models.ts
git -C /home/bcloud/npu-sandbox/npu-infer/1bit commit -m "feat: dynamic model discovery from config.json

Replace hardcoded KNOWN_MODELS and knownDim dicts with dynamic
directory scan. Reads HF config.json to derive model dimensions,
type, and architecture flags. Generates xclbin tags from model
type + hidden_size. Falls back to generic xclbins when no
tag-specific binary exists."
```

---

